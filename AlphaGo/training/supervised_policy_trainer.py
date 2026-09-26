import os
import json
import time
import types

# BEFORE `import tensorflow`, deliberately. The flag must be in the environment ahead of
# the first XLA compilation, and importing TF first is exactly the ordering that was
# observed to defeat it (the autotuner then fails on the first jit_compile=True step).
# ensure_xla_conv_nhwc() warns if TF is already imported, so calling it after the import
# below would also make that warning fire on every normal run and train everyone to
# ignore it.
from AlphaGo.training.xla_workarounds import ensure_xla_conv_nhwc  # noqa: E402

ensure_xla_conv_nhwc()

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402,F401

from keras import mixed_precision, ops, utils as keras_utils  # noqa: E402
from keras.metrics import TopKCategoricalAccuracy  # noqa: E402
from keras.optimizers import SGD  # noqa: E402
from keras.optimizers.schedules import CosineDecay, LearningRateSchedule  # noqa: E402
from keras.callbacks import (  # noqa: E402
    ModelCheckpoint, Callback, ReduceLROnPlateau, TerminateOnNaN)
from AlphaGo.models.policy import CNNPolicy  # noqa: E402
# Unused directly, but importing it registers ResTowerPolicy (via the @neuralnet
# decorator) so CNNPolicy.load_model() can find it by name in a model.json's "class"
# field - registration only happens when a class's defining module is actually imported
# somewhere in the process, and nothing else here pulls this one in.
import AlphaGo.models.resnet_tower_policy  # noqa: E402,F401
from AlphaGo.training.shard_stream import (  # noqa: E402
    BATCH_TRANSFORMATIONS, find_split_shards, dataset_info, shard_batch_generator,
    validation_arrays)


def prediction_entropy(y_true, y_pred):
    """Shannon entropy of the predicted move distribution, averaged over the batch.

    Ignores y_true - a diagnostic on the model's own confidence, not its correctness.
    Starts near log(361) (uniform, untrained) and should fall as training progresses;
    comparing the train vs. val curves can also surface overconfidence before loss alone
    would show it.
    """
    p = ops.clip(y_pred, 1e-7, 1.0)
    return -ops.sum(p * ops.log(p), axis=-1)


def sanity_checked_generator(base_generator, out_directory, label, check_every=50):
    """Wraps a batch generator, periodically checking yielded batches for basic sanity
    (proper one-hot Y rows, valid X range, no NaN) and logging loudly the moment
    something looks wrong - catches data corruption at the exact step it starts, rather
    than only showing up in a loss/accuracy number at the next epoch boundary. This
    project has seen a training run stay stuck at the random-guessing baseline (loss ==
    entropy) from step 0 - epoch-boundary checks alone missed that entirely, since
    nothing ever *changed*, it just never looked right from the start. Checks every
    Nth step plus the first few unconditionally (to catch a from-the-start problem too)
    rather than every step, to keep this cheap.
    """
    log_path = os.path.join(out_directory, "batch_sanity_log.json")
    step = 0
    for X, Y in base_generator:
        step += 1
        if step % check_every == 0 or step <= 5:
            row_sums = Y.sum(axis=1)
            x_nan = int(np.isnan(X).sum())
            x_min, x_max = float(X.min()), float(X.max())
            y_min, y_max = float(row_sums.min()), float(row_sums.max())
            bad = (x_nan > 0 or x_min < -1e-3 or x_max > 1 + 1e-3
                   or abs(y_min - 1.0) > 1e-3 or abs(y_max - 1.0) > 1e-3)
            if bad:
                print("\n*** BAD BATCH detected: {} step {}: X in [{:.4f}, {:.4f}] "
                      "({} NaN), Y row-sums in [{:.4f}, {:.4f}] (should be exactly 1.0) "
                      "***".format(label, step, x_min, x_max, x_nan, y_min, y_max))
                entry = {"label": label, "step": step, "X_min": x_min, "X_max": x_max,
                         "X_nan_count": x_nan, "Y_row_sum_min": y_min, "Y_row_sum_max": y_max}
                existing = []
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        existing = json.load(f)
                existing.append(entry)
                with open(log_path, "w") as f:
                    json.dump(existing, f, indent=2)
        yield X, Y


class _ResumedLRSchedule(LearningRateSchedule):
    """Wraps a base schedule, shifting its step input by a fixed offset.

    Resuming only reloads model weights (save_weights_only=True, no optimizer state), so a
    resumed run's freshly-created SGD optimizer always starts its own step counter at 0.
    Without this, a resumed run would re-enter warmup and restart the cosine decay curve
    from its beginning every time instead of continuing the single, whole-run schedule
    total_steps was originally shaped for.
    """

    def __init__(self, base_schedule, offset):
        super().__init__()
        self.base_schedule = base_schedule
        self.offset = offset

    def __call__(self, step):
        return self.base_schedule(step + self.offset)

    def get_config(self):
        return {"base_schedule": self.base_schedule.get_config(), "offset": self.offset}


class WarmupCallback(Callback):
    """Linearly ramps the optimizer's learning rate from start_lr to target_lr over
    warmup_steps, then stops touching it.

    Only used for --lr-schedule plateau: ReduceLROnPlateau reduces the optimizer's
    learning_rate by direct assignment (new_lr = old_lr * factor), which requires it to
    be a plain mutable value rather than a LearningRateSchedule object - so unlike
    --lr-schedule cosine (where warmup is folded into CosineDecay's own
    warmup_steps/warmup_target), warmup has to happen here as a callback instead, before
    handing control of the (now plain) learning_rate over to ReduceLROnPlateau for the
    rest of the run.
    """

    def __init__(self, warmup_steps, start_lr, target_lr):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.target_lr = target_lr
        self._step = 0

    def on_train_batch_begin(self, batch, logs=None):
        if self._step > self.warmup_steps:
            return
        frac = self._step / max(1, self.warmup_steps)
        self.model.optimizer.learning_rate = self.start_lr + (self.target_lr - self.start_lr) * frac
        self._step += 1

    @property
    def is_done(self):
        """True once warmup has stopped touching the learning_rate - lets other
        epoch-boundary callbacks (LROverrideCallback) know it's safe to act without
        fighting warmup's own per-batch ramp."""
        return self._step > self.warmup_steps


class LROverrideCallback(Callback):
    """--lr-schedule plateau only: at each epoch boundary, check out_directory/
    lr_override.txt for a manually-specified learning rate - if the file exists and its
    value differs (beyond float-precision noise) from the optimizer's current
    learning_rate, force the optimizer to that value instead.

    Runs after ReduceLROnPlateau in the callback list, so an override always wins over
    whatever the plateau logic just decided that epoch. The check re-runs every epoch,
    so a standing file acts as a persistent pin (re-asserted for as long as it exists
    and disagrees), not a one-shot nudge - lets an LR be changed live mid-run (e.g. for
    a manual cut) without stopping the process, editing metadata.json, and resuming.

    Never applies while warmup_cb (if any) is still actively ramping the learning_rate -
    warmup owns it exclusively until done, the same way it already precedes
    ReduceLROnPlateau itself. Only meaningful under --lr-schedule plateau, where the
    optimizer's learning_rate is a plain mutable value; --lr-schedule cosine drives it
    from a CosineDecay schedule object instead, which this doesn't attempt to override.
    """

    def __init__(self, out_directory, warmup_cb=None, verbose=False, tol=1e-6):
        super().__init__()
        self.override_path = os.path.join(out_directory, "lr_override.txt")
        self.warmup_cb = warmup_cb
        self.verbose = verbose
        self.tol = tol

    def on_epoch_end(self, epoch, logs=None):
        if self.warmup_cb is not None and not self.warmup_cb.is_done:
            return
        if not os.path.exists(self.override_path):
            return
        with open(self.override_path) as f:
            text = f.read().strip()
        if not text:
            return
        try:
            override_lr = float(text)
        except ValueError:
            # Always printed, not gated behind --verbose: a malformed override file is
            # very likely a live typo the person editing it right now wants to know
            # about immediately, not routine informational logging.
            print("lr_override.txt: could not parse {!r} as a float, ignoring".format(text))
            return
        current_lr = float(self.model.optimizer.learning_rate)
        if abs(override_lr - current_lr) > self.tol:
            if self.verbose:
                print("lr override file: {:.6g} -> {:.6g}".format(current_lr, override_lr))
            self.model.optimizer.learning_rate = override_lr


class OptimizerStateCallback(Callback):
    """--lr-schedule plateau only: saves the optimizer's own variables (SGD momentum,
    and under --mixed-precision the wrapping LossScaleOptimizer's own dynamic
    loss-scale state) to a single rolling file every epoch, so a later --weights resume
    can restore momentum instead of starting it at 0. That reset has been a real,
    repeated source of trouble in this project - an un-cushioned LR jump onto a
    momentum-less optimizer diverged to nan the first time a manual LR cut was tried
    via resume, and even a cushioned (warmup-ramped) resume afterward still showed
    several epochs of visibly disrupted training loss/entropy before settling. The
    underlying cause isn't specific to a deliberate LR change, either - ANY --weights
    resume loses momentum today, including one forced by a plain interruption (power
    loss, needing the GPU for something else) with no LR change at all.

    Overwrites the same file every epoch (out_directory/optimizer_state.npz) rather
    than keeping one per epoch like weights.NNNNN.weights.h5 does - unlike model
    weights, nothing in this project has ever resumed from anything but the most
    recent checkpoint, and SGD-with-momentum's state is roughly the same size as the
    model's own weights, so keeping historical copies would roughly double checkpoint
    storage for no benefit anything here actually uses.
    """

    def __init__(self, out_directory):
        super().__init__()
        self.path = os.path.join(out_directory, "optimizer_state.npz")

    def on_epoch_end(self, epoch, logs=None):
        store = {}
        self.model.optimizer.save_own_variables(store)
        np.savez(self.path, **store)


class RangeTestLRCallback(Callback):
    """--lr-range-test only: gentle linear warmup up to a conservative floor, then an
    EXPONENTIAL sweep from that floor to a ceiling over the rest of the run.

    The warmup phase exists because the model's raw gradient magnitude at
    initialization is very large on real data (observed ~100K at step 0 in an earlier
    ramp test on this project's data/architecture) - hitting a real candidate LR
    immediately, before that settles, conflates "unstable at this LR" with "unstable
    because weights are still fresh". Warming up to a floor low enough that it
    obviously isn't itself the target LR gives the optimizer a chance to leave that
    initial regime before the sweep starts probing.

    The sweep itself is exponential rather than linear so it gets even resolution per
    decade on a log-LR axis (the standard shape for this kind of test - Smith,
    "Cyclical Learning Rates for Training Neural Networks"). This is a coarse
    localizer only, not a verdict: a ramp's apparent tolerance for a given LR is not
    the same as genuine stability at that LR held constant (confirmed on this exact
    project - a prior linear ramp climbed to LR=0.8 with no visible break, while later
    held-constant tests failed well below that) - candidate LRs from this sweep must
    still be verified by a separate held-constant run before being trusted.
    """

    def __init__(self, warmup_steps, warmup_start_lr, floor_lr, ceiling_lr, total_steps):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.floor_lr = floor_lr
        self.ceiling_lr = ceiling_lr
        self.sweep_steps = max(1, total_steps - warmup_steps)
        self._step = 0

    def on_train_batch_begin(self, batch, logs=None):
        if self._step <= self.warmup_steps:
            frac = self._step / max(1, self.warmup_steps)
            lr = self.warmup_start_lr + (self.floor_lr - self.warmup_start_lr) * frac
        else:
            frac = min(1.0, (self._step - self.warmup_steps) / self.sweep_steps)
            lr = self.floor_lr * (self.ceiling_lr / self.floor_lr) ** frac
        self.model.optimizer.learning_rate = lr
        self._step += 1


class RangeTestDiagnosticsCallback(Callback):
    """--lr-range-test only: step-granularity visibility into the sweep.

    Logs step/lr/loss/weight_norm/grad_norm/loss_scale to <out_directory>/
    step_diagnostics.jsonl every check_every steps. Deliberately has NO automatic
    stop-on-weight_norm-ratio (an earlier version of this diagnostic, in
    lr_testing_trainer.py, stopped training once weight_norm exceeded a fixed
    multiple of its starting value) - that heuristic produced misleading verdicts in
    this project's past LR investigations (weight_norm grew smoothly through LRs that
    later turned out to be well past the real stability boundary, and looked alarming
    at LRs that turned out fine). This callback only records; judging the sweep is a
    manual/offline read of the full curve afterward, primarily via the loss trend, not
    a live threshold. TerminateOnNaN (added alongside this callback where it's used)
    is the only automatic stop, as a backstop against a genuine runaway wasting the
    rest of the sweep's step budget.

    grad_norm/loss_scale are read from logs["grad_norm"]/logs["loss_scale"], only
    present when the model's train_step has been monkey-patched (see
    _grad_norm_and_loss_scale_train_step) - the logs.get(...) guards keep this
    callback harmless if that patch isn't present.
    """

    def __init__(self, check_every, out_path):
        super().__init__()
        self.check_every = check_every
        self._step = 0
        self._f = open(out_path, "w")

    def on_train_batch_end(self, batch, logs=None):
        if self._step % self.check_every == 0:
            weight_norm = float(tf.linalg.global_norm(self.model.trainable_variables))
            lr = float(self.model.optimizer.learning_rate)
            loss = float(logs.get("loss")) if logs and "loss" in logs else None
            grad_norm = logs.get("grad_norm") if logs else None
            grad_norm = float(grad_norm) if grad_norm is not None else None
            loss_scale = logs.get("loss_scale") if logs else None
            loss_scale = float(loss_scale) if loss_scale is not None else None
            record = {"step": self._step, "lr": lr, "loss": loss,
                      "weight_norm": weight_norm, "grad_norm": grad_norm,
                      "loss_scale": loss_scale}
            self._f.write(json.dumps(record) + "\n")
            self._f.flush()
        self._step += 1

    def on_train_end(self, logs=None):
        self._f.close()


def _grad_norm_and_loss_scale_train_step(self, data):
    """--lr-range-test only: replaces model.train_step so grad_norm and (under mixed
    precision) the optimizer's current dynamic loss scale reach RangeTestDiagnosticsCallback
    via logs["grad_norm"]/logs["loss_scale"] every step - model.fit() doesn't expose either
    to callbacks otherwise.

    Mirrors Model.train_step (keras/src/models/model.py): under mixed_float16, gradients
    must be computed w.r.t. the SCALED loss (keeps small gradient values representable
    through the fp16 backward pass; apply_gradients then unscales internally before
    actually updating weights) - computing them against the raw loss instead would starve
    the effective step size, since apply_gradients always assumes its input was
    pre-scaled.

    loss_scale: under --mixed-precision, model.compile() wraps the optimizer in
    keras.src.optimizers.loss_scale_optimizer.LossScaleOptimizer, whose current dynamic
    scale isn't a plain attribute - it's one of the optimizer's own tracked variables
    (named "dynamic_scale", created lazily on the optimizer's first apply_gradients
    call - i.e. by the line below, on this very first traced step). It halves
    automatically whenever an inf/nan gradient triggers a skipped step, then grows back
    over time - logging it directly lets an occasional Infinity grad_norm reading during
    a sweep be told apart from genuine instability (a self-correcting mixed-precision
    artifact vs a real blowup) instead of guessed at. Looked up AFTER apply_gradients
    (not before) so the lookup - itself plain Python running once at trace time, not
    per-call - finds the variable already built; under plain float32 (no mixed
    precision) the optimizer has no such variable and loss_scale is left out of results
    entirely.
    """
    x, y = data
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        scaled_loss = self.optimizer.scale_loss(loss) if self.optimizer is not None else loss
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(scaled_loss, trainable_vars)
    grad_norm = tf.linalg.global_norm(gradients)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    for metric in self.metrics:
        if metric.name == "loss":
            metric.update_state(loss)
        else:
            metric.update_state(y, y_pred)
    results = {m.name: m.result() for m in self.metrics}
    results["grad_norm"] = grad_norm
    for v in self.optimizer.variables:
        if v.name == "dynamic_scale":
            results["loss_scale"] = tf.convert_to_tensor(v)
            break
    return results


def _replay_plateau_state(epoch_logs, factor, patience, cooldown, min_lr, min_delta=1e-4):
    """Reconstructs ReduceLROnPlateau's best/wait/cooldown_counter for a --weights resume
    under --lr-schedule plateau, by replaying its exact algorithm (see
    keras.callbacks.ReduceLROnPlateau.on_epoch_end) over the historical per-epoch
    val_loss/learning_rate sequence already sitting in metadata.json.

    No new persisted state needed: TrainingDiagnosticsCallback already logs
    learning_rate every epoch (read before ReduceLROnPlateau's own callback runs, i.e.
    the value actually used that epoch - the same "old_lr" ReduceLROnPlateau itself
    reads at this same point), and val_loss is logged by Keras directly. Without this, a
    resume would silently forget how close training already was to a cut (wait resets
    to 0 even if it was, say, 3/5 when the run stopped) - only `best` would otherwise
    carry over.
    """
    best = float("inf")
    wait = 0
    cooldown_counter = 0
    for log in epoch_logs:
        if "val_loss" not in log or "learning_rate" not in log:
            continue
        current = log["val_loss"]
        old_lr = log["learning_rate"]
        if cooldown_counter > 0:
            cooldown_counter -= 1
            wait = 0
        if current < best - min_delta:
            best = current
            wait = 0
        elif cooldown_counter == 0:
            wait += 1
            if wait >= patience and old_lr > min_lr:
                cooldown_counter = cooldown
                wait = 0
    return best, wait, cooldown_counter


class _PlateauStateRestorer(Callback):
    """Re-applies a resumed wait/cooldown_counter onto a ReduceLROnPlateau callback.

    ReduceLROnPlateau.on_train_begin() unconditionally resets wait and cooldown_counter
    to 0 (unlike best, which on_train_begin never touches) - so setting them on the
    plateau callback before model.fit() starts would just get silently overwritten the
    moment training begins. Keras calls on_train_begin on every callback in list order,
    once per model.fit() call, so placing an instance of this AFTER the plateau
    callback in the callbacks list makes it run second, re-applying the resumed values
    right after the plateau callback's own reset.
    """

    def __init__(self, plateau_cb, wait, cooldown_counter):
        super().__init__()
        self.plateau_cb = plateau_cb
        self.wait = wait
        self.cooldown_counter = cooldown_counter

    def on_train_begin(self, logs=None):
        self.plateau_cb.wait = self.wait
        self.plateau_cb.cooldown_counter = self.cooldown_counter


class TrainingDiagnosticsCallback(Callback):
    """Adds a few per-epoch numbers to logs that Keras doesn't track on its own, so they
    end up in metadata.json (via MetadataWriterCallback, which must run AFTER this one in
    the callbacks list - Keras passes the same logs dict through every callback for a
    given event, in list order):
    - learning_rate: the actual LR used this epoch. With --lr-schedule cosine, read from
      the schedule object directly rather than guessed from the args. With --lr-schedule
      plateau there is no schedule object (ReduceLROnPlateau mutates the optimizer's
      learning_rate directly), so lr_schedule is None and this instead reads the
      optimizer's live learning_rate value - the same way ReduceLROnPlateau itself does.
    - epoch_seconds / steps_per_second: wall-clock throughput, so a long run degrading
      partway through (as happened once already this session, on spinning-disk hardware)
      shows up in the data instead of only being caught by watching it live.
    """

    def __init__(self, lr_schedule, steps_per_epoch):
        super().__init__()
        self.lr_schedule = lr_schedule
        self.steps_per_epoch = steps_per_epoch
        self._epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        if self.lr_schedule is not None:
            logs["learning_rate"] = float(self.lr_schedule(self.model.optimizer.iterations))
        else:
            # ReduceLROnPlateau's own source reads this the same way (float() on the
            # optimizer's learning_rate directly) - keras.backend.convert_to_numpy, which
            # its source also references, isn't actually exposed on the public
            # keras.backend namespace in this installed version (AttributeError, caught by
            # a smoke test before this ever hit a real training run).
            logs["learning_rate"] = float(self.model.optimizer.learning_rate)
        elapsed = time.time() - self._epoch_start
        logs["epoch_seconds"] = elapsed
        logs["steps_per_second"] = self.steps_per_epoch / elapsed if elapsed > 0 else 0.0


class MetadataWriterCallback(Callback):

    def __init__(self, path):
        self.file = path
        self.metadata = {
            "epochs": [],
            "best_epoch": 0
        }

    def on_epoch_end(self, epoch, logs={}):
        # in case appending to logs (resuming training), get epoch number ourselves
        epoch = len(self.metadata["epochs"])

        self.metadata["epochs"].append(logs)

        if "val_loss" in logs:
            key = "val_loss"
        else:
            key = "loss"

        best_loss = self.metadata["epochs"][self.metadata["best_epoch"]][key]
        if logs.get(key) < best_loss:
            self.metadata["best_epoch"] = epoch

        with open(self.file, "w") as f:
            json.dump(self.metadata, f, indent=2)


def run_training(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list

    Tuned large-batch recipe. Differences from the original RocAlphaGo trainer (which
    this file replaced; see git history):
    - momentum+Nesterov SGD, and a warmup + cosine-decay learning rate schedule in place
      of InverseTimeDecay.
    - train_data is the output directory of convert_shuffled.py: train/ and val/
      subdirectories of shards. The train/val/test split is made by GAME upstream, in
      select_games.py, so no game appears in more than one split.
    - the shards of each split are already ONE uniformly random permutation of all its
      positions, so training simply streams train/ from start to finish (wrapping at the
      end) with no shuffle buffer. Every pass sees the same order; each position gets a
      random board symmetry.
    - the stream is deterministic given --seed, including symmetry choices, so a resumed
      run continues from exactly the position an uninterrupted run would have reached.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network (tuned large-batch recipe: momentum+Nesterov, warmup+cosine LR, streamed pre-shuffled shards).')  # noqa: E501
    # required args
    parser.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicy.save_model())")  # noqa: E501
    parser.add_argument("train_data", help="Output directory of convert_shuffled.py, containing train/ and val/ shard subdirectories")  # noqa: E501
    parser.add_argument("out_directory", help="directory where metadata and weights will be saved")
    # frequently used args
    parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 256", type=int, default=256)  # noqa: E501
    parser.add_argument("--epochs", "-E", help="Total number of iterations on the data across the WHOLE training run, including any epochs already completed in a prior --weights resume (not additional epochs on top of those) - this also shapes the LR schedule's decay horizon. Default: 20", type=int, default=20)  # noqa: E501
    parser.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)  # noqa: E501
    parser.add_argument("--validation-length", help="Number of validation examples to check per epoch. Default: # validation data (full validation set every epoch). The first N positions of val/ are used - a uniform random sample, identical every epoch", type=int, default=None)  # noqa: E501
    parser.add_argument("--learning-rate", "-r", help="Peak learning rate, reached at the end of warmup. Default: .055 (from an LR range test on this model/data/optimizer - see benchmarks/lr_range_test.py; loss was stable through .167, unstable by .183, so .055 is ~1/3 of the instability threshold)", type=float, default=.055)  # noqa: E501
    parser.add_argument("--warmup-steps", help="Number of steps to linearly warm up the learning rate over before decay (cosine) or plateau-monitoring (plateau) begins. Default: 1500", type=int, default=1500)  # noqa: E501
    parser.add_argument("--warmup-start-lr", help="Learning rate at step 0, before warmup begins. Default: .0001", type=float, default=.0001)  # noqa: E501
    parser.add_argument("--momentum", help="SGD momentum, used with Nesterov. Default: .9", type=float, default=.9)  # noqa: E501
    parser.add_argument("--lr-schedule", choices=["cosine", "plateau"], default="cosine", help="How the learning rate decays after warmup. 'cosine' (default): smooth cosine decay to 0 over the whole run, shaped by --epochs up front - the original behavior. 'plateau': after warmup, hold at --learning-rate and let keras.callbacks.ReduceLROnPlateau cut it (by --plateau-factor) whenever val_loss stops improving for --plateau-patience epochs, floored at --plateau-min-lr - reacts to the real training curve instead of committing to a fixed decay shape in advance, and (with a nonzero --plateau-min-lr) never crushes all the way to 0 the way cosine's default alpha=0 does.")  # noqa: E501
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="--lr-schedule plateau only: multiplier applied to the learning rate on each plateau cut (new_lr = lr * factor). Default: 0.5 (a 2x cut) - gentler than Keras's own ReduceLROnPlateau default of 0.1 (10x), chosen because this project's LR range tests found training fairly sensitive to large LR swings.")  # noqa: E501
    parser.add_argument("--plateau-patience", type=int, default=5, help="--lr-schedule plateau only: epochs with no val_loss improvement before cutting the learning rate. Default: 5. Counts in *epochs* as shaped by --epoch-length, not real dataset passes - a small --epoch-length reacts faster in wall-clock terms but each epoch's val_loss reading is noisier (fewer steps backing it), so pick patience relative to whatever --epoch-length this run actually uses.")  # noqa: E501
    parser.add_argument("--plateau-cooldown", type=int, default=2, help="--lr-schedule plateau only: epochs to wait after a cut before monitoring for a new plateau again, so each cut gets a fair chance to show its effect before another one can fire. Default: 2 (Keras's own default is 0, which allows immediate back-to-back cuts).")  # noqa: E501
    parser.add_argument("--plateau-min-lr", type=float, default=0.0, help="--lr-schedule plateau only: floor - the learning rate is never cut below this. Default: 0.0 (matches Keras's own default, i.e. no floor) - set this explicitly (e.g. some fraction of --learning-rate) to keep training from grinding to a near-zero LR the way cosine's default alpha=0 does.")  # noqa: E501
    parser.add_argument("--plateau-min-delta", type=float, default=0.005, help="--lr-schedule plateau only: minimum val_loss improvement to count as 'still improving' and reset --plateau-patience's wait counter. Default: 0.005 - NOT Keras's own default of 1e-4, which measured 30-50x smaller than this project's real epoch-to-epoch val_loss noise (stdev ~0.0048-0.0162 across the b15c192 mb1024 lr1p6 run's two LR phases). At 1e-4, noise alone registers a 'new best' often enough that the patience counter rarely reaches --plateau-patience even during a genuine multi-epoch plateau - that run needed a manual LR cut at epoch 36 and ground for 39 more epochs (avg 0.0014/epoch) before the next one. 0.005 sits just above the quieter (lower-LR) phase's noise floor and comfortably below real early-training gains, so it filters noise-driven bests without masking genuine progress.")  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # slightly fancier args
    parser.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training", default=None)  # noqa: E501
    parser.add_argument("--mixed-precision", help="Enable the mixed_float16 policy (fp16 compute, fp32 weights). Off by default: measured no benefit on this GPU/driver/model combo - 103.5ms/step with XLA+mixed precision together vs 103ms/step for XLA alone (statistically the same), despite this being an Ada GPU with Tensor Cores. XLA's fusion is apparently already capturing the available speedup here, leaving mixed precision nothing to add while still carrying its own numerical-stability surface (see the forced float32 softmax in policy.py). Only enable to re-test under different conditions (e.g. a larger batch size)", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--symmetries", help="Comma-separated list of transforms, subset of noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2", default='noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2')  # noqa: E501
    parser.add_argument("--seed", help="Seed for weight initialization and for the per-position symmetry choices (train and val use seed and seed+1). Default: unseeded (a fresh, unrecoverable draw from OS entropy every run) - set this to make the exact position stream fed to the network reproducible, e.g. to replay a run that hit an anomaly.", type=int, default=None)  # noqa: E501
    parser.add_argument("--val-dataset-gpu-resident", action="store_true",
                        help="Diagnostic-only escape hatch: build val_dataset WITHOUT the "
                             "CPU-pin fix, letting TF's default placement put it on GPU "
                             "again - reproduces the original ResourceExhaustedError risk. "
                             "Exists solely to A/B the CPU-pin fix's effect on training-time "
                             "GPU utilization against an otherwise-identical run (same "
                             "batch size, same prefetch) - never use this "
                             "for a real training run.")
    # LR range test - a coarse, cheap localizer for a new architecture/batch-size
    # combination's viable LR region, meant to be followed by separate held-constant
    # verification runs (this mode only ramps, it never holds). Off by default, does not
    # affect normal training runs. Bypasses --lr-schedule/--learning-rate/--warmup-steps/
    # --warmup-start-lr entirely.
    parser.add_argument("--lr-range-test", action="store_true",
                        help="Diagnostic mode: ramp the LR through a gentle linear warmup "
                             "(--range-warmup-start-lr -> --range-floor-lr over "
                             "--range-warmup-steps) and then an EXPONENTIAL sweep from "
                             "--range-floor-lr up to --range-ceiling-lr over the rest of the "
                             "run (shaped by --epochs/--epoch-length as usual). Ignores "
                             "--lr-schedule/--learning-rate/--warmup-steps/--warmup-start-lr "
                             "entirely - not compatible with --weights (resume). Logs "
                             "step/lr/loss/weight_norm/grad_norm/loss_scale to "
                             "<out_directory>/step_diagnostics.jsonl every "
                             "--range-check-every steps, and adds TerminateOnNaN as a "
                             "safety net. This is a coarse localizer only, NOT a verdict on "
                             "a safe LR - ramping tolerance is not the same as genuine "
                             "stability at a held LR, so treat its output as candidates for "
                             "a follow-up held-constant test, not an answer. Default: False "
                             "(normal training).")
    parser.add_argument("--range-warmup-steps", type=int, default=None,
                        help="--lr-range-test only: steps to linearly ramp from "
                             "--range-warmup-start-lr to --range-floor-lr before the "
                             "exponential sweep begins. Default: one epoch's worth of steps "
                             "(steps_per_epoch, from --epoch-length/--minibatch) - a minimum "
                             "gentle warmup so the sweep doesn't start while the model's "
                             "initial (very large) raw gradient magnitude is still settling.")
    parser.add_argument("--range-warmup-start-lr", type=float, default=1e-4,
                        help="--lr-range-test only: LR at step 0. Default: .0001")
    parser.add_argument("--range-floor-lr", type=float, default=1e-3,
                        help="--lr-range-test only: LR reached at the end of warmup, and the "
                             "starting point of the exponential sweep. Should be low enough "
                             "that it obviously isn't itself the target LR. Default: .001")
    parser.add_argument("--range-ceiling-lr", type=float, default=2.0,
                        help="--lr-range-test only: LR reached at the end of the run (the "
                             "last step of the last epoch). The exponential sweep moves from "
                             "--range-floor-lr to this value over the steps remaining after "
                             "warmup. Default: 2.0")
    parser.add_argument("--range-check-every", type=int, default=50,
                        help="--lr-range-test only: log step/lr/loss/weight_norm/grad_norm/"
                             "loss_scale to <out_directory>/step_diagnostics.jsonl every this "
                             "many steps. Default: 50")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    resume = args.weights is not None

    # --lr-range-test + --weights is a supported combination (unlike plateau/cosine's
    # --weights resume, which continues one specific schedule's own step accounting):
    # RangeTestLRCallback's step counter always starts fresh at 0 regardless of resume,
    # so this just warm-starts a NEW, independent sweep from a checkpoint's weights
    # (e.g. continuing an earlier sweep's LR curve from where it left off, without
    # re-running the cheap-to-skip low end of the range again) - not an attempt to
    # resume the SAME sweep mid-step-count, which would need offset-aware accounting
    # this callback doesn't have. The optimizer's own state (SGD momentum) is not
    # reloaded either way (--weights only ever reloads model weights - see the
    # epochs_already_trained handling below), so a warm-started sweep has a brief
    # (~10-50 step, matching momentum's 1/(1-momentum) memory) transient while momentum
    # rebuilds, rather than being a bit-for-bit continuation of the original run.

    if args.verbose:
        if resume:
            print("trying to resume from %s with weights %s" %
                  (args.out_directory, os.path.join(args.out_directory, args.weights)))
        else:
            if os.path.exists(args.out_directory):
                print("directory %s exists. any previous data will be overwritten" %
                      args.out_directory)
            else:
                print("starting fresh output directory %s" % args.out_directory)

    # Must be set before the model is constructed - controls the actual random draw
    # used for weight initialization. Without this, --seed only ever controlled the
    # data stream (symmetry choice) - weight init drew from Keras's own global RNG
    # regardless of --seed, silently breaking the "same --seed -> same run" assumption for anything
    # sensitive to initial weights.
    if args.seed is not None:
        keras_utils.set_random_seed(args.seed)

    # Must be set before the model is constructed - it controls the dtype policy new
    # layers are built with. The final softmax layer overrides back to float32 regardless
    # (see policy.py) - the standard mixed-precision recommendation to keep the
    # loss-adjacent computation at full precision.
    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        if args.verbose:
            policy_obj = mixed_precision.global_policy()
            print("mixed precision enabled: compute dtype=%s, variable dtype=%s" %
                  (policy_obj.compute_dtype, policy_obj.variable_dtype))
    elif args.verbose:
        print("mixed precision disabled (default - see --help)")

    # load model from json spec
    policy = CNNPolicy.load_model(args.model)
    model_features = policy.preprocessor.get_feature_list()
    model = policy.model
    if resume:
        model.load_weights(os.path.join(args.out_directory, args.weights))

    # discover the pre-shuffled shards of each split
    train_shards = find_split_shards(args.train_data, "train")
    val_shards = find_split_shards(args.train_data, "val")
    dataset_features, board_size, n_features, train_sizes = dataset_info(train_shards)
    val_features, _vb, _vf, val_sizes = dataset_info(val_shards)
    if val_features != dataset_features:
        raise ValueError("train/ and val/ shards were built with different feature lists")

    if dataset_features != model_features:
        raise ValueError("Model JSON file expects features \n\t%s\n"
                         "But shards contain \n\t%s" % ("\n\t".join(model_features),
                                                        "\n\t".join(dataset_features)))
    elif args.verbose:
        print("Verified that shard features and model features exactly match.")

    # ensure output directory is available
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    n_train_data = sum(train_sizes)
    n_val_data = sum(val_sizes)
    if args.verbose:
        print("dataset loaded from %s" % args.train_data)
        print("\t%d training positions in %d shards" % (n_train_data, len(train_shards)))
        print("\t%d validation positions in %d shards" % (n_val_data, len(val_shards)))

    # create metadata file and the callback object that will write to it
    meta_file = os.path.join(args.out_directory, "metadata.json")
    meta_writer = MetadataWriterCallback(meta_file)
    # load prior data if it already exists
    if os.path.exists(meta_file) and resume:
        with open(meta_file, "r") as f:
            meta_writer.metadata = json.load(f)
        if args.verbose:
            print("previous metadata loaded: %d epochs. new epochs will be appended." %
                  len(meta_writer.metadata["epochs"]))
    elif args.verbose:
        print("starting with empty metadata")

    epochs_already_trained = len(meta_writer.metadata["epochs"]) if resume else 0
    if resume and epochs_already_trained >= args.epochs:
        raise ValueError(
            "{} already has {} recorded epochs, which is >= --epochs {}. Increase --epochs "
            "to train further.".format(meta_file, epochs_already_trained, args.epochs))

    if resume and meta_writer.metadata.get("cmd_line_args"):
        # _ResumedLRSchedule's step offset (epochs_already_trained * steps_per_epoch,
        # below) is only correct if steps_per_epoch is the same value it was in every
        # prior invocation - which depends on --minibatch and --epoch-length, and the
        # schedule's own decay_steps also depends on --warmup-steps. A silent change to
        # any of these would silently point the resumed run at the wrong spot on the LR
        # curve with no error, so this is checked explicitly rather than trusted.
        prev_args = meta_writer.metadata["cmd_line_args"][-1]
        for key in ("minibatch", "epoch_length", "warmup_steps"):
            prev_value = prev_args.get(key)
            cur_value = getattr(args, key)
            if prev_value != cur_value:
                raise ValueError(
                    "--{} changed across resume ({} -> {}): this would silently break the "
                    "learning-rate schedule's step accounting, since warmup/decay timing "
                    "depends on steps_per_epoch (minibatch, epoch-length) and the decay "
                    "horizon (warmup-steps). Keep these identical across a resume, or "
                    "start a fresh out_directory.".format(
                        key.replace('_', '-'), prev_value, cur_value))

    meta_writer.metadata["training_data"] = args.train_data
    meta_writer.metadata["model_file"] = args.model
    # Record all command line args in a list so that all args are recorded even
    # when training is stopped and resumed.
    meta_writer.metadata["cmd_line_args"] = meta_writer.metadata.get("cmd_line_args", [])
    meta_writer.metadata["cmd_line_args"].append(vars(args))

    # create ModelCheckpoint to save weights every epoch
    checkpoint_template = os.path.join(args.out_directory, "weights.{epoch:05d}.weights.h5")
    checkpointer = ModelCheckpoint(checkpoint_template, save_weights_only=True)

    symmetries = args.symmetries.strip().split(",")
    unknown = [name for name in symmetries if name not in BATCH_TRANSFORMATIONS]
    if unknown:
        raise ValueError("unknown symmetries: {}".format(unknown))

    # Computed before the training stream is built: a resumed run starts the stream at
    # exactly the position an uninterrupted run would have reached, which needs the
    # step count per epoch.
    samples_per_epoch = args.epoch_length or n_train_data
    steps_per_epoch = samples_per_epoch // args.minibatch
    total_steps = steps_per_epoch * args.epochs
    start_position = epochs_already_trained * steps_per_epoch * args.minibatch
    if args.verbose and start_position:
        print("resuming the training stream at position %d" % start_position)

    # train and val use distinct seeds so their symmetry choices are independent
    train_seed = args.seed
    val_seed = None if args.seed is None else args.seed + 1
    train_data_generator = sanity_checked_generator(
        shard_batch_generator(train_shards, train_sizes, args.minibatch, board_size,
                              symmetries, seed=train_seed, start_position=start_position),
        args.out_directory, "train")

    # Validation is a fixed prefix of val/. The val shards are a uniform random sample
    # already, so the first N positions are too, and taking a prefix keeps the set
    # identical every epoch - val_loss then moves only because the model does.
    n_val_eval = min(args.validation_length or n_val_data, n_val_data)
    print("materializing {} validation positions into fixed arrays...".format(n_val_eval))
    X_val, Y_val = validation_arrays(val_shards, val_sizes, n_val_eval, board_size,
                                     symmetries, seed=val_seed)

    # tf.data.Dataset.from_tensor_slices() embeds the whole array as an in-graph
    # constant, and under TF2 eager execution with a GPU visible, TF's default device
    # placement puts that constant on the GPU immediately at construction time - not
    # lazily streamed per-batch like the training shuffle-buffer generator is. At
    # validation_length=100000 that's an extra ~6.93GB permanently resident in VRAM
    # (confirmed via benchmarks/_val_dataset_gpu_placement_test.py: GPU memory jumped
    # from ~0GB to ~7GB the instant this line ran, before any iteration), which was
    # enough on its own to push a minibatch=1024 run into a GPU ResourceExhaustedError
    # that leaner minibatch=1024 diagnostics (which never build a validation_data set at
    # all) never caught. Pinning construction to CPU keeps X_val/Y_val host-resident and
    # streams one batch at a time onto the GPU, same as training data already does.
    if args.val_dataset_gpu_resident:
        # Diagnostic escape hatch only - see --val-dataset-gpu-resident's help text. Not
        # the default: this is exactly the placement that caused the original GPU
        # ResourceExhaustedError this whole CPU-pin fix exists to prevent.
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(args.minibatch)
    else:
        with tf.device('/cpu:0'):
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(args.minibatch)
    # from_tensor_slices() makes its own internal copy of the array into the dataset
    # regardless of device placement - the CPU pinning above only controls where THAT
    # copy lives, it doesn't stop X_val/Y_val (the original numpy arrays) from also
    # still being held in memory afterward. Confirmed via a real end-to-end run: with
    # both copies alive at once (~6.93GB each at validation_length=100000) on top of the
    # persistent ~6.8GB training shuffle buffer, host RAM hit the WSL2 cap and the
    # kernel OOM-killed the process (dmesg: "Out of memory: Killed process ... python3",
    # anon-rss ~17.76GB at kill time) - even though GPU memory never moved. Dropping the
    # now-redundant references lets the original arrays be reclaimed once val_dataset
    # has its own copy.
    del X_val, Y_val
    import gc
    gc.collect()

    warmup_cb = None
    plateau_cb = None
    plateau_state_restorer = None
    range_lr_cb = None
    lr_override_cb = None
    optimizer_state_cb = None
    if args.lr_range_test:
        # No LearningRateSchedule object and no ReduceLROnPlateau - RangeTestLRCallback
        # owns the optimizer's learning_rate directly for the whole run, the same way
        # WarmupCallback does for --lr-schedule plateau (see there for why a callback,
        # not a schedule, is needed for a plain mutable LR).
        lr_schedule = None
        sgd = SGD(learning_rate=args.range_warmup_start_lr, momentum=args.momentum,
                  nesterov=True)
        range_warmup_steps = (args.range_warmup_steps if args.range_warmup_steps is not None
                              else steps_per_epoch)
        range_lr_cb = RangeTestLRCallback(
            range_warmup_steps, args.range_warmup_start_lr, args.range_floor_lr,
            args.range_ceiling_lr, total_steps)
        if args.verbose:
            print("LR range test: warmup {} -> {} over {} steps, then exponential sweep "
                  "{} -> {} over the remaining {} steps".format(
                     args.range_warmup_start_lr, args.range_floor_lr, range_warmup_steps,
                     args.range_floor_lr, args.range_ceiling_lr,
                     max(1, total_steps - range_warmup_steps)))
    elif args.lr_schedule == "cosine":
        # Warmup (linear, warmup_start_lr -> learning_rate over warmup_steps) then cosine
        # decay to zero over the rest of total_steps, replacing InverseTimeDecay's
        # lr/(1+decay*step) - InverseTimeDecay's decay_rate was tuned for a step count this
        # recipe doesn't use, whereas cosine-to-a-known-horizon needs no per-run retuning.
        #
        # CosineDecay's decay_steps is the length of the decay phase AFTER warmup ends, not
        # the total schedule length - confirmed empirically (a schedule with warmup_steps=10,
        # decay_steps=100 reaches its floor at step 110, not step 100). So decay_steps here is
        # total_steps minus the warmup already spent, not total_steps itself - otherwise the
        # schedule would run warmup_steps longer than intended and never reach its floor
        # within the actual training run.
        lr_schedule = CosineDecay(
            initial_learning_rate=args.warmup_start_lr,
            decay_steps=max(1, total_steps - args.warmup_steps),
            warmup_target=args.learning_rate,
            warmup_steps=args.warmup_steps)
        if epochs_already_trained:
            # See _ResumedLRSchedule: the resumed optimizer's own step counter restarts at 0,
            # so shift the schedule's input by however many steps were already trained in
            # prior invocations, keeping this one continuous warmup+decay curve across resumes.
            lr_schedule = _ResumedLRSchedule(lr_schedule, epochs_already_trained * steps_per_epoch)
        sgd = SGD(learning_rate=lr_schedule, momentum=args.momentum, nesterov=True)
    else:
        # plateau: no LearningRateSchedule object - ReduceLROnPlateau mutates the
        # optimizer's learning_rate directly (new_lr = old_lr * factor), which requires a
        # plain mutable value rather than a schedule. Warmup is therefore done via a
        # callback (WarmupCallback) instead of folded into the schedule; lr_schedule stays
        # None so TrainingDiagnosticsCallback knows to read the live optimizer value
        # instead of calling a schedule object.
        #
        # Resume handling: --weights only reloads model weights, never optimizer state, so
        # a naive resume would restart the optimizer's learning_rate at warmup_start_lr and
        # re-run warmup, forgetting wherever ReduceLROnPlateau had actually left it (and
        # forgetting its own best/wait/cooldown bookkeeping too - see
        # _replay_plateau_state). Fix: read the last completed epoch's logged
        # learning_rate (TrainingDiagnosticsCallback already writes this every epoch) and
        # start the optimizer there instead, skipping warmup entirely on a resume (it only
        # makes sense for a genuinely fresh start), and replay history into
        # ReduceLROnPlateau's own best/wait/cooldown_counter - best doesn't reset on
        # on_train_begin (only wait/cooldown do), but all three are set explicitly here for
        # clarity.
        #
        # (A --resume-warmup-steps option - ramping into the resumed LR instead of jumping
        # straight to it - was tried and removed again: it avoided the instant-jump nan
        # divergence, but the resumed run still spent several epochs with visibly disrupted
        # training loss/entropy before settling, which wasn't judged worth keeping over the
        # simpler instant-jump behavior.)
        if resume and meta_writer.metadata["epochs"]:
            resumed_target_lr = meta_writer.metadata["epochs"][-1]["learning_rate"]
            initial_lr = resumed_target_lr
            resumed_best, resumed_wait, resumed_cooldown = _replay_plateau_state(
                meta_writer.metadata["epochs"], args.plateau_factor, args.plateau_patience,
                args.plateau_cooldown, args.plateau_min_lr, min_delta=args.plateau_min_delta)
        else:
            resumed_target_lr = None
            initial_lr = args.warmup_start_lr
            resumed_best, resumed_wait, resumed_cooldown = None, 0, 0
        lr_schedule = None
        sgd = SGD(learning_rate=initial_lr, momentum=args.momentum, nesterov=True)
        if not resume:
            warmup_cb = WarmupCallback(args.warmup_steps, args.warmup_start_lr, args.learning_rate)
        plateau_cb = ReduceLROnPlateau(
            monitor="val_loss", factor=args.plateau_factor, patience=args.plateau_patience,
            cooldown=args.plateau_cooldown, min_lr=args.plateau_min_lr,
            min_delta=args.plateau_min_delta, verbose=1)
        lr_override_cb = LROverrideCallback(
            args.out_directory, warmup_cb=warmup_cb, verbose=args.verbose)
        optimizer_state_cb = OptimizerStateCallback(args.out_directory)
        if resumed_best is not None:
            plateau_cb.best = resumed_best
            # wait/cooldown_counter can't just be set here - ReduceLROnPlateau's own
            # on_train_begin() would reset them to 0 the moment model.fit() starts (it
            # never touches best, so that one alone is safe to set directly). See
            # _PlateauStateRestorer: placed after plateau_cb in the callbacks list below,
            # so its on_train_begin re-applies these right after that reset happens.
            plateau_state_restorer = _PlateauStateRestorer(
                plateau_cb, resumed_wait, resumed_cooldown)
            if args.verbose:
                print("resuming plateau state: best={:.4f} wait={}/{} cooldown_counter={}"
                      .format(resumed_best, resumed_wait, args.plateau_patience,
                              resumed_cooldown))
    # jit_compile=True (XLA): safe here because of the ensure_xla_conv_nhwc() call at
    # module load time above - see xla_workarounds.py for why it's needed. Measured
    # 102.5ms/step vs 113ms/step without XLA.
    model.compile(
        loss='categorical_crossentropy', optimizer=sgd,
        metrics=["accuracy", TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
                 prediction_entropy],
        jit_compile=True)

    # Restore optimizer state (SGD momentum, plus the LossScaleOptimizer's own dynamic
    # loss-scale state under --mixed-precision - see OptimizerStateCallback) saved by a
    # prior run, if present. Must happen after compile() (the optimizer isn't attached,
    # and under mixed precision isn't wrapped, until then) and needs an explicit build()
    # first: optimizer variables are created lazily on the first apply_gradients() call,
    # so load_own_variables() has nothing to load into otherwise.
    #
    # DO NOT remove/reorder the build() call below - confirmed via
    # benchmarks/_optimizer_state_roundtrip_test.py that calling load_own_variables()
    # on an unbuilt optimizer does NOT raise: it silently no-ops (a UserWarning about a
    # variable-count mismatch, easy to miss without --verbose) and leaves momentum at 0,
    # i.e. exactly the failure mode this whole feature exists to prevent, just silent
    # instead of loud.
    if args.lr_schedule == "plateau" and resume:
        optimizer_state_path = os.path.join(args.out_directory, "optimizer_state.npz")
        if os.path.exists(optimizer_state_path):
            model.optimizer.build(model.trainable_variables)
            with np.load(optimizer_state_path) as f:
                model.optimizer.load_own_variables(dict(f))
            if args.verbose:
                print("restored optimizer state (momentum) from {}".format(optimizer_state_path))
        elif args.verbose:
            print("no optimizer_state.npz found - resuming with momentum reset to 0")

    range_diagnostics = None
    if args.lr_range_test:
        # Monkey-patch AFTER compile() (so the patched step still picks up
        # jit_compile=True via make_train_function()) and BEFORE fit() - see
        # _grad_norm_and_loss_scale_train_step's docstring for why this is a
        # monkey-patch rather than a model_class kwarg.
        model.train_step = types.MethodType(_grad_norm_and_loss_scale_train_step, model)
        range_diagnostics = RangeTestDiagnosticsCallback(
            args.range_check_every, os.path.join(args.out_directory, "step_diagnostics.jsonl"))

    diagnostics = TrainingDiagnosticsCallback(lr_schedule, steps_per_epoch)

    if args.verbose:
        print("STARTING TRAINING")

    # Order matters: Keras passes the same logs dict through every callback for a given
    # event, in list order. diagnostics must run before plateau_cb - both set
    # logs["learning_rate"], and diagnostics needs to capture the LR actually used this
    # epoch before plateau_cb potentially reduces it for the next one, or the logged value
    # would show next epoch's (reduced) LR against this epoch's loss. diagnostics must also
    # run before meta_writer, which just persists whatever's in logs by the time it sees it.
    # plateau_state_restorer must run after plateau_cb (see its docstring) - both only
    # hook on_train_begin/on_epoch_end respectively, so its position relative to
    # diagnostics/meta_writer doesn't matter, only relative to plateau_cb. lr_override_cb
    # must also run after plateau_cb (see its docstring) - an override should always win
    # over whatever plateau_cb just decided this epoch, not get silently clobbered by it.
    # optimizer_state_cb has no ordering requirement - it only reads/writes
    # model.optimizer's variables (momentum), which none of these other callbacks touch,
    # only its learning_rate (a separate, unrelated attribute).
    callbacks = [checkpointer]
    if warmup_cb is not None:
        callbacks.append(warmup_cb)
    if range_lr_cb is not None:
        callbacks.append(range_lr_cb)
    callbacks.append(diagnostics)
    if range_diagnostics is not None:
        callbacks.append(range_diagnostics)
        callbacks.append(TerminateOnNaN())
    if plateau_cb is not None:
        callbacks.append(plateau_cb)
    if lr_override_cb is not None:
        callbacks.append(lr_override_cb)
    if optimizer_state_cb is not None:
        callbacks.append(optimizer_state_cb)
    if plateau_state_restorer is not None:
        callbacks.append(plateau_state_restorer)
    callbacks.append(meta_writer)

    model.fit(
        x=train_data_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        initial_epoch=epochs_already_trained,
        callbacks=callbacks,
        validation_data=val_dataset)


if __name__ == '__main__':
    run_training()
