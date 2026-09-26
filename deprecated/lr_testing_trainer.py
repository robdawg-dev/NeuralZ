import os
import json
import time
import types
import numpy as np
from AlphaGo.training.xla_workarounds import ensure_xla_conv_nhwc

# Must run before the first XLA compilation, i.e. before model.compile(jit_compile=True)
# below is ever exercised - module level, ahead of everything else, is the safest place.
ensure_xla_conv_nhwc()

import tensorflow as tf  # noqa: E402
from keras import mixed_precision, ops, utils as keras_utils  # noqa: E402
from keras.metrics import TopKCategoricalAccuracy  # noqa: E402
from keras.optimizers import SGD  # noqa: E402
from keras.optimizers.schedules import CosineDecay, LearningRateSchedule  # noqa: E402
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, TerminateOnNaN  # noqa: E402
from AlphaGo.models.policy import CNNPolicy  # noqa: E402
# Unused directly, but importing it registers ResTowerPolicy (via the @neuralnet
# decorator) so CNNPolicy.load_model() can find it by name in a model.json's "class"
# field - registration only happens when a class's defining module is actually imported
# somewhere in the process, and nothing else here pulls this one in.
import AlphaGo.models.resnet_tower_policy  # noqa: E402,F401
from shuffle_buffer import (  # noqa: E402
    BOARD_TRANSFORMATIONS, find_shard_files, build_game_index, get_or_create_game_split,
    shuffle_buffer_batch_generator, build_validation_arrays)


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
            bad = x_nan > 0 or x_min < -1e-3 or x_max > 1 + 1e-3 or abs(y_min - 1.0) > 1e-3 or abs(y_max - 1.0) > 1e-3
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
    warmup_steps, then stops touching it (unless uncapped=True - see below).

    Only used for --lr-schedule plateau: ReduceLROnPlateau reduces the optimizer's
    learning_rate by direct assignment (new_lr = old_lr * factor), which requires it to
    be a plain mutable value rather than a LearningRateSchedule object - so unlike
    --lr-schedule cosine (where warmup is folded into CosineDecay's own
    warmup_steps/warmup_target), warmup has to happen here as a callback instead, before
    handing control of the (now plain) learning_rate over to ReduceLROnPlateau for the
    rest of the run.

    uncapped=True (used by --uncapped-warmup, an LR-to-collapse diagnostic - see
    run_training_v2's argparse help): skips the freeze entirely and keeps evaluating
    the same linear formula forever. Algebraically this is identical to
    lr(step) = start_lr + slope*step for step > warmup_steps too (slope =
    (target_lr - start_lr) / warmup_steps) - it's the exact same line, just no longer
    clamped to frac <= 1.0.
    """

    def __init__(self, warmup_steps, start_lr, target_lr, uncapped=False):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.uncapped = uncapped
        self._step = 0

    def on_train_batch_begin(self, batch, logs=None):
        if not self.uncapped and self._step > self.warmup_steps:
            return
        frac = self._step / max(1, self.warmup_steps)
        self.model.optimizer.learning_rate = self.start_lr + (self.target_lr - self.start_lr) * frac
        self._step += 1


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


class StepDiagnosticsCallback(Callback):
    """--uncapped-warmup only: step-granularity weight_norm/grad_norm visibility.

    A real production epoch here is ~2,700+ steps (~10-12 minutes) - logging
    weight_norm/grad_norm only at on_epoch_end could miss a collapse's actual onset
    entirely (all that would be seen is the aftermath, up to ~10 minutes late). This
    instead checks every check_every steps, writing each checked point to its own
    JSONL file (kept separate from metadata.json's one-record-per-epoch convention,
    so nothing else that reads metadata.json is affected) and stopping training the
    moment weight_norm exceeds divergence_norm_multiple times its value at the first
    checked step - catching the *onset* of a blowup, not just its eventual NaN
    (TerminateOnNaN, added alongside this callback, is the backstop for that).

    grad_norm is read from logs["grad_norm"], only present when run_training_v2 has
    monkey-patched the model's train_step to return it there (see --uncapped-warmup
    and _grad_norm_train_step) - the logs.get(...) guard keeps this callback
    harmless/reusable if that patch isn't present.
    """

    def __init__(self, check_every, out_path, divergence_norm_multiple=5.0):
        super().__init__()
        self.check_every = check_every
        self.out_path = out_path
        self.divergence_norm_multiple = divergence_norm_multiple
        self._step = 0
        self._start_weight_norm = None
        self._f = open(out_path, "w")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_max_grad_norm = 0.0
        self._epoch_last_weight_norm = None

    def on_train_batch_end(self, batch, logs=None):
        # By the time logs reaches a callback, Keras has already converted train_step's
        # whole returned dict (including "grad_norm" - see _grad_norm_train_step) from
        # tensors to concrete Python values, so this is already a plain float here.
        grad_norm = logs.get("grad_norm") if logs else None
        if grad_norm is not None:
            self._epoch_max_grad_norm = max(self._epoch_max_grad_norm, grad_norm)
        if self._step % self.check_every == 0:
            weight_norm = float(tf.linalg.global_norm(self.model.trainable_variables))
            self._epoch_last_weight_norm = weight_norm
            if self._start_weight_norm is None:
                self._start_weight_norm = weight_norm
            lr = float(self.model.optimizer.learning_rate)
            loss = float(logs.get("loss")) if logs and "loss" in logs else None
            record = {"step": self._step, "lr": lr, "loss": loss,
                     "weight_norm": weight_norm, "grad_norm": grad_norm}
            self._f.write(json.dumps(record) + "\n")
            self._f.flush()
            if weight_norm > self._start_weight_norm * self.divergence_norm_multiple:
                print("*** weight_norm {:.2f} exceeds {}x starting norm {:.2f} at step "
                     "{} (lr={:.5g}) - stopping early ***".format(
                         weight_norm, self.divergence_norm_multiple,
                         self._start_weight_norm, self._step, lr), flush=True)
                self.model.stop_training = True
        self._step += 1

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["grad_norm_max"] = self._epoch_max_grad_norm
            logs["weight_norm"] = self._epoch_last_weight_norm

    def on_train_end(self, logs=None):
        self._f.close()


def _grad_norm_train_step(self, data):
    """--uncapped-warmup only: replaces model.train_step so grad_norm is available to
    StepDiagnosticsCallback via logs["grad_norm"] every step.

    model.fit() doesn't expose per-step gradients to callbacks - getting real
    grad_norm needs a custom train_step. Monkey-patched onto the already-built model
    instance (a plain bound-method reassignment) rather than threaded through as a
    model_class kwarg, since the model here is loaded via CNNPolicy.load_model()
    (AlphaGo/models/nn_util.py), which reconstructs the architecture from a saved
    Keras JSON config - not by re-invoking ResTowerPolicy.create_network() - so a
    model_class kwarg wouldn't be reachable from this trainer's normal load path.

    grad_norm is returned in the same dict as every other metric rather than stashed
    on self as a plain attribute - a first attempt at the latter failed
    (TypeError: float() argument must be a string or a real number, not
    'SymbolicTensor') because with jit_compile=True this method is traced once as a
    graph function: a raw `self._x = tensor` assignment only actually executes during
    that initial trace, so self._x stays bound to the trace-time symbolic tensor
    forever after, never updated on later concrete calls. Returning grad_norm in this
    function's own result dict works because Keras's fit() loop already converts
    that whole dict from tensors to concrete Python values before handing it to
    callbacks as `logs` - the exact same mechanism that already makes plain metrics
    like loss/accuracy safely readable in callbacks, reused here rather than
    reinvented. Deliberately NOT wrapped in a keras.metrics.Mean (which would report a
    running average since epoch start, diluting exactly the kind of brief spike this
    diagnostic exists to catch) - this is the raw, instantaneous per-step value.

    Reproduces the same loss/metrics computation as this file's own model.compile()
    call (loss='categorical_crossentropy', metrics=[accuracy, top5_accuracy,
    prediction_entropy]) so logged numbers stay comparable to every other run's
    metadata.json.
    """
    x, y = data
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # Mirrors Model.train_step exactly (keras/src/models/model.py) - under
        # mixed_float16, gradients must be computed w.r.t. the SCALED loss (keeps
        # small gradient values representable through the fp16 backward pass;
        # apply_gradients then unscales them back down internally before actually
        # updating weights). A first version of this monkey-patch skipped this call
        # entirely - grad_norm still looked plausible (computed directly from the raw
        # tape output before any scaling), but weight_norm barely moved and loss
        # stayed pinned near the random-baseline ceiling for 7000+ steps: unscaled
        # gradients both lost precision in the fp16 backward pass AND then got
        # divided by the loss-scale factor a second time by apply_gradients (which
        # always assumes its input was pre-scaled), crushing the effective step size.
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
    return results


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


def run_training_v2(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list

    Tuned large-batch recipe, kept as a separate file from supervised_policy_trainer.py
    (which is left untouched) rather than changing it in place. Differences from that
    file:
    - momentum+Nesterov SGD, and a warmup + cosine-decay learning rate schedule in place
      of InverseTimeDecay.
    - train_data is a DIRECTORY of sharded .h5 files (see game_converter_parallel.py),
      not a single .h5 file - a single-file dataset large enough for real training
      degrades badly on spinning-disk hardware past roughly 100-150GB (confirmed the
      hard way), so data now lives in several smaller shard files instead.
    - reads via a streaming shuffle buffer (AlphaGo/training/shuffle_buffer.py) instead
      of random single-row HDF5 access: the old generator's random access pattern reads
      one HDF5 chunk (compressed as a unit) to serve one row, and measured 7x slower than
      pure GPU compute for this reason. The shuffle buffer reads whole games (each
      already one contiguous slice) and shuffles at the individual-position level from an
      in-memory buffer, giving comparable randomness with far less disk overhead. See
      that module's docstrings for the full design.
    - train/val/test split is at the GAME level, not the position level - positions
      within one game are highly correlated, so a position-level split (what both the
      original trainer and the position-permutation approach this replaces actually do)
      risks the same game appearing in both train and validation.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network (tuned large-batch recipe: momentum+Nesterov, warmup+cosine LR, shuffle-buffer data loading).')  # noqa: E501
    # required args
    parser.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicy.save_model())")  # noqa: E501
    parser.add_argument("train_data", help="Directory containing sharded .h5 training data files (see game_converter_parallel.py)")  # noqa: E501
    parser.add_argument("out_directory", help="directory where metadata and weights will be saved")
    # frequently used args
    parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 256", type=int, default=256)  # noqa: E501
    parser.add_argument("--epochs", "-E", help="Total number of iterations on the data across the WHOLE training run, including any epochs already completed in a prior --weights resume (not additional epochs on top of those) - this also shapes the LR schedule's decay horizon. Default: 20", type=int, default=20)  # noqa: E501
    parser.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)  # noqa: E501
    parser.add_argument("--validation-length", help="Number of validation examples to check per epoch. Default: # validation data (full validation set every epoch). Validation reads whole games too, so this only needs to be much smaller than the full validation set if --epoch-length is also set much smaller than the full training set - otherwise a fast per-epoch training pass ends up dominated by a fixed-cost full validation pass every time", type=int, default=None)  # noqa: E501
    parser.add_argument("--learning-rate", "-r", help="Peak learning rate, reached at the end of warmup. Default: .055 (from an LR range test on this model/data/optimizer - see benchmarks/lr_range_test.py; loss was stable through .167, unstable by .183, so .055 is ~1/3 of the instability threshold)", type=float, default=.055)  # noqa: E501
    parser.add_argument("--warmup-steps", help="Number of steps to linearly warm up the learning rate over before decay (cosine) or plateau-monitoring (plateau) begins. Default: 1500", type=int, default=1500)  # noqa: E501
    parser.add_argument("--warmup-start-lr", help="Learning rate at step 0, before warmup begins. Default: .0001", type=float, default=.0001)  # noqa: E501
    parser.add_argument("--momentum", help="SGD momentum, used with Nesterov. Default: .9", type=float, default=.9)  # noqa: E501
    parser.add_argument("--lr-schedule", choices=["cosine", "plateau"], default="cosine", help="How the learning rate decays after warmup. 'cosine' (default): smooth cosine decay to 0 over the whole run, shaped by --epochs up front - the original behavior. 'plateau': after warmup, hold at --learning-rate and let keras.callbacks.ReduceLROnPlateau cut it (by --plateau-factor) whenever val_loss stops improving for --plateau-patience epochs, floored at --plateau-min-lr - reacts to the real training curve instead of committing to a fixed decay shape in advance, and (with a nonzero --plateau-min-lr) never crushes all the way to 0 the way cosine's default alpha=0 does.")  # noqa: E501
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="--lr-schedule plateau only: multiplier applied to the learning rate on each plateau cut (new_lr = lr * factor). Default: 0.5 (a 2x cut) - gentler than Keras's own ReduceLROnPlateau default of 0.1 (10x), chosen because this project's LR range tests found training fairly sensitive to large LR swings.")  # noqa: E501
    parser.add_argument("--plateau-patience", type=int, default=5, help="--lr-schedule plateau only: epochs with no val_loss improvement before cutting the learning rate. Default: 5. Counts in *epochs* as shaped by --epoch-length, not real dataset passes - a small --epoch-length reacts faster in wall-clock terms but each epoch's val_loss reading is noisier (fewer steps backing it), so pick patience relative to whatever --epoch-length this run actually uses.")  # noqa: E501
    parser.add_argument("--plateau-cooldown", type=int, default=2, help="--lr-schedule plateau only: epochs to wait after a cut before monitoring for a new plateau again, so each cut gets a fair chance to show its effect before another one can fire. Default: 2 (Keras's own default is 0, which allows immediate back-to-back cuts).")  # noqa: E501
    parser.add_argument("--plateau-min-lr", type=float, default=0.0, help="--lr-schedule plateau only: floor - the learning rate is never cut below this. Default: 0.0 (matches Keras's own default, i.e. no floor) - set this explicitly (e.g. some fraction of --learning-rate) to keep training from grinding to a near-zero LR the way cosine's default alpha=0 does.")  # noqa: E501
    parser.add_argument("--buffer-size", help="Number of positions held in the shuffle buffer at once. Default: 400000 (~7GB at 19x19x48 planes)", type=int, default=400000)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # slightly fancier args
    parser.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training", default=None)  # noqa: E501
    parser.add_argument("--mixed-precision", help="Enable the mixed_float16 policy (fp16 compute, fp32 weights). Off by default: measured no benefit on this GPU/driver/model combo - 103.5ms/step with XLA+mixed precision together vs 103ms/step for XLA alone (statistically the same), despite this being an Ada GPU with Tensor Cores. XLA's fusion is apparently already capturing the available speedup here, leaving mixed precision nothing to add while still carrying its own numerical-stability surface (see the forced float32 softmax in policy.py). Only enable to re-test under different conditions (e.g. a larger batch size)", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--train-val-test", help="Fraction of games to use for training/val/test. Must sum to 1. Only used the first time (see game_split.json in out_directory)", nargs=3, type=float, default=[0.93, .05, .02])  # noqa: E501
    parser.add_argument("--symmetries", help="Comma-separated list of transforms, subset of noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2", default='noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2')  # noqa: E501
    parser.add_argument("--seed", help="Seed for the shuffle buffer's game order, buffer sampling, and symmetry choice (both train and val, offset by 1 from each other since they read disjoint game pools). Default: unseeded (a fresh, unrecoverable draw from OS entropy every run) - set this to make the exact position stream fed to the network reproducible, e.g. to replay a run that hit an anomaly.", type=int, default=None)  # noqa: E501
    # LR-to-collapse diagnostic (see LR_COLLAPSE_TEST_PLAN.md) - off by default, does
    # not affect normal training runs.
    parser.add_argument("--uncapped-warmup", help="Diagnostic mode: never cap/freeze WarmupCallback's linear ramp - let it keep climbing at the same slope indefinitely instead of leveling off at --learning-rate, until TerminateOnNaN or the weight_norm-multiple check (see --check-every) stops it, or --epochs runs out. Requires --lr-schedule plateau (that's the branch WarmupCallback lives in). Adds keras.callbacks.TerminateOnNaN, skips ReduceLROnPlateau entirely (no re-hold/re-cut should compete with the uncapped ramp), and enables step-granularity weight_norm/grad_norm logging (StepDiagnosticsCallback) via a monkey-patched train_step. Default: False (normal capped warmup).", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--check-every", help="--uncapped-warmup only: log weight_norm/grad_norm/loss/lr to <out_directory>/step_diagnostics.jsonl every this many steps, and check for divergence (weight_norm exceeding --divergence-norm-multiple times its value at the first checked step) at the same cadence. Default: 50 (matches the cadence the now-superseded benchmarks/_lr_collapse_test.py already validated as cheap).", type=int, default=50)  # noqa: E501
    parser.add_argument("--divergence-norm-multiple", help="--uncapped-warmup only: treat weight_norm as diverged (and stop training) once it exceeds this many times its value at the first --check-every checkpoint. Default: 5.0", type=float, default=5.0)  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    resume = args.weights is not None

    if args.uncapped_warmup and args.lr_schedule != "plateau":
        raise ValueError(
            "--uncapped-warmup requires --lr-schedule plateau (that's the branch "
            "WarmupCallback lives in - --lr-schedule cosine folds warmup into "
            "CosineDecay instead, which has no uncapped option).")

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
    # data stream (shuffle_buffer_batch_generator's game order/sampling/symmetry
    # choice) - weight init drew from Keras's own global RNG regardless of --seed,
    # silently breaking the "same --seed -> same run" assumption for anything
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

    # discover shards and build the combined game index
    shard_files = find_shard_files(args.train_data)
    if args.verbose:
        print("found {} shard files in {}".format(len(shard_files), args.train_data))
    games, dataset_features, board_size, n_features = build_game_index(
        shard_files, verbose=args.verbose)

    if dataset_features != model_features:
        raise ValueError("Model JSON file expects features \n\t%s\n"
                         "But shards contain \n\t%s" % ("\n\t".join(model_features),
                                                        "\n\t".join(dataset_features)))
    elif args.verbose:
        print("Verified that shard features and model features exactly match.")

    # ensure output directory is available
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    # Game-level train/val/test split, persisted so a resumed run uses the exact same
    # split rather than a freshly (and differently) drawn one.
    train_games, val_games, test_games = get_or_create_game_split(
        games, args.out_directory, args.train_val_test, verbose=args.verbose)

    n_train_data = sum(g["length"] for g in train_games)
    n_val_data = sum(g["length"] for g in val_games)
    n_test_data = sum(g["length"] for g in test_games)

    if args.verbose:
        print("dataset loaded")
        print("\t%d games, %d total positions" % (
            len(games), n_train_data + n_val_data + n_test_data))
        print("\t%d training games, %d training positions" % (len(train_games), n_train_data))
        print("\t%d validation games, %d validation positions" % (len(val_games), n_val_data))
        print("\t%d test games, %d test positions" % (len(test_games), n_test_data))

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

    symmetries = [BOARD_TRANSFORMATIONS[name] for name in args.symmetries.strip().split(",")]

    # If --validation-length caps validation below the full set, pick a STABLE subset of
    # val_games up front rather than letting the shuffle buffer's per-epoch reshuffle pick
    # a different random slice each time (which it would, if the generator's output were
    # simply truncated after N steps every epoch) - otherwise val_loss becomes noisy
    # epoch-to-epoch for reasons unrelated to the model actually improving, and
    # MetadataWriterCallback's best_epoch (which just tracks lowest val_loss seen) could
    # end up picking a noise-driven "best" epoch rather than a genuinely better one. The
    # existing order of val_games (from game_split.json) is already randomized once at
    # split time, so taking a prefix of it needs no further shuffling.
    if args.validation_length is not None and args.validation_length < n_val_data:
        val_games_for_eval = []
        n_val_eval = 0
        for g in val_games:
            if n_val_eval >= args.validation_length:
                break
            val_games_for_eval.append(g)
            n_val_eval += g["length"]
    else:
        val_games_for_eval = val_games
        n_val_eval = n_val_data

    # create dataset generators. Unlike the position-permutation generator this replaces
    # (which shuffles once at the very start of training and then cycles that same fixed
    # order forever), the shuffle buffer re-shuffles at the start of every full pass over
    # the data, so every epoch gets its own independent shuffle.
    # Offset the val seed by 1 from train's: they read disjoint game pools (per
    # game_split.json) so there's no real correlation risk either way, but distinct
    # seeds rule it out entirely rather than relying on that reasoning holding up.
    train_seed = args.seed
    val_seed = None if args.seed is None else args.seed + 1
    train_data_generator = sanity_checked_generator(
        shuffle_buffer_batch_generator(
            train_games, args.buffer_size, args.minibatch, board_size, n_features, symmetries,
            seed=train_seed),
        args.out_directory, "train")

    # Validation, unlike training, is small enough (tens of thousands of positions, not
    # tens of millions) to fully materialize in memory once rather than stream through
    # the infinite shuffle-buffer generator. That infinite-generator approach had a real
    # correctness bug: validation_steps = n_val_eval // minibatch (floor division) never
    # evenly divides n_val_eval, so Keras stops pulling batches partway through a full
    # pass over the shuffle buffer every single epoch, leaving a rolling handful of
    # unconsumed positions to bleed into the next epoch's validation - contradicting the
    # "fresh independent shuffle every epoch" this generator was designed to give. A
    # finite tf.data.Dataset sidesteps this entirely: every epoch sees every validation
    # position exactly once, with TF's own dataset machinery handling the reset.
    print("materializing {} validation positions into fixed arrays...".format(n_val_eval))
    X_val, Y_val = build_validation_arrays(
        val_games_for_eval, board_size, n_features, symmetries, seed=val_seed)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(args.minibatch)

    # Computed here (before building the LR schedule) rather than after model.compile():
    # CosineDecay needs the total step budget up front to shape its decay curve.
    samples_per_epoch = args.epoch_length or n_train_data
    steps_per_epoch = samples_per_epoch // args.minibatch
    total_steps = steps_per_epoch * args.epochs

    warmup_cb = None
    plateau_cb = None
    plateau_state_restorer = None
    if args.lr_schedule == "cosine":
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
        # start the optimizer there instead, skip warmup entirely (it only makes sense
        # for a genuinely fresh start), and replay history into ReduceLROnPlateau's own
        # best/wait/cooldown_counter - best doesn't reset on on_train_begin (only
        # wait/cooldown do), but all three are set explicitly here for clarity.
        if resume and meta_writer.metadata["epochs"]:
            initial_lr = meta_writer.metadata["epochs"][-1]["learning_rate"]
            resumed_best, resumed_wait, resumed_cooldown = _replay_plateau_state(
                meta_writer.metadata["epochs"], args.plateau_factor, args.plateau_patience,
                args.plateau_cooldown, args.plateau_min_lr)
        else:
            initial_lr = args.warmup_start_lr
            resumed_best, resumed_wait, resumed_cooldown = None, 0, 0
        lr_schedule = None
        sgd = SGD(learning_rate=initial_lr, momentum=args.momentum, nesterov=True)
        if not resume:
            warmup_cb = WarmupCallback(args.warmup_steps, args.warmup_start_lr, args.learning_rate,
                                       uncapped=args.uncapped_warmup)
        # --uncapped-warmup: no re-hold/re-cut behavior should compete with the
        # uncapped ramp, so ReduceLROnPlateau is skipped entirely in this mode.
        if not args.uncapped_warmup:
            plateau_cb = ReduceLROnPlateau(
                monitor="val_loss", factor=args.plateau_factor, patience=args.plateau_patience,
                cooldown=args.plateau_cooldown, min_lr=args.plateau_min_lr, verbose=1)
        if resumed_best is not None and plateau_cb is not None:
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

    step_diagnostics = None
    if args.uncapped_warmup:
        # Monkey-patch AFTER compile() (so the patched step still picks up
        # jit_compile=True via make_train_function()) and BEFORE fit() - see
        # _grad_norm_train_step's docstring for why this is a monkey-patch rather
        # than a model_class kwarg.
        model.train_step = types.MethodType(_grad_norm_train_step, model)
        step_diagnostics = StepDiagnosticsCallback(
            args.check_every, os.path.join(args.out_directory, "step_diagnostics.jsonl"),
            divergence_norm_multiple=args.divergence_norm_multiple)

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
    # diagnostics/meta_writer doesn't matter, only relative to plateau_cb.
    callbacks = [checkpointer]
    if warmup_cb is not None:
        callbacks.append(warmup_cb)
    callbacks.append(diagnostics)
    if step_diagnostics is not None:
        # Must also run before meta_writer, same reasoning as diagnostics above - it
        # sets logs["weight_norm"]/logs["grad_norm_max"] each epoch.
        callbacks.append(step_diagnostics)
        callbacks.append(TerminateOnNaN())
    if plateau_cb is not None:
        callbacks.append(plateau_cb)
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
    run_training_v2()
