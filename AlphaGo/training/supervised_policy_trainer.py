import os
import json
import re
import time
import types
import warnings

import h5py
import numpy as np
import tensorflow as tf  # noqa: F401

from keras import mixed_precision, ops, utils as keras_utils
from keras.metrics import TopKCategoricalAccuracy
from keras.optimizers import SGD
from keras.optimizers.schedules import CosineDecay
from keras.callbacks import (
    ModelCheckpoint, Callback, ReduceLROnPlateau)
from AlphaGo.models.policy import CNNPolicy
# Unused directly, but importing it registers ResTowerPolicy (via the @neuralnet
# decorator) so CNNPolicy.load_model() can find it by name in a model.json's "class"
# field - registration only happens when a class's defining module is actually imported
# somewhere in the process, and nothing else here pulls this one in.
import AlphaGo.models.resnet_tower_policy  # noqa: F401
from AlphaGo.training.shard_stream import (
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

    start_step: batches of warmup already done, for a --weights resume that stopped
    partway through warmup - the ramp then continues exactly where the interrupted run's
    left off. Counted in batches, as the ramp itself is (the optimizer's own iteration
    count can lag it under --mixed-precision, which skips steps with non-finite
    gradients).
    """

    def __init__(self, warmup_steps, start_lr, target_lr, start_step=0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.target_lr = target_lr
        self._step = start_step

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


def load_checkpoint(model, path, with_optimizer=True):
    """Loads a weights.NNNNN.weights.h5 checkpoint into a compiled model - the weights,
    and (with_optimizer) the optimizer's state as it was at that epoch's end.

    A checkpoint saved while the model is compiled - always the case for ModelCheckpoint
    during fit() - holds the optimizer's variables alongside the weights, so a --weights
    resume continues with the exact optimizer the interrupted run had:
    - SGD momentum. Resetting it to 0 on resume has been a real, repeated source of
      trouble in this project - an un-cushioned LR jump onto a momentum-less optimizer
      diverged to nan the first time a manual LR cut was tried via resume, and even a
      cushioned (warmup-ramped) resume afterward still showed several epochs of visibly
      disrupted training loss/entropy before settling.
    - the iteration counter, which --lr-schedule cosine's CosineDecay reads its position
      on the warmup+decay curve from - restoring it continues the one whole-run curve.
    - under --lr-schedule plateau, the learning_rate itself - a plain variable there,
      holding whatever ReduceLROnPlateau / lr_override.txt left it at, including a cut
      made at the end of the last epoch (the checkpoint callback runs after both).
    - under --mixed-precision, the wrapping LossScaleOptimizer's dynamic loss-scale state.

    Keras only restores that state into an optimizer that is already built, and
    otherwise does NOT raise: it warns "Skipping variable loading for optimizer ..." and
    leaves momentum at 0. The same warning is all a mismatched checkpoint produces (other
    --lr-schedule, --mixed-precision toggled). So the optimizer is built explicitly
    first, and that warning is turned into an error.

    with_optimizer=False loads only the weights and leaves the optimizer fresh
    (--lr-range-test starts every sweep with one).
    """
    if not with_optimizer:
        model.load_weights(path, objects_to_skip=[model.optimizer])
        return
    with h5py.File(path, "r") as f:
        if "optimizer" not in f:
            raise ValueError(
                "{} holds no optimizer state (it was saved from an uncompiled model), so a "
                "resume from it would start with momentum at 0.".format(path))
    model.optimizer.build(model.trainable_variables)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.load_weights(path)
    skipped = [w for w in caught if "Skipping variable loading for optimizer" in str(w.message)]
    for w in caught:
        if w not in skipped:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if skipped:
        raise ValueError(
            "the optimizer state in {} doesn't match this run's optimizer - was it saved "
            "with a different --lr-schedule or --mixed-precision setting? ({})".format(
                path, skipped[0].message))


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
            # float32, as ReduceLROnPlateau compares: the logged LR is the optimizer's
            # float32 variable, which lands a hair above a plain-float min_lr once cut to it.
            if wait >= patience and old_lr > np.float32(min_lr):
                cooldown_counter = cooldown
                wait = 0
    return best, wait, cooldown_counter


class _PlateauStateRestorer(Callback):
    """Re-applies a resumed best/wait/cooldown_counter onto a ReduceLROnPlateau callback.

    ReduceLROnPlateau.on_train_begin() resets its state, so setting it on the plateau
    callback before model.fit() starts would just get silently overwritten the moment
    training begins. What it resets depends on the Keras version: wait and
    cooldown_counter always, and since Keras 3.15 best as well (set to None, which makes
    the first epoch an automatic "improvement"). Keras calls on_train_begin on every
    callback in list order, once per model.fit() call, so placing an instance of this
    AFTER the plateau callback in the callbacks list makes it run second, re-applying all
    three resumed values right after the plateau callback's own reset.
    """

    def __init__(self, plateau_cb, best, wait, cooldown_counter):
        super().__init__()
        self.plateau_cb = plateau_cb
        self.best = best
        self.wait = wait
        self.cooldown_counter = cooldown_counter

    def on_train_begin(self, logs=None):
        self.plateau_cb.best = self.best
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


# --- the run pipeline, shared with lr_range_test.py ----------------------------------------
#
# A training run and an LR range test must see exactly the same model, data stream,
# validation set, precision and compilation - otherwise an LR found by one doesn't carry
# over to the other (an earlier range-test script that kept its own copy of this setup
# drifted from the trainer and gave misleading results). Both entry points therefore build
# their runs from these functions; only the optimizer's learning rate handling differs.

def add_run_arguments(parser):
    """Arguments every run takes, training or LR range test. --weights is left to each
    entry point, since resuming means something different to each."""
    # required args
    parser.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicy.save_model())")  # noqa: E501
    parser.add_argument("train_data", help="Output directory of convert_shuffled.py, containing train/ and val/ shard subdirectories")  # noqa: E501
    parser.add_argument("out_directory", help="directory where metadata and weights will be saved")
    # frequently used args
    parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 256", type=int, default=256)  # noqa: E501
    parser.add_argument("--epochs", "-E", help="Total number of iterations on the data across the WHOLE run, including any epochs already completed before a --weights resume (not additional epochs on top of those) - this also shapes the learning rate curve's horizon. Default: 20", type=int, default=20)  # noqa: E501
    parser.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)  # noqa: E501
    parser.add_argument("--validation-length", help="Number of validation examples to check per epoch. Default: # validation data (full validation set every epoch). The first N positions of val/ are used - a uniform random sample, identical every epoch", type=int, default=None)  # noqa: E501
    parser.add_argument("--momentum", help="SGD momentum, used with Nesterov. Default: .9", type=float, default=.9)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # slightly fancier args
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


def set_up_run(args, resume_setting_keys, require_latest_checkpoint):
    """Everything a run needs before its optimizer exists: seeding and precision, the
    model, the data stream and validation set, metadata, and - on a --weights resume -
    the checks that the resume lines up with the run it continues.

    resume_setting_keys: the args that must be unchanged since the previous invocation
    recorded in metadata.json. require_latest_checkpoint: refuse a --weights checkpoint
    older than the metadata's last epoch.

    Returns a SimpleNamespace: model, resume, weights_path, meta_writer,
    epochs_already_trained, steps_per_epoch, total_steps, train_data_generator,
    val_dataset, checkpointer.
    """
    resume = args.weights is not None

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
    # On a resume the checkpoint is loaded after compile() (see load_checkpoint) - only
    # then is there an optimizer for its saved state to go into. Checked here so a wrong
    # path fails before the (slow) validation set is materialized.
    weights_path = os.path.join(args.out_directory, args.weights) if resume else None
    if resume and not os.path.exists(weights_path):
        raise ValueError("--weights {} not found".format(weights_path))

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
        # Settings a resume must keep: each would otherwise silently misalign the
        # continued run with no error, so they're checked explicitly rather than trusted.
        # Which ones is up to the entry point (see resume_setting_keys at its call site).
        prev_args = meta_writer.metadata["cmd_line_args"][-1]
        for key in resume_setting_keys:
            prev_value = prev_args.get(key)
            cur_value = getattr(args, key)
            if prev_value != cur_value:
                raise ValueError(
                    "--{} changed across resume ({} -> {}): this would silently misalign "
                    "the continued run - see the resume checks where it's run from. Keep "
                    "it identical across a resume, or start a fresh out_directory.".format(
                        key.replace('_', '-'), prev_value, cur_value))

    if resume and require_latest_checkpoint:
        # Everything about the resume point is keyed off epochs_already_trained (the
        # metadata's epoch count): the data stream position, the epoch numbering, the
        # plateau replay. Resuming from an older checkpoint would pair its weights and
        # optimizer state with all of those from a later epoch, so the checkpoint must be
        # from that same epoch. A --weights name outside the checkpoint pattern can't be
        # checked and is taken at its word.
        match = re.fullmatch(r"weights\.(\d+)\.weights\.h5", os.path.basename(args.weights))
        if match and int(match.group(1)) != epochs_already_trained:
            raise ValueError(
                "--weights {} is the checkpoint from epoch {}, but {} records {} completed "
                "epochs: resume from the latest checkpoint.".format(
                    args.weights, int(match.group(1)), meta_file, epochs_already_trained))

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

    return types.SimpleNamespace(
        model=model, resume=resume, weights_path=weights_path, meta_writer=meta_writer,
        epochs_already_trained=epochs_already_trained, steps_per_epoch=steps_per_epoch,
        total_steps=total_steps, train_data_generator=train_data_generator,
        val_dataset=val_dataset, checkpointer=checkpointer)


def compile_model(model, optimizer):
    """The one compile() every run uses: loss, metrics and XLA."""
    # jit_compile=True (XLA): measured 102.5ms/step vs 113ms/step without XLA.
    model.compile(
        loss='categorical_crossentropy', optimizer=optimizer,
        metrics=["accuracy", TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
                 prediction_entropy],
        jit_compile=True)


def fit_run(run, args, lr_schedule, lr_callbacks=(), epoch_callbacks=(),
            restore_callbacks=()):
    """model.fit() over the run's data, with the callbacks in the order they depend on.

    Order matters: Keras passes the same logs dict through every callback for a given
    event, in list order.
    - lr_callbacks (whatever sets the LR per batch - warmup, the range test's sweep) come
      first.
    - diagnostics (TrainingDiagnosticsCallback) next: it logs the LR used this epoch, so
      it must run before any epoch_callbacks that change the LR for the next epoch
      (ReduceLROnPlateau, the LR override) - otherwise the logged value would show next
      epoch's LR against this epoch's loss.
    - epoch_callbacks, in the order given.
    - the checkpointer after them: under plateau the checkpoint's optimizer state includes
      the learning_rate variable, which must be the value they leave for the next epoch,
      so a resume picks up a cut made in the last epoch.
    - restore_callbacks (on_train_begin re-application of resumed state - see
      _PlateauStateRestorer) after the epoch_callbacks whose own on_train_begin resets
      that state.
    - meta_writer last, persisting whatever is in logs by then - and only after the
      checkpoint was written, so metadata.json never records an epoch without one.
    """
    diagnostics = TrainingDiagnosticsCallback(lr_schedule, run.steps_per_epoch)
    callbacks = (list(lr_callbacks) + [diagnostics] + list(epoch_callbacks) +
                 [run.checkpointer] + list(restore_callbacks) + [run.meta_writer])
    if args.verbose:
        print("STARTING TRAINING")
    run.model.fit(
        x=run.train_data_generator,
        steps_per_epoch=run.steps_per_epoch,
        epochs=args.epochs,
        initial_epoch=run.epochs_already_trained,
        callbacks=callbacks,
        validation_data=run.val_dataset)


# --- training ----------------------------------------------------------------------------

def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network (tuned large-batch recipe: momentum+Nesterov, warmup+cosine LR, streamed pre-shuffled shards).')  # noqa: E501
    add_run_arguments(parser)
    parser.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training. Must be the latest checkpoint (weights.NNNNN.weights.h5 with NNNNN = epochs recorded in metadata.json), which also restores the optimizer's state (momentum, step count, learning rate)", default=None)  # noqa: E501
    parser.add_argument("--learning-rate", "-r", help="Peak learning rate, reached at the end of warmup. Default: .055 (from an LR range test on this model/data/optimizer - see AlphaGo/training/lr_range_test.py; loss was stable through .167, unstable by .183, so .055 is ~1/3 of the instability threshold)", type=float, default=.055)  # noqa: E501
    parser.add_argument("--warmup-steps", help="Number of steps to linearly warm up the learning rate over before decay (cosine) or plateau-monitoring (plateau) begins. Default: 1500", type=int, default=1500)  # noqa: E501
    parser.add_argument("--warmup-start-lr", help="Learning rate at step 0, before warmup begins. Default: .0001", type=float, default=.0001)  # noqa: E501
    parser.add_argument("--lr-schedule", choices=["cosine", "plateau"], default="cosine", help="How the learning rate decays after warmup. 'cosine' (default): smooth cosine decay to 0 over the whole run, shaped by --epochs up front - the original behavior. 'plateau': after warmup, hold at --learning-rate and let keras.callbacks.ReduceLROnPlateau cut it (by --plateau-factor) whenever val_loss stops improving for --plateau-patience epochs, floored at --plateau-min-lr - reacts to the real training curve instead of committing to a fixed decay shape in advance, and (with a nonzero --plateau-min-lr) never crushes all the way to 0 the way cosine's default alpha=0 does.")  # noqa: E501
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="--lr-schedule plateau only: multiplier applied to the learning rate on each plateau cut (new_lr = lr * factor). Default: 0.5 (a 2x cut) - gentler than Keras's own ReduceLROnPlateau default of 0.1 (10x), chosen because this project's LR range tests found training fairly sensitive to large LR swings.")  # noqa: E501
    parser.add_argument("--plateau-patience", type=int, default=5, help="--lr-schedule plateau only: epochs with no val_loss improvement before cutting the learning rate. Default: 5. Counts in *epochs* as shaped by --epoch-length, not real dataset passes - a small --epoch-length reacts faster in wall-clock terms but each epoch's val_loss reading is noisier (fewer steps backing it), so pick patience relative to whatever --epoch-length this run actually uses.")  # noqa: E501
    parser.add_argument("--plateau-cooldown", type=int, default=2, help="--lr-schedule plateau only: epochs to wait after a cut before monitoring for a new plateau again, so each cut gets a fair chance to show its effect before another one can fire. Default: 2 (Keras's own default is 0, which allows immediate back-to-back cuts).")  # noqa: E501
    parser.add_argument("--plateau-min-lr", type=float, default=0.0, help="--lr-schedule plateau only: floor - the learning rate is never cut below this. Default: 0.0 (matches Keras's own default, i.e. no floor) - set this explicitly (e.g. some fraction of --learning-rate) to keep training from grinding to a near-zero LR the way cosine's default alpha=0 does.")  # noqa: E501
    parser.add_argument("--plateau-min-delta", type=float, default=0.005, help="--lr-schedule plateau only: minimum val_loss improvement to count as 'still improving' and reset --plateau-patience's wait counter. Default: 0.005 - NOT Keras's own default of 1e-4, which measured 30-50x smaller than this project's real epoch-to-epoch val_loss noise (stdev ~0.0048-0.0162 across the b15c192 mb1024 lr1p6 run's two LR phases). At 1e-4, noise alone registers a 'new best' often enough that the patience counter rarely reaches --plateau-patience even during a genuine multi-epoch plateau - that run needed a manual LR cut at epoch 36 and ground for 39 more epochs (avg 0.0014/epoch) before the next one. 0.005 sits just above the quieter (lower-LR) phase's noise floor and comfortably below real early-training gains, so it filters noise-driven bests without masking genuine progress.")  # noqa: E501
    return parser


# Settings a training resume must keep (see set_up_run):
# - minibatch, epoch_length: steps_per_epoch, which places the training stream's
#   start_position and (with --epochs) the cosine schedule's decay horizon, while the
#   restored optimizer iteration count says where on that curve the run is.
# - warmup_steps: the warmup length (both schedules) and the cosine schedule's decay_steps.
# - lr_schedule: the checkpoint's optimizer state has a different shape per schedule
#   (plateau's learning_rate is a variable, cosine's is computed from iterations).
TRAINING_RESUME_SETTINGS = ("minibatch", "epoch_length", "warmup_steps", "lr_schedule")


def _cosine_schedule(args, run):
    """Warmup (linear, warmup_start_lr -> learning_rate over warmup_steps) then cosine
    decay to zero over the rest of total_steps, replacing InverseTimeDecay's
    lr/(1+decay*step) - InverseTimeDecay's decay_rate was tuned for a step count this
    recipe doesn't use, whereas cosine-to-a-known-horizon needs no per-run retuning.

    CosineDecay's decay_steps is the length of the decay phase AFTER warmup ends, not
    the total schedule length - confirmed empirically (a schedule with warmup_steps=10,
    decay_steps=100 reaches its floor at step 110, not step 100). So decay_steps here is
    total_steps minus the warmup already spent, not total_steps itself - otherwise the
    schedule would run warmup_steps longer than intended and never reach its floor
    within the actual training run.

    On a resume, the optimizer state loaded from the checkpoint (see load_checkpoint)
    brings back the iteration counter this schedule is evaluated at, so the run
    continues the same curve.

    Returns (optimizer, lr_schedule, fit_run callback groups).
    """
    lr_schedule = CosineDecay(
        initial_learning_rate=args.warmup_start_lr,
        decay_steps=max(1, run.total_steps - args.warmup_steps),
        warmup_target=args.learning_rate,
        warmup_steps=args.warmup_steps)
    sgd = SGD(learning_rate=lr_schedule, momentum=args.momentum, nesterov=True)
    return sgd, lr_schedule, {}


def _plateau_schedule(args, run):
    """No LearningRateSchedule object - ReduceLROnPlateau mutates the optimizer's
    learning_rate directly (new_lr = old_lr * factor), which requires a plain mutable
    value rather than a schedule. Warmup is therefore done via a callback
    (WarmupCallback) instead of folded into the schedule; lr_schedule stays None so
    TrainingDiagnosticsCallback knows to read the live optimizer value instead of
    calling a schedule object.

    Resume handling: the optimizer state loaded from the checkpoint brings back the
    learning_rate wherever ReduceLROnPlateau / lr_override.txt left it, so the
    optimizer's initial value here only matters for a fresh start. A resume that
    stopped partway through warmup continues the ramp from where it was (the
    checkpoint's LR is overwritten on the first batch, as warmup owns the LR until
    it's done); one past warmup gets no warmup callback at all.
    ReduceLROnPlateau's own best/wait/cooldown_counter bookkeeping isn't optimizer
    state, so it's replayed from the metadata history instead - see
    _replay_plateau_state, and re-applied at the start of fit() by
    _PlateauStateRestorer.

    Returns (optimizer, lr_schedule, fit_run callback groups).
    """
    epoch_logs = run.meta_writer.metadata["epochs"]
    if run.resume and epoch_logs:
        resumed_best, resumed_wait, resumed_cooldown = _replay_plateau_state(
            epoch_logs, args.plateau_factor, args.plateau_patience,
            args.plateau_cooldown, args.plateau_min_lr, min_delta=args.plateau_min_delta)
    else:
        resumed_best, resumed_wait, resumed_cooldown = None, 0, 0
    sgd = SGD(learning_rate=args.warmup_start_lr, momentum=args.momentum, nesterov=True)

    warmup_cb = None
    # <=, not <: the ramp sets the target LR itself on step warmup_steps.
    completed_steps = run.epochs_already_trained * run.steps_per_epoch
    if completed_steps <= args.warmup_steps:
        warmup_cb = WarmupCallback(args.warmup_steps, args.warmup_start_lr,
                                   args.learning_rate, start_step=completed_steps)
    plateau_cb = ReduceLROnPlateau(
        monitor="val_loss", factor=args.plateau_factor, patience=args.plateau_patience,
        cooldown=args.plateau_cooldown, min_lr=args.plateau_min_lr,
        min_delta=args.plateau_min_delta, verbose=1)
    # After plateau_cb (see its docstring): an override always wins over whatever
    # plateau_cb just decided this epoch, rather than being clobbered by it.
    lr_override_cb = LROverrideCallback(
        args.out_directory, warmup_cb=warmup_cb, verbose=args.verbose)
    restore_callbacks = []
    if resumed_best is not None:
        # Can't just be set on plateau_cb here - its own on_train_begin() resets them
        # the moment model.fit() starts. _PlateauStateRestorer re-applies them right
        # after that reset (fit_run places it after plateau_cb).
        restore_callbacks.append(_PlateauStateRestorer(
            plateau_cb, resumed_best, resumed_wait, resumed_cooldown))
        if args.verbose:
            print("resuming plateau state: best={:.4f} wait={}/{} cooldown_counter={}"
                  .format(resumed_best, resumed_wait, args.plateau_patience,
                          resumed_cooldown))
    return sgd, None, {
        "lr_callbacks": [warmup_cb] if warmup_cb is not None else [],
        "epoch_callbacks": [plateau_cb, lr_override_cb],
        "restore_callbacks": restore_callbacks}


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

    The learning rate to train at comes from an LR range test on the same pipeline - see
    AlphaGo/training/lr_range_test.py.
    """
    args = build_parser().parse_args(cmd_line_args)
    run = set_up_run(args, TRAINING_RESUME_SETTINGS, require_latest_checkpoint=True)
    schedule = _cosine_schedule if args.lr_schedule == "cosine" else _plateau_schedule
    optimizer, lr_schedule, callback_groups = schedule(args, run)
    compile_model(run.model, optimizer)
    if run.resume:
        # After compile() - see load_checkpoint.
        load_checkpoint(run.model, run.weights_path)
        if args.verbose:
            print("loaded weights and optimizer state from {}".format(run.weights_path))
    fit_run(run, args, lr_schedule, **callback_groups)


if __name__ == '__main__':
    run_training()
