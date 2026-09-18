import os
import json
import time
import numpy as np
from AlphaGo.training.xla_workarounds import ensure_xla_conv_nhwc

# Must run before the first XLA compilation, i.e. before model.compile(jit_compile=True)
# below is ever exercised - module level, ahead of everything else, is the safest place.
ensure_xla_conv_nhwc()

from keras import mixed_precision, ops, utils as keras_utils  # noqa: E402
from keras.metrics import TopKCategoricalAccuracy  # noqa: E402
from keras.optimizers import SGD  # noqa: E402
from keras.optimizers.schedules import CosineDecay, LearningRateSchedule  # noqa: E402
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau  # noqa: E402
from AlphaGo.models.policy import CNNPolicy  # noqa: E402
# Unused directly, but importing it registers ResTowerPolicy (via the @neuralnet
# decorator) so CNNPolicy.load_model() can find it by name in a model.json's "class"
# field - registration only happens when a class's defining module is actually imported
# somewhere in the process, and nothing else here pulls this one in.
import AlphaGo.models.resnet_tower_policy  # noqa: E402,F401
from AlphaGo.training.shuffle_buffer import (  # noqa: E402
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
    parser.add_argument("--val-dataset-gpu-resident", action="store_true",
                        help="Diagnostic-only escape hatch: build val_dataset WITHOUT the "
                             "CPU-pin fix, letting TF's default placement put it on GPU "
                             "again - reproduces the original ResourceExhaustedError risk. "
                             "Exists solely to A/B the CPU-pin fix's effect on training-time "
                             "GPU utilization against an otherwise-identical run (same "
                             "buffer_size, same batch size, same prefetch) - never use this "
                             "for a real training run.")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

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

    import tensorflow as tf
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
            warmup_cb = WarmupCallback(args.warmup_steps, args.warmup_start_lr, args.learning_rate)
        plateau_cb = ReduceLROnPlateau(
            monitor="val_loss", factor=args.plateau_factor, patience=args.plateau_patience,
            cooldown=args.plateau_cooldown, min_lr=args.plateau_min_lr, verbose=1)
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
