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
# Imported only to register ResTowerPolicy, so CNNPolicy.load_model() can find it by the
# "class" name in a model JSON.
import AlphaGo.models.resnet_tower_policy  # noqa: F401
from AlphaGo.training.shard_stream import (
    BATCH_TRANSFORMATIONS, find_split_shards, dataset_info, shard_batch_generator,
    validation_arrays)


def prediction_entropy(y_true, y_pred):
    """Shannon entropy of the predicted move distribution, averaged over the batch - the
    model's confidence, not its correctness (y_true is ignored). Starts near log(361) and
    falls as training progresses."""
    p = ops.clip(y_pred, 1e-7, 1.0)
    return -ops.sum(p * ops.log(p), axis=-1)


def sanity_checked_generator(base_generator, out_directory, label, check_every=50):
    """Wraps a batch generator, checking the first 5 batches and every check_every-th for
    NaNs, X outside [0, 1] and Y rows that don't sum to 1. A bad batch is printed and
    appended to out_directory/batch_sanity_log.json, so data corruption shows up at the
    step it starts - including a stream that's wrong from step 0, which epoch-level
    metrics alone don't reveal.
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
    warmup_steps batches, then stops touching it.

    --lr-schedule plateau only: ReduceLROnPlateau needs the learning rate as a plain
    variable it can assign to, not a schedule, so warmup can't be folded into the
    schedule as cosine's is.

    start_step: batches of warmup already done, for a --weights resume that stopped
    partway through warmup. Counted in batches like the ramp itself (the optimizer's
    iteration count can lag under --mixed-precision, which skips non-finite steps).
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
        """True once warmup has stopped setting the learning rate."""
        return self._step > self.warmup_steps


class LROverrideCallback(Callback):
    """--lr-schedule plateau only: at each epoch's end, if out_directory/lr_override.txt
    holds a learning rate that differs from the optimizer's, set the optimizer to it -
    a way to change the LR of a running job (e.g. a manual cut) without stopping it.

    Re-applied every epoch for as long as the file exists, so it acts as a pin. Runs
    after ReduceLROnPlateau, so an override wins over whatever that decided, and does
    nothing while warmup_cb is still ramping.
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
            # Printed even without --verbose: most likely a typo being made right now.
            print("lr_override.txt: could not parse {!r} as a float, ignoring".format(text))
            return
        current_lr = float(self.model.optimizer.learning_rate)
        if abs(override_lr - current_lr) > self.tol:
            if self.verbose:
                print("lr override file: {:.6g} -> {:.6g}".format(current_lr, override_lr))
            self.model.optimizer.learning_rate = override_lr


def load_checkpoint(model, path, with_optimizer=True):
    """Loads a weights.NNNNN.weights.h5 checkpoint into a compiled model: the weights and,
    with_optimizer, the optimizer's state at that epoch's end.

    Checkpoints written during fit() include the optimizer's variables, so a --weights
    resume continues with the interrupted run's exact optimizer: SGD momentum (resetting
    it to 0 has caused divergence here), the iteration count cosine's schedule is read at,
    plateau's current learning rate, and the mixed-precision loss scale.

    Keras only restores that state into an optimizer that is already built; otherwise,
    or when the state doesn't fit (another --lr-schedule, --mixed-precision toggled), it
    merely warns "Skipping variable loading for optimizer ..." and leaves momentum at 0.
    So the optimizer is built first and that warning is raised as an error.

    with_optimizer=False loads only the weights, leaving the optimizer fresh (the LR
    range test's warm start).
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
    """ReduceLROnPlateau's best/wait/cooldown_counter for a --weights resume, rebuilt by
    replaying its algorithm (keras.callbacks.ReduceLROnPlateau.on_epoch_end) over the
    val_loss/learning_rate history in metadata.json - so a resume remembers how close
    training already was to the next cut.
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
    """Re-applies a resumed best/wait/cooldown_counter to a ReduceLROnPlateau callback.

    ReduceLROnPlateau.on_train_begin() resets that state (wait and cooldown_counter
    always; best too since Keras 3.15), so it can't just be set before fit(). Placed
    after the plateau callback, this runs right after that reset and puts it back.
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
    """Adds per-epoch numbers Keras doesn't track to the logs (and so to metadata.json):
    - learning_rate: the LR used this epoch - from the schedule for cosine, from the
      optimizer for plateau (lr_schedule None).
    - epoch_seconds, steps_per_second: throughput, so a run slowing down shows in the data.
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
            logs["learning_rate"] = float(self.model.optimizer.learning_rate)
        elapsed = time.time() - self._epoch_start
        logs["epoch_seconds"] = elapsed
        logs["steps_per_second"] = self.steps_per_epoch / elapsed if elapsed > 0 else 0.0


class MetadataWriterCallback(Callback):
    """Appends each epoch's logs to metadata.json, tracking the best epoch by val_loss
    (loss when there is no validation)."""

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
# validation set, precision and compilation, or an LR found by one won't carry over to
# the other. Both entry points build their runs from these functions; only the
# optimizer's learning rate handling differs.

def add_run_arguments(parser):
    """Arguments every run takes, training or LR range test. --weights is left to each
    entry point, since resuming means something different to each."""
    # required args
    parser.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicy.save_model())")  # noqa: E501
    parser.add_argument("train_data", help="Output directory of convert_shuffled.py, containing train/ and val/ shard subdirectories")  # noqa: E501
    parser.add_argument("out_directory", help="directory where metadata and weights will be saved")
    # frequently used args
    parser.add_argument("--minibatch", "-B", type=int, required=True, help="Size of training data minibatches - as large as GPU memory allows; the learning rate is tuned per minibatch size.")  # noqa: E501
    parser.add_argument("--epochs", "-E", type=int, required=True, help="Total number of epochs across the WHOLE run, including any already completed before a --weights resume (not additional epochs on top of those) - this also sets the horizon of a cosine learning rate schedule or LR range test sweep.")  # noqa: E501
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Minibatch steps per epoch. An epoch is when validation runs and a checkpoint and metadata.json entry are written; --epochs and the plateau schedule's patience/cooldown count epochs. Default: one pass over the training set (positions / --minibatch)")  # noqa: E501
    parser.add_argument("--validation-length", help="Number of validation examples to check per epoch. Default: # validation data (full validation set every epoch). The first N positions of val/ are used - a uniform random sample, identical every epoch", type=int, default=None)  # noqa: E501
    parser.add_argument("--momentum", help="SGD momentum, used with Nesterov. Default: .9", type=float, default=.9)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # slightly fancier args
    parser.add_argument("--mixed-precision", default=False, action="store_true", help="Train with the mixed_float16 policy: fp16 compute, fp32 weights, with dynamic loss scaling. The final softmax stays float32 (see policy.py). Default: off (float32)")  # noqa: E501
    parser.add_argument("--symmetries", help="Comma-separated list of transforms, subset of noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2", default='noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2')  # noqa: E501
    parser.add_argument("--seed", help="Seed for weight initialization and for the per-position symmetry choices (train and val use seed and seed+1), making a run reproducible. Default: unseeded", type=int, default=None)  # noqa: E501


def set_up_run(args, resume_setting_keys, require_latest_checkpoint):
    """Everything a run needs before its optimizer exists: seeding and precision, the
    model, metadata and - on a --weights resume - the checks that the resume lines up with
    the run it continues, and the data stream and validation set.

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

    # Both before the model is built: the seed decides weight initialization as well as
    # the data stream, and the precision policy decides the dtype new layers get (the
    # final softmax stays float32 regardless - see policy.py).
    if args.seed is not None:
        keras_utils.set_random_seed(args.seed)
    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        if args.verbose:
            policy_obj = mixed_precision.global_policy()
            print("mixed precision enabled: compute dtype=%s, variable dtype=%s" %
                  (policy_obj.compute_dtype, policy_obj.variable_dtype))
    elif args.verbose:
        print("mixed precision disabled (default - see --help)")

    model, shards = _load_model_and_shards(args)
    # The checkpoint itself is loaded after compile() (see load_checkpoint); checked here
    # so a wrong path fails before the slow validation set is built.
    weights_path = os.path.join(args.out_directory, args.weights) if resume else None
    if resume and not os.path.exists(weights_path):
        raise ValueError("--weights {} not found".format(weights_path))

    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    meta_writer, epochs_already_trained = _load_metadata(
        args, resume, resume_setting_keys, require_latest_checkpoint)

    checkpoint_template = os.path.join(args.out_directory, "weights.{epoch:05d}.weights.h5")
    checkpointer = ModelCheckpoint(checkpoint_template, save_weights_only=True)

    steps_per_epoch, total_steps, train_data_generator, val_dataset = _open_data(
        args, shards, epochs_already_trained)
    # Data seen per epoch, for comparing runs with different --minibatch sizes (whose
    # epochs then cover different amounts of data).
    meta_writer.metadata["positions_per_epoch"] = steps_per_epoch * args.minibatch

    return types.SimpleNamespace(
        model=model, resume=resume, weights_path=weights_path, meta_writer=meta_writer,
        epochs_already_trained=epochs_already_trained, steps_per_epoch=steps_per_epoch,
        total_steps=total_steps, train_data_generator=train_data_generator,
        val_dataset=val_dataset, checkpointer=checkpointer)


def _load_model_and_shards(args):
    """The model from its JSON, and the train/ and val/ shards, checked to be built with
    the feature planes the model expects."""
    policy = CNNPolicy.load_model(args.model)
    model_features = policy.preprocessor.get_feature_list()

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
        print("dataset loaded from %s" % args.train_data)
        print("\t%d training positions in %d shards" % (sum(train_sizes), len(train_shards)))
        print("\t%d validation positions in %d shards" % (sum(val_sizes), len(val_shards)))

    shards = types.SimpleNamespace(train=train_shards, train_sizes=train_sizes,
                                   val=val_shards, val_sizes=val_sizes,
                                   board_size=board_size)
    return policy.model, shards


def _load_metadata(args, resume, resume_setting_keys, require_latest_checkpoint):
    """The run's metadata writer - carrying on a resumed run's metadata.json - after the
    checks that a resume lines up with it. Returns (meta_writer, epochs_already_trained).
    """
    meta_file = os.path.join(args.out_directory, "metadata.json")
    meta_writer = MetadataWriterCallback(meta_file)
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
        # Settings that would silently misalign the continued run if changed (which ones
        # depends on the entry point - see resume_setting_keys where it's called from).
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
        # The data stream position, epoch numbering and plateau replay all follow the
        # metadata's epoch count, so the checkpoint must be from that same epoch. A
        # --weights name outside the checkpoint pattern can't be checked.
        match = re.fullmatch(r"weights\.(\d+)\.weights\.h5", os.path.basename(args.weights))
        if match and int(match.group(1)) != epochs_already_trained:
            raise ValueError(
                "--weights {} is the checkpoint from epoch {}, but {} records {} completed "
                "epochs: resume from the latest checkpoint.".format(
                    args.weights, int(match.group(1)), meta_file, epochs_already_trained))

    meta_writer.metadata["training_data"] = args.train_data
    meta_writer.metadata["model_file"] = args.model
    # A list, so every invocation of a resumed run is recorded.
    meta_writer.metadata["cmd_line_args"] = meta_writer.metadata.get("cmd_line_args", [])
    meta_writer.metadata["cmd_line_args"].append(vars(args))
    return meta_writer, epochs_already_trained


def _open_data(args, shards, epochs_already_trained):
    """The training stream - resuming at the position an uninterrupted run would have
    reached - and the fixed validation set. Returns (steps_per_epoch, total_steps,
    train_data_generator, val_dataset)."""
    symmetries = args.symmetries.strip().split(",")
    unknown = [name for name in symmetries if name not in BATCH_TRANSFORMATIONS]
    if unknown:
        raise ValueError("unknown symmetries: {}".format(unknown))

    steps_per_epoch = args.steps_per_epoch or sum(shards.train_sizes) // args.minibatch
    if steps_per_epoch < 1:
        raise ValueError("an epoch must be at least one step (--steps-per-epoch {}, "
                         "--minibatch {}, {} training positions)".format(
                             args.steps_per_epoch, args.minibatch, sum(shards.train_sizes)))
    total_steps = steps_per_epoch * args.epochs
    start_position = epochs_already_trained * steps_per_epoch * args.minibatch
    if args.verbose and start_position:
        print("resuming the training stream at position %d" % start_position)

    # train and val use distinct seeds so their symmetry choices are independent
    train_seed = args.seed
    val_seed = None if args.seed is None else args.seed + 1
    train_data_generator = sanity_checked_generator(
        shard_batch_generator(shards.train, shards.train_sizes, args.minibatch,
                              shards.board_size, symmetries, seed=train_seed,
                              start_position=start_position),
        args.out_directory, "train")

    # Validation is a fixed prefix of val/: the shards are already a uniform random
    # sample, and a fixed set means val_loss only moves because the model does.
    n_val_data = sum(shards.val_sizes)
    n_val_eval = min(args.validation_length or n_val_data, n_val_data)
    print("materializing {} validation positions into fixed arrays...".format(n_val_eval))
    X_val, Y_val = validation_arrays(shards.val, shards.val_sizes, n_val_eval,
                                     shards.board_size, symmetries, seed=val_seed)
    # Built on the CPU: from_tensor_slices() embeds the arrays in the graph, and TF would
    # otherwise place them in GPU memory whole - ~7 GB at 100k positions, enough to run
    # minibatch-1024 training out of GPU memory. Batches then stream to the GPU as the
    # training data does.
    with tf.device('/cpu:0'):
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(args.minibatch)
    # The dataset holds its own copy; dropping the originals keeps host memory from
    # holding it twice (at 100k positions that got the process OOM-killed).
    del X_val, Y_val
    import gc
    gc.collect()
    return steps_per_epoch, total_steps, train_data_generator, val_dataset


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
    Keras passes one logs dict through every callback for an event, in list order:
    - lr_callbacks (whatever sets the LR per batch: warmup, the range test's sweep).
    - TrainingDiagnosticsCallback, logging the LR used this epoch - so before any
      epoch_callbacks that change it for the next epoch.
    - epoch_callbacks (ReduceLROnPlateau, the LR override), in the order given.
    - the checkpointer, after them: its optimizer state then holds the LR they leave for
      the next epoch, so a resume keeps a cut made in the last one.
    - restore_callbacks, re-applying resumed state after the epoch_callbacks'
      on_train_begin resets it (see _PlateauStateRestorer).
    - meta_writer last: it saves the logs as they end up, and only once the epoch's
      checkpoint exists.
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
    parser.add_argument("--learning-rate", "-r", type=float, required=True, help="Peak learning rate, reached at the end of warmup - find a candidate for this model and minibatch size with an LR range test (AlphaGo/training/lr_range_test.py), then confirm it with a training run.")  # noqa: E501
    parser.add_argument("--warmup-steps", type=int, required=True, help="Number of minibatch steps to linearly ramp the learning rate from --warmup-start-lr up to --learning-rate, before cosine decay or plateau monitoring begins. Counted in steps, like --steps-per-epoch.")  # noqa: E501
    parser.add_argument("--warmup-start-lr", help="Learning rate at step 0, before warmup begins. Default: .0001", type=float, default=.0001)  # noqa: E501
    parser.add_argument("--lr-schedule", choices=["cosine", "plateau"], required=True, help="How the learning rate decays after warmup. 'cosine': cosine decay to 0 over the whole run, shaped by --epochs up front. 'plateau': hold at --learning-rate and cut by --plateau-factor whenever val_loss stops improving for --plateau-patience epochs, down to --plateau-min-lr - reacts to the real training curve instead of a fixed shape.")  # noqa: E501
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="--lr-schedule plateau only: multiplier applied to the learning rate on each cut. Default: 0.5 (a 2x cut; Keras's own default of 0.1 is harsher than training here tolerated well).")  # noqa: E501
    parser.add_argument("--plateau-patience", type=int, default=5, help="--lr-schedule plateau only: epochs with no val_loss improvement before a cut. Counts epochs as set by --steps-per-epoch, so choose it relative to that. Default: 5")  # noqa: E501
    parser.add_argument("--plateau-cooldown", type=int, default=5, help="--lr-schedule plateau only: epochs after a cut before monitoring resumes, so each cut can show its effect. Default: 5")  # noqa: E501
    parser.add_argument("--plateau-min-lr", type=float, default=1e-5, help="--lr-schedule plateau only: the learning rate is never cut below this. Default: 1e-5")  # noqa: E501
    parser.add_argument("--plateau-min-delta", type=float, default=0.005, help="--lr-schedule plateau only: minimum val_loss improvement that counts as 'still improving'. Default: 0.005 - Keras's own 1e-4 is below this project's epoch-to-epoch val_loss noise, so noise alone kept resetting the patience count.")  # noqa: E501
    return parser


# Settings a training resume must keep (see _load_metadata):
# - minibatch, steps_per_epoch: together they place the data stream (positions already
#   seen = epochs x steps x minibatch), and steps_per_epoch with --epochs sets the cosine
#   schedule's horizon.
# - warmup_steps: the warmup length, and cosine's decay_steps.
# - lr_schedule: the checkpoint's optimizer state differs in shape per schedule.
TRAINING_RESUME_SETTINGS = ("minibatch", "steps_per_epoch", "warmup_steps", "lr_schedule")


def _cosine_schedule(args, run):
    """Linear warmup from warmup_start_lr to learning_rate over warmup_steps, then cosine
    decay to 0 by the end of the run. CosineDecay's decay_steps counts only the decay
    phase after warmup, so it's total_steps minus warmup_steps.

    On a resume the checkpoint restores the optimizer's iteration count, which the
    schedule is evaluated at, so the same curve continues.

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
    """Warmup via WarmupCallback, then ReduceLROnPlateau cutting a plain learning-rate
    variable (no schedule object - so lr_schedule is None), with LROverrideCallback able to
    set it by hand.

    On a resume the checkpoint restores the learning rate wherever the plateau callback or
    an override left it; a resume still inside warmup continues the ramp instead; and
    ReduceLROnPlateau's own best/wait/cooldown_counter are replayed from metadata.json
    (_replay_plateau_state) and re-applied at fit() start (_PlateauStateRestorer).

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
    lr_override_cb = LROverrideCallback(
        args.out_directory, warmup_cb=warmup_cb, verbose=args.verbose)
    restore_callbacks = []
    if resumed_best is not None:
        restore_callbacks.append(_PlateauStateRestorer(
            plateau_cb, resumed_best, resumed_wait, resumed_cooldown))
        if args.verbose:
            print("resuming plateau state: best={:.4f} wait={}/{} cooldown_counter={}"
                  .format(resumed_best, resumed_wait, args.plateau_patience,
                          resumed_cooldown))
    return sgd, None, {
        "lr_callbacks": [warmup_cb] if warmup_cb is not None else [],
        # override after plateau: it wins over whatever plateau just decided
        "epoch_callbacks": [plateau_cb, lr_override_cb],
        "restore_callbacks": restore_callbacks}


def run_training(cmd_line_args=None):
    """Run training; command-line args may be passed in as a list.

    Momentum+Nesterov SGD with warmup and a cosine or plateau learning-rate schedule,
    streaming the output of convert_shuffled.py: train/ and val/ shards, split by game
    upstream (select_games.py), each already one random permutation of its positions, so
    training reads train/ start to finish with a random board symmetry per position. With
    --seed the stream is deterministic, so a resumed run continues from exactly the
    position an uninterrupted run would have reached.

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
