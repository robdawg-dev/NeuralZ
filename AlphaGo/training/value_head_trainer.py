"""Train a value head on a trained policy network.

    python -m AlphaGo.training.value_head_trainer <model.json> <weights.h5> \\
        <shards_dir> <out_directory> --minibatch 512 --epochs 20 --steps-per-epoch 2000

<model.json> is either a policy network - PolicyValueNet.from_policy() then adds a new
value head - or a PolicyValueNet from an earlier run, to train further. It fits the value
and score outputs to the targets add_value_targets.py wrote beside the shards
(<split>/value_NNNNN.h5, one row per shard record); positions without a target
(has_target 0) get sample weight 0.

--trainable-blocks 0 (default) trains the value head alone, the policy network frozen.
--trainable-blocks N also trains the last N residual blocks and the policy head, with the
policy's own loss (the move played, under the same board symmetry) alongside the value
losses, so those layers stay good at predicting moves while learning features that help
the value. The starting model is evaluated on the validation set before training, so a
change in policy accuracy shows against the starting network's.

The data stream is shard_stream's: the shards read in order, bit-packed planes unpacked on
the GPU under a per-position board symmetry (which changes the planes and the move label,
but not the value, score or komi).

Each epoch writes <out_directory>/model.json + weights.NNNNN.weights.h5 - the whole
policy+value network, loadable with NeuralNetBase.load_model - and appends the epoch's
metrics to metadata.json. See VALUE_HEAD_PLAN.md.
"""
import argparse
import json
import os
import time

import h5py as h5
import numpy as np
import tensorflow as tf
import keras
from keras import mixed_precision, ops
from keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers.schedules import CosineDecay

from AlphaGo.models.nn_util import NeuralNetBase
from AlphaGo.models.value import (PolicyValueNet, VALUE, SCORE, KOMI_SCALE,
                                  set_trainable_blocks)
from AlphaGo.preprocessing.add_value_targets import sidecar_path
from AlphaGo.training.shard_stream import (
    BATCH_TRANSFORMATIONS, PACKED_STATES, _symmetry_choices, _resolve_seed,
    dataset_info, decode_on_device, encode_labels, find_split_shards)

_OPEN_FILES = 4


class ValueReader(object):
    """Contiguous reads from shards and their value sidecars, as one endless wrapping
    array: read() returns (packed states, actions, komi, value, score, has_target)."""

    def __init__(self, shards, sizes):
        self.shards = shards
        self.starts = np.concatenate([[0], np.cumsum(sizes)])
        self.total = int(self.starts[-1])
        self.handles = {}
        for shard, n in zip(shards, sizes):
            with h5.File(sidecar_path(shard), "r") as f:
                if len(f["value"]) != n:
                    raise ValueError("{} has {} rows, its shard {}".format(
                        sidecar_path(shard), len(f["value"]), n))

    def _files(self, i):
        if i not in self.handles:
            self.handles[i] = (h5.File(self.shards[i], "r"),
                               h5.File(sidecar_path(self.shards[i]), "r"))
            while len(self.handles) > _OPEN_FILES:
                for f in self.handles.pop(next(iter(self.handles))):
                    f.close()
        return self.handles[i]

    def read(self, position, n):
        cols = {k: [] for k in ("packed", "actions", "komi", "value", "score", "has")}
        while n > 0:
            p = position % self.total
            i = int(np.searchsorted(self.starts, p, side="right") - 1)
            offset = p - int(self.starts[i])
            take = min(n, int(self.starts[i + 1]) - p)
            shard, side = self._files(i)
            sl = slice(offset, offset + take)
            cols["packed"].append(shard[PACKED_STATES][sl])
            cols["actions"].append(shard["actions"][sl])
            cols["komi"].append(side["komi"][sl])
            cols["value"].append(side["value"][sl])
            cols["score"].append(side["score"][sl])
            cols["has"].append(side["has_target"][sl])
            position += take
            n -= take
        out = {k: np.concatenate(v) for k, v in cols.items()}
        return (out["packed"], out["actions"], out["komi"].astype(np.float32)[:, None],
                out["value"].astype(np.float32)[:, None], out["score"].astype(np.float32)[:, None],
                out["has"].astype(np.float32))

    def close(self):
        for files in self.handles.values():
            for f in files:
                f.close()
        self.handles.clear()


def _batch(reader, position, n, seed, symmetries, board_size, with_policy):
    packed, actions, komi, value, score, has = reader.read(position, n)
    choices = _symmetry_choices(seed, position, n, len(symmetries)).astype(np.int32)
    # tuples in output order ([policy,] value, score): Keras hands dicts to the step as a
    # wrapper that cannot be unpacked like the stream's tuples
    y, w = (value, score), (has, has)
    if with_policy:
        y = (encode_labels(actions, choices, symmetries, board_size),) + y
        w = (np.ones(len(has), np.float32),) + w
    return (packed, choices, komi), y, w


def value_batch_generator(shards, sizes, batch_size, symmetries, board_size, with_policy,
                          seed=None, start_position=0):
    """Endless ((packed, choices, komi), ([policy,] value, score), weights) batches."""
    seed = _resolve_seed(seed)
    reader = ValueReader(shards, sizes)
    position = int(start_position)
    try:
        while True:
            yield _batch(reader, position, batch_size, seed, symmetries, board_size, with_policy)
            position += batch_size
    finally:
        reader.close()


def value_validation_dataset(shards, sizes, n, batch_size, symmetries, board_size,
                             with_policy, seed):
    """The first n val positions as a fixed tf.data set (a uniform sample already)."""
    reader = ValueReader(shards, sizes)
    try:
        x, y, w = _batch(reader, 0, min(n, reader.total), _resolve_seed(seed), symmetries,
                         board_size, with_policy)
    finally:
        reader.close()
    with tf.device("/cpu:0"):
        return tf.data.Dataset.from_tensor_slices((x, y, w)).batch(batch_size)


def score_loss(y_true, y_pred):
    """Huber loss on the score in units of KOMI_SCALE points: a few points matter, but a
    lopsided game's 80-point lead doesn't swamp the win-rate loss."""
    return keras.losses.huber(y_true / KOMI_SCALE, y_pred / KOMI_SCALE, delta=1.0)


def side_agreement(y_true, y_pred):
    """Share of positions where the head and KataGo agree on which side is ahead."""
    return ops.cast(ops.equal(y_true > 0.5, y_pred > 0.5), "float32")


def training_model(net, with_policy):
    """The outputs of a PolicyValueNet that fit() trains: value and score, and the policy
    when layers it depends on train too (otherwise it would only cost compute)."""
    outputs = [net.model.get_layer(VALUE).output, net.model.get_layer(SCORE).output]
    if with_policy:
        outputs = [net.model.outputs[0]] + outputs
    return Model(net.model.inputs, outputs)


def decode_in_steps(model, board_size, n_features, symmetries):
    """Unpack the packed planes on the GPU as the first op of each train/test step."""
    dtype = model.inputs[0].dtype
    train_step, test_step = model.train_step, model.test_step

    def decoded(data):
        (packed, choices, komi), y, w = data
        planes = decode_on_device(packed, choices, board_size, n_features, symmetries, dtype)
        return (planes, komi), y, w

    model.train_step = lambda data: train_step(decoded(data))
    model.test_step = lambda data: test_step(decoded(data))


class LinearWarmup(Callback):
    """Ramps a constant learning rate linearly from 0 to target over the first steps (the
    plateau schedule's warmup: ReduceLROnPlateau needs a plain learning-rate variable, not
    a schedule object)."""

    def __init__(self, target, steps):
        super().__init__()
        self.target, self.steps, self.step = target, steps, 0

    def on_train_batch_begin(self, batch, logs=None):
        if self.step < self.steps:
            self.model.optimizer.learning_rate.assign(
                self.target * (self.step + 1) / self.steps)
        self.step += 1


class SaveEpoch(Callback):
    """After each epoch: the whole policy+value network, and the epoch's metrics."""

    def __init__(self, net, out_directory, args):
        super().__init__()
        self.net = net
        self.out = out_directory
        self.meta_path = os.path.join(out_directory, "metadata.json")
        self.meta = {"args": vars(args), "before_training": None, "epochs": []}
        self.t0 = time.time()

    def record_start(self, results):
        self.meta["before_training"] = {"val_" + k: float(v) for k, v in results.items()}
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)

    def on_epoch_end(self, epoch, logs=None):
        weights = os.path.join(self.out, "weights.{:05d}.weights.h5".format(epoch + 1))
        self.net.save_model(os.path.join(self.out, "model.json"), weights)
        entry = {k: float(v) for k, v in (logs or {}).items()}
        entry["epoch"] = epoch + 1
        entry["learning_rate"] = float(ops.convert_to_numpy(self.model.optimizer.learning_rate))
        entry["minutes"] = round((time.time() - self.t0) / 60, 1)
        self.meta["epochs"].append(entry)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="Policy network JSON (e.g. b20c256's), or a PolicyValueNet's "
                                 "to continue training it")
    p.add_argument("weights", help="Its weights")
    p.add_argument("train_data", help="Shards directory with train/ and val/, and value sidecars")
    p.add_argument("out_directory")
    p.add_argument("--minibatch", "-B", type=int, default=512)
    p.add_argument("--epochs", "-E", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=2000,
                   help="Steps between validations/checkpoints. Default: 2000 (~1M positions "
                        "at minibatch 512)")
    p.add_argument("--learning-rate", type=float, default=1e-3,
                   help="Adam's peak learning rate. Default: 1e-3")
    p.add_argument("--lr-schedule", choices=("cosine", "plateau"), default="cosine",
                   help="cosine: decay to 0 over the whole run. plateau: hold the peak and "
                        "multiply it by --plateau-factor whenever val_value_loss hasn't "
                        "improved for --plateau-patience epochs. Default: cosine")
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Steps over which the learning rate ramps linearly from 0 to "
                        "--learning-rate first. Default: 0")
    p.add_argument("--trainable-blocks", type=int, default=0,
                   help="Also train the last N residual blocks and the policy head, with the "
                        "policy loss added. Default: 0 (value head only)")
    p.add_argument("--policy-weight", type=float, default=1.0,
                   help="Weight of the policy loss when --trainable-blocks > 0. Default: 1.0")
    p.add_argument("--score-weight", type=float, default=1.0,
                   help="Weight of the score loss against the win-rate loss. Default: 1.0")
    p.add_argument("--value-channels", type=int, default=32)
    p.add_argument("--value-hidden", type=int, default=64)
    p.add_argument("--validation-length", type=int, default=200000)
    p.add_argument("--symmetries", default=",".join(BATCH_TRANSFORMATIONS))
    p.add_argument("--mixed-precision", action="store_true",
                   help="mixed_float16 compute (outputs stay float32). Default: float32")
    p.add_argument("--seed", type=int, default=None)
    return p


def run(argv=None):
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        keras.utils.set_random_seed(args.seed)
    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
    os.makedirs(args.out_directory, exist_ok=True)

    loaded = NeuralNetBase.load_model(args.model)
    loaded.model.load_weights(args.weights)
    if isinstance(loaded, PolicyValueNet):
        net = loaded
    else:
        net = PolicyValueNet.from_policy(loaded, value_channels=args.value_channels,
                                         value_hidden=args.value_hidden)
    set_trainable_blocks(net, args.trainable_blocks)
    with_policy = args.trainable_blocks > 0
    model = training_model(net, with_policy)

    train_shards = find_split_shards(args.train_data, "train")
    val_shards = find_split_shards(args.train_data, "val")
    features, board_size, n_features, train_sizes = dataset_info(train_shards)
    _vf, _vb, _vn, val_sizes = dataset_info(val_shards)
    if features != net.preprocessor.get_feature_list():
        raise ValueError("shards were built with features {}, the model expects {}".format(
            features, net.preprocessor.get_feature_list()))
    symmetries = args.symmetries.split(",")

    total_steps = args.steps_per_epoch * args.epochs
    lr_callbacks = []
    if args.lr_schedule == "plateau":
        schedule = args.learning_rate
        if args.warmup_steps:
            lr_callbacks.append(LinearWarmup(args.learning_rate, args.warmup_steps))
        lr_callbacks.append(ReduceLROnPlateau(monitor="val_value_loss", mode="min",
                                              factor=args.plateau_factor,
                                              patience=args.plateau_patience, verbose=1))
    elif args.warmup_steps:
        schedule = CosineDecay(0.0, max(total_steps - args.warmup_steps, 1),
                               warmup_target=args.learning_rate,
                               warmup_steps=args.warmup_steps)
    else:
        schedule = CosineDecay(args.learning_rate, total_steps)
    # lists in output order: [policy,] value, score (see training_model)
    losses = [keras.losses.BinaryCrossentropy(), score_loss]
    weights = [1.0, args.score_weight]
    metrics = [[keras.metrics.MeanAbsoluteError(name="mae"), side_agreement],
               [keras.metrics.MeanAbsoluteError(name="mae")]]
    if with_policy:
        losses = ["categorical_crossentropy"] + losses
        weights = [args.policy_weight] + weights
        metrics = [["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
                   ] + metrics
    model.compile(optimizer=Adam(schedule), loss=losses, loss_weights=weights,
                  weighted_metrics=metrics, jit_compile=True)
    decode_in_steps(model, board_size, n_features, symmetries)

    print("training {} of {} weights; {} train / {} val positions".format(
        sum(int(np.prod(w.shape)) for w in model.trainable_weights),
        sum(int(np.prod(w.shape)) for w in net.model.weights), sum(train_sizes),
        sum(val_sizes)), flush=True)
    train = value_batch_generator(train_shards, train_sizes, args.minibatch, symmetries,
                                  board_size, with_policy, seed=args.seed)
    val = value_validation_dataset(val_shards, val_sizes, args.validation_length,
                                   args.minibatch, symmetries, board_size, with_policy,
                                   None if args.seed is None else args.seed + 1)
    saver = SaveEpoch(net, args.out_directory, args)
    saver.record_start(model.evaluate(val, return_dict=True, verbose=0))
    print("before training:", saver.meta["before_training"], flush=True)
    # The saver before ReduceLROnPlateau, so an epoch's recorded learning rate is the one it
    # ended with, and a cut shows from the next epoch on.
    warmup = [cb for cb in lr_callbacks if isinstance(cb, LinearWarmup)]
    plateau = [cb for cb in lr_callbacks if not isinstance(cb, LinearWarmup)]
    model.fit(train, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
              validation_data=val, callbacks=warmup + [TerminateOnNaN(), saver] + plateau)


if __name__ == "__main__":
    run()
