"""The data stream for joint policy + value + score + ownership training (see
JOINT_TRAINING_PLAN.md): shard_stream's shards, read in order, together with

- the value sidecars beside them (<split>/value_NNNNN.h5, add_value_targets.py): KataGo's
  win rate and score for the player to move, has_target, komi and black_to_move;
- the split's ownership table (<split>/ownership.bin, add_ownership_targets.py): KataGo's
  final-position ownership per game, from Black's side, looked up by the record's game_id
  and turned to the player to move's side.

Batches are ((packed, choices, komi), (policy, value, score, ownership),
(policy weight, value weight, score weight, ownership weight)): packed planes and
ownership stay in their original orientation, unpacked and transformed under the symmetry
choices on the GPU (decode_joint_on_device); the move label is transformed here, as the
policy-only stream does. Value and score weigh 0 where KataGo left no annotation.
"""
import json
import os

import h5py as h5
import numpy as np

from AlphaGo.preprocessing.add_value_targets import sidecar_path
from AlphaGo.training.shard_stream import (
    PACKED_STATES, _resolve_seed, _symmetry_choices, decode_on_device, encode_labels,
    symmetry_permutations)

_OPEN_FILES = 4


def load_ownership(split_dir):
    """(n_games, points) int8 table of KataGo's final ownership, Black's side, x127."""
    with open(os.path.join(split_dir, "ownership.json")) as f:
        meta = json.load(f)
    table = np.fromfile(os.path.join(split_dir, "ownership.bin"), dtype=np.int8)
    return table.reshape(meta["games"], meta["points"])


class JointReader(object):
    """Contiguous reads from shards, their value sidecars and the ownership table, as one
    endless wrapping array."""

    def __init__(self, shards, sizes, ownership):
        self.shards = shards
        self.starts = np.concatenate([[0], np.cumsum(sizes)])
        self.total = int(self.starts[-1])
        self.ownership = ownership
        self.handles = {}
        for shard, n in zip(shards, sizes):
            with h5.File(sidecar_path(shard), "r") as f:
                if len(f["value"]) != n or "black_to_move" not in f:
                    raise ValueError("{}: {} rows (shard has {}), black_to_move {} - rerun "
                                     "add_value_targets".format(sidecar_path(shard),
                                                               len(f["value"]), n,
                                                               "black_to_move" in f))

    def _files(self, i):
        if i not in self.handles:
            self.handles[i] = (h5.File(self.shards[i], "r"),
                               h5.File(sidecar_path(self.shards[i]), "r"))
            while len(self.handles) > _OPEN_FILES:
                for f in self.handles.pop(next(iter(self.handles))):
                    f.close()
        return self.handles[i]

    def read(self, position, n):
        """dict of packed, actions, komi, value, score, has, ownership for n positions."""
        cols = {k: [] for k in ("packed", "actions", "game_id", "komi", "value", "score",
                                "has", "black")}
        while n > 0:
            p = position % self.total
            i = int(np.searchsorted(self.starts, p, side="right") - 1)
            offset = p - int(self.starts[i])
            take = min(n, int(self.starts[i + 1]) - p)
            shard, side = self._files(i)
            sl = slice(offset, offset + take)
            cols["packed"].append(shard[PACKED_STATES][sl])
            cols["actions"].append(shard["actions"][sl])
            cols["game_id"].append(shard["game_id"][sl])
            for key, name in (("komi", "komi"), ("value", "value"), ("score", "score"),
                              ("has", "has_target"), ("black", "black_to_move")):
                cols[key].append(side[name][sl])
            position += take
            n -= take
        out = {k: np.concatenate(v) for k, v in cols.items()}
        sign = np.where(out["black"] == 1, 1.0, -1.0).astype(np.float32)[:, None]
        out["ownership"] = self.ownership[out["game_id"]].astype(np.float32) / 127.0 * sign
        return out

    def close(self):
        for files in self.handles.values():
            for f in files:
                f.close()
        self.handles.clear()


def joint_batch(reader, position, n, seed, symmetries, board_size):
    d = reader.read(position, n)
    choices = _symmetry_choices(seed, position, n, len(symmetries)).astype(np.int32)
    ones = np.ones(n, np.float32)
    has = d["has"].astype(np.float32)
    x = (d["packed"], choices, d["komi"].astype(np.float32)[:, None])
    y = (encode_labels(d["actions"], choices, symmetries, board_size),
         d["value"].astype(np.float32)[:, None], d["score"].astype(np.float32)[:, None],
         d["ownership"])
    return x, y, (ones, has, has, ones)


def joint_batch_generator(shards, sizes, ownership, batch_size, symmetries, board_size,
                          seed=None, start_position=0):
    """Endless joint batches, reading the shards in order from start_position."""
    seed = _resolve_seed(seed)
    reader = JointReader(shards, sizes, ownership)
    position = int(start_position)
    try:
        while True:
            yield joint_batch(reader, position, batch_size, seed, symmetries, board_size)
            position += batch_size
    finally:
        reader.close()


def joint_validation_arrays(shards, sizes, ownership, n, symmetries, board_size, seed=None):
    """The first n positions of the validation shards as one joint batch."""
    reader = JointReader(shards, sizes, ownership)
    try:
        return joint_batch(reader, 0, min(n, reader.total), _resolve_seed(seed), symmetries,
                           board_size)
    finally:
        reader.close()


def decode_joint_on_device(x, y, w, board_size, n_features, symmetries, dtype):
    """TensorFlow, inside the train/test step: unpack the planes and put the ownership map
    under each position's symmetry (the move label already is)."""
    import tensorflow as tf
    packed, choices, komi = x
    planes = decode_on_device(packed, choices, board_size, n_features, symmetries, dtype)
    perms = tf.constant(symmetry_permutations(board_size, symmetries))
    ownership = tf.gather(y[3], tf.gather(perms, choices), batch_dims=1)
    return (planes, komi), (y[0], y[1], y[2], ownership), w
