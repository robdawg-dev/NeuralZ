"""Read training data from pre-shuffled shards (convert_shuffled.py) as a plain stream.

The shards of a split, in file order, already form one uniformly random permutation of
every position in that split, so there is nothing to shuffle here. Training reads them
start to finish, wrapping around at the end; each position gets a random board symmetry.

Symmetry choices are a pure function of (seed, position in the stream): the stream is
split into fixed-size blocks and each block's choices come from its own seeded
generator. So a run resumed at stream position P sees exactly the batches, symmetries
included, that an uninterrupted run would have seen from P.
"""
import glob
import os

import h5py as h5
import numpy as np

# Batch versions of shuffle_buffer.BOARD_TRANSFORMATIONS, acting on axes (1, 2) of an
# (N, size, size[, F]) array rather than axes (0, 1) of one position. The test suite
# checks each against its single-position original.
BATCH_TRANSFORMATIONS = {
    "noop": lambda a: a,
    "rot90": lambda a: np.rot90(a, 1, axes=(1, 2)),
    "rot180": lambda a: np.rot90(a, 2, axes=(1, 2)),
    "rot270": lambda a: np.rot90(a, 3, axes=(1, 2)),
    "fliplr": lambda a: np.flip(a, axis=2),
    "flipud": lambda a: np.flip(a, axis=1),
    "diag1": lambda a: np.swapaxes(a, 1, 2),
    "diag2": lambda a: np.flip(np.rot90(a, 1, axes=(1, 2)), axis=2),
}

SYMMETRY_BLOCK = 4096

# Shard handles kept open at once. Two is enough for a forward stream: the shard being read
# and the next one, which a batch spanning a boundary also touches. Python 3.7+ dicts keep
# insertion order, so the oldest is the one evicted.
_OPEN_SHARDS = 2


def find_split_shards(root, split):
    shards = sorted(glob.glob(os.path.join(root, split, "shard_*.h5")))
    if not shards:
        raise ValueError("no shard_*.h5 files in {}".format(os.path.join(root, split)))
    return shards


def dataset_info(shards):
    """(feature_list, board_size, n_features, positions per shard). Every shard must
    agree on features and tensor shape."""
    features = shape = None
    sizes = []
    for path in shards:
        with h5.File(path, "r") as f:
            feats = f["features"][()]
            feats = (feats.decode("ascii") if isinstance(feats, bytes) else feats).split(",")
            n, board, board2, planes = f["states"].shape
            if len(f["actions"]) != n:
                raise ValueError("{}: {} states but {} actions".format(path, n, len(f["actions"])))
        if features is None:
            features, shape = feats, (board, board2, planes)
        elif feats != features or (board, board2, planes) != shape:
            raise ValueError("{} does not match {}: features or tensor shape differ".format(
                path, shards[0]))
        sizes.append(n)
    return features, shape[0], shape[2], sizes


class _Reader(object):
    """Contiguous reads from a list of shards treated as one endless, wrapping array."""

    def __init__(self, shards, sizes):
        self.shards = shards
        self.starts = np.concatenate([[0], np.cumsum(sizes)])
        self.total = int(self.starts[-1])
        self.handles = {}

    def _file(self, i):
        """Open shard i, keeping only the last few handles.

        Reading is strictly forward, so a handle the stream has moved past is dead weight:
        HDF5 keeps a chunk cache per open dataset, which at a few hundred shards adds up to
        hundreds of MB for data that will not be read again until the next pass.
        """
        if i not in self.handles:
            self.handles[i] = h5.File(self.shards[i], "r")
            while len(self.handles) > _OPEN_SHARDS:
                self.handles.pop(next(iter(self.handles))).close()
        return self.handles[i]

    def read(self, position, n):
        states, actions = [], []
        while n > 0:
            p = position % self.total
            i = int(np.searchsorted(self.starts, p, side="right") - 1)
            offset = p - int(self.starts[i])
            take = min(n, int(self.starts[i + 1]) - p)
            f = self._file(i)
            states.append(f["states"][offset:offset + take])
            actions.append(f["actions"][offset:offset + take])
            position += take
            n -= take
        return np.concatenate(states), np.concatenate(actions)

    def close(self):
        for h in self.handles.values():
            h.close()
        self.handles.clear()


def _symmetry_choices(seed, position, n, n_transforms):
    """Symmetry index for each of stream positions [position, position + n)."""
    first = position // SYMMETRY_BLOCK
    last = (position + n - 1) // SYMMETRY_BLOCK
    blocks = [np.random.default_rng([seed, b]).integers(0, n_transforms, SYMMETRY_BLOCK)
              for b in range(first, last + 1)]
    start = position - first * SYMMETRY_BLOCK
    return np.concatenate(blocks)[start:start + n]


def encode(states, actions, choices, transform_names, board_size):
    """Float32 (X, Y) with symmetry transform_names[choices[i]] applied to position i."""
    n = len(states)
    labels = np.zeros((n, board_size, board_size), dtype=np.float32)
    labels[np.arange(n), actions[:, 0], actions[:, 1]] = 1.0
    X = np.empty(states.shape, dtype=np.float32)
    Y = np.empty((n, board_size * board_size), dtype=np.float32)
    for t, name in enumerate(transform_names):
        idx = np.flatnonzero(choices == t)
        if len(idx) == 0:
            continue
        fn = BATCH_TRANSFORMATIONS[name]
        X[idx] = fn(states[idx])
        Y[idx] = fn(labels[idx]).reshape(len(idx), -1)
    return X, Y


def _resolve_seed(seed):
    return int(np.random.SeedSequence().entropy % (2 ** 63)) if seed is None else int(seed)


def shard_batch_generator(shards, sizes, batch_size, board_size, transform_names, seed=None,
                          start_position=0):
    """Endless (X, Y) batches, reading the shards in order from start_position."""
    seed = _resolve_seed(seed)
    reader = _Reader(shards, sizes)
    position = int(start_position)
    try:
        while True:
            states, actions = reader.read(position, batch_size)
            choices = _symmetry_choices(seed, position, batch_size, len(transform_names))
            yield encode(states, actions, choices, transform_names, board_size)
            position += batch_size
    finally:
        reader.close()


def validation_arrays(shards, sizes, n, board_size, transform_names, seed=None):
    """The first n positions of the validation shards, fully materialised. They are a
    uniform sample already, and taking a prefix keeps the set identical every epoch."""
    reader = _Reader(shards, sizes)
    try:
        n = min(n, reader.total)
        states, actions = reader.read(0, n)
    finally:
        reader.close()
    choices = _symmetry_choices(_resolve_seed(seed), 0, n, len(transform_names))
    return encode(states, actions, choices, transform_names, board_size)
