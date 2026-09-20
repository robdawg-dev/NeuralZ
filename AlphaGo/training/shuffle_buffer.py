import glob
import json
import os

import h5py as h5
import numpy as np


def one_hot_action(action, size=19):
    """Convert an (x,y) action into a size x size array of zeros with a 1 at x,y
    """
    categorical = np.zeros((size, size), dtype=np.float32)
    categorical[action] = 1
    return categorical


# Each of these acts on the first two (spatial) axes of its input, whether that input is a
# 2D one-hot action array (size, size) or a 3D channels_last state array (size, size, features).
# np.rot90/fliplr/flipud already default to axes (0, 1), but np.transpose reverses *all* axes by
# default, so diag1 must be told explicitly to only swap the first two.
BOARD_TRANSFORMATIONS = {
    "noop": lambda feature: feature,
    "rot90": lambda feature: np.rot90(feature, 1),
    "rot180": lambda feature: np.rot90(feature, 2),
    "rot270": lambda feature: np.rot90(feature, 3),
    "fliplr": lambda feature: np.fliplr(feature),
    "flipud": lambda feature: np.flipud(feature),
    "diag1": lambda feature: np.transpose(feature, (1, 0) + tuple(range(2, feature.ndim))),
    "diag2": lambda feature: np.fliplr(np.rot90(feature, 1))
}


def find_shard_files(directory):
    """Find all .h5 shard files in a directory, sorted for reproducible ordering."""
    return sorted(glob.glob(os.path.join(directory, "*.h5")))


def _game_id(shard_file, start):
    """Stable identifier for one game, used for persisting train/val/test split
    assignments across resumes independent of in-memory list order."""
    return "{}::{}".format(os.path.basename(shard_file), start)


def build_game_index(shard_files, verbose=False):
    """Read every shard's file_offsets group (already written by
    game_converter_parallel.py / game_converter.py for every converted game) and build
    one combined list of games across all shards.

    Returns (games, feature_list, board_size, n_features) where games is a list of dicts:
        {"id": str, "shard": path, "start": int, "length": int}
    feature_list, board_size and n_features are read from the first shard and verified
    identical across the rest - shards from different feature sets/board sizes can't be
    combined into one training run. board_size/n_features come from the states dataset's
    own shape (there's no public accessor for this on the Cython Preprocess class), same
    source the generator this replaces used.
    """
    if not shard_files:
        raise ValueError("No shard files found")

    games = []
    feature_list = None
    board_size = None
    n_features = None
    for shard in shard_files:
        with h5.File(shard, 'r') as f:
            shard_features = f['features'][()]
            if isinstance(shard_features, bytes):
                shard_features = shard_features.decode('ascii')
            shard_features = shard_features.split(",")
            _, shard_board_size, shard_board_size2, shard_n_features = f['states'].shape
            if feature_list is None:
                feature_list = shard_features
                board_size = shard_board_size
                n_features = shard_n_features
            else:
                if shard_features != feature_list:
                    raise ValueError(
                        "{} has features {}\nbut {} has features {}\nall shards in a "
                        "training run must use the same feature list".format(
                            shard, shard_features, shard_files[0], feature_list))
                if shard_board_size != board_size or shard_n_features != n_features:
                    raise ValueError(
                        "{} has states shape (.., {}, {}, {})\nbut {} has (.., {}, {}, {})"
                        "\nall shards must use the same board size and feature count"
                        .format(shard, shard_board_size, shard_board_size2, shard_n_features,
                               shard_files[0], board_size, board_size, n_features))

            file_offsets = f['file_offsets']
            n_before = len(games)
            for key in file_offsets:
                start, length = file_offsets[key][()]
                games.append({
                    "id": _game_id(shard, int(start)),
                    "shard": shard,
                    "start": int(start),
                    "length": int(length),
                })
            if verbose:
                print("{}: {} games, {} positions".format(
                    shard, len(games) - n_before,
                    sum(g["length"] for g in games[n_before:])))

    return games, feature_list, board_size, n_features


def get_or_create_game_split(games, out_directory, train_val_test, verbose=False, seed=None):
    """Split games into train/val/test at the GAME level (not position level - positions
    within one game are highly correlated, so splitting at the position level risks the
    same game appearing in both train and validation).

    Persists the split (by stable game id, not list position) to
    <out_directory>/game_split.json so a resumed run uses the exact same split rather than
    silently drawing a new random one. If that file exists, the current set of game ids
    found must match exactly what's recorded - a changed shard directory since the split
    was created is treated as an error rather than silently guessed at.

    seed: controls the split drawn for a FRESH out_directory (no existing game_split.json).
    Without this, every fresh out_directory silently drew an independently-random split
    even when the caller passed the same --seed everywhere else (that seed only ever
    reached the shuffle buffer's position ordering/sampling, never which games land in
    train/val/test to begin with) - confirmed the hard way: two runs compared at matched
    steps under supposedly identical seeds/LR schedules turned out to differ because their
    train sets only overlapped ~93%, not because of anything LR-related. Passing the same
    seed here now makes a fresh split reproducible across separate out_directories, the
    same way the position-level seed already was.
    """
    split_file = os.path.join(out_directory, "game_split.json")
    by_id = {g["id"]: g for g in games}

    if os.path.exists(split_file):
        with open(split_file) as f:
            split = json.load(f)
        recorded_ids = set(split["train"]) | set(split["val"]) | set(split["test"])
        current_ids = set(by_id.keys())
        if recorded_ids != current_ids:
            raise ValueError(
                "{} was created from a different set of games than what's found now "
                "({} recorded, {} found; {} missing, {} new). The shard directory "
                "appears to have changed since this split was created - remove {} to "
                "start a fresh split (this will invalidate resume-comparability of any "
                "existing train/val history), or restore the original shard set."
                .format(split_file, len(recorded_ids), len(current_ids),
                       len(recorded_ids - current_ids), len(current_ids - recorded_ids),
                       split_file))
        if verbose:
            print("loaded existing game split from {}".format(split_file))
    else:
        ids = list(by_id.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(train_val_test[0] * n)
        n_val = int(train_val_test[1] * n)
        split = {
            "train": ids[:n_train],
            "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:],
        }
        with open(split_file, "w") as f:
            json.dump(split, f)
        if verbose:
            print("created new game split, saved to {}".format(split_file))

    train_games = [by_id[i] for i in split["train"]]
    val_games = [by_id[i] for i in split["val"]]
    test_games = [by_id[i] for i in split["test"]]
    return train_games, val_games, test_games


class _ShardCache:
    """Keeps shard files open (h5py.File handles) across many small reads instead of
    reopening per game."""

    def __init__(self):
        self._handles = {}

    def get(self, path):
        if path not in self._handles:
            self._handles[path] = h5.File(path, 'r')
        return self._handles[path]

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def _position_source(games, rng, shard_cache):
    """Yield every position from `games` exactly once, in a freshly game-shuffled order.
    Reads one game (a contiguous HDF5 slice) at a time - not one row at a time - so disk
    access stays close to sequential regardless of how finely the output gets shuffled.
    """
    order = list(games)
    rng.shuffle(order)
    for g in order:
        f = shard_cache.get(g["shard"])
        states = f["states"][g["start"]:g["start"] + g["length"]]
        actions = f["actions"][g["start"]:g["start"] + g["length"]]
        for i in range(g["length"]):
            yield states[i], actions[i]


def _epoch_positions(games, buffer_size, rng, shard_cache, board_size, n_features):
    """One full pass over every position in `games`, exactly once each, via a streaming
    reservoir/shuffle buffer: fill a buffer of `buffer_size` positions (reading whole
    games at a time), then repeatedly draw a uniformly random occupied slot, yield it, and
    refill that slot from the next position in the read-ahead stream. When the stream is
    exhausted, drains the remaining buffer (still via random slot draws) until empty.

    Positions can only end up near each other in the output if they were both resident in
    the buffer at overlapping times - i.e. within roughly `buffer_size` positions of each
    other in read order. Since a single game contributes only its own (small) length to
    the buffer before the read pointer moves to a different (shuffled-in) game, this
    keeps same-game correlation from showing up in the output at anywhere near
    buffer_size scale.
    """
    source = _position_source(games, rng, shard_cache)

    buffer_states = np.zeros((buffer_size, board_size, board_size, n_features), dtype=np.uint8)
    buffer_actions = np.zeros((buffer_size, 2), dtype=np.uint8)
    filled = 0

    for i in range(buffer_size):
        try:
            s, a = next(source)
        except StopIteration:
            break
        buffer_states[i] = s
        buffer_actions[i] = a
        filled += 1

    exhausted = filled < buffer_size

    while filled > 0:
        idx = int(rng.integers(0, filled))
        out_s = buffer_states[idx].copy()
        out_a = buffer_actions[idx].copy()

        if not exhausted:
            try:
                s, a = next(source)
                buffer_states[idx] = s
                buffer_actions[idx] = a
            except StopIteration:
                exhausted = True

        if exhausted:
            # Drain: swap the last occupied slot into idx's place and shrink by one,
            # rather than shifting a range - O(1) removal from an unordered buffer.
            filled -= 1
            if idx != filled:
                buffer_states[idx] = buffer_states[filled]
                buffer_actions[idx] = buffer_actions[filled]

        yield out_s, out_a


def shuffle_buffer_batch_generator(games, buffer_size, batch_size, board_size, n_features,
                                   transforms, seed=None):
    """Drop-in replacement for supervised_policy_trainer.shuffled_hdf5_batch_generator,
    backed by the streaming shuffle buffer instead of random single-row HDF5 access.

    Re-shuffles (fresh game order, fresh buffer fill/drain) at the start of every full
    pass over `games` - unlike the position-permutation generator it replaces, which only
    ever shuffles once at the very start of training and then cycles the same fixed order
    forever, this gives every epoch its own independent shuffle.
    """
    rng = np.random.default_rng(seed)
    shard_cache = _ShardCache()

    # float32, not float64: the model computes in float32 (or float16 under a mixed
    # precision policy) regardless, so float64 here only doubled host memory, batch
    # assembly, and host->device transfer size for no benefit.
    Xbatch = np.zeros((batch_size, board_size, board_size, n_features), dtype=np.float32)
    Ybatch = np.zeros((batch_size, board_size * board_size), dtype=np.float32)
    batch_idx = 0

    try:
        while True:
            for state, action_xy in _epoch_positions(games, buffer_size, rng, shard_cache,
                                                       board_size, n_features):
                # Drawn from the same seeded `rng` as everything else in this generator,
                # not the bare np.random.choice - that draws from numpy's global RNG
                # state instead, which `seed` above has no control over, silently
                # breaking reproducibility even with an explicit seed passed in.
                transform = transforms[rng.integers(0, len(transforms))]
                state = transform(state)
                action = transform(one_hot_action(tuple(action_xy), board_size))
                Xbatch[batch_idx] = state
                Ybatch[batch_idx] = action.flatten()
                batch_idx += 1
                if batch_idx == batch_size:
                    batch_idx = 0
                    yield (Xbatch, Ybatch)
    finally:
        shard_cache.close()


def build_validation_arrays(games, board_size, n_features, transforms, seed=None):
    """Fully materialize every position in `games` into complete (X, Y) arrays, applying
    one seeded-random symmetry per position, instead of streaming through the infinite
    shuffle buffer that shuffle_buffer_batch_generator uses for training.

    Meant for validation, where the position count (tens of thousands, not tens of
    millions) comfortably fits in memory as a single array. Every position appears
    exactly once, with no reservoir sampling and no partial-pass carryover - handing the
    result to tf.data.Dataset.from_tensor_slices(...).batch(...) lets Keras's own
    per-epoch dataset reset apply cleanly, instead of a validation_steps count that (with
    the infinite generator) can stop mid-buffer-pass and leave leftover positions to
    bleed into the next epoch's validation.
    """
    rng = np.random.default_rng(seed)
    shard_cache = _ShardCache()

    n_positions = sum(g["length"] for g in games)
    X = np.zeros((n_positions, board_size, board_size, n_features), dtype=np.float32)
    Y = np.zeros((n_positions, board_size * board_size), dtype=np.float32)

    try:
        for i, (state, action_xy) in enumerate(_position_source(games, rng, shard_cache)):
            transform = transforms[rng.integers(0, len(transforms))]
            X[i] = transform(state)
            Y[i] = transform(one_hot_action(tuple(action_xy), board_size)).flatten()
    finally:
        shard_cache.close()

    return X, Y
