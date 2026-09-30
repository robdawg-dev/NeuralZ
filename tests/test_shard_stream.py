"""Tests for the training stream over pre-shuffled shards."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import h5py as h5
import numpy as np
import pytest
import tensorflow as tf

from AlphaGo.training import shard_stream as ss
from tests.test_convert_shuffled import FEATURES, _run, _selection

ALL = list(ss.BATCH_TRANSFORMATIONS)

# Single-position reference versions of the symmetries, written with different numpy calls
# than BATCH_TRANSFORMATIONS so the comparison below is a real cross-check. Each acts on
# axes (0, 1) of a (size, size) label or a (size, size, features) state.
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


def one_hot_action(action, size=19):
    categorical = np.zeros((size, size), dtype=np.float32)
    categorical[action] = 1
    return categorical


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stream")
    sel = _selection(tmp, {"train": [(s, None) for s in range(12)],
                           "val": [(100 + s, None) for s in range(3)]})
    out = str(tmp / "out")
    _run(sel, out, "--splits", "train", "val")
    return out


@pytest.mark.parametrize("name", ALL)
def test_batch_transforms_match_the_single_position_originals(name):
    rng = np.random.default_rng(0)
    states = rng.integers(0, 2, size=(5, 19, 19, 7)).astype(np.uint8)
    batch = ss.BATCH_TRANSFORMATIONS[name](states)
    for i in range(5):
        assert np.array_equal(batch[i], BOARD_TRANSFORMATIONS[name](states[i]))


def test_encode_applies_the_same_symmetry_to_state_and_label():
    rng = np.random.default_rng(1)
    states = rng.integers(0, 2, size=(8, 19, 19, 3)).astype(np.uint8)
    actions = rng.integers(0, 19, size=(8, 2)).astype(np.uint8)
    choices = np.arange(8)
    X, Y = ss.encode(states, actions, choices, ALL, 19)
    for i, name in enumerate(ALL):
        fn = BOARD_TRANSFORMATIONS[name]
        assert np.array_equal(X[i], fn(states[i]))
        assert np.array_equal(Y[i], fn(one_hot_action(tuple(actions[i]), 19)).flatten())
        assert Y[i].sum() == 1


@pytest.mark.parametrize("board,planes", [(19, 3), (9, 5), (19, 48)])
def test_pack_and_unpack_round_trip(board, planes):
    """Including sizes whose bit count isn't a multiple of 8 (19 * 19 * 3 = 1083)."""
    rng = np.random.default_rng(0)
    states = rng.integers(0, 2, size=(6, board, board, planes)).astype(np.uint8)
    packed = ss.pack(states)
    assert packed.shape == (6, (board * board * planes + 7) // 8)
    assert np.array_equal(ss.unpack(packed, (board, board, planes)), states)


@pytest.mark.parametrize("board,planes", [(19, 3), (9, 5)])
def test_decode_on_device_matches_the_cpu_encoding(board, planes):
    """The GPU-side unpacking + symmetry gives exactly encode()'s X - every symmetry, and
    a bit count that isn't a multiple of 8."""
    rng = np.random.default_rng(2)
    states = rng.integers(0, 2, size=(24, board, board, planes)).astype(np.uint8)
    actions = rng.integers(0, board, size=(24, 2)).astype(np.uint8)
    choices = np.arange(24) % len(ALL)
    X, _Y = ss.encode(states, actions, choices, ALL, board)
    decoded = ss.decode_on_device(tf.constant(ss.pack(states)),
                                  tf.constant(choices.astype(np.int32)), board, planes, ALL,
                                  tf.float32)
    assert decoded.dtype == tf.float32
    assert np.array_equal(decoded.numpy(), X)


def test_encode_labels_matches_encode():
    rng = np.random.default_rng(3)
    states = rng.integers(0, 2, size=(8, 19, 19, 2)).astype(np.uint8)
    actions = rng.integers(0, 19, size=(8, 2)).astype(np.uint8)
    choices = np.arange(8)
    assert np.array_equal(ss.encode_labels(actions, choices, ALL, 19),
                          ss.encode(states, actions, choices, ALL, 19)[1])


def test_stream_reads_in_order_and_wraps(shards):
    train = ss.find_split_shards(shards, "train")
    feats, board, planes, sizes = ss.dataset_info(train)
    assert feats == FEATURES.split(",")
    total = sum(sizes)
    reader = ss._Reader(train, sizes)
    expected, _a = reader.read(0, total)
    wrapped, _a = reader.read(total - 10, 20)
    reader.close()
    assert np.array_equal(wrapped[:10], expected[-10:])
    assert np.array_equal(wrapped[10:], expected[:10])

    gen = ss.shard_batch_generator(train, sizes, 64, board, ["noop"], seed=3)
    (packed, choices), Y = next(gen)
    assert np.array_equal(packed, expected[:64])
    assert ss.unpack(packed, (board, board, planes)).shape == (64, 19, 19, planes)
    assert choices.dtype == np.int32 and np.all(choices == 0)
    assert Y.shape == (64, 361) and np.all(Y.sum(axis=1) == 1)


def test_resuming_at_a_position_reproduces_the_uninterrupted_stream(shards):
    train = ss.find_split_shards(shards, "train")
    _f, board, _p, sizes = ss.dataset_info(train)
    full = ss.shard_batch_generator(train, sizes, 50, board, ALL, seed=7)
    batches = [next(full) for _ in range(6)]
    resumed = ss.shard_batch_generator(train, sizes, 50, board, ALL, seed=7,
                                       start_position=4 * 50)
    for ((e_packed, e_choices), e_Y) in batches[4:]:
        (packed, choices), Y = next(resumed)
        assert np.array_equal(packed, e_packed) and np.array_equal(choices, e_choices)
        assert np.array_equal(Y, e_Y)


def test_validation_is_a_stable_prefix(shards):
    val = ss.find_split_shards(shards, "val")
    _f, board, _p, sizes = ss.dataset_info(val)
    (a_packed, a_choices), a_Y = ss.validation_arrays(val, sizes, 40, board, ALL, seed=2)
    (b_packed, b_choices), b_Y = ss.validation_arrays(val, sizes, 40, board, ALL, seed=2)
    assert a_packed.shape[0] == a_choices.shape[0] == 40
    assert np.array_equal(a_packed, b_packed) and np.array_equal(a_choices, b_choices)
    assert np.array_equal(a_Y, b_Y) and np.all(a_Y.sum(axis=1) == 1)


def _shard(path, feats, packed=True):
    with h5.File(path, "w") as f:
        states = np.zeros((2, 19, 19, 4), np.uint8)
        if packed:
            f.create_dataset(ss.PACKED_STATES, data=ss.pack(states))
            f[ss.PACKED_STATES].attrs[ss.PLANES_SHAPE] = (19, 19, 4)
        else:
            f.create_dataset("states", data=states)
        f.create_dataset("actions", data=np.zeros((2, 2), np.uint8))
        f["features"] = np.bytes_(feats)
    return str(path)


def test_dataset_info_rejects_mismatched_shards(tmp_path):
    paths = [_shard(tmp_path / "s{}.h5".format(i), feats)
             for i, feats in enumerate(("board,ones", "board,zeros"))]
    assert ss.dataset_info(paths[:1]) == (["board", "ones"], 19, 4, [2])
    with pytest.raises(ValueError):
        ss.dataset_info(paths)


def test_dataset_info_points_old_format_shards_to_the_repacker(tmp_path):
    with pytest.raises(ValueError, match="repack_shards"):
        ss.dataset_info([_shard(tmp_path / "old.h5", "board", packed=False)])


def test_reader_does_not_hoard_open_shard_handles(shards):
    """Reading is forward-only, so handles left open just hold HDF5 chunk caches - at a few
    hundred shards that is hundreds of MB for data not read again until the next pass."""
    train = ss.find_split_shards(shards, "train")
    _f, board, _p, sizes = ss.dataset_info(train)
    assert len(train) > ss._OPEN_SHARDS, "test needs more shards than the handle cap"
    reader = ss._Reader(train, sizes)
    try:
        for position in range(0, sum(sizes), max(1, sum(sizes) // 20)):
            reader.read(position, 16)
            assert len(reader.handles) <= ss._OPEN_SHARDS
    finally:
        reader.close()
    assert reader.handles == {}
