"""Tests for the v4 training stream over pre-shuffled shards."""
import numpy as np
import pytest

from AlphaGo.training import shard_stream as ss
from AlphaGo.training.shuffle_buffer import BOARD_TRANSFORMATIONS, one_hot_action
from tests.test_convert_shuffled import FEATURES, _run, _selection

ALL = list(ss.BATCH_TRANSFORMATIONS)


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
    X, Y = next(gen)
    assert X.shape == (64, 19, 19, planes) and Y.shape == (64, 361)
    assert np.array_equal(X, expected[:64].astype(np.float32))


def test_resuming_at_a_position_reproduces_the_uninterrupted_stream(shards):
    train = ss.find_split_shards(shards, "train")
    _f, board, _p, sizes = ss.dataset_info(train)
    full = ss.shard_batch_generator(train, sizes, 50, board, ALL, seed=7)
    batches = [next(full) for _ in range(6)]
    resumed = ss.shard_batch_generator(train, sizes, 50, board, ALL, seed=7,
                                       start_position=4 * 50)
    for expected in batches[4:]:
        X, Y = next(resumed)
        assert np.array_equal(X, expected[0]) and np.array_equal(Y, expected[1])


def test_validation_is_a_stable_prefix(shards):
    val = ss.find_split_shards(shards, "val")
    _f, board, _p, sizes = ss.dataset_info(val)
    a = ss.validation_arrays(val, sizes, 40, board, ALL, seed=2)
    b = ss.validation_arrays(val, sizes, 40, board, ALL, seed=2)
    assert a[0].shape[0] == 40
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.all(a[1].sum(axis=1) == 1)


def test_dataset_info_rejects_mismatched_shards(tmp_path):
    import h5py as h5
    paths = []
    for i, feats in enumerate(("board,ones", "board,zeros")):
        p = str(tmp_path / "s{}.h5".format(i))
        with h5.File(p, "w") as f:
            f.create_dataset("states", data=np.zeros((2, 19, 19, 4), np.uint8))
            f.create_dataset("actions", data=np.zeros((2, 2), np.uint8))
            f["features"] = np.bytes_(feats)
        paths.append(p)
    with pytest.raises(ValueError):
        ss.dataset_info(paths)


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
