"""Tests for AlphaGo/preprocessing/repack_shards.py: old one-byte-per-plane shards ->
the current bit-packed format."""
import glob
import os
import shutil

import h5py as h5
import numpy as np
import pytest

from AlphaGo.preprocessing import repack_shards
from AlphaGo.training import shard_stream as ss
from tests.test_convert_shuffled import _run, _selection

COLUMNS = ("actions", "game_id", "move")


@pytest.fixture(scope="module")
def current(tmp_path_factory):
    """Shards in the current format, from convert_shuffled."""
    tmp = tmp_path_factory.mktemp("repack")
    sel = _selection(tmp, {"train": [(s, None) for s in range(8)],
                           "val": [(100 + s, None) for s in range(2)]})
    out = str(tmp / "current")
    _run(sel, out, "--splits", "train", "val")
    return out


def _paths(root):
    return sorted(glob.glob(os.path.join(root, "*", "shard_*.h5")))


def _to_old_format(src_root, dst_root):
    """A copy of src_root as convert_shuffled wrote it before packing: `states` with one
    uint8 per plane, LZF."""
    shutil.copytree(src_root, dst_root)
    for path in _paths(dst_root):
        with h5.File(path, "a") as f:
            packed = f[ss.PACKED_STATES]
            states = ss.unpack(packed[:], packed.attrs[ss.PLANES_SHAPE])
            del f[ss.PACKED_STATES]
            f.create_dataset("states", data=states, chunks=(64,) + states.shape[1:],
                             compression="lzf")


def _contents(path):
    with h5.File(path, "r") as f:
        return {name: f[name][()] for name in (ss.PACKED_STATES,) + COLUMNS +
                ("features", "conversion_args")}, dict(f[ss.PACKED_STATES].attrs)


def test_repacks_old_shards_to_exactly_the_current_format(current, tmp_path):
    old = str(tmp_path / "old")
    _to_old_format(current, old)
    with pytest.raises(ValueError, match="repack_shards"):
        ss.dataset_info(ss.find_split_shards(old, "train"))

    repack_shards.main([old])

    assert [os.path.relpath(p, old) for p in _paths(old)] == [
        os.path.relpath(p, current) for p in _paths(current)]
    for repacked, original in zip(_paths(old), _paths(current)):
        got, got_attrs = _contents(repacked)
        want, want_attrs = _contents(original)
        for name in want:
            assert np.array_equal(got[name], want[name]), name
        assert tuple(got_attrs[ss.PLANES_SHAPE]) == tuple(want_attrs[ss.PLANES_SHAPE])
        with h5.File(repacked, "r") as f:
            assert "states" not in f
            assert f[ss.PACKED_STATES].compression == "gzip"
    assert not glob.glob(os.path.join(old, "*", "*.partial"))
    ss.dataset_info(ss.find_split_shards(old, "train"))  # readable again


def test_skips_shards_already_in_the_current_format(current, tmp_path, capsys):
    root = str(tmp_path / "copy")
    shutil.copytree(current, root)
    before = {p: os.path.getmtime(p) for p in _paths(root)}
    repack_shards.main([root])
    assert "already packed" in capsys.readouterr().out
    assert {p: os.path.getmtime(p) for p in _paths(root)} == before


def test_no_shards_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no shard"):
        repack_shards.main([str(tmp_path)])
