"""Convert shards from before positions were stored bit-packed to the current format.

    python -m AlphaGo.preprocessing.repack_shards workspace/lr_study/shards

Older convert_shuffled.py output stored each feature plane as a byte (`states`, LZF); the
training stream now reads `packed_states` (see shard_stream). This rewrites every old
shard under <shards directory>/{train,val,test}/ in place, position for position - the
order, and everything else in the shard, is unchanged - so a repacked set trains exactly
as the original did. Shards already in the current format are skipped, so an interrupted
run can simply be started again. About 8x smaller on disk afterwards.
"""
import argparse
import glob
import os
import sys
import time

import h5py as h5
import numpy as np

from AlphaGo.preprocessing.convert_shuffled import SPLITS, _create
from AlphaGo.training.shard_stream import PACKED_STATES, PLANES_SHAPE, pack


def repack_shard(path, block=8192):
    """Rewrites one old-format shard in the current format, via a temp file, so the shard
    on disk is always complete. Returns False if it was already in the current format."""
    tmp = path + ".partial"
    with h5.File(path, "r") as src:
        if PACKED_STATES in src:
            return False
        states = src["states"]
        n, shape = len(states), states.shape[1:]
        with h5.File(tmp, "w") as dst:
            packed = _create(dst, PACKED_STATES, ((int(np.prod(shape)) + 7) // 8,),
                             np.uint8, rows=n)
            packed.attrs[PLANES_SHAPE] = shape
            for start in range(0, n, block):
                packed[start:start + block] = pack(states[start:start + block])
            for name in ("actions", "game_id", "move"):
                column = _create(dst, name, src[name].shape[1:], src[name].dtype, rows=n)
                column[:] = src[name][:]
            for name in ("features", "conversion_args"):
                dst[name] = src[name][()]
    os.replace(tmp, path)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("shards_directory",
                        help="A convert_shuffled.py output directory (with train/, val/...)")
    args = parser.parse_args(argv)
    paths = [p for split in SPLITS for p in sorted(
        glob.glob(os.path.join(args.shards_directory, split, "shard_*.h5")))]
    if not paths:
        raise ValueError("no shard_*.h5 files under {}/{{{}}}".format(
            args.shards_directory, ",".join(SPLITS)))
    before = after = 0
    for i, path in enumerate(paths):
        size = os.path.getsize(path)
        start = time.time()
        if repack_shard(path):
            before, after = before + size, after + os.path.getsize(path)
            print("[{}/{}] {}: {:.2f} GB -> {:.2f} GB in {:.0f}s".format(
                i + 1, len(paths), path, size / 1e9, os.path.getsize(path) / 1e9,
                time.time() - start), flush=True)
        else:
            print("[{}/{}] {}: already packed".format(i + 1, len(paths), path), flush=True)
    if before:
        print("repacked: {:.1f} GB -> {:.1f} GB".format(before / 1e9, after / 1e9))


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        sys.exit("error: {}".format(e))
