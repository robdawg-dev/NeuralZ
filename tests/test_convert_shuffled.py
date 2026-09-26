"""Tests for the two-pass shuffling converter.

The properties that matter: every emitted position appears exactly once, it is still the
same tensor and label it would be without the shuffle, and the order across shards no
longer follows games."""
import glob
import os
import random

import h5py as h5
import numpy as np
import pytest

import AlphaGo.go as go
from AlphaGo.preprocessing import convert_shuffled as conv

FEATURES = "board,ones,turns_since,sensibleness,zeros"


def _random_game(seed, n_moves=40, blunder_at=None):
    """A legal random game as KataGo-style SGF, every move annotated. If blunder_at is set,
    the move at that index gives up 0.30 winrate."""
    rng = random.Random(seed)
    gs = go.GameState(19, enforce_superko=False)
    nodes, white_w = [], 0.50
    for i in range(n_moves):
        legal = gs.get_legal_moves(include_eyes=False)
        if not legal:
            break
        x, y = rng.choice(legal)
        color = "B" if gs.get_current_player() == go.BLACK else "W"
        gs.do_move((x, y))
        nodes.append((color, "{}{}".format(chr(97 + x), chr(97 + y))))
    parts = []
    for i, (color, coord) in enumerate(nodes):
        parts.append(";{}[{}]C[{:.2f} {:.2f} 0.00 0.0 v=400 weight=1.00]".format(
            color, coord, white_w, 1 - white_w))
        if i == blunder_at:
            # the mover's position worsens by 0.30 at the next annotation
            white_w = white_w + 0.30 if color == "B" else white_w - 0.30
    return ("(;GM[1]FF[4]SZ[19]KM[7.5]C[gameHash={:04X},gtype=normal]".format(seed)
            + "".join(parts) + ")")


def _selection(tmp_path, splits):
    """Write games and keep-lists. splits: {name: [(seed, blunder_at), ...]}."""
    sel = tmp_path / "sel"
    sgfs = tmp_path / "sgf"
    sel.mkdir()
    sgfs.mkdir(exist_ok=True)
    for name, specs in splits.items():
        with open(str(sel / (name + ".txt")), "w", newline="\n") as f:
            for seed, blunder_at in specs:
                path = str(sgfs / "g{}.sgf".format(seed))
                with open(path, "w") as g:
                    g.write(_random_game(seed, blunder_at=blunder_at))
                f.write("{}\t40\tnormal\n".format(path))
    return str(sel)


def _run(sel, out, *extra):
    conv.main([sel, out, "--features", FEATURES, "--workers", "2", "--quiet",
               "--positions-per-bucket", "150", "--positions-per-file", "150"]
              + list(extra))


def _load(split_dir):
    cols = {"states": [], "actions": [], "game_id": [], "move": []}
    shards = sorted(glob.glob(os.path.join(split_dir, "shard_*.h5")))
    for path in shards:
        with h5.File(path, "r") as f:
            for k in cols:
                cols[k].append(f[k][:])
            assert f["features"][()].decode() == FEATURES
    return shards, {k: np.concatenate(v) for k, v in cols.items()}


def _direct(path, max_loss=None):
    conv._init_worker(FEATURES.split(","), 19, max_loss, keep_unpacked=True)
    return conv.convert_game((0, path))


def test_every_position_appears_once_and_matches_direct_conversion(tmp_path):
    games = [(s, None) for s in range(30)]
    sel = _selection(tmp_path, {"train": games})
    out = str(tmp_path / "out")
    _run(sel, out, "--splits", "train")
    shards, data = _load(os.path.join(out, "train"))
    assert len(shards) > 1, "test needs several shards to exercise the shuffle"
    assert not glob.glob(os.path.join(out, "train", "_buckets", "*")), "buckets left behind"

    keys = list(zip(data["game_id"].tolist(), data["move"].tolist()))
    assert len(keys) == len(set(keys)), "a position was emitted twice"

    total = 0
    for gid, (seed, _b) in enumerate(games):
        ref = _direct(str(tmp_path / "sgf" / "g{}.sgf".format(seed)))
        rows = np.flatnonzero(data["game_id"] == gid)
        rows = rows[np.argsort(data["move"][rows])]
        assert np.array_equal(data["move"][rows], ref["moves"])
        assert np.array_equal(data["states"][rows], ref["states"]), "tensor/label misaligned"
        assert np.array_equal(data["actions"][rows], ref["actions"])
        total += len(ref["moves"])
    assert len(keys) == total


def test_order_across_shards_no_longer_follows_games(tmp_path):
    sel = _selection(tmp_path, {"train": [(s, None) for s in range(30)]})
    out = str(tmp_path / "out")
    _run(sel, out, "--splits", "train")
    _shards, data = _load(os.path.join(out, "train"))
    ids = data["game_id"]
    same_next = np.mean(ids[1:] == ids[:-1])
    assert same_next < 0.15, "consecutive positions still come from the same game"
    # every game with enough positions is spread over more than one shard
    per_shard = []
    for path in sorted(glob.glob(os.path.join(out, "train", "shard_*.h5"))):
        with h5.File(path, "r") as f:
            per_shard.append(set(f["game_id"][:].tolist()))
    for gid in range(30):
        assert sum(gid in s for s in per_shard) > 1


def test_max_winrate_loss_drops_only_the_blunder(tmp_path):
    sel = _selection(tmp_path, {"train": [(1, 10), (2, None)]})
    out = str(tmp_path / "out")
    _run(sel, out, "--splits", "train", "--max-winrate-loss", "0.10")
    _shards, data = _load(os.path.join(out, "train"))
    blunder_moves = data["move"][data["game_id"] == 0]
    assert 10 not in blunder_moves
    assert 11 in blunder_moves, "the position after the blunder must be kept"
    clean = _direct(str(tmp_path / "sgf" / "g2.sgf"))
    assert np.sum(data["game_id"] == 1) == len(clean["moves"])


def test_splits_are_written_separately_with_game_tables(tmp_path):
    sel = _selection(tmp_path, {"train": [(1, None), (2, None)],
                                "val": [(3, None)], "test": [(4, None)]})
    out = str(tmp_path / "out")
    _run(sel, out)
    for split, seeds in (("train", [1, 2]), ("val", [3]), ("test", [4])):
        with open(os.path.join(out, split, "games.tsv")) as f:
            rows = [line.rstrip("\n").split("\t") for line in f][1:]
        assert [r[1].endswith("g{}.sgf".format(s)) for r, s in zip(rows, seeds)] == \
            [True] * len(seeds)
    assert os.path.exists(os.path.join(out, "conversion.json"))


def test_refuses_to_overwrite_existing_output(tmp_path):
    sel = _selection(tmp_path, {"train": [(1, None)]})
    out = str(tmp_path / "out")
    _run(sel, out, "--splits", "train")
    with pytest.raises(SystemExit):
        _run(sel, out, "--splits", "train")


def test_shard_size_is_independent_of_bucket_size(tmp_path):
    """Buckets are a memory knob, shards are packaging. Grouping several buckets into one
    shard must leave the position SEQUENCE identical - only where the file boundaries fall
    changes."""
    games = [(s, None) for s in range(20)]
    sel = _selection(tmp_path, {"train": games})
    one = str(tmp_path / "one")      # one bucket per shard
    many = str(tmp_path / "many")    # four buckets per shard
    _run(sel, one, "--splits", "train")
    conv.main([sel, many, "--features", FEATURES, "--workers", "2", "--quiet",
               "--splits", "train", "--positions-per-bucket", "150",
               "--positions-per-file", "600"])

    shards_one, data_one = _load(os.path.join(one, "train"))
    shards_many, data_many = _load(os.path.join(many, "train"))
    assert len(shards_many) < len(shards_one), "grouping did not reduce the file count"
    for key in ("game_id", "move", "actions", "states"):
        assert np.array_equal(data_one[key], data_many[key]), (
            "{} differs: grouping changed the position order".format(key))
