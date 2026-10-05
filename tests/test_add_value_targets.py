"""Tests for add_value_targets: every shard record gets the annotation of its own move node,
turned to the player to move's side, and a record that doesn't line up stops the run."""
import csv
import glob
import io
import os
import random

import h5py as h5
import numpy as np
import pytest

import AlphaGo.go as go
from AlphaGo.preprocessing import add_value_targets as avt
from AlphaGo.preprocessing import convert_shuffled as conv
from AlphaGo.training import shard_stream as ss

FEATURES = "board,ones,sensibleness"
KOMI = 6.5
START_TURN = 2   # like KataGo's startTurnIdx: the first nodes carry no annotation
PASS_AT = 17     # a pass node: no shard record, but it still counts in the node index


def _annotation(i):
    """Distinct per node, so a record matched to the wrong node shows up: White win rate,
    White score."""
    return 0.10 + (i % 40) / 50.0, i - 30.0


def _game(seed, n_moves=40):
    rng = random.Random(seed)
    gs = go.GameState(19, enforce_superko=False)
    parts = []
    for i in range(n_moves):
        color = "B" if gs.get_current_player() == go.BLACK else "W"
        if i == PASS_AT:
            gs.do_move(go.PASS)
            coord = ""
        else:
            legal = gs.get_legal_moves(include_eyes=False)
            if not legal:
                break
            x, y = rng.choice(legal)
            gs.do_move((x, y))
            coord = chr(97 + x) + chr(97 + y)
        node = ";{}[{}]".format(color, coord)
        if i >= START_TURN:
            win, score = _annotation(i)
            node += "C[{:.2f} {:.2f} 0.00 {:.1f} v=400 weight=1.00]".format(win, 1 - win, score)
        parts.append(node)
    return "(;GM[1]FF[4]SZ[19]KM[{}]C[gtype=normal]".format(KOMI) + "".join(parts) + ")"


def _shards(tmp_path, n_games=12):
    sel, sgfs = tmp_path / "sel", tmp_path / "sgf"
    sel.mkdir()
    sgfs.mkdir()
    with open(str(sel / "train.txt"), "w", newline="\n") as f:
        for seed in range(n_games):
            path = str(sgfs / "g{}.sgf".format(seed))
            with open(path, "w") as g:
                g.write(_game(seed))
            f.write("{}\t40\tnormal\n".format(path))
    out = str(tmp_path / "out")
    conv.main([str(sel), out, "--splits", "train", "--features", FEATURES, "--workers", "2",
               "--quiet", "--positions-per-bucket", "100", "--positions-per-file", "100"])
    return out


def test_parse_game_indexes_move_nodes_like_sgf_iter_states():
    komi, black, index, win, score = avt.parse_game(_game(3))
    assert komi == KOMI
    assert len(index) == 40
    assert black[0] and not black[1]
    assert index[PASS_AT] == avt.NO_MOVE
    assert np.isnan(win[:START_TURN]).all() and not np.isnan(win[START_TURN:]).any()
    assert win[5] == pytest.approx(_annotation(5)[0], abs=1e-6)
    assert score[5] == pytest.approx(_annotation(5)[1])


def test_every_record_gets_its_own_nodes_targets_for_the_player_to_move(tmp_path):
    out = _shards(tmp_path)
    avt.main([out, "--splits", "train", "--workers", "2", "--quiet"])
    shards = sorted(glob.glob(os.path.join(out, "train", "shard_*.h5")))
    assert len(shards) > 1
    checked = with_target = 0
    for shard in shards:
        with h5.File(shard, "r") as f:
            moves = f["move"][()]
        with h5.File(avt.sidecar_path(shard), "r") as f:
            assert f.attrs["shard"] == os.path.basename(shard)
            value, score = f["value"][()], f["score"][()]
            komi, has = f["komi"][()], f["has_target"][()]
            black = f["black_to_move"][()]
        assert len(value) == len(moves)
        for r, i in enumerate(moves.tolist()):
            assert i != PASS_AT, "passes are never positions"
            white_to_move = i % 2 == 1  # Black moves first and the colours alternate
            assert black[r] == (0 if white_to_move else 1)
            assert komi[r] == pytest.approx(KOMI if white_to_move else -KOMI)
            if i < START_TURN:
                assert has[r] == 0
                continue
            win, w_score = _annotation(i)
            assert has[r] == 1
            assert value[r] == pytest.approx(win if white_to_move else 1 - win, abs=2e-3)
            assert score[r] == pytest.approx(w_score if white_to_move else -w_score, abs=0.05)
            with_target += 1
        checked += len(moves)
    assert with_target > 0 and checked > with_target


def test_sidecars_are_not_found_as_shards(tmp_path):
    out = _shards(tmp_path, n_games=3)
    before = ss.find_split_shards(out, "train")
    avt.main([out, "--splits", "train", "--workers", "2", "--quiet"])
    assert ss.find_split_shards(out, "train") == before
    assert all(os.path.basename(p).startswith("value_")
               for p in glob.glob(os.path.join(out, "train", "*.h5")) if p not in before)


def test_a_record_that_does_not_match_its_node_stops_the_run(tmp_path):
    out = _shards(tmp_path, n_games=4)
    tsv = os.path.join(out, "train", "games.tsv")
    with io.open(tsv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows[0]["path"], rows[1]["path"] = rows[1]["path"], rows[0]["path"]  # wrong SGF per id
    with io.open(tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(RuntimeError, match="does not match the stored action"):
        avt.main([out, "--splits", "train", "--workers", "2", "--quiet"])
