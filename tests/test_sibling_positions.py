"""Tests for build_sibling_positions (a policy's top-k moves played out from sampled shard
positions) and label_sibling_positions' query/perspective helpers."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import csv
import glob

import h5py as h5
import numpy as np

import AlphaGo.go as go
from AlphaGo.models.policy import NewResPolicy
from AlphaGo.preprocessing import add_value_targets as avt
from AlphaGo.preprocessing import build_sibling_positions as bsp
from AlphaGo.preprocessing import convert_shuffled as conv
from AlphaGo.preprocessing import label_sibling_positions as lsp
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.training.shard_stream import PACKED_STATES, PLANES_SHAPE, unpack
from AlphaGo.util import sgf_iter_states
from tests.test_add_value_targets import FEATURES, _game

TOP = 4


def _shards(tmp_path):
    sel, sgfs = tmp_path / "sel", tmp_path / "sgf"
    sel.mkdir()
    sgfs.mkdir()
    with open(str(sel / "val.txt"), "w", newline="\n") as f:
        for seed in range(6):
            path = str(sgfs / "g{}.sgf".format(seed))
            with open(path, "w") as g:
                g.write(_game(seed))
            f.write("{}\t40\tnormal\n".format(path))
    out = str(tmp_path / "shards")
    conv.main([str(sel), out, "--splits", "val", "--features", FEATURES, "--workers", "2",
               "--quiet", "--positions-per-bucket", "100", "--positions-per-file", "100"])
    avt.main([out, "--splits", "val", "--workers", "2", "--quiet"])
    return out


def test_siblings_are_the_policys_top_moves_played_out(tmp_path):
    shards = _shards(tmp_path)
    policy = NewResPolicy(FEATURES.split(","), num_blocks=1, filters=4, gpool_every=0,
                          head_channels=2)
    out = str(tmp_path / "sib")
    meta = bsp.build(shards, out, policy, "val", 20, TOP, seed=1, workers=2, quiet=True)
    assert meta["sources"] == 20
    with h5.File(glob.glob(os.path.join(out, "val", "sib_*.h5"))[0], "r") as f:
        planes = unpack(f[PACKED_STATES][()], tuple(f[PACKED_STATES].attrs[PLANES_SHAPE]))
        group, rank, prior = f["group"][()], f["rank"][()], f["prior"][()]
    with open(os.path.join(out, "val", "queries.tsv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == len(planes) == 20 * TOP
    assert list(rank[:TOP]) == list(range(TOP)) and len(set(group[:TOP])) == 1
    assert np.allclose(prior.reshape(-1, TOP).astype(np.float32).sum(axis=1), 1, atol=1e-2)

    proc = Preprocess(FEATURES.split(","))
    for r in (0, TOP + 1, len(rows) - 1):  # a few siblings, rebuilt from scratch
        row = rows[r]
        with open(row["path"]) as f:
            text = f.read()
        for i, (state, _move, player) in enumerate(sgf_iter_states(text, include_end=False)):
            if i == int(row["node"]):
                state.set_current_player(player)
                assert int(row["black_to_move"]) == (0 if player == go.BLACK else 1)
                if r % TOP == 0:  # the first sibling of a group is the policy's top move
                    probs = policy.forward(proc.state_to_tensor(state).astype(np.float32))[0]
                    sensible = state.get_legal_moves(include_eyes=False)
                    top = max(sensible, key=lambda m: probs[m[0] * 19 + m[1]])
                    assert row["move"] == bsp.COLS[top[0]] + bsp.COLS[top[1]]
                m = row["move"]
                state.do_move((bsp.COLS.index(m[0]), bsp.COLS.index(m[1])))
                assert np.array_equal(proc.state_to_tensor(state)[0], planes[r])
                break


def test_exclude_skips_an_earlier_builds_sources(tmp_path):
    shards = _shards(tmp_path)
    policy = NewResPolicy(FEATURES.split(","), num_blocks=1, filters=4, gpool_every=0,
                          head_channels=2)

    def sources(out):
        with open(os.path.join(out, "val", "queries.tsv"), encoding="utf-8") as f:
            return {(r["path"], r["node"]) for r in csv.DictReader(f, delimiter="	")}

    first, second = str(tmp_path / "a"), str(tmp_path / "b")
    bsp.build(shards, first, policy, "val", 20, TOP, seed=1, workers=2, quiet=True)
    meta = bsp.build(shards, second, policy, "val", 20, TOP, seed=1, workers=2, quiet=True,
                     exclude=[os.path.join(first, "val", "queries.tsv")])
    assert meta["excluded"] == 20 and meta["sources"] == 20
    assert len(sources(first)) == len(sources(second)) == 20
    assert not sources(first) & sources(second)


def test_sibling_query_plays_the_candidate_for_the_source_mover():
    base = {"boardXSize": 19, "initialStones": [], "rules": "chinese", "komi": 7.5,
            "boardYSize": 19}
    moves = [["B", "Q16"], ["W", "D4"], ["B", "Q4"]]
    q = lsp.sibling_query("5", base, moves, 2, "dd", black_to_move=False, visits=1)
    # node 2 is Black's move; the sibling's player to move is White
    assert q["moves"] == [["B", "Q16"], ["W", "D4"], ["B", "D16"]]
    assert q["analyzeTurns"] == [3] and q["maxVisits"] == 1 and q["id"] == "5"


def test_labels_turn_to_the_siblings_player_to_move():
    assert lsp.to_player_to_move(4.0, 0.7, black_to_move=True) == (4.0, 0.7)
    score, win = lsp.to_player_to_move(4.0, 0.7, black_to_move=False)
    assert score == -4.0 and abs(win - 0.3) < 1e-9


def test_game_setup_reads_handicap_stones_and_komi():
    base, moves = lsp.game_setup("(;SZ[19]KM[0.5]AB[dd][pp]HA[2];W[dp];B[])")
    assert base["initialStones"] == [["B", "D16"], ["B", "Q4"]] and base["komi"] == 0.5
    assert moves == [["W", "D4"], ["B", "pass"]]
