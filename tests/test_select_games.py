"""Tests for select_games: pools, sampling, and a clean game-level split."""
import json
import os

from AlphaGo.preprocessing import select_games


def _manifest(tmp_path, rows):
    path = str(tmp_path / "m.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _row(name, gtype="normal", komi=7.5, wr=0.5, moves=100):
    return {"path": "game_data\\d\\{}.sgf".format(name), "gtype": gtype, "komi": str(komi),
            "first_searched_winrate": wr, "n_moves": moves}


def _read(out, split):
    with open(os.path.join(out, split + ".txt")) as f:
        return [line.rstrip("\n").split("\t") for line in f]


def test_pools_apply_the_agreed_criteria(tmp_path):
    m = _manifest(tmp_path, [
        _row("keep_normal"),
        _row("keep_hcap", gtype="handicap", komi=45.0),         # compensated
        _row("low_komi", komi=4.5),
        _row("high_komi", komi=9.5),
        _row("flip", komi=-6.5, wr=0.01),
        _row("lopsided", wr=0.85),
        _row("uncomp_hcap", gtype="handicap", komi=7.0, wr=0.0),
        _row("sgfpos", gtype="sgfpos"),
    ])
    normal, handicap = select_games._pools(m, 5, 9, 0.3, 0.7)
    assert [p[0] for p in normal] == ["game_data/d/keep_normal.sgf"]
    assert [p[0] for p in handicap] == ["game_data/d/keep_hcap.sgf"]


def test_split_is_by_game_stratified_and_reproducible(tmp_path):
    rows = [_row("n{}".format(i)) for i in range(400)]
    rows += [_row("h{}".format(i), gtype="handicap", komi=40) for i in range(40)]
    m = _manifest(tmp_path, rows)
    out = str(tmp_path / "sel")
    select_games.main([m, out, "--normal-positions", "1e9", "--handicap-positions", "1e9"])

    splits = {s: _read(out, s) for s in ("train", "val", "test")}
    paths = [r[0] for s in splits.values() for r in s]
    assert len(paths) == len(set(paths)) == 440, "a game is missing or in two splits"
    for name, lo in (("train", 0.9), ("val", 0.03), ("test", 0.01)):
        assert len(splits[name]) >= lo * 440
        assert any(r[2] == "handicap" for r in splits[name]), name + " has no handicap"
    assert all("\\" not in p for p in paths)

    out2 = str(tmp_path / "sel2")
    select_games.main([m, out2, "--normal-positions", "1e9", "--handicap-positions", "1e9"])
    assert _read(out2, "train") == splits["train"]


def test_position_targets_bound_the_sample(tmp_path):
    rows = [_row("n{}".format(i), moves=100) for i in range(1000)]
    m = _manifest(tmp_path, rows)
    out = str(tmp_path / "sel")
    select_games.main([m, out, "--normal-positions", "5000", "--handicap-positions", "0"])
    chosen = sum(len(_read(out, s)) for s in ("train", "val", "test"))
    # 5,000 positions at 0.97 positions per move of a 100-move game is 52 games
    assert chosen == 52
