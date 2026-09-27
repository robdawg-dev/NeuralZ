"""Tests for AlphaGo/preprocessing/sgf_analyze.py, the read-only corpus survey tool."""
import collections
import json
import os

import pytest

from AlphaGo.preprocessing import sgf_analyze as sa

RULES = "RU[koSITUATIONALscoreAREAtaxNONEsui1]"
ROOT_C = "C[startTurnIdx=3,initTurnNum=0,gtype={},gameHash=ABCD]"


def _annot(v=400, rv=None, weight="1.00"):
    parts = ["0.45 0.55 0.00 -1.2", "v={}".format(v)]
    if rv is not None:
        parts.append("rv={}".format(rv))
    if weight is not None:
        parts.append("weight={}".format(weight))
    return "C[{}]".format(" ".join(parts))


def _katago(moves="", size=19, gtype="normal", extra=""):
    return ("(;FF[4]GM[1]SZ[{}]PB[kata]PW[kata]HA[0]KM[7.5]{}RE[B+R]{}{}{})"
            .format(size, RULES, ROOT_C.format(gtype), extra, moves))


GAME = _katago(";B[pd]{};W[dp]{};B[]{}".format(
    _annot(v=150), _annot(v=900, rv=2000, weight="0.00"), _annot(v=5000, weight="1.50")))


def _scan(text, **kwargs):
    return sa.scan_file_cheap("mem.sgf", text=text, **kwargs)


# --- _bucket ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,label", [(0, "<100"), (99, "<100"), (100, "100-200"),
                                         (3999, "2000-4000"), (4000, ">=4000")])
def test_bucket(value, label):
    assert sa._bucket(value, sa.VISIT_EDGES) == label


# --- scan_file_cheap -------------------------------------------------------------------

def test_clean_katago_game():
    rec = _scan(GAME)
    assert rec["reasons"] == ["suicide_allowed"]
    assert rec["size"] == "19"
    assert (rec["gtype"], rec["start_turn_idx"], rec["init_turn_num"], rec["game_hash"]) == (
        "normal", "3", "0", "ABCD")
    assert (rec["ko"], rec["score"], rec["tax"], rec["sui"], rec["rules_extra"]) == (
        "SITUATIONAL", "AREA", "NONE", "1", None)
    assert (rec["komi"], rec["result"], rec["handicap"]) == ("7.5", "B+R", "0")
    assert (rec["n_moves"], rec["n_passes"], rec["n_moveless_nodes"]) == (3, 1, 0)
    assert (rec["n_annotated"], rec["n_reanalyzed"], rec["n_weighted"]) == (3, 1, 3)
    assert (rec["n_weight_zero"], rec["n_weight_pos"]) == (1, 2)
    assert rec["visit_buckets"] == collections.Counter(
        {"100-200": 1, "500-1000": 1, ">=4000": 1})
    assert rec["weight_buckets"] == collections.Counter({"<0.001": 1, "1.0-2.0": 2})
    assert rec["bytes"] == len(GAME)


def test_wrong_board_size():
    rec = _scan(_katago(";B[cc]", size=9))
    assert rec["reasons"][0] == "not_19x19"
    assert rec["n_moves"] == 1  # scanning continued


def test_wrong_board_size_short_circuits():
    rec = _scan(_katago(";B[cc]", size=9), short_circuit=True)
    assert rec["reasons"] == ["not_19x19"]
    assert "n_moves" not in rec


def test_board_size_is_configurable():
    assert "not_9x9" not in _scan(_katago(";B[cc]", size=9), board_size="9")["reasons"]


def test_missing_size_is_flagged():
    rec = _scan("(;GM[1];B[pd])")
    assert rec["size"] is None
    assert "not_19x19" in rec["reasons"]


@pytest.mark.parametrize("gtype", sorted(sa.SUSPECT_GTYPES))
def test_suspect_gtype(gtype):
    assert "suspect_gtype:" + gtype in _scan(_katago(";B[pd]", gtype=gtype))["reasons"]


def test_no_provenance_comment():
    rec = _scan("(;GM[1]SZ[19];B[pd])")
    assert rec["gtype"] is None
    assert "start_turn_idx" not in rec


def test_setup_stones_are_counted():
    rec = _scan(_katago(";W[pd]", gtype="sgfpos", extra="AB[aa][bb][cc]AW[dd]"))
    assert (rec["n_ab"], rec["n_aw"]) == (3, 1)
    assert "has_setup_stones" in rec["reasons"]


def test_suicide_not_allowed_is_not_flagged():
    rec = _scan(GAME.replace("sui1", "sui0"))
    assert rec["sui"] == "0"
    assert "suicide_allowed" not in rec["reasons"]


def test_extra_rule_flags():
    rec = _scan(GAME.replace("sui1]", "sui1button1]"))
    assert rec["rules_extra"] == "button1"


def test_unparsed_ruleset():
    rec = _scan(GAME.replace(RULES, "RU[Japanese]"))
    assert rec["rules"] == "Japanese"
    assert "unparsed_ruleset" in rec["reasons"]
    assert "ko" not in rec


def test_no_moves():
    rec = _scan(_katago(""))
    assert rec["n_moves"] == 0
    assert "no_moves" in rec["reasons"]
    assert "no_move_annotations" not in rec["reasons"]


def test_moveless_nodes_are_counted():
    rec = _scan(_katago(";B[pd];C[a comment];W[dp];TR[aa]"))
    assert rec["n_moveless_nodes"] == 2
    assert "moveless_node" in rec["reasons"]


def test_rating_game_has_annotations_but_no_weights():
    rec = _scan(_katago(";B[pd]{};W[dp]{}".format(_annot(weight=None), _annot(weight=None))))
    assert (rec["n_annotated"], rec["n_weighted"]) == (2, 0)
    assert rec["weight_buckets"] == collections.Counter()
    assert "no_training_weights" in rec["reasons"]


def test_moves_without_annotations():
    rec = _scan(_katago(";B[pd];W[dp]"))
    assert (rec["n_annotated"], rec["n_weighted"], rec["n_weight_zero"]) == (0, 0, 0)
    assert "no_move_annotations" in rec["reasons"]


def test_reads_the_file_when_no_text_given(tmp_path):
    path = tmp_path / "g.sgf"
    path.write_text(GAME)
    assert sa.scan_file_cheap(str(path))["n_moves"] == 3


def test_unreadable_file(tmp_path):
    rec = sa.scan_file_cheap(str(tmp_path / "missing.sgf"))
    assert rec["reasons"] == ["unreadable"]
    assert "error" in rec


# --- scan_file_deep --------------------------------------------------------------------

# (x, y) -> SGF coordinate: column letter then row letter
def _c(x, y):
    return "abcdefghijklmnopqrs"[x] + "abcdefghijklmnopqrs"[y]


def _game(*moves):
    colors = "BW"
    return "(;GM[1]SZ[19]" + "".join(
        ";{}[{}]".format(colors[i % 2], "" if m is None else _c(*m))
        for i, m in enumerate(moves)) + ")"


def _deep(tmp_path, text):
    path = tmp_path / "g.sgf"
    path.write_text(text)
    return sa.scan_file_deep(str(path))


def test_deep_clean_game(tmp_path):
    out = _deep(tmp_path, _game((3, 3), (15, 15), None))
    assert out == {"replay": "ok", "replay_moves_ok": 3, "replay_moves_declared": 3}


def test_deep_occupied(tmp_path):
    out = _deep(tmp_path, _game((3, 3), (3, 3)))
    assert (out["replay"], out["replay_fail_move"]) == ("illegal_occupied", 1)


def test_deep_suicide(tmp_path):
    # black surrounds the corner; white then plays into it
    out = _deep(tmp_path, _game((1, 0), (10, 10), (0, 1), (0, 0)))
    assert (out["replay"], out["replay_fail_move"]) == ("illegal_suicide", 3)


def test_deep_ko(tmp_path):
    # . B W .
    # B W . W     black takes at (2, 1); white retaking at (1, 1) at once is ko
    # . B W .
    out = _deep(tmp_path, _game((1, 0), (2, 0), (0, 1), (1, 1), (1, 2), (3, 1), (10, 10),
                                (2, 2), (2, 1), (1, 1)))
    assert (out["replay"], out["replay_fail_move"]) == ("illegal_ko", 9)


def test_deep_no_moves(tmp_path):
    assert _deep(tmp_path, "(;GM[1]SZ[19])")["replay"] == "no_moves"


def test_deep_parse_error(tmp_path):
    out = _deep(tmp_path, "this is not an sgf")
    assert out["replay"] == "parse_error"
    assert out["replay_error"]


def test_deep_ignores_moveless_nodes(tmp_path):
    out = _deep(tmp_path, "(;GM[1]SZ[19];B[dd];C[note];W[pp])")
    assert (out["replay"], out["replay_moves_declared"]) == ("ok", 2)


# --- file discovery / chunking ---------------------------------------------------------

def _tree(root, n_top=3, n_nested=2):
    (root / "a" / "b").mkdir(parents=True)
    for i in range(n_top):
        (root / "g{}.sgf".format(i)).write_text(GAME)
    for i in range(n_nested):
        (root / "a" / "b" / "n{}.sgf".format(i)).write_text(_katago(";B[pd]", size=9))
    (root / "notes.txt").write_text("not a game")
    return root


def test_iter_sgf_files_recurses_and_filters(tmp_path):
    paths = sa.iter_sgf_files(str(_tree(tmp_path)))
    found = sorted(os.path.relpath(p, str(tmp_path)) for p in paths)
    assert found == sorted(["g0.sgf", "g1.sgf", "g2.sgf", os.path.join("a", "b", "n0.sgf"),
                            os.path.join("a", "b", "n1.sgf")])


def test_iter_sgf_files_limit(tmp_path):
    assert len(list(sa.iter_sgf_files(str(_tree(tmp_path)), limit=2))) == 2


def test_chunks():
    assert list(sa._chunks(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    assert list(sa._chunks([], 3)) == []


def test_scan_chunk_adds_deep_results_only_when_asked(tmp_path):
    path = str(_tree(tmp_path) / "g0.sgf")
    missing = str(tmp_path / "missing.sgf")
    cheap, gone = sa._scan_chunk(([path, missing], False))
    assert "replay" not in cheap
    deep, gone_deep = sa._scan_chunk(([path, missing], True))
    assert deep["replay"] == "ok"
    assert "replay" not in gone_deep  # unreadable files skip the replay


# --- Aggregator ------------------------------------------------------------------------

def test_aggregator_counts_totals_and_caps_examples():
    agg = sa.Aggregator(keep_examples=2)
    for i in range(3):
        rec = _scan(GAME)
        rec["path"] = "g{}.sgf".format(i)
        agg.add(rec)
    agg.add(dict(_scan(_katago(";B[pd]", size=9)), path="small.sgf",
                 replay="illegal_ko", replay_moves_declared=1, replay_moves_ok=0))

    assert agg.n_files == 4
    assert agg.counters["size"] == collections.Counter({"19": 3, "9": 1})
    assert agg.counters["replay"] == collections.Counter({"illegal_ko": 1})
    assert agg.totals["moves"] == 10 and agg.totals["passes"] == 3
    assert agg.totals["weight_zero"] == 3
    assert (agg.totals["replay_declared"], agg.totals["replay_ok"]) == (1, 0)
    assert agg.reasons["suicide_allowed"] == 4
    assert agg.examples["suicide_allowed"] == ["g0.sgf", "g1.sgf"]
    assert agg.reasons["replay:illegal_ko"] == 1
    assert agg.examples["not_19x19"] == ["small.sgf"]


# --- analyze / main --------------------------------------------------------------------

@pytest.mark.parametrize("workers", [1, 2])
def test_analyze_totals(tmp_path, capsys, workers):
    agg = sa.analyze(str(_tree(tmp_path / "corpus")), workers=workers, chunk_size=2)
    assert agg.n_files == 5
    assert agg.counters["size"] == collections.Counter({"19": 3, "9": 2})
    assert agg.totals["moves"] == 3 * 3 + 2
    out = capsys.readouterr().out
    assert "SGF CORPUS ANALYSIS" in out
    assert "files scanned : 5" in out
    assert "ENGINE REPLAY OUTCOME" not in out


def test_analyze_deep_reports_replay(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bad.sgf").write_text(_game((3, 3), (3, 3)))
    agg = sa.analyze(str(corpus), workers=1, deep=True)
    assert agg.counters["replay"] == collections.Counter({"illegal_occupied": 1})
    out = capsys.readouterr().out
    assert "ENGINE REPLAY OUTCOME" in out
    assert "replay:illegal_occupied" in out


def test_analyze_empty_directory(tmp_path, capsys):
    agg = sa.analyze(str(tmp_path), workers=1)
    assert agg.n_files == 0
    assert "(none)" in capsys.readouterr().out


def test_main_writes_json_summary(tmp_path, capsys):
    corpus = _tree(tmp_path / "corpus")
    out_json = tmp_path / "summary.json"
    sa.main([str(corpus), "--workers", "1", "--sample", "4", "--quiet",
             "--json", str(out_json)])
    summary = json.loads(out_json.read_text())
    assert summary["n_files"] == 4
    assert summary["deep"] is False
    assert summary["reasons"]["suicide_allowed"] == 4
    assert set(summary["counters"]["size"]) <= {"19", "9"}
    assert "machine-readable summary written" in capsys.readouterr().out


def test_main_progress_output_with_workers(tmp_path, capsys):
    sa.main([str(_tree(tmp_path / "corpus")), "--workers", "2", "--chunk-size", "1"])
    assert "files scanned : 5" in capsys.readouterr().out
