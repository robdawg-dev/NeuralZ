"""Tests for the two-phase SGF corpus preparation tool.

The contract that matters: `scan` reads the corpus exactly once and records facts; `select`
applies policy and can be re-run freely. Anything that makes a policy decision at scan
time, or that makes `select` need the original SGFs, breaks the reason the tool is split
in two - conversion is a 60+ hour job on a real corpus, so re-deriving anything from the
files is the expensive path.
"""
import gzip
import json
import os

import pytest

from AlphaGo.preprocessing import sgf_preparation as prep


# --- an SGF with known per-move annotations -------------------------------------------
# KataGo comment format: <win> <loss> <noResult> <score> v=<visits> [weight=<w>], with
# win ALWAYS from White's perspective. So the winrate change across a move measures what
# the player who moved gave up.
#
#   idx mover  white_w   loss for mover
#    0    B      0.50    0.80-0.50 = +0.30   <- a blunder at every threshold
#    1    W      0.80    0.80-0.82 = -0.02
#    2    B      0.82    0.81-0.82 = -0.01
#    3    W      0.81    (no next move)
ANNOTATED = (
    "(;GM[1]FF[4]SZ[19]KM[7.5]RU[koSIMPLEscoreAREAtaxNONEsui0]"
    "C[startTurnIdx=0,initTurnNum=0,gameHash=DEADBEEF,gtype=normal]"
    ";B[aa]C[0.50 0.50 0.00 0.0 v=100 weight=1.00]"
    ";W[bb]C[0.80 0.20 0.00 5.0 v=100 weight=1.00]"
    ";B[cc]C[0.82 0.18 0.00 5.0 v=100 weight=0.00]"
    ";W[dd]C[0.81 0.19 0.00 5.0 v=100 weight=1.00])"
)


def _corpus(tmp_path, **files):
    d = tmp_path / "sgf"
    d.mkdir(exist_ok=True)
    for name, text in files.items():
        (d / name).write_text(text)
    return str(d)


def _rows(path):
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# move_stats
# ---------------------------------------------------------------------------

def test_move_stats_measures_loss_from_the_movers_perspective():
    st = prep.move_stats(ANNOTATED)
    assert st["first_searched_winrate"] == 0.50
    assert st["visits_median"] == 100
    assert st["visits_p10"] == 100
    # only the first move loses ground, and it loses 0.30
    assert st["n_blunder_gt05"] == 1
    assert st["n_blunder_gt10"] == 1
    assert st["n_blunder_gt20"] == 1
    # nothing is anywhere near decided
    assert st["n_decided"] == 0
    assert st["n_mover_hopeless"] == 0


def test_move_stats_counts_decided_positions_from_the_right_side():
    """A position at white_w=0.99 is decided; it is HOPELESS only for the player to move
    when that player is Black. Getting the perspective backwards would invert the
    asymmetric decided-position filter this field exists to support."""
    text = ("(;GM[1]FF[4]SZ[19]"
            ";B[aa]C[0.99 0.01 0.00 9.0 v=100 weight=1.00]"    # Black to move, hopeless
            ";W[bb]C[0.99 0.01 0.00 9.0 v=100 weight=1.00])")  # White to move, winning
    st = prep.move_stats(text)
    assert st["n_decided"] == 2
    assert st["n_mover_hopeless"] == 1


def test_move_stats_is_empty_but_safe_without_annotations():
    st = prep.move_stats("(;GM[1]FF[4]SZ[19];B[aa];W[bb])")
    assert st["first_searched_winrate"] is None
    assert st["visits_median"] is None
    assert st["n_blunder_gt10"] == 0


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def test_scan_writes_one_row_per_file_and_never_touches_the_corpus(tmp_path):
    src = _corpus(tmp_path, a=ANNOTATED, b=ANNOTATED,
                  c="(;GM[1]FF[4]SZ[9];B[aa];W[bb])")
    before = {n: (tmp_path / "sgf" / n).read_text() for n in ("a", "b", "c")}
    # ".sgf" suffix is what the walker looks for
    for n in ("a", "b", "c"):
        os.rename(os.path.join(src, n), os.path.join(src, n + ".sgf"))

    manifest = str(tmp_path / "m.jsonl")
    prep.main(["scan", src, manifest, "--quiet", "--workers", "1"])

    rows = _rows(manifest)
    assert len(rows) == 3
    for r in rows:
        assert set(r) >= {"path", "size", "gtype", "komi", "n_moves", "sgf_ok",
                          "n_blunder_gt10", "n_decided", "game_hash"}
    # corpus untouched
    for n in ("a", "b", "c"):
        assert (tmp_path / "sgf" / (n + ".sgf")).read_text() == before[n]


def test_scan_records_rejections_as_advisory_without_acting_on_them(tmp_path):
    """The whole point of the split: scan must not drop anything. A 9x9 game and a
    hintpos game both belong in the manifest, flagged, so select can change its mind
    without a re-scan."""
    src = _corpus(tmp_path, **{
        "ok.sgf": ANNOTATED,
        "small.sgf": "(;GM[1]FF[4]SZ[9]KM[7]C[gtype=normal];B[aa];W[bb])",
        "hint.sgf": ANNOTATED.replace("gtype=normal", "gtype=hintpos"),
    })
    manifest = str(tmp_path / "m.jsonl")
    prep.main(["scan", src, manifest, "--quiet", "--workers", "1"])
    rows = {os.path.basename(r["path"]): r for r in _rows(manifest)}
    assert len(rows) == 3, "scan dropped a file"
    assert rows["small.sgf"]["size"] == "9"
    assert rows["hint.sgf"]["gtype"] == "hintpos"


def test_every_sgf_gets_a_row_however_malformed(tmp_path):
    """The manifest must account for every .sgf walked. A file that cannot be parsed is
    a finding to record, not a reason to omit it - otherwise "rows == files" stops being
    a check you can rely on."""
    src = _corpus(tmp_path, **{
        "good.sgf": ANNOTATED,
        "small.sgf": "(;GM[1]FF[4]SZ[9];B[aa];W[bb])",
        "empty.sgf": "",
        "garbage.sgf": "not an sgf at all",
        "nomoves.sgf": "(;GM[1]FF[4]SZ[19]KM[7.5])",
        "truncated.sgf": "(;GM[1]FF[4]SZ[19];B[pd]C[0.5",
        "notsgf.txt": ANNOTATED,          # wrong extension - must NOT be scanned
    })
    manifest = str(tmp_path / "m.jsonl")
    prep.main(["scan", src, manifest, "--quiet", "--workers", "1"])
    rows = _rows(manifest)
    assert len(rows) == 6, "one row per .sgf, and nothing else"
    assert not any(r["path"].endswith(".txt") for r in rows)


def test_a_malformed_annotation_does_not_kill_the_scan(tmp_path):
    """The move regexes are permissive ([0-9.]+ matches "1.2.3"), so float() can raise on a
    corrupt comment. That exception used to escape the worker, tear down the pool and
    leave an EMPTY manifest - losing a multi-hour scan to one bad file out of millions."""
    src = _corpus(tmp_path, **{
        "ok1.sgf": ANNOTATED,
        "bad.sgf": "(;GM[1]FF[4]SZ[19];B[pd]C[1.2.3 0.5 0.0 0.0 v=600 weight=1.00])",
        "ok2.sgf": ANNOTATED,
    })
    manifest = str(tmp_path / "m.jsonl")
    prep.main(["scan", src, manifest, "--quiet", "--workers", "1"])
    rows = {os.path.basename(r["path"]): r for r in _rows(manifest)}
    assert len(rows) == 3, "a malformed file cost other files their rows"
    assert any("move_stats_error" in r for r in rows["bad.sgf"]["reasons"])
    # the healthy files are unaffected
    assert rows["ok1.sgf"]["n_blunder_gt10"] == 1
    assert rows["ok2.sgf"]["reasons"] == []


def test_scan_resume_appends_without_duplicating(tmp_path):
    names = {"f{}.sgf".format(i): ANNOTATED for i in range(10)}
    src = _corpus(tmp_path, **names)
    manifest = str(tmp_path / "m.jsonl")
    prep.main(["scan", src, manifest, "--quiet", "--workers", "1", "--sample", "4"])
    assert len(_rows(manifest)) == 4
    prep.main(["scan", src, manifest, "--quiet", "--workers", "1", "--resume"])
    rows = _rows(manifest)
    assert len(rows) == 10
    assert len({r["path"] for r in rows}) == 10, "resume produced duplicate rows"


def test_scan_manifest_round_trips_through_gzip(tmp_path):
    src = _corpus(tmp_path, **{"a.sgf": ANNOTATED})
    manifest = str(tmp_path / "m.jsonl.gz")
    prep.main(["scan", src, manifest, "--quiet", "--workers", "1"])
    rows = _rows(manifest)
    assert len(rows) == 1 and rows[0]["gtype"] == "normal"


def test_no_move_stats_skips_only_the_per_move_aggregates(tmp_path):
    src = _corpus(tmp_path, **{"a.sgf": ANNOTATED})
    full, fast = str(tmp_path / "f.jsonl"), str(tmp_path / "q.jsonl")
    prep.main(["scan", src, full, "--quiet", "--workers", "1"])
    prep.main(["scan", src, fast, "--quiet", "--workers", "1", "--no-move-stats"])
    a, b = _rows(full)[0], _rows(fast)[0]
    assert a["n_moves"] == b["n_moves"] == 4          # header facts still present
    assert "n_blunder_gt10" in a and "n_blunder_gt10" not in b


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def _select(tmp_path, rows, *extra):
    manifest = tmp_path / "m.jsonl"
    with open(str(manifest), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    keep = tmp_path / "keep.txt"
    prep.main(["select", str(manifest), str(keep)] + list(extra))
    with open(str(keep)) as f:
        return [line.strip() for line in f if line.strip()]


def _row(**over):
    base = {"path": "x.sgf", "sgf_ok": True, "size": "19", "gtype": "normal",
            "komi": "7.5", "n_moves": 300, "n_moveless_nodes": 0, "n_ab": 0, "n_aw": 0,
            "n_annotated": 300, "n_weighted": 300, "n_blunder_gt10": 0}
    base.update(over)
    return base


@pytest.mark.parametrize("over,kept", [
    ({}, True),
    ({"size": "9"}, False),
    ({"gtype": "hintpos"}, False),
    ({"gtype": "hintfork"}, False),
    ({"gtype": "cleanuptraining"}, False),
    ({"gtype": "handicap"}, True),    # kept: its setup stones are genuine placements
    ({"gtype": "asym"}, False),       # asymmetric playouts - weakened side
    ({"gtype": "sgfpos"}, False),     # AB block is a serialized board, not a move order
    ({"gtype": "fork"}, False),
    ({"gtype": None}, False),         # allowlist fails safe on an unknown gtype
    ({"komi": "-40"}, False),
    ({"komi": "200"}, False),
    ({"komi": None}, False),
    ({"n_moves": 0}, False),
    ({"sgf_ok": False}, False),
    # moveless nodes are NOT a reason to reject the file. sgf_iter_states now skips
    # annotation-only nodes cleanly and raises only on board-altering ones, so the
    # converter handles both correctly. Rejecting here would discard every KataGo rating
    # game (89/90 carry a terminal result comment) for no benefit.
    ({"n_moveless_nodes": 3}, True),
])
def test_select_default_criteria(tmp_path, over, kept):
    rows = [_row(path="keepme.sgf", **over)]
    assert (_select(tmp_path, rows) == ["keepme.sgf"]) is kept


def test_select_does_not_filter_on_setup_stones_or_blunder_counts(tmp_path):
    """Both were removed. Setup stones are meaningful only for gtypes the allowlist
    already drops, and game-level blunder counts are superseded by the converter's
    position-level --max-winrate-loss."""
    rows = [_row(path="a.sgf", n_ab=5, n_aw=0, n_blunder_gt10=9)]
    assert _select(tmp_path, rows) == ["a.sgf"]


def test_select_komi_band_is_configurable(tmp_path):
    rows = [_row(path="a.sgf", komi="20")]
    assert _select(tmp_path, rows) == ["a.sgf"]                       # generous default
    assert _select(tmp_path, rows, "--komi-max", "16") == []


def test_komi_band_does_not_apply_to_handicap_games(tmp_path):
    """KataGo expresses handicap as komi compensation at ~13 pts/stone, so a 5-stone game
    sits near komi 53. Judging that by the even-board band discarded ~50% of handicap
    games - the data most wanted for play against handicap opponents."""
    hcap = [_row(path="h.sgf", komi="53.5", handicap="5", n_ab=4)]
    even = [_row(path="e.sgf", komi="53.5", handicap="0")]
    assert _select(tmp_path, hcap) == ["h.sgf"], "handicap komi must not hit the band"
    assert _select(tmp_path, even) == [], "even-board komi 53.5 is still out of range"
    # tightening the band must still leave handicap games alone
    assert _select(tmp_path, hcap, "--komi-max", "8") == ["h.sgf"]


def test_handicap_games_still_get_a_sanity_bound(tmp_path):
    """Loose, but not unbounded - the raw corpus holds komi of -303 and +359."""
    assert _select(tmp_path, [_row(path="h.sgf", komi="200.5", handicap="5")]) == []
    assert _select(tmp_path, [_row(path="h.sgf", komi="115.5", handicap="9")]) == ["h.sgf"]
    assert _select(tmp_path, [_row(path="h.sgf", komi="200.5", handicap="5"),
                              _row(path="i.sgf", komi="115.5", handicap="9")],
                   "--handicap-komi-max", "250") == ["h.sgf", "i.sgf"]



def test_allow_no_komi_opt_out(tmp_path):
    rows = [_row(path="a.sgf", komi=None)]
    assert _select(tmp_path, rows) == []
    assert _select(tmp_path, rows, "--allow-no-komi") == ["a.sgf"]


def test_select_limit_stops_early(tmp_path):
    rows = [_row(path="f{}.sgf".format(i)) for i in range(50)]
    assert len(_select(tmp_path, rows, "--limit", "7")) == 7


def test_select_is_repeatable_from_the_manifest_alone(tmp_path):
    """No SGF is read during select. This is what makes re-tuning criteria cheap, so it
    is worth pinning: the manifest must be self-sufficient."""
    rows = [_row(path="/nonexistent/never/created.sgf")]
    assert _select(tmp_path, rows) == ["/nonexistent/never/created.sgf"]


def test_gtype_allowlist_keeps_only_normal_and_handicap(tmp_path):
    rows = [_row(path="n.sgf", gtype="normal"), _row(path="h.sgf", gtype="handicap"),
            _row(path="s.sgf", gtype="sgfpos"), _row(path="a.sgf", gtype="asym"),
            _row(path="f.sgf", gtype="fork"), _row(path="x.sgf", gtype="brand_new_type")]
    assert _select(tmp_path, rows) == ["n.sgf", "h.sgf"]
    assert _select(tmp_path, rows, "--include-gtype", "sgfpos") == ["s.sgf"]
