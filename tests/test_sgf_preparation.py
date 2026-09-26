"""Tests for the SGF corpus scanner.

The contract that matters: `scan` reads the corpus exactly once, records facts about every
file it walks, and never modifies, deletes or filters anything. Selection lives in
select_games.py and works from the manifest alone, so a criterion can change without
re-reading a corpus that takes hours to scan.
"""
import gzip
import json
import os

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
    hintpos game both belong in the manifest, flagged, so selection can change its mind
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
