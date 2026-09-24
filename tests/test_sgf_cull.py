"""Tests for sgf_cull: scan decides, delete acts, and nothing else is ever removed."""
import os

import pytest

from AlphaGo.preprocessing import sgf_cull


def _game(gtype="normal", size="19", game_hash="AAAA", moves=";B[pd];W[dp]", extra=""):
    comment = "startTurnIdx=0,initTurnNum=0,gameHash={}".format(game_hash)
    if gtype is not None:
        comment += ",gtype={}".format(gtype)
    return "(;FF[4]GM[1]SZ[{}]HA[0]KM[7.5]{}C[{}]{})".format(size, extra, comment, moves)


def _write(root, rel, text):
    path = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="latin-1", newline="") as f:
        f.write(text)
    return path


def _scan(root, out):
    sgf_cull.main(["scan", str(root), str(out), "--quiet", "--workers", "2"])
    listed = {}
    with open(os.path.join(str(out), "to_delete.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            listed[os.path.basename(parts[0])] = parts[1:]
    return listed


def test_each_reason_is_detected_and_good_games_are_kept(tmp_path):
    root = tmp_path / "data"
    _write(root, "d/keep_normal.sgf", _game("normal", game_hash="01"))
    _write(root, "d/keep_handicap.sgf",
           _game("handicap", game_hash="02", extra="AB[dd][pp]", moves=";W[dp];B[pd]"))
    _write(root, "d/empty.sgf", "")
    _write(root, "d/blank.sgf", "   \n")
    _write(root, "d/garbage.sgf", "not an sgf at all")
    _write(root, "d/nomoves.sgf", _game(game_hash="03", moves=""))
    _write(root, "d/nine.sgf", _game(size="9", game_hash="04", moves=";B[cc]"))
    _write(root, "d/sgfpos.sgf", _game("sgfpos", game_hash="05"))
    _write(root, "d/asym.sgf", _game("asym", game_hash="06"))
    _write(root, "d/nogtype.sgf", _game(None, game_hash="07"))
    _write(root, "d/notes.txt", "ignored - not an .sgf")

    listed = _scan(root, tmp_path / "out")
    assert listed == {
        "empty.sgf": ["empty"],
        "blank.sgf": ["empty"],
        "garbage.sgf": ["unreadable"],
        "nomoves.sgf": ["no_moves"],
        "nine.sgf": ["not_19x19"],
        "sgfpos.sgf": ["gtype_sgfpos"],
        "asym.sgf": ["gtype_asym"],
        "nogtype.sgf": ["gtype_none"],
    }


def test_missing_sz_counts_as_19(tmp_path):
    root = tmp_path / "data"
    _write(root, "d/a.sgf", "(;FF[4]GM[1]C[gameHash=01,gtype=normal];B[pd];W[dp])")
    assert _scan(root, tmp_path / "out") == {}


def test_escaped_bracket_in_a_root_value_does_not_break_parsing(tmp_path):
    root = tmp_path / "data"
    _write(root, "d/a.sgf", _game(extra="GN[odd \\] name]"))
    assert _scan(root, tmp_path / "out") == {}


def test_root_node_longer_than_the_head_read_is_still_parsed(tmp_path):
    root = tmp_path / "data"
    _write(root, "d/a.sgf", _game(extra="GC[{}]".format("x" * (sgf_cull.HEAD_BYTES * 2))))
    assert _scan(root, tmp_path / "out") == {}


def test_byte_identical_duplicate_is_listed_and_the_earliest_copy_kept(tmp_path):
    root = tmp_path / "data"
    text = _game(game_hash="ABCD")
    first = _write(root, "2025-07-01/g.sgf", text)
    _write(root, "2025-07-02/g.sgf", text)
    listed = _scan(root, tmp_path / "out")
    assert list(listed) == ["g.sgf"]
    reason, kept = listed["g.sgf"]
    assert reason == "duplicate"
    assert kept == first


def test_same_hash_different_bytes_is_logged_not_deleted(tmp_path):
    root = tmp_path / "data"
    _write(root, "2025-07-01/g.sgf", _game(game_hash="ABCD", moves=";B[pd];W[dp]"))
    _write(root, "2025-07-02/g.sgf", _game(game_hash="ABCD", moves=";B[dd];W[pp]"))
    out = tmp_path / "out"
    assert _scan(root, out) == {}
    with open(os.path.join(str(out), "hash_conflicts.txt"), encoding="utf-8") as f:
        rows = [line.split("\t") for line in f if line.strip()]
    assert len(rows) == 1 and rows[0][0] == "ABCD"


def test_scan_deletes_nothing_and_refuses_to_overwrite(tmp_path):
    root = tmp_path / "data"
    bad = _write(root, "d/asym.sgf", _game("asym"))
    out = tmp_path / "out"
    _scan(root, out)
    assert os.path.exists(bad), "scan must never delete"
    with pytest.raises(SystemExit):
        sgf_cull.main(["scan", str(root), str(out), "--quiet"])


def test_delete_removes_exactly_the_listed_files(tmp_path):
    root = tmp_path / "data"
    keep = _write(root, "d/keep.sgf", _game("normal", game_hash="01"))
    bad = _write(root, "d/asym.sgf", _game("asym", game_hash="02"))
    out = tmp_path / "out"
    _scan(root, out)

    sgf_cull.main(["delete", str(out), "--dry-run"])
    assert os.path.exists(bad), "dry run must not delete"

    sgf_cull.main(["delete", str(out)])
    assert not os.path.exists(bad)
    assert os.path.exists(keep)


def test_duplicate_is_not_deleted_if_its_kept_copy_has_gone(tmp_path):
    """Deleting a duplicate is only safe while the copy it duplicates still exists -
    otherwise the last copy of the game would go."""
    root = tmp_path / "data"
    text = _game(game_hash="ABCD")
    first = _write(root, "2025-07-01/g.sgf", text)
    second = _write(root, "2025-07-02/g.sgf", text)
    out = tmp_path / "out"
    _scan(root, out)
    os.remove(first)
    sgf_cull.main(["delete", str(out)])
    assert os.path.exists(second)
