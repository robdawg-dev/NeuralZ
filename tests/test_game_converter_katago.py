"""Tests for the KataGo-specific converter.

The contract: every position written to an h5 is one we intend to train on. So the tests
care about (a) which positions get emitted, (b) that a game the engine cannot fully replay
still contributes its prefix and is classified, and (c) that the output stays readable by
the EXISTING trainer with no changes.
"""
import json
import os

import h5py as h5
import pytest

from AlphaGo.preprocessing import game_converter_katago_data as conv
from AlphaGo.training.shuffle_buffer import build_game_index, find_shard_files

FEATURES = ["board", "ones", "turns_since", "sensibleness", "zeros"]

# Four annotated moves. win/loss/noResult/score are from White's perspective, so the
# change across a move is what the MOVER gave up:
#   idx mover white_w  loss for mover
#    0    B     0.50   0.80-0.50 = +0.30   <- a blunder at every threshold
#    1    W     0.80   0.80-0.82 = -0.02
#    2    B     0.82   0.81-0.82 = -0.01
#    3    W     0.81   (no next)
CLEAN = (
    "(;GM[1]FF[4]SZ[19]KM[7.5]RU[koSIMPLEscoreAREAtaxNONEsui0]"
    "C[startTurnIdx=0,initTurnNum=0,gameHash=AAAA,gtype=normal]"
    ";B[pd]C[0.50 0.50 0.00 0.0 v=600 weight=1.00]"
    ";W[dp]C[0.80 0.20 0.00 5.0 v=600 weight=1.00]"
    ";B[pp]C[0.82 0.18 0.00 5.0 v=600 weight=1.00]"
    ";W[dd]C[0.81 0.19 0.00 5.0 v=600 weight=1.00])"
)

# Black (0,0)+(0,1) has exactly one liberty at (0,2); White holds (1,0),(1,1),(1,2),(0,3).
# Black filling (0,2) kills its own three stones and captures nothing - multi-stone
# suicide, legal under KataGo's sui1 rulesets and rejected by this engine.
SUICIDE = (
    "(;GM[1]FF[4]SZ[19]KM[7.5]RU[koSIMPLEscoreAREAtaxNONEsui1]"
    "C[gtype=normal]"
    "AB[aa][ab]AW[ba][bb][bc][ad]PL[B]"
    ";B[pp]C[0.50 0.50 0.00 0.0 v=600 weight=1.00]"
    ";W[qq]C[0.50 0.50 0.00 0.0 v=600 weight=1.00]"
    ";B[ac]C[0.50 0.50 0.00 0.0 v=600 weight=1.00])"
)

# A board-altering node mid-game: the position can no longer track the record.
SETUP_NODE = (
    "(;GM[1]FF[4]SZ[19]KM[7.5]C[gtype=normal]"
    ";B[pd]C[0.50 0.50 0.00 0.0 v=600 weight=1.00]"
    ";W[dp]C[0.50 0.50 0.00 0.0 v=600 weight=1.00]"
    ";AE[pd]"
    ";B[pp]C[0.50 0.50 0.00 0.0 v=600 weight=1.00])"
)

# Setup stones, so the first N positions are the ones turns_since misreports.
WITH_SETUP = (
    "(;GM[1]FF[4]SZ[19]KM[7.5]C[gtype=sgfpos]"
    "AB[aa][bb][cc]AW[dd][ee]PL[B]"
    # columns j..s (9..18) on rows j and k (9..10): 20 distinct on-board points, clear of
    # the setup stones at (0,0)..(4,4). Letters past 's' would be off a 19x19 board.
    + "".join(";{}[{}]C[0.50 0.50 0.00 0.0 v=600 weight=1.00]".format(
        "BW"[i % 2], "jklmnopqrs"[i % 10] + "jk"[i // 10])
        for i in range(20))
    + ")"
)


def _corpus(tmp_path, **files):
    d = tmp_path / "sgf"
    d.mkdir(exist_ok=True)
    for name, text in files.items():
        (d / (name + ".sgf")).write_text(text)
    return str(d)


def _run(src, out, **kw):
    kw.setdefault("features", FEATURES)
    kw.setdefault("workers", 1)
    kw.setdefault("quiet", True)
    return conv.convert(src, str(out), **kw)


def _positions(out):
    shards = find_shard_files(str(out))
    games, _f, _bs, _nf = build_game_index(shards)
    return shards, games, sum(g["length"] for g in games)


# ---------------------------------------------------------------------------
# annotation parsing
# ---------------------------------------------------------------------------

def test_annotations_are_scored_from_the_movers_perspective():
    ann = conv._annotations(CLEAN)
    assert len(ann) == 4
    # move 0 is Black at white_w 0.50 -> mover sees 0.50, and gives up 0.30
    assert ann[0][0] == pytest.approx(0.50)
    assert ann[0][1] == pytest.approx(0.30)
    # move 1 is White at white_w 0.80 -> mover sees 0.80, and gains
    assert ann[1][0] == pytest.approx(0.80)
    assert ann[1][1] < 0
    assert ann[3][1] is None, "last move has no following position to compare against"


def test_unannotated_moves_are_recorded_as_none():
    ann = conv._annotations("(;GM[1]FF[4]SZ[19];B[pd];W[dp])")
    assert ann == [None, None]


# ---------------------------------------------------------------------------
# what gets emitted
# ---------------------------------------------------------------------------

def test_clean_game_emits_every_move(tmp_path):
    src = _corpus(tmp_path, a=CLEAN)
    _run(src, tmp_path / "out")
    _shards, games, n = _positions(tmp_path / "out")
    assert len(games) == 1 and n == 4


def test_suicide_truncates_and_never_emits_the_suicide_itself(tmp_path):
    """The prefix is valid data and is kept; the suicide is neither applied nor emitted.
    Suicide is illegal under both rulesets KGS offers, so it must not become a label."""
    src = _corpus(tmp_path, a=SUICIDE)
    _run(src, tmp_path / "out")
    shards, games, n = _positions(tmp_path / "out")
    assert n == 2, "expected the two legal moves before the suicide"
    with h5.File(shards[0]) as f:
        actions = [tuple(a) for a in f["actions"][:]]
    assert (0, 2) not in actions, "the suicide was emitted as a training label"


def test_board_altering_node_truncates_rather_than_desyncing(tmp_path):
    """A mid-game AE changes the position without being a move. Continuing past it would
    replay every later move against a board that never existed."""
    src = _corpus(tmp_path, a=SETUP_NODE)
    _run(src, tmp_path / "out")
    _shards, _games, n = _positions(tmp_path / "out")
    assert n == 2, "expected only the prefix before the setup node"


@pytest.mark.parametrize("text,reason", [
    (SUICIDE, "illegal_suicide"),
    (SETUP_NODE, "setup_node"),
])
def test_truncation_reason_is_classified(text, reason, tmp_path):
    """Classification is the point: once suicide is an expected, counted event, any other
    reason becomes a real alarm instead of noise."""
    src = _corpus(tmp_path, a=text)
    conv._init_worker(conv._Cfg(FEATURES, 19, 0, None, None))
    res = conv.convert_one(os.path.join(src, "a.sgf"))
    assert res["truncation"] == reason


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def test_max_winrate_loss_drops_only_the_offending_position(tmp_path):
    src = _corpus(tmp_path, a=CLEAN)
    _run(src, tmp_path / "off")
    _run(src, tmp_path / "on", max_winrate_loss=0.10)
    assert _positions(tmp_path / "off")[2] == 4
    assert _positions(tmp_path / "on")[2] == 3


def test_drop_hopeless_mover_is_asymmetric(tmp_path):
    """white_w=0.99 is decided, but hopeless only for the player to move when that is
    Black. A symmetric filter would drop both; this must drop one."""
    text = ("(;GM[1]FF[4]SZ[19]KM[7.5]C[gtype=normal]"
            ";B[pd]C[0.99 0.01 0.00 9.0 v=600 weight=1.00]"
            ";W[dp]C[0.99 0.01 0.00 9.0 v=600 weight=1.00])")
    src = _corpus(tmp_path, a=text)
    _run(src, tmp_path / "out", drop_hopeless_mover=0.05)
    assert _positions(tmp_path / "out")[2] == 1


def test_skip_setup_positions_only_affects_setup_stone_games(tmp_path):
    src = _corpus(tmp_path, setup=WITH_SETUP, clean=CLEAN)
    _run(src, tmp_path / "off")
    _run(src, tmp_path / "on", skip_setup_positions=7)
    before = _positions(tmp_path / "off")[2]
    after = _positions(tmp_path / "on")[2]
    assert before - after == 7, "expected exactly 7 positions dropped, from one game only"


def test_filters_are_off_by_default(tmp_path):
    src = _corpus(tmp_path, a=CLEAN, b=WITH_SETUP)
    _run(src, tmp_path / "out")
    assert _positions(tmp_path / "out")[2] == 4 + 20


# ---------------------------------------------------------------------------
# sharding, resume, provenance, trainer compatibility
# ---------------------------------------------------------------------------

def test_sharding_rolls_and_leaves_no_empty_shard(tmp_path):
    """Measure the unsharded size first rather than hard-coding a byte threshold - these
    planes are extremely sparse and lzf compresses them by an amount that would make any
    fixed number fragile."""
    src = _corpus(tmp_path, **{"g{}".format(i): CLEAN for i in range(40)})

    _run(src, tmp_path / "whole", shard_bytes=10 ** 12)
    whole = find_shard_files(str(tmp_path / "whole"))
    assert len(whole) == 1
    total_bytes = os.path.getsize(whole[0])

    _run(src, tmp_path / "out", shard_bytes=max(4096, total_bytes // 3))
    shards, _games, n = _positions(tmp_path / "out")
    assert len(shards) > 1, "byte target never triggered a roll"
    assert n == 40 * 4, "sharding changed the number of positions written"
    for shard in shards:
        with h5.File(shard) as f:
            assert len(f["file_offsets"]) > 0, "empty shard left behind: " + shard


def test_resume_tops_up_to_the_limit_without_duplicating(tmp_path):
    src = _corpus(tmp_path, **{"g{}".format(i): CLEAN for i in range(20)})
    out = tmp_path / "out"
    _run(src, out, limit=8)
    assert _positions(out)[2] == 8 * 4
    # --limit counts the TOTAL target, so this adds 7 more rather than another 15
    _run(src, out, limit=15, resume=True)
    shards, games, n = _positions(out)
    assert len(games) == 15 and n == 15 * 4
    assert len({g["id"] for g in games}) == 15, "resume duplicated a game"


def test_conversion_args_are_recorded_for_provenance(tmp_path):
    """A final shard set outlives the session that made it; 'which filters produced
    this?' is not answerable from a directory listing."""
    src = _corpus(tmp_path, a=CLEAN)
    _run(src, tmp_path / "out", max_winrate_loss=0.10,
         conversion_args=json.dumps({"max_winrate_loss": 0.10}))
    shards = find_shard_files(str(tmp_path / "out"))
    with h5.File(shards[0]) as f:
        assert json.loads(f["conversion_args"][()].decode())["max_winrate_loss"] == 0.10


def test_output_is_readable_by_the_existing_trainer_path(tmp_path):
    """Shards must work with the current shuffle_buffer/trainer unchanged - the whole
    point of keeping the schema a backward-compatible superset."""
    src = _corpus(tmp_path, a=CLEAN, b=WITH_SETUP)
    _run(src, tmp_path / "out")
    shards = find_shard_files(str(tmp_path / "out"))
    games, feature_list, board_size, n_features = build_game_index(shards)
    assert feature_list == FEATURES
    assert board_size == 19
    assert n_features == 3 + 1 + 8 + 1 + 1      # board, ones, turns_since, sensible, zeros
    assert len(games) == 2


def test_accepts_a_keep_list_as_well_as_a_directory(tmp_path):
    src = _corpus(tmp_path, a=CLEAN, b=CLEAN, c=CLEAN)
    keep = tmp_path / "keep.txt"
    keep.write_text("\n".join(os.path.join(src, n + ".sgf") for n in ("a", "c")) + "\n")
    _run(str(keep), tmp_path / "out")
    assert len(_positions(tmp_path / "out")[1]) == 2


def test_refuses_to_write_into_a_non_empty_directory_without_resume(tmp_path):
    src = _corpus(tmp_path, a=CLEAN)
    _run(src, tmp_path / "out")
    with pytest.raises(ValueError):
        _run(src, tmp_path / "out")
