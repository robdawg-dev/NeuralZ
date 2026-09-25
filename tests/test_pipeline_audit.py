"""Regression tests from the data-generation / training audit.

Two kinds of test live here:

1. Tests that PASS and lock in behaviour the trained model's correctness depends on
   (coordinate frame, symmetry consistency, shuffle-buffer coverage, batch identity).
   These are the ones that would catch a catastrophic silent regression.

2. Tests that lock in behaviour which LOOKS like a defect until you know why it is
   intended - each carries the reasoning, so nobody "fixes" it back. The defects this
   file originally documented as xfail have all been fixed; their tests now assert the
   corrected behaviour and are marked FIXED with a note on observed real-corpus impact.
"""
import os
import subprocess
import sys
import textwrap

import h5py as h5
import numpy as np
import pytest
import sgf as sgflib

import AlphaGo.go as go
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.training.shuffle_buffer import (
    BOARD_TRANSFORMATIONS, _ShardCache, _epoch_positions, build_game_index,
    get_or_create_game_split, one_hot_action, shuffle_buffer_batch_generator)
from AlphaGo.util import _sgf_init_gamestate, flatten_idx, sgf_iter_states

ALL_FEATURES = ["board", "ones", "turns_since", "liberties", "capture_size",
                "self_atari_size", "liberties_after", "ladder_capture",
                "ladder_escape", "sensibleness", "zeros"]
BOARD, NFEAT = 19, 48


# ---------------------------------------------------------------------------
# Coordinate frame: the single thing that would silently destroy the model.
# ---------------------------------------------------------------------------

def test_state_tensor_and_action_share_a_coordinate_frame():
    """state_to_tensor indexes [x][y][plane] and one_hot_action indexes [x][y].

    If these disagreed, every symmetry transform would rotate the board one way and
    the label the other, and the label would point at an intersection unrelated to the
    position. Checked by playing a stone at an ASYMMETRIC point and confirming the
    tensor marks the same cell the action one-hot does.
    """
    proc = Preprocess(ALL_FEATURES, size=BOARD)
    gs = go.GameState(BOARD, enforce_superko=False)
    pt = (2, 15)                      # deliberately x != y
    gs.do_move(pt)
    tensor = proc.state_to_tensor(gs)[0]

    # plane 0 = current player, plane 1 = opponent. Black just played, so the stone at
    # `pt` belongs to the OPPONENT of the player to move.
    assert tensor[pt[0]][pt[1]][1] == 1, "stone not found at [x][y] in the state tensor"
    assert tensor[pt[1]][pt[0]][1] == 0, "stone found at the TRANSPOSED cell"

    onehot = one_hot_action(pt, BOARD)
    assert onehot[pt[0]][pt[1]] == 1
    assert onehot.flatten()[flatten_idx(pt, BOARD)] == 1


@pytest.mark.parametrize("name", sorted(BOARD_TRANSFORMATIONS))
def test_symmetry_transforms_keep_state_and_label_aligned(name):
    """Applying a transform to the state and to the one-hot label must move both to the
    same intersection - otherwise 7 of the 8 symmetries feed the network mislabelled data.
    """
    fn = BOARD_TRANSFORMATIONS[name]
    proc = Preprocess(ALL_FEATURES, size=BOARD)
    gs = go.GameState(BOARD, enforce_superko=False)
    for mv in [(3, 15), (15, 3), (2, 2), (16, 16)]:
        gs.do_move(mv)
    state = proc.state_to_tensor(gs)[0]
    move = (4, 11)
    assert gs.is_legal(move)

    t_state = fn(state)
    t_label = fn(one_hot_action(move, BOARD))
    assert t_label.sum() == 1, "transform destroyed the one-hot label"
    tx, ty = divmod(int(np.argmax(t_label.flatten())), BOARD)
    # plane 2 == EMPTY: the transformed label must land on the transformed empty point
    assert t_state[tx][ty][2] == 1, (
        "{}: label landed on a non-empty point - state and label frames disagree".format(name))


# ---------------------------------------------------------------------------
# Shuffle buffer
# ---------------------------------------------------------------------------

def _make_synthetic_shard(path, game_lengths):
    """Shard where every position carries a recoverable unique id, so coverage and
    batch-identity can be checked exactly."""
    total = sum(game_lengths)
    with h5.File(path, 'w') as f:
        st = f.create_dataset('states', shape=(total, BOARD, BOARD, NFEAT), dtype=np.uint8)
        ac = f.create_dataset('actions', shape=(total, 2), dtype=np.uint8)
        grp = f.create_group('file_offsets')
        f['features'] = np.bytes_("board,ones")
        cur = 0
        for gi, length in enumerate(game_lengths):
            for j in range(length):
                uid = cur + j
                arr = np.zeros((BOARD, BOARD, NFEAT), dtype=np.uint8)
                arr[0, 0, 0] = uid % 251
                arr[0, 1, 0] = (uid // 251) % 251
                st[cur + j] = arr
                ac[cur + j] = [uid % BOARD, (uid // BOARD) % BOARD]
            grp["game{}".format(gi)] = [cur, length]
            cur += length
    return total


def _uid(state):
    return int(state[0, 0, 0]) + 251 * int(state[0, 1, 0])


@pytest.mark.parametrize("buffer_size", [1, 3, 25, 100, 300])
def test_epoch_positions_yields_every_position_exactly_once(tmp_path, buffer_size):
    shard = str(tmp_path / "s.h5")
    total = _make_synthetic_shard(shard, [7, 13, 5, 40, 1, 22, 9, 3])
    games, _, _, _ = build_game_index([shard])
    cache = _ShardCache()
    try:
        seen = [_uid(s) for s, _a in _epoch_positions(
            games, buffer_size, np.random.default_rng(7), cache, BOARD, NFEAT)]
    finally:
        cache.close()
    assert len(seen) == total
    assert sorted(seen) == list(range(total))


def test_game_split_is_position_disjoint(tmp_path):
    shard = str(tmp_path / "s.h5")
    total = _make_synthetic_shard(shard, [7, 13, 5, 40, 1, 22, 9, 3])
    games, _, _, _ = build_game_index([shard])
    out = tmp_path / "out"
    out.mkdir()
    train, val, test = get_or_create_game_split(games, str(out), [0.5, 0.25, 0.25], seed=1)

    def positions(group):
        found = set()
        for g in group:
            found.update(range(g["start"], g["start"] + g["length"]))
        return found

    ptr, pva, pte = positions(train), positions(val), positions(test)
    assert not (ptr & pva) and not (ptr & pte) and not (pva & pte)
    assert len(ptr | pva | pte) == total


def test_generator_yields_independent_batch_arrays(tmp_path):
    """The generator reuses Xbatch/Ybatch in place. Keras's GeneratorDataAdapter peeks
    two batches via itertools.islice and keeps both, so without a defensive copy the
    first peeked batch is silently overwritten by the second and a real batch is lost.
    """
    import itertools
    shard = str(tmp_path / "s.h5")
    _make_synthetic_shard(shard, [20, 20, 20])
    games, _, _, _ = build_game_index([shard])
    gen = shuffle_buffer_batch_generator(
        games, 10, 4, BOARD, NFEAT, [BOARD_TRANSFORMATIONS['noop']], seed=5)
    try:
        first, second = list(itertools.islice(gen, 2))
    finally:
        gen.close()
    assert first[0] is not second[0], "batches alias the same array object"
    assert not np.array_equal(first[0], second[0]), "batch 1 was overwritten by batch 2"
    assert not (set(_uid(x) for x in first[0]) & set(_uid(x) for x in second[0]))


def test_seed_makes_the_position_stream_reproducible(tmp_path):
    shard = str(tmp_path / "s.h5")
    _make_synthetic_shard(shard, [20, 20, 20])
    games, _, _, _ = build_game_index([shard])

    def stream(seed):
        gen = shuffle_buffer_batch_generator(
            games, 10, 4, BOARD, NFEAT, [BOARD_TRANSFORMATIONS['noop']], seed=seed)
        try:
            return np.stack([next(gen)[0].copy() for _ in range(5)])
        finally:
            gen.close()

    assert np.array_equal(stream(42), stream(42))
    assert not np.array_equal(stream(42), stream(43))


# ---------------------------------------------------------------------------
# Feature-plane invariants
# ---------------------------------------------------------------------------

def test_feature_planes_are_internally_consistent():
    proc = Preprocess(ALL_FEATURES, size=BOARD)
    gs = go.GameState(BOARD, enforce_superko=False)
    for mv in [(3, 3), (15, 15), (3, 15), (15, 3), (9, 9)]:
        gs.do_move(mv)
    t = proc.state_to_tensor(gs)[0]
    assert t.shape == (BOARD, BOARD, NFEAT)
    # board planes 0/1/2 are a 3-way one-hot at every intersection
    assert np.all(t[:, :, [0, 1, 2]].sum(axis=2) == 1)
    assert np.all(t[:, :, 3] == 1), "'ones' plane must be all ones"
    assert np.all(t[:, :, 47] == 0), "'zeros' plane must be all zeros"


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_each_feature_reports_the_plane_count_it_writes(feature):
    """Guards the offset chaining in state_to_tensor: the tensor depth must equal the
    output_dim the processor list advertised."""
    proc = Preprocess([feature], size=BOARD)
    t = proc.state_to_tensor(go.GameState(BOARD, enforce_superko=False))
    assert t.shape[3] == proc.get_output_dimension()


# ---------------------------------------------------------------------------
# Confirmed defects - assert the CORRECT behaviour, xfail until fixed.
# ---------------------------------------------------------------------------

# FIXED: sgf_iter_states used to reuse the previous node's move/player when a node carried
# neither W nor B, replaying that move - an IllegalMove that truncated the game, or an
# UnboundLocalError that dropped the whole file when such a node came first. Measured at
# 0/60,136 KataGo training games but 89/90 rating games, and live for KGS/GoGoD records.
@pytest.mark.parametrize("text,expected_moves", [
    ("(;GM[1]FF[4]SZ[19];B[pd];W[dp];C[comment];B[pp])", 3),
    ("(;GM[1]FF[4]SZ[19];C[hello];B[pd];W[dp])", 2),
])
def test_moveless_nodes_are_skipped_not_replayed(text, expected_moves):
    moves = [m for (_gs, m, _p) in sgf_iter_states(text, include_end=False)]
    assert len(moves) == expected_moves


# FIXED: AW (white setup) stones are placed through place_handicap_stone(), which used to
# increment the single num_handicap counter - so get_handicaps() reported white stones as
# black handicap (an HA[0] KataGo game with 15 AB + 13 AW returned 28). The counter is now
# split: num_handicap is the whole setup block (both colours, and the boundary the superko
# pre-filter keys off), num_black_handicap is only the genuine black handicap stones.
@pytest.mark.parametrize("text", [
    "(;GM[1]FF[4]SZ[19];B[pd];W[dp];TR[aa][bb];B[pp])",      # markup only
    "(;GM[1]FF[4]SZ[19];B[pd];W[dp];C[note];B[pp])",         # comment only
    "(;GM[1]FF[4]SZ[19];B[pd];W[dp];BL[12.5];B[pp])",        # timing only
])
def test_annotation_only_nodes_are_skipped(text):
    """Nodes with no board effect (comments, markup, timing) must be skipped silently.
    This is 100% of the moveless nodes in every corpus measured - KataGo rating games all
    end with a terminal C[...result=...]."""
    moves = [m for (_gs, m, _p) in sgf_iter_states(text, include_end=False)]
    assert len(moves) == 3


@pytest.mark.parametrize("text,prop", [
    ("(;GM[1]FF[4]SZ[19];B[pd];W[dp];AE[pd];B[pp])", "AE"),
    ("(;GM[1]FF[4]SZ[19];B[pd];W[dp];AB[cc];B[pp])", "AB"),
    ("(;GM[1]FF[4]SZ[19];B[pd];W[dp];AW[cc];B[pp])", "AW"),
    ("(;GM[1]FF[4]SZ[19];B[pd];W[dp];PL[B];B[pp])", "PL"),
])
def test_board_altering_nodes_stop_iteration_rather_than_desync(text, prop):
    """AB/AW/AE/PL outside the root change the position without being a move. We cannot
    apply them, and SKIPPING them would leave the board silently out of step with the
    record - every later move replayed against a position that never occurred.

    That is the same desync proven corrupting for "skip the suicide and carry on" (11
    'occupied' rejections after a skip, and 6 of 13 games silently accepting every later
    move on a wrong board). So it must raise, letting the caller keep the prefix and drop
    the remainder - never continue.
    """
    seen = []
    with pytest.raises(go.IllegalMove) as excinfo:
        for (_gs, move, _p) in sgf_iter_states(text, include_end=False):
            seen.append(move)
    assert prop in str(excinfo.value)
    # the prefix before the offending node is still delivered
    assert len(seen) == 2


def test_iterator_yields_the_offending_move_before_raising():
    """Documents a subtlety that matters downstream: sgf_iter_states yields
    (position, move) BEFORE applying the move, so a move the engine will reject still
    arrives at the consumer once, and only the FOLLOWING iteration raises.

    That is why convert_game needs its own legality guard - see the test below.
    """
    # black fills its own last liberty; (2, 0) is occupied by white, so this is rejected
    text = ("(;GM[1]FF[4]SZ[19]"
            "AB[aa][ba]"
            "AW[ab][bb][cb][ca]"
            "PL[B]"
            ";B[da]"
            ";W[ea]"
            ";B[ca])")
    seen = []
    with pytest.raises(go.IllegalMove):
        for (_gs, move, _p) in sgf_iter_states(text, include_end=False):
            seen.append(move)
    # two legal moves PLUS the rejected one, which was yielded but never applied
    assert len(seen) == 3, seen
    assert seen[-1] == (2, 0)


def test_converter_does_not_emit_an_illegal_move_as_a_label(tmp_path):
    """The position before an unplayable move is valid, but the MOVE is not - and
    emitting it teaches the network exactly what it must never play.

    On real data this is a multi-stone suicide from a KataGo sui1 ruleset: 0.43% of games
    carry one, and before the guard in convert_game each contributed one such label.
    Suicide is illegal under both rulesets KGS offers.
    """
    from AlphaGo.preprocessing.game_converter import GameConverter, SizeMismatchError

    text = ("(;GM[1]FF[4]SZ[19]"
            "AB[aa][ba]"
            "AW[ab][bb][cb][ca]"
            "PL[B]"
            ";B[da]"
            ";W[ea]"
            ";B[ca])")
    f = tmp_path / "g.sgf"
    f.write_text(text)

    conv = GameConverter(["board", "ones"])
    emitted = []
    try:
        for _state, move in conv.convert_game(str(f), 19):
            emitted.append(tuple(move))
    except (go.IllegalMove, SizeMismatchError):
        pass

    assert (2, 0) not in emitted, "the illegal move was emitted as a training label"
    assert emitted == [(3, 0), (4, 0)], emitted


def test_converter_keeps_the_prefix_of_a_truncated_game(tmp_path):
    """A game the engine cannot fully replay must still contribute the positions it
    managed, not be dropped entirely. Suicides land around move 290 of 320 on real data,
    so discarding whole games would waste ~91% of their usable moves."""
    from AlphaGo.preprocessing.game_converter import GameConverter, SizeMismatchError

    good = "(;GM[1]FF[4]SZ[19];B[pd];W[dp];B[pp];W[dd])"
    bad = "(;GM[1]FF[4]SZ[19];B[pd];W[dp];B[pp];W[dd];AE[pd];B[cc])"
    conv = GameConverter(["board", "ones"])

    def convert(text):
        f = tmp_path / "g.sgf"
        f.write_text(text)
        out = []
        try:
            for _state, move in conv.convert_game(str(f), 19):
                out.append(tuple(move))
        except (go.IllegalMove, SizeMismatchError):
            pass
        return out

    assert convert(good) == convert(bad), (
        "the prefix before the unreplayable node must match the clean game exactly")
    assert len(convert(bad)) == 4


def test_only_black_setup_stones_count_as_handicap():
    text = "(;GM[1]FF[4]SZ[19]HA[0]AB[aa][bb]AW[cc][dd]PL[B];B[pp])"
    gs = _sgf_init_gamestate(sgflib.parse(text)[0].root)
    # stone placement itself must stay correct
    board = gs.get_board()
    assert board[0][0] == go.BLACK and board[1][1] == go.BLACK
    assert board[2][2] == go.WHITE and board[3][3] == go.WHITE
    # exactly the two BLACK setup stones, and neither white one
    handicaps = gs.get_handicaps()
    assert sorted(handicaps) == [(0, 0), (1, 1)], handicaps
    assert (2, 2) not in handicaps and (3, 3) not in handicaps
    # the full setup block is still 4 long - that boundary is what superko needs
    assert len(gs.get_history()) == 4


def test_turns_since_ages_handicap_stones_as_played_moves():
    """Handicap stones occupy the recent-age planes, in placement order. This is
    deliberate, not a defect.

    In a handicap game Black really does place its stones in sequence immediately before
    White's first move, so they really are the most recent events on the board - and the
    live GTP path builds the identical history (cmd_set_free_handicap -> place_handicaps
    -> place_handicap_stone -> do_move). Routing them into the "age >= 7" plane instead
    would describe a board where stones were played 7+ turns ago followed by 7 turns of
    nobody playing, which cannot occur, and would differ from what the bot sees from KGS.

    The genuinely unrepresentable case is a serialized mid-game board (gtype=sgfpos/fork:
    50-100 AB/AW stones in raster order, captured stones absent). Those game types are
    excluded at selection instead - see DATA_PIPELINE.md.
    """
    # HA[3]: KataGo writes HA-1 setup stones and lets Black play the last one as a move
    text = "(;GM[1]FF[4]SZ[19]HA[3]AB[dd][pp];B[dp];W[pd])"
    gs = _sgf_init_gamestate(sgflib.parse(text)[0].root)
    proc = Preprocess(["turns_since"], size=BOARD)
    t = proc.state_to_tensor(gs)[0]

    # before any move is played: the two setup stones are the two most recent events,
    # newest first, and nothing else is marked anywhere.
    assert t[15][15][0] == 1, "the last-placed handicap stone (pp) is not at age 0"
    assert t[3][3][1] == 1, "the first-placed handicap stone (dd) is not at age 1"
    assert int(t[:, :, 0:8].sum()) == 2, "planes hold something other than the two stones"

    # after two real moves they shift back by two, still ahead of the played moves
    for mv in [(3, 15), (15, 3)]:
        gs.do_move(mv)
    t = proc.state_to_tensor(gs)[0]
    assert t[15][3][0] == 1 and t[3][15][1] == 1, "played moves are not the most recent"
    assert t[15][15][2] == 1 and t[3][3][3] == 1, "handicap stones did not age by two"
    assert int(t[:, :, 0:8].sum()) == 4


# FIXED: Preprocess.zeros() had no reachable return - "return offset + 1" had been
# absorbed into the trailing comment on the "# Nothing to do" line, so the cdef int
# function returned 0 and silently reset the plane offset. Harmless only while "zeros" was
# last in the default feature list; color() delegates to it and was broken regardless.
def test_zeros_feature_does_not_reset_the_plane_offset():
    # put 'zeros' FIRST so a bad return value corrupts what follows
    proc = Preprocess(["zeros", "board", "ones"], size=BOARD)
    gs = go.GameState(BOARD, enforce_superko=False)
    gs.do_move((3, 3))
    t = proc.state_to_tensor(gs)[0]
    assert np.all(t[:, :, 0] == 0), "plane 0 ('zeros') was overwritten"
    assert np.all(t[:, :, 1:4].sum(axis=2) == 1), "board planes are not a 3-way one-hot"
    assert np.all(t[:, :, 4] == 1), "'ones' plane missing - offset chain broken"


def test_mixed_board_sizes_do_not_corrupt_live_gamestates():
    """FIXED: GameState holds raw pointers into global neighbour/zobrist lookup tables.
    Those used to be single vectors reassigned in place whenever a state of a different
    size was constructed, silently invalidating the pointers held by every live state of
    the old size - and with boundscheck=False that reads out of bounds rather than
    raising. Building a 19x19 state, constructing a 9x9 state, then calling
    state_to_tensor on the 19x19 state produced a SIGSEGV (exit 139).

    The tables are now keyed by board size in a std::map, whose references to existing
    elements stay valid across insertions, so states of different sizes coexist safely.

    Still run in a subprocess: a regression here is a segfault, which would take the whole
    test runner down rather than failing a single test.
    """
    script = textwrap.dedent("""
        import numpy as np
        import AlphaGo.go as go
        from AlphaGo.preprocessing.preprocessing import Preprocess
        proc = Preprocess(["board", "ones", "turns_since", "liberties", "capture_size",
                           "self_atari_size", "liberties_after", "ladder_capture",
                           "ladder_escape", "sensibleness", "zeros"], size=19)
        big = go.GameState(19, enforce_superko=False)
        for mv in [(3, 3), (15, 15), (3, 15), (15, 3)]:
            big.do_move(mv)
        before = proc.state_to_tensor(big)
        small = go.GameState(9, enforce_superko=False)    # must NOT disturb big's tables
        small.do_move((2, 2))
        after = proc.state_to_tensor(big)
        print("SURVIVED")
        if (before == after).all():
            print("TENSORS-MATCH")
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, env=env)
    assert result.returncode == 0, (
        "constructing a 9x9 GameState while a 19x19 one is live crashed the process "
        "(exit {}) - the per-size lookup tables have regressed:\n{}".format(
            result.returncode, result.stderr.decode(errors="replace")[-2000:]))
    assert b"SURVIVED" in result.stdout
    assert b"TENSORS-MATCH" in result.stdout, (
        "the 19x19 state's tensor changed after a 9x9 state was constructed - its lookup "
        "tables were overwritten even though the process did not crash")
