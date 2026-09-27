"""Tests for AlphaGo/util.py: SGF move/setup parsing, SGF export, and the heatmap plot.

sgf_iter_states' handling of moveless and setup nodes is covered in test_pipeline_audit.py;
this file covers the rest. Tests marked xfail pin the intended behavior of known bugs,
to be decided and fixed separately.
"""
import os

import numpy as np
import pytest
import sgf as sgflib

from AlphaGo import go
from AlphaGo.util import (
    _parse_sgf_move, _sgf_init_gamestate, flatten_idx, plot_network_output,
    save_gamestate_to_sgf, sgf_iter_states, sgf_to_gamestate)


def _root(text):
    return sgflib.parse(text)[0].root


def _saved(tmp_path, state, **kwargs):
    save_gamestate_to_sgf(state, str(tmp_path), "game.sgf", **kwargs)
    return (tmp_path / "game.sgf").read_text()


def _played(moves, size=19, handicaps=None):
    state = go.GameState(size, enforce_superko=False)
    if handicaps:
        state.place_handicaps(handicaps)
    for move in moves:
        state.do_move(move)
    return state


# Deliberately asymmetric under every board flip, so a mirrored round trip can't match.
ASYMMETRIC_MOVES = [(3, 2), (15, 16), (2, 13), None, (10, 4)]


# --- move / setup parsing --------------------------------------------------------------

def test_flatten_idx_is_x_major():
    assert flatten_idx((0, 0), 19) == 0
    assert flatten_idx((0, 5), 19) == 5
    assert flatten_idx((2, 3), 19) == 2 * 19 + 3
    assert flatten_idx((8, 8), 9) == 80


@pytest.mark.parametrize("value,expected", [
    ("", go.PASS), ("tt", go.PASS), ("aa", (0, 0)), ("ab", (0, 1)), ("ba", (1, 0)),
    ("sa", (18, 0)), ("AS", (0, 18)),
])
def test_parse_sgf_move(value, expected):
    assert _parse_sgf_move(value) == expected


def test_init_gamestate_defaults_to_empty_19x19_black_to_move():
    gs = _sgf_init_gamestate(_root("(;GM[1]FF[4])"))
    assert gs.get_size() == 19
    assert gs.get_current_player() == go.BLACK
    assert gs.get_history() == []


def test_init_gamestate_reads_size_setup_stones_and_player():
    gs = _sgf_init_gamestate(_root("(;GM[1]SZ[9]AB[aa][cc]AW[ee]PL[W])"))
    assert gs.get_size() == 9
    board = gs.get_board()
    assert board[0][0] == go.BLACK and board[2][2] == go.BLACK
    assert board[4][4] == go.WHITE
    assert gs.get_current_player() == go.WHITE


def test_sgf_to_gamestate_returns_the_final_position():
    gs = sgf_to_gamestate("(;GM[1]SZ[9];B[cd];W[ge];B[])")
    assert gs.get_history() == [(2, 3), (6, 4), None]
    board = gs.get_board()
    assert board[2][3] == go.BLACK and board[6][4] == go.WHITE
    assert gs.get_current_player() == go.WHITE


def test_sgf_iter_states_yields_each_move_then_the_end():
    seen = [(move, player) for (_gs, move, player) in
            sgf_iter_states("(;GM[1]SZ[9];B[cd];W[ge])")]
    assert seen == [((2, 3), go.BLACK), ((6, 4), go.WHITE), (None, None)]


def test_sgf_iter_states_without_end():
    seen = [move for (_gs, move, _p) in
            sgf_iter_states("(;GM[1]SZ[9];B[cd];W[ge])", include_end=False)]
    assert seen == [(2, 3), (6, 4)]


# --- save_gamestate_to_sgf -------------------------------------------------------------

def test_save_writes_header_and_alternating_moves(tmp_path):
    text = _saved(tmp_path, _played([(0, 0), None]), black_player_name="Alice",
                  white_player_name="Bob", komi=6.5)
    assert text.startswith("(;GM[1]FF[4]CA[UTF-8]SZ[19]KM[6.5]PB[Alice]PW[Bob]")
    assert ";B[" in text and ";W[tt]" in text
    assert text.endswith(")")
    assert "HA[" not in text


def test_save_writes_handicap_count_and_white_moves_first(tmp_path):
    text = _saved(tmp_path, _played([(9, 9)], handicaps=[(3, 3), (15, 15)]))
    assert "HA[2]" in text
    assert text.count(";W[") == 1 and ";B[" not in text


def test_saved_file_parses(tmp_path):
    text = _saved(tmp_path, _played(ASYMMETRIC_MOVES))
    game = sgflib.parse(text)[0]
    assert len(list(game.rest)) == len(ASYMMETRIC_MOVES)


@pytest.mark.xfail(strict=True, reason="bug 3: save writes rows via REV_LETTERS (flipped) "
                                       "but _parse_sgf_move reads them unflipped")
def test_save_round_trips_moves(tmp_path):
    state = _played(ASYMMETRIC_MOVES)
    reread = sgf_to_gamestate(_saved(tmp_path, state))
    assert reread.get_history() == state.get_history()


@pytest.mark.xfail(strict=True, reason="bug 4: SZ comes from the size argument (default "
                                       "19), not the state; REV_LETTERS assumes 19 rows")
def test_save_round_trips_a_9x9_game(tmp_path):
    state = _played([(0, 0), (8, 1), (4, 6)], size=9)
    reread = sgf_to_gamestate(_saved(tmp_path, state))
    assert reread.get_size() == 9
    assert reread.get_history() == state.get_history()


@pytest.mark.xfail(strict=True, reason="bug 5: handicap stones are written as ;AB on a "
                                       "non-root node, which sgf_iter_states rejects")
def test_save_round_trips_a_handicap_game(tmp_path):
    state = _played([(9, 9), (2, 5)], handicaps=[(3, 3), (15, 15)])
    reread = sgf_to_gamestate(_saved(tmp_path, state))
    assert sorted(reread.get_handicaps()) == sorted(state.get_handicaps())
    assert reread.get_history() == state.get_history()


# --- plot_network_output ---------------------------------------------------------------

class _BoardView:
    """The (size, [i][j]) board interface plot_network_output expects - as built by
    benchmarks/_plot_sgf_heatmaps.py."""

    def __init__(self, state):
        self.size = state.get_size()
        self._board = state.get_board()

    def __getitem__(self, i):
        return self._board[i]


@pytest.mark.parametrize("western", [True, False])
@pytest.mark.parametrize("history", [[(3, 2), (15, 16)], [(3, 2), None], []])
def test_plot_network_output_writes_an_image(tmp_path, western, history):
    pytest.importorskip("matplotlib")
    state = _played([m for m in history if m is not None])
    scores = np.random.default_rng(0).dirichlet(np.ones(361))
    plot_network_output(scores, _BoardView(state), history, str(tmp_path), "heat.png",
                        western_column_notation=western)
    path = tmp_path / "heat.png"
    assert path.exists() and os.path.getsize(str(path)) > 0


def test_plot_network_output_without_a_file_saves_nothing(tmp_path):
    pytest.importorskip("matplotlib")
    plot_network_output(np.zeros(361), _BoardView(_played([])), [], str(tmp_path), None)
    assert list(tmp_path.iterdir()) == []
