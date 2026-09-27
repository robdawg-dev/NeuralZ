"""Tests for interface/gtp_wrapper.py, driving the GTP engine in-process through
Engine.send() so every reply can be checked.

Tests marked xfail pin the intended behavior of known bugs, to be decided and fixed
separately.
"""
import os

import gtp
import pytest

from AlphaGo import go
from interface import gtp_wrapper
from interface.gtp_wrapper import ExtendedGtpEngine, GTPGameConnector, run_gtp


class ScriptedPlayer(object):
    """Returns the given moves in order, recording whose turn each request was for."""

    def __init__(self, moves=()):
        self.moves = list(moves)
        self.asked_for = []

    def get_move(self, state):
        self.asked_for.append(state.get_current_player())
        return self.moves.pop(0) if self.moves else go.PASS


def _engine(player=None):
    game = GTPGameConnector(player or ScriptedPlayer())
    return ExtendedGtpEngine(game, "Test Player", "1.2"), game


def _ok(response=None):
    return "= {}\n\n".format(response) if response is not None else "=\n\n"


def _err(message):
    return "? {}\n\n".format(message)


# --- identity / board setup ------------------------------------------------------------

def test_name_and_version():
    engine, _ = _engine()
    assert engine.send("name") == _ok("Test Player")
    assert engine.send("version") == _ok("1.2")


def test_message_ids_are_echoed():
    engine, _ = _engine()
    assert engine.send("7 name") == "=7 Test Player\n\n"


def test_boardsize_resizes_the_game():
    engine, game = _engine()
    assert engine.send("boardsize 9") == _ok()
    assert game._state.get_size() == 9


def test_clear_board_empties_the_game_and_keeps_its_size():
    engine, game = _engine()
    engine.send("boardsize 13")
    engine.send("play black D4")
    assert engine.send("clear_board") == _ok()
    assert game._state.get_history() == []
    assert game._state.get_size() == 13


def test_komi_is_recorded():
    engine, game = _engine()
    assert engine.send("komi 6.5") == _ok()
    assert game._komi == 6.5


@pytest.mark.parametrize("command", ["time_left B 60 0", "load_sgf x.sgf", "save_sgf x.sgf"])
def test_accepted_no_op_commands(command):
    engine, _ = _engine()
    assert engine.send(command) == _ok()


# --- play / genmove --------------------------------------------------------------------

def test_play_translates_vertex_and_color():
    engine, game = _engine()
    assert engine.send("play black D4") == _ok()   # D = 4th column (GTP skips I)
    assert engine.send("play white Q16") == _ok()
    board = game._state.get_board()
    assert game._state.get_history() == [(3, 3), (15, 15)]
    assert board[3][3] == go.BLACK and board[15][15] == go.WHITE


def test_play_same_color_twice_uses_the_given_color():
    engine, game = _engine()
    engine.send("play black D4")
    engine.send("play black Q16")
    assert game._state.get_board()[15][15] == go.BLACK


def test_play_pass():
    engine, game = _engine()
    assert engine.send("play black pass") == _ok()
    assert game._state.get_history() == [None]
    assert game._state.get_current_player() == go.WHITE


@pytest.mark.xfail(strict=True, reason="bug 1: a pass is played without its color, so it "
                                       "goes to whoever's turn the engine thinks it is")
def test_play_pass_out_of_turn_is_that_colors_pass():
    engine, game = _engine()
    assert engine.send("play white pass") == _ok()   # black to move, white passes
    assert game._state.get_current_player() == go.BLACK


@pytest.mark.parametrize("move", ["black D4", "black T20", "black Z1", "purple D4", "black"])
def test_illegal_or_malformed_play_is_rejected(move):
    engine, _ = _engine()
    engine.send("play white D4")
    assert engine.send("play " + move) == _err("illegal move")


def test_genmove_asks_the_player_and_plays_its_move():
    player = ScriptedPlayer([(3, 3)])
    engine, game = _engine(player)
    assert engine.send("genmove white") == _ok("D4")
    assert player.asked_for == [go.WHITE]
    assert game._state.get_board()[3][3] == go.WHITE


def test_genmove_pass():
    engine, game = _engine()
    assert engine.send("genmove b") == _ok("PASS")
    assert game._state.get_history() == [None]


def test_genmove_unknown_color():
    engine, _ = _engine()
    assert engine.send("genmove purple") == _err("unknown player: purple")


# --- handicap --------------------------------------------------------------------------

@pytest.mark.parametrize("stones", range(2, 10))
def test_place_free_handicap_uses_the_recommended_points(stones):
    engine, game = _engine()
    vertices = ExtendedGtpEngine.recommended_handicaps[stones]
    assert engine.send("place_free_handicap {}".format(stones)) == _ok(vertices)
    expected = [(x - 1, y - 1) for x, y in map(gtp.parse_vertex, vertices.split())]
    assert game._state.get_handicaps() == expected
    assert game._state.get_current_player() == go.WHITE


@pytest.mark.parametrize("arg,message", [
    ("1", "Invalid number of handicap stones: 1"),
    ("10", "Invalid number of handicap stones: 10"),
    ("many", "Number of handicaps could not be parsed: many"),
])
def test_place_free_handicap_rejects_bad_counts(arg, message):
    engine, game = _engine()
    assert engine.send("place_free_handicap " + arg) == _err(message)
    assert game._state.get_history() == []


def test_set_free_handicap_places_the_given_stones():
    engine, game = _engine()
    assert engine.send("set_free_handicap C3 R17 K10") == _ok()
    assert game._state.get_handicaps() == [(2, 2), (16, 16), (9, 9)]


# --- scoring through gnugo -------------------------------------------------------------

def test_call_gnugo_without_gnugo_installed_returns_nothing(tmp_path):
    engine, _ = _engine()
    sgf = tmp_path / "g.sgf"
    sgf.write_text("(;GM[1]SZ[19])")
    assert engine.call_gnugo(str(sgf), "final_score\n") == ""


@pytest.mark.parametrize("command,gnugo_command", [
    ("final_score", "final_score\n"),
    ("final_status_list dead", "final_status_list dead\n"),
])
def test_scoring_commands_send_the_game_to_gnugo(monkeypatch, command, gnugo_command):
    engine, _ = _engine()
    engine.send("play black D4")
    calls = []

    def fake_call_gnugo(sgf_file_name, cmd):
        with open(sgf_file_name) as f:
            calls.append((f.read(), cmd))
        return "B+7.5"

    monkeypatch.setattr(engine, "call_gnugo", fake_call_gnugo)
    assert engine.send(command) == _ok("B+7.5")
    [(sgf_text, sent)] = calls
    assert sent == gnugo_command
    assert ";B[" in sgf_text


@pytest.mark.xfail(strict=True, reason="bug 2: get_current_state_as_sgf never deletes its "
                                       "temp file (and writes it while still open)")
def test_scoring_does_not_leave_temp_files(monkeypatch):
    engine, _ = _engine()
    paths = []
    monkeypatch.setattr(engine, "call_gnugo", lambda path, cmd: paths.append(path) or "")
    engine.send("final_score")
    [path] = paths
    assert not os.path.exists(path)


def test_run_gnugo_without_gnugo_returns_nothing(tmp_path):
    assert gtp_wrapper.run_gnugo(str(tmp_path / "g.sgf"), "final_score\n") == ""


# --- run_gtp loop ----------------------------------------------------------------------

def test_run_gtp_answers_each_command_until_quit(capsys):
    lines = iter(["1 name\n2 boardsize 9\n3 play black C3", "4 genmove white", "5 quit",
                  "never read"])
    run_gtp(ScriptedPlayer([(6, 6)]), inpt_fn=lambda: next(lines), name="Loop", version="3")
    out, err = capsys.readouterr()
    assert out == "=1 Loop\n\n=2\n\n=3\n\n=4 G7\n\n=5\n\n"
    assert "GTP engine ready" in err
    assert next(lines) == "never read"
