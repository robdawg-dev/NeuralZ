import datetime
import sys
import multiprocessing
import os
import shutil
import tempfile
import gtp
from AlphaGo import go
from AlphaGo.util import save_gamestate_to_sgf
from builtins import input

# A dedicated log file (rather than stderr) so the command trail survives even if stderr
# has been redirected to /dev/null - written next to this file regardless of cwd, so it
# lands in a predictable place no matter where the process is launched from.
_CMD_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gtp_commands.log")


def _log_gtp_command(cmd):
    try:
        with open(_CMD_LOG_PATH, "a") as f:
            f.write("{} pid={} recv: {!r}\n".format(
                datetime.datetime.now().isoformat(), os.getpid(), cmd))
    except OSError:
        pass  # never let logging itself take down the bot


# The gtp package's own BLACK/WHITE constants (1/-1) don't match this engine's
# BLACK/WHITE values, so GTP-supplied colors must be translated before they reach
# GameState.do_move()/set_current_player().
_GTP_TO_GO_COLOR = {gtp.BLACK: go.BLACK, gtp.WHITE: go.WHITE}


# GTP vertices are 1-indexed with row 1 at the BOTTOM of the board. GameState uses SGF's
# orientation, the one all training data is read in: 0-indexed with y=0 the TOP row (SGF
# row 'a'). So the column only shifts by one while the row is inverted - e.g. on 19x19
# GTP C3 is (2, 16), written to SGF as [cq]. This keeps a position arriving over GTP
# oriented exactly as it would be read from that game's SGF record.
def _gtp_to_engine(vertex, size):
    (x, y) = vertex
    return (x - 1, size - y)


def _engine_to_gtp(point, size):
    (x, y) = point
    return (x + 1, size - y)


def run_gnugo(sgf_file_name, command):
    if shutil.which('gnugo'):
        from subprocess import Popen, PIPE
        p = Popen(['gnugo', '--chinese-rules', '--mode', 'gtp', '-l', sgf_file_name],
                  stdout=PIPE, stdin=PIPE, stderr=PIPE)
        out_bytes = p.communicate(input=command.encode('utf-8'))[0]
        return out_bytes.decode('utf-8')[2:]
    else:
        return ''


class ExtendedGtpEngine(gtp.Engine):

    recommended_handicaps = {
        2: "D4 Q16",
        3: "D4 Q16 D16",
        4: "D4 Q16 D16 Q4",
        5: "D4 Q16 D16 Q4 K10",
        6: "D4 Q16 D16 Q4 D10 Q10",
        7: "D4 Q16 D16 Q4 D10 Q10 K10",
        8: "D4 Q16 D16 Q4 D10 Q10 K4 K16",
        9: "D4 Q16 D16 Q4 D10 Q10 K4 K16 K10"
    }

    def call_gnugo(self, sgf_file_name, command):
        try:
            pool = multiprocessing.Pool(processes=1)
            result = pool.apply_async(run_gnugo, (sgf_file_name, command))
            output = result.get(timeout=10)
            pool.close()
            return output
        except multiprocessing.TimeoutError:
            pool.terminate()
            # if can't get answer from GnuGo, return no result
            return ''

    def cmd_play(self, arguments):
        # Overrides gtp.Engine.cmd_play to record the move leniently - see
        # GTPGameConnector.record_move.
        move = gtp.parse_move(arguments)
        if move:
            color, vertex = move
            if self.vertex_in_range(vertex) and self._game.record_move(color, vertex):
                return
        raise ValueError("illegal move")

    def cmd_genmove(self, arguments):
        # Overrides gtp.Engine.cmd_genmove, which ignores whether the engine accepted the
        # generated move - it would report a move to the server that this engine never
        # recorded, and the two boards would silently diverge. Fail loudly instead.
        color = gtp.parse_color(arguments)
        if not color:
            raise ValueError("unknown player: {}".format(arguments))
        move = self._game.get_move(color)
        if not self._game.make_move(color, move):
            raise ValueError("engine rejected its own move {}".format(gtp.gtp_vertex(move)))
        return gtp.gtp_vertex(move)

    def cmd_time_left(self, arguments):
        pass

    def cmd_place_free_handicap(self, arguments):
        try:
            number_of_stones = int(arguments)
        except Exception:
            raise ValueError('Number of handicaps could not be parsed: {}'.format(arguments))
        if number_of_stones < 2 or number_of_stones > 9:
            raise ValueError('Invalid number of handicap stones: {}'.format(number_of_stones))
        vertex_string = ExtendedGtpEngine.recommended_handicaps[number_of_stones]
        self.cmd_set_free_handicap(vertex_string)
        return vertex_string

    def cmd_set_free_handicap(self, arguments):
        vertices = arguments.strip().split()
        moves = [gtp.parse_vertex(vertex) for vertex in vertices]
        self._game.place_handicaps(moves)

    def _ask_gnugo_about_current_game(self, command):
        sgf_file_name = self._game.get_current_state_as_sgf()
        try:
            return self.call_gnugo(sgf_file_name, command)
        finally:
            os.remove(sgf_file_name)

    def cmd_final_score(self, arguments):
        return self._ask_gnugo_about_current_game('final_score\n')

    def cmd_final_status_list(self, arguments):
        return self._ask_gnugo_about_current_game('final_status_list {}\n'.format(arguments))

    def cmd_load_sgf(self, arguments):
        pass

    def cmd_save_sgf(self, arguments):
        pass

    # def cmd_kgs_genmove_cleanup(self, arguments):
    #     return self.cmd_genmove(arguments)


class GTPGameConnector(object):
    """A class implementing the functions of a 'game' object required by the GTP
    Engine by wrapping a GameState and Player instance
    """

    def __init__(self, player):
        self._state = go.GameState(enforce_superko=True)
        self._player = player
        # genmoves answered this game, per color - the player's own move count. Counted
        # here rather than from the board: a controller may send handicap stones as
        # ordinary 'play' moves, which the board can't tell apart from real ones. Resets
        # with the board; a controller that reconnects and replays a game restarts it.
        self._own_moves = {}
        # Not currently read anywhere (final scoring goes through an external gnugo
        # process via SGF export, not through GameState) - kept only so 'set_komi'
        # has somewhere to write to, matching the previous (already unused) behavior.
        self._komi = 7.5

    def clear(self):
        self._state = go.GameState(self._state.get_size(), enforce_superko=True)
        self._own_moves = {}

    def make_move(self, color, vertex):
        """Play a move under this engine's own rules (including positional superko) - for
        the bot's own moves. Returns False, leaving the game unchanged, if it is illegal."""
        return self._apply(self._state.do_move, color, vertex)

    def record_move(self, color, vertex):
        """Apply a move the controller reports as played. The server is the authority on
        its own game's rules, so ko/superko aren't re-checked (KGS Japanese rules allow
        repeating a position); only a move that can't physically be placed - an occupied
        point, or suicide - returns False, leaving the game unchanged."""
        return self._apply(self._state.record_move, color, vertex)

    def _apply(self, move_fn, color, vertex):
        # with its color: GTP lets either side move at any time, passes included
        move = go.PASS if vertex == gtp.PASS else _gtp_to_engine(vertex, self._state.get_size())
        try:
            move_fn(move, _GTP_TO_GO_COLOR[color])
            return True
        except go.IllegalMove:
            return False

    def set_size(self, n):
        self._state = go.GameState(n, enforce_superko=True)
        self._own_moves = {}

    def set_komi(self, k):
        self._komi = k

    def get_move(self, color):
        self._state.set_current_player(_GTP_TO_GO_COLOR[color])
        move = self._player.get_move(self._state, own_moves=self._own_moves.get(color, 0))
        self._own_moves[color] = self._own_moves.get(color, 0) + 1
        if move == go.PASS:
            return gtp.PASS
        else:
            return _engine_to_gtp(move, self._state.get_size())

    def get_current_state_as_sgf(self):
        """Writes the game to a new temp file and returns its path; the caller deletes it.

        The handle mkstemp opens is closed before writing: on Windows a file can't be
        opened a second time while its first handle is still open."""
        fd, path = tempfile.mkstemp(suffix='.sgf')
        os.close(fd)
        save_gamestate_to_sgf(self._state, '', path)
        return path

    def place_handicaps(self, vertices):
        size = self._state.get_size()
        self._state.place_handicaps([_gtp_to_engine(vertex, size) for vertex in vertices])


def run_gtp(player_obj, inpt_fn=None, name="Gtp Player", version="0.0"):
    gtp_game = GTPGameConnector(player_obj)
    gtp_engine = ExtendedGtpEngine(gtp_game, name, version)
    if inpt_fn is None:
        inpt_fn = input

    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not gtp_engine.disconnect:
        inpt = inpt_fn()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        cmd_list = inpt.split("\n")
        for cmd in cmd_list:
            # _log_gtp_command(cmd)
            engine_reply = gtp_engine.send(cmd)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()
