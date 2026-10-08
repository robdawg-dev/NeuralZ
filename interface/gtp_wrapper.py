import datetime
import sys
import os
import shutil
import subprocess
import tempfile
import gtp
from AlphaGo import go
from AlphaGo.util import save_gamestate_to_sgf
from builtins import input


class GtpLog(object):
    """Appends each GTP command (">") and reply ("<") to a file, one line each with the time
    and process id: the whole exchange with the controller (e.g. kgsGtp), to see afterwards
    what a game's controller asked and what the bot answered. Its own file rather than
    stderr, which controllers often discard. path=None logs nothing. A failed write is
    reported once on stderr and never stops the bot."""

    def __init__(self, path=None):
        self.path = path
        self._failed = False

    def write(self, direction, text):
        if self.path is None:
            return
        try:
            # opened per line: commands are seconds apart, and a log moved or deleted
            # while the bot runs simply starts again
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("{} pid={} {} {!r}\n".format(
                    datetime.datetime.now().isoformat(timespec="milliseconds"), os.getpid(),
                    direction, text.rstrip("\n")))
        except OSError as e:
            if not self._failed:
                self._failed = True
                sys.stderr.write("gtp log: cannot write {}: {}\n".format(self.path, e))
                sys.stderr.flush()


# The gtp package's own BLACK/WHITE constants (1/-1) don't match this engine's
# BLACK/WHITE values, so GTP-supplied colors must be translated before they reach
# GameState.do_move()/set_current_player().
_GTP_TO_GO_COLOR = {gtp.BLACK: go.BLACK, gtp.WHITE: go.WHITE}
_GO_TO_GTP_COLOR = {go.BLACK: gtp.BLACK, go.WHITE: gtp.WHITE}

# Most contested points (KataGo ownership still open) at which the bot passes back after the
# opponent's pass. Measured on 212 KGS games: finished boards 0-2 (median 0), mid-game
# positions 12-275 - so 10 lets no mid-game position end, and stopped exactly the games that
# ended on an unsettled board (171-211).
SETTLED_MAX_CONTESTED = 10
# Most of the bot's own points that may be left open (KataGo gives them to the bot, but the
# border isn't closed, so a count gives them to no one) when it passes back. Over the 540
# pass-backs the gate allowed after its deploy, the bot had 0-2 open points on 531 boards and
# 10-47 on the other 9 - five of them games lost that way (MEASUREMENTS.md).
OPEN_MAX = 4
# Border moves per game at most: if KataGo keeps seeing an open border, the bot passes back
# rather than fill its own area forever.
BORDER_MOVES_MAX = 20


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


def run_gnugo(sgf_file_name, command, timeout=10):
    """GNU Go's answer to one GTP command about the game in sgf_file_name, or None if GNU Go
    isn't installed, answered with an error or took over timeout seconds."""
    if not shutil.which('gnugo'):
        return None
    p = subprocess.Popen(['gnugo', '--chinese-rules', '--mode', 'gtp', '-l', sgf_file_name],
                         stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out = p.communicate(input=command.encode('utf-8'), timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return None
    out = out.decode('utf-8')
    if not out.startswith('='):
        return None
    return out[2:].strip()


class ExtendedGtpEngine(gtp.Engine):
    """The bot's GTP engine. End of the game: final_status_list / final_score ask the scorer
    first (KataGo behind go_server --katago, see interface/katago_scorer.py), then GNU Go,
    and answer a GTP error if neither does - kgsGtp then leaves the dead stones to the
    opponent, rather than reading an empty answer as "no dead stones". With cleanup (and a
    scorer), kgs-genmove_cleanup is supported: KGS uses it when the opponent disputes the
    dead stones in a non-Japanese game, and the bot captures the stones KataGo judges dead
    before it passes.

    stop_file: while this file exists, new games are declined (kgsGtp checks each challenge
    with boardsize) and the engine exits when a game ends (kgs-game_over) - so a deployment
    can let every bot finish its game and stop, instead of killing it mid-game.

    board_size: the only size accepted (the network's), so a challenge on another board is
    declined up front rather than failing at the first genmove. None: any size."""

    def __init__(self, game_obj, name="gtp (python library)", version="0.2", scorer=None,
                 cleanup=False, stop_file=None, board_size=None):
        super(ExtendedGtpEngine, self).__init__(game_obj, name, version)
        self._scorer = scorer
        self._stop_file = stop_file
        self._board_size = board_size
        # GTP command names with a hyphen can't be method names: registered by hand
        setattr(self, "cmd_kgs-rules", self._kgs_rules)
        self.known_commands.append("kgs-rules")
        setattr(self, "cmd_kgs-game_over", self._kgs_game_over)
        self.known_commands.append("kgs-game_over")
        if cleanup and scorer is not None:
            setattr(self, "cmd_kgs-genmove_cleanup", self._kgs_genmove_cleanup)
            self.known_commands.append("kgs-genmove_cleanup")

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
        """GNU Go's answer, or None if it isn't installed, errs or takes over 10 s."""
        return run_gnugo(sgf_file_name, command)

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
        allowed, border_open = self._pass_check(color)
        move = self._border_move(color) if border_open else None
        if move is None:
            move = self._game.get_move(color, pass_when_offered=allowed)
            if not self._game.make_move(color, move):
                raise ValueError("engine rejected its own move {}".format(
                    gtp.gtp_vertex(move)))
        return gtp.gtp_vertex(move)

    def _pass_check(self, color):
        """When the opponent has just passed and the player would pass back, may it? With a
        scorer, only if
        - the board is settled: at most SETTLED_MAX_CONTESTED points whose owner KataGo
          still sees as open. Otherwise the game would end on an unfinished board, judged
          as it stands (opponents passing right after move 100 got results off by up to ~90
          points that way); and
        - the player's border is closed: at most OPEN_MAX of its points that KataGo gives
          it but a count would give no one. Otherwise it closes the border first
          (_border_move) - opponents passing on its open border took up to 47 points.
        -> (pass_when_offered, border_open): None leaves the player's own rule, False makes
        it play on (asking again at the next pass); border_open asks for a border move."""
        if self._scorer is None or not self._game.opponent_just_passed():
            return None, False
        try:
            verdict = self._scorer.final_status(**self._game.position(color))
        except Exception as e:  # noqa: BLE001 - fall back to the player's own rule
            sys.stderr.write("gtp: KataGo pass check failed ({})\n".format(e))
            sys.stderr.flush()
            return None, False
        if verdict.get("contested", 0) > SETTLED_MAX_CONTESTED:
            return False, False
        # a server older than the border check sends no "open": the player's own rule
        own_open = verdict.get("open", {}).get("B" if color == gtp.BLACK else "W", [])
        if len(own_open) > OPEN_MAX and self._game.border_moves < BORDER_MOVES_MAX:
            return False, True
        return None, False

    def _border_move(self, color):
        """KataGo's move closing color's open border, played and returned; None (nothing
        played) if it has none, fails, or this engine rejects it - the caller then plays the
        player's own move, without passing."""
        try:
            vertex = self._scorer.border_move(**self._game.position(color))
        except Exception as e:  # noqa: BLE001 - e.g. a server older than border moves
            sys.stderr.write("gtp: KataGo border move failed ({})\n".format(e))
            sys.stderr.flush()
            return None
        move = gtp.parse_vertex(vertex) if vertex else None
        if move is None or not self._game.make_move(color, move):
            if vertex:
                sys.stderr.write("gtp: unusable KataGo border move {}\n".format(vertex))
                sys.stderr.flush()
            return None
        self._game.border_moves += 1
        return move

    def cmd_time_left(self, arguments):
        pass

    def _stopping(self):
        return self._stop_file is not None and os.path.exists(self._stop_file)

    def cmd_boardsize(self, arguments):
        # kgsGtp sends boardsize for every challenge: an error declines it
        if self._stopping():
            raise ValueError("not accepting games: {} exists".format(self._stop_file))
        if self._board_size is not None and arguments.strip() != str(self._board_size):
            raise ValueError("unacceptable size: this bot plays {0}x{0} only".format(
                self._board_size))
        return super(ExtendedGtpEngine, self).cmd_boardsize(arguments)

    def _kgs_game_over(self, arguments):
        if self._stopping():
            sys.stderr.write("gtp: game over and {} exists - exiting\n".format(self._stop_file))
            sys.stderr.flush()
            self.disconnect = True

    def cmd_undo(self, arguments):
        # Without undo, kgsGtp replays the game after a clear_board - which would also
        # restart the bot's sampling window (replayed moves aren't genmoves)
        if not self._game.undo():
            raise ValueError("cannot undo")

    def cmd_place_free_handicap(self, arguments):
        try:
            number_of_stones = int(arguments)
        except Exception:
            raise ValueError('Number of handicaps could not be parsed: {}'.format(arguments))
        if number_of_stones < 2 or number_of_stones > 9:
            raise ValueError('Invalid number of handicap stones: {}'.format(number_of_stones))
        if self.size != 19:
            raise ValueError('Free handicap placement is only known for 19x19')
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

    def _judged(self):
        """The scorer's verdict on the current position, or None (no scorer, or it failed)."""
        if self._scorer is None:
            return None
        try:
            return self._scorer.final_status(**self._game.position())
        except Exception as e:  # noqa: BLE001 - fall back to GNU Go
            sys.stderr.write("gtp: KataGo scoring failed ({}); trying GNU Go\n".format(e))
            sys.stderr.flush()
            return None

    def cmd_final_score(self, arguments):
        judged = self._judged()
        if judged is not None:
            lead = judged["score_lead"]
            if abs(lead) < 0.25:
                return "0"
            return "{}+{:.1f}".format("B" if lead > 0 else "W", abs(lead))
        answer = self._ask_gnugo_about_current_game('final_score\n')
        if answer is None:
            raise ValueError("cannot score: no KataGo, and GNU Go did not answer")
        return answer

    def cmd_final_status_list(self, arguments):
        status = arguments.strip().lower()
        if status not in ("dead", "alive", "seki"):
            raise ValueError("final_status_list takes dead, alive or seki")
        judged = self._judged()
        if judged is not None:
            if status == "seki":
                return ""
            dead = set(judged["dead"])
            if status == "dead":
                return " ".join(sorted(dead))
            return " ".join(sorted(v for _c, v in self._game.position()["stones"]
                                   if v not in dead))
        answer = self._ask_gnugo_about_current_game('final_status_list {}\n'.format(status))
        if answer is None:
            raise ValueError("cannot judge dead stones: no KataGo, and GNU Go did not answer")
        return answer

    def _kgs_rules(self, arguments):
        # "kgs-rules japanese" - KGS may add parameters after the rules in future
        words = arguments.split()
        self._game.set_rules(words[0].lower() if words else "chinese")

    def _kgs_genmove_cleanup(self, arguments):
        """Like genmove, but no pass while KataGo judges any of the opponent's stones dead:
        its move if it can be played here; otherwise (KataGo failed, or a move this engine
        rejects) the bot's own move, without passing just because the opponent did."""
        color = gtp.parse_color(arguments)
        if not color:
            raise ValueError("unknown player: {}".format(arguments))
        move = None
        try:
            vertex = self._scorer.cleanup_move(**self._game.position(color))
            move = gtp.PASS if vertex.lower() == "pass" else gtp.parse_vertex(vertex)
            if move is None or not self._game.make_move(color, move):
                sys.stderr.write("gtp: unusable KataGo cleanup move {}\n".format(vertex))
                move = None
        except Exception as e:  # noqa: BLE001 - fall back to the bot's own move
            sys.stderr.write("gtp: KataGo cleanup failed ({})\n".format(e))
        if move is None:
            move = self._game.get_move(color, pass_when_offered=False)
            if not self._game.make_move(color, move):
                raise ValueError("engine rejected its own move {}".format(gtp.gtp_vertex(move)))
        sys.stderr.flush()
        return gtp.gtp_vertex(move)


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
        # border moves played this game (ExtendedGtpEngine._border_move), for its cap
        self.border_moves = 0
        # komi and the rules (kgs-rules) - read only when judging the finished game
        self._komi = 7.5
        self._rules = "chinese"

    def clear(self):
        self._state = go.GameState(self._state.get_size(), enforce_superko=True)
        self._own_moves = {}
        self.border_moves = 0

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
        self.border_moves = 0

    def set_komi(self, k):
        self._komi = k

    def set_rules(self, rules):
        self._rules = rules

    def undo(self):
        """Take back the last move: the board is rebuilt from scratch - handicap stones, then
        every move but the last, in order - so captures, ko and superko come out exactly as
        when the game was played (GameState has no way to reverse a move). If the move was
        the bot's own (a color it was asked to genmove), its own-move count goes down too,
        keeping the sampling window at sample_moves bot moves per game. False if there is
        no move to take back."""
        handicaps = self._state.get_handicaps()
        history = self._state.get_history_with_colors()[len(handicaps):]
        if not history:
            return False
        state = go.GameState(self._state.get_size(), enforce_superko=True)
        if handicaps:
            state.place_handicaps(handicaps)
        for move, color in history[:-1]:
            state.record_move(move, color)
        self._state = state
        color = _GO_TO_GTP_COLOR[history[-1][1]]
        if self._own_moves.get(color, 0) > 0:
            self._own_moves[color] -= 1
        return True

    def opponent_just_passed(self):
        """The last move was a pass, after move 100 - when the player's pass_when_offered
        rule would pass back."""
        history = self._state.get_history()
        return len(history) > 100 and history[-1] == go.PASS

    def position(self, to_move=None):
        """The position for the scorer: {"stones": [[color, GTP vertex], ...], "to_move",
        "komi", "rules"} - to_move a GTP color, default the side to move."""
        size = self._state.get_size()
        board = self._state.get_board()
        stones = [["B" if board[x, y] == go.BLACK else "W",
                   gtp.gtp_vertex(_engine_to_gtp((x, y), size))]
                  for x in range(size) for y in range(size) if board[x, y] != go.EMPTY]
        if to_move is None:
            mover = "B" if self._state.get_current_player() == go.BLACK else "W"
        else:
            mover = "B" if to_move == gtp.BLACK else "W"
        return {"stones": stones, "to_move": mover, "komi": self._komi, "rules": self._rules}

    def get_move(self, color, pass_when_offered=None):
        """The player's move for color. pass_when_offered=False overrides the player's own
        "pass when the opponent just passed" for this move (cleanup must not pass early)."""
        self._state.set_current_player(_GTP_TO_GO_COLOR[color])
        saved = getattr(self._player, "pass_when_offered", None)
        if pass_when_offered is not None and saved is not None:
            self._player.pass_when_offered = pass_when_offered
        try:
            move = self._player.get_move(self._state, own_moves=self._own_moves.get(color, 0))
        finally:
            if pass_when_offered is not None and saved is not None:
                self._player.pass_when_offered = saved
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


def run_gtp(player_obj, inpt_fn=None, name="Gtp Player", version="0.0", scorer=None,
            cleanup=False, stop_file=None, board_size=None, log_path=None):
    """log_path: append the GTP exchange to this file (GtpLog); None logs nothing."""
    log = GtpLog(log_path)
    gtp_game = GTPGameConnector(player_obj)
    gtp_engine = ExtendedGtpEngine(gtp_game, name, version, scorer=scorer, cleanup=cleanup,
                                   stop_file=stop_file, board_size=board_size)
    if inpt_fn is None:
        inpt_fn = input

    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not gtp_engine.disconnect:
        try:
            inpt = inpt_fn()
        except EOFError:  # the controller closed our stdin: nothing more will come
            break
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        cmd_list = inpt.split("\n")
        for cmd in cmd_list:
            if cmd.strip():
                log.write(">", cmd)
            engine_reply = gtp_engine.send(cmd)
            if engine_reply:
                log.write("<", engine_reply)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()
