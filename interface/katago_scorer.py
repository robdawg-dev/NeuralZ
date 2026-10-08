"""End-of-game judgement for the bots from one shared KataGo analysis engine (go_server.py
--katago): which stones are dead (GTP final_status_list / final_score), and - when a bot
plays the cleanup phase - a move that removes them (kgs-genmove_cleanup).

Only the current position is judged (the stones, the side to move, komi and rules), not
the move history: dead stones don't depend on it, and KGS games can hold moves KataGo's
rules reject. A stone is dead if KataGo's ownership gives its point to the other color.
Measured on 263 scored KGS games: the small b10c128 network at 1 visit agrees with a large
network at 400 visits on 258 final positions, at ~30 ms per query on a CPU - while GNU Go's
list had cost the bot a ranked game it had won (MEASUREMENTS.md, "End of game").

Standard library only. Positions and moves are GTP vertices ("D4", "pass").
"""
import collections
import itertools
import json
import os
import subprocess
import sys
import threading
import time

GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"  # GTP skips I
# KGS's kgs-rules names -> KataGo's
RULES = {"chinese": "chinese", "japanese": "japanese", "aga": "aga",
         "new_zealand": "new-zealand", "tromp-taylor": "tromp-taylor"}


def vertex_index(vertex, size):
    """A GTP vertex's index in KataGo's ownership array (row-major from the top-left)."""
    x = GTP_COLUMNS.index(vertex[0].upper())
    row = int(vertex[1:])
    return (size - row) * size + x


class KataGoScorer(object):
    """One KataGo analysis engine process answering queries from any number of threads.

    command: the full command line (default: <exe> analysis -config <config> -model <model>).
    visits: per final_status query - 1 (the network alone) is enough on finished positions.
    cleanup_visits: per cleanup move, which needs a real move choice.
    restart_wait: if KataGo has exited, the next query restarts it (waiting for its network
    to load) - but at most once per this many seconds, so one that keeps failing is not
    respawned on every call. A query in flight when it exits fails; the next one restarts.
    """

    def __init__(self, exe=None, model=None, config=None, command=None, visits=1,
                 cleanup_visits=32, timeout=10.0, board_size=19, restart_wait=30.0):
        # ownership and score from Black's side, whatever the config file says
        self.command = command or [exe, "analysis", "-config", config, "-model", model,
                                   "-override-config", "reportAnalysisWinratesAs=BLACK"]
        self.visits, self.cleanup_visits = visits, cleanup_visits
        self.timeout, self.size = timeout, board_size
        self.restart_wait = restart_wait
        self._ids = itertools.count()
        self._lock = threading.Lock()  # the current process, its stdin and pending queries
        self._restart_lock = threading.Lock()
        self._closed = False
        # KataGo's last stderr lines and warnings, quoted in errors: why it failed or exited
        self._stderr = collections.deque(maxlen=20)
        self._start()
        self._warm_up()  # a failure here is a startup error

    def _start(self):
        # KataGo's Linux releases are AppImages, which need FUSE unless told to unpack
        # themselves first (servers and containers often lack FUSE; other builds ignore it)
        env = dict(os.environ, APPIMAGE_EXTRACT_AND_RUN="1")
        proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        # Each process has its own pending queries: when an old one's reader sees it exit
        # and wakes its waiters, a restarted process's queries are not among them.
        pending = {}
        stderr_reader = threading.Thread(target=self._read_stderr, args=(proc,),
                                         name="katago-stderr", daemon=True)
        with self._lock:
            self.proc, self._pending, self._stderr_reader = proc, pending, stderr_reader
            self._started = time.monotonic()
        threading.Thread(target=self._read, args=(proc, pending), name="katago-reader",
                         daemon=True).start()
        stderr_reader.start()

    def _warm_up(self):
        """The first query, which waits for the network to load."""
        self._ask(self._position([], "B", 7.5, "chinese", self.visits),
                  max(self.timeout, 120.0))

    def _ensure_running(self):
        """Restart KataGo if it has exited - unless it was (re)started under restart_wait
        seconds ago - and wait for it to load. Other queries wait meanwhile."""
        if self.alive():
            return
        with self._restart_lock:
            if self.alive():
                return
            if self._closed or time.monotonic() - self._started < self.restart_wait:
                raise RuntimeError(self._with_stderr(
                    "KataGo is not running (exit code {})".format(self.proc.poll())))
            sys.stderr.write("katago_scorer: KataGo exited (code {}); restarting\n".format(
                self.proc.poll()))
            sys.stderr.flush()
            self._start()
            self._warm_up()

    def _read_stderr(self, proc):
        # also keeps the pipe drained, so a chatty KataGo never blocks on a full buffer
        for line in proc.stderr:
            if line.strip():
                self._stderr.append(line.rstrip())

    def _with_stderr(self, message):
        if not self.alive():
            self._stderr_reader.join(timeout=1.0)  # let it read KataGo's final words
        lines = list(self._stderr)
        if not lines:
            return message
        return "{}; KataGo's last output:\n  {}".format(message, "\n  ".join(lines[-5:]))

    def _read(self, proc, pending):
        for line in proc.stdout:
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            if "warning" in reply and "error" not in reply:
                # about the query (e.g. a field KataGo ignores), sent before its answer
                # under the same id: note it, and keep waiting for the answer
                warning = "warning: {} ({})".format(reply["warning"], reply.get("field", ""))
                self._stderr.append(warning)
                sys.stderr.write("katago_scorer: KataGo {}\n".format(warning))
                sys.stderr.flush()
                continue
            with self._lock:
                waiter = pending.pop(reply.get("id"), None)
            if waiter is not None:
                waiter["reply"] = reply
                waiter["done"].set()
        with self._lock:  # KataGo exited: wake everyone still waiting on it
            for waiter in pending.values():
                waiter["done"].set()
            pending.clear()

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        self._closed = True
        if self.alive():
            self.proc.terminate()

    def _query(self, query, timeout=None):
        self._ensure_running()
        return self._ask(query, self.timeout if timeout is None else timeout)

    def _ask(self, query, wait):
        """Send one query to the current process and wait up to wait seconds for its answer."""
        qid = str(next(self._ids))
        waiter = {"done": threading.Event(), "reply": None}
        with self._lock:
            proc, pending = self.proc, self._pending
            pending[qid] = waiter
            try:
                proc.stdin.write(json.dumps(dict(query, id=qid)) + "\n")
                proc.stdin.flush()
            except OSError:  # it has exited
                pending.pop(qid, None)
                raise RuntimeError(self._with_stderr("KataGo exited"))
        if not waiter["done"].wait(wait):
            with self._lock:
                pending.pop(qid, None)
            raise RuntimeError(self._with_stderr(
                "KataGo did not answer within {:.0f} s".format(wait)))
        reply = waiter["reply"]
        if reply is None:
            try:  # its stdout closes a moment before the process is gone
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError(self._with_stderr("KataGo exited"))
        if "error" in reply:
            raise ValueError("KataGo: {}".format(reply["error"]))
        return reply

    def _position(self, stones, to_move, komi, rules, visits):
        return {"initialStones": [[c.upper(), v.upper()] for c, v in stones], "moves": [],
                "initialPlayer": to_move.upper(), "komi": float(komi),
                "rules": RULES.get(rules, "chinese"), "boardXSize": self.size,
                "boardYSize": self.size, "maxVisits": visits, "includeOwnership": True}

    def final_status(self, stones, to_move, komi, rules, timeout=None):
        """stones: [[color, vertex], ...] on the board. -> {"dead": [vertex, ...],
        "score_lead": Black's estimated lead, "contested": points whose owner is still open
        (0.3 <= |ownership| < 0.9) - 0-2 on finished boards, 12-275 in mid-game (212 KGS
        games, MEASUREMENTS.md); dame and seki read near 0, so they count as settled}."""
        reply = self._query(self._position(stones, to_move, komi, rules, self.visits), timeout)
        own = reply["ownership"]  # + = Black's point
        dead = [v.upper() for c, v in stones
                if (own[vertex_index(v, self.size)] < 0) == (c.upper() == "B")]
        contested = sum(1 for o in own if 0.3 <= abs(o) < 0.9)
        return {"dead": dead, "score_lead": reply["rootInfo"]["scoreLead"],
                "contested": contested}

    def cleanup_move(self, stones, color, komi, rules):
        """For color's kgs-genmove_cleanup: "pass" once none of the opponent's stones are
        dead, else KataGo's best move other than a pass."""
        color = color.upper()
        dead = self.final_status(stones, color, komi, rules)["dead"]
        board = {v.upper(): c.upper() for c, v in stones}
        if not any(board.get(v) != color for v in dead):
            return "pass"
        query = self._position(stones, color, komi, rules, self.cleanup_visits)
        query["avoidMoves"] = [{"player": color, "moves": ["pass"], "untilDepth": 1}]
        infos = self._query(query).get("moveInfos") or []
        moves = [m["move"] for m in sorted(infos, key=lambda m: m.get("order", 0))
                 if m["move"].lower() != "pass"]
        return moves[0] if moves else "pass"
