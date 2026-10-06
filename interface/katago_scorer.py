"""End-of-game judgement for the bots from one shared KataGo analysis engine (go_server.py
--katago): which stones are dead (GTP final_status_list / final_score), and - when a bot
plays the cleanup phase - a move that removes them (kgs-genmove_cleanup).

Only the current position is judged (the stones, the side to move, komi and rules), not
the move history: dead stones don't depend on it, and KGS games can hold moves KataGo's
rules reject. A stone is dead if KataGo's ownership gives its point to the other color.
Measured on 263 scored KGS games (workspace/kgs_scoring/): the small b10c128 network at 1
visit agrees with a large network at 400 visits on 258 final positions, at ~30 ms per query
on a CPU - while GNU Go's list had cost the bot a ranked game it had won.

Standard library only. Positions and moves are GTP vertices ("D4", "pass").
"""
import itertools
import json
import os
import subprocess
import threading

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
    """

    def __init__(self, exe=None, model=None, config=None, command=None, visits=1,
                 cleanup_visits=32, timeout=10.0, board_size=19):
        # ownership and score from Black's side, whatever the config file says
        self.command = command or [exe, "analysis", "-config", config, "-model", model,
                                   "-override-config", "reportAnalysisWinratesAs=BLACK"]
        self.visits, self.cleanup_visits = visits, cleanup_visits
        self.timeout, self.size = timeout, board_size
        self._ids = itertools.count()
        self._pending = {}
        self._lock = threading.Lock()
        self._start()
        # the first query waits for the network to load; a failure here is a startup error
        self.final_status([], "B", 7.5, "chinese", timeout=max(timeout, 120.0))

    def _start(self):
        # KataGo's Linux releases are AppImages, which need FUSE unless told to unpack
        # themselves first (servers and containers often lack FUSE; other builds ignore it)
        env = dict(os.environ, APPIMAGE_EXTRACT_AND_RUN="1")
        self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        threading.Thread(target=self._read, name="katago-reader", daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                waiter = self._pending.pop(reply.get("id"), None)
            if waiter is not None:
                waiter["reply"] = reply
                waiter["done"].set()
        with self._lock:  # KataGo exited: wake everyone still waiting
            for waiter in self._pending.values():
                waiter["done"].set()
            self._pending.clear()

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        if self.alive():
            self.proc.terminate()

    def _query(self, query, timeout=None):
        if not self.alive():
            raise RuntimeError("KataGo is not running (exit code {})".format(self.proc.poll()))
        qid = str(next(self._ids))
        waiter = {"done": threading.Event(), "reply": None}
        with self._lock:
            self._pending[qid] = waiter
            self.proc.stdin.write(json.dumps(dict(query, id=qid)) + "\n")
            self.proc.stdin.flush()
        if not waiter["done"].wait(self.timeout if timeout is None else timeout):
            with self._lock:
                self._pending.pop(qid, None)
            raise RuntimeError("KataGo did not answer within {:.0f} s".format(self.timeout))
        reply = waiter["reply"]
        if reply is None:
            raise RuntimeError("KataGo exited")
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
        (0.3 <= |ownership| < 0.9) - 0-2 on finished boards, 12-275 in mid-game (measured on
        212 KGS games, workspace/samples2/contested_dist.py); dame and seki read near 0, so
        they count as settled}."""
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
