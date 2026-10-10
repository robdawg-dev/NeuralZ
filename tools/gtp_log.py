"""Read a bot's GTP log (go_client.py --gtp-log): the games in it, how long the bot took to
answer, and - with KataGo - whether its dead-stone lists and cleanup moves were right.

    uv run python tools/gtp_log.py games  logs/NeuralZ05.log
    uv run python tools/gtp_log.py timing logs/NeuralZ05.log [--game 3]
    uv run python tools/gtp_log.py replay logs/NeuralZ05.log --game 3 [--visits 200]

Games are numbered from 1 in log order (each starts at a clear_board). The played line is
rebuilt from the controller's commands: set_free_handicap / place_free_handicap, play,
genmove and kgs-genmove_cleanup (their replies), and undo. "replay" judges the end of the
game: KataGo's dead stones against each final_status_list answer, and through any cleanup
phase which opponent stones were still dead when the bot moved. KataGo:
--katago/--katago-model/--katago-config or KATAGO_EXE/KATAGO_MODEL/KATAGO_CONFIG.
"""
import argparse
import ast
import collections
import datetime
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from katago_util import GTP_COLS, add_katago_args, katago_command, run  # noqa: E402

LINE = re.compile(r"(\S+) pid=(\d+) ([<>]) (.*)$")


def exchanges(lines):
    """[(pid, command, reply, sent time, answer time), ...] from log lines; a command whose
    reply is missing (the log ends) gets reply None."""
    out, pending = [], {}
    for line in lines:
        m = LINE.match(line.rstrip("\n"))
        if not m:
            continue
        t, pid, direction = datetime.datetime.fromisoformat(m.group(1)), m.group(2), m.group(3)
        text = ast.literal_eval(m.group(4))
        if direction == ">":
            if pid in pending:
                out.append(pending.pop(pid) + (None, None))
            pending[pid] = (pid, text, t)
        elif pid in pending:
            p, cmd, sent = pending.pop(pid)
            out.append((p, cmd, text, sent, t))
    for p, cmd, sent in pending.values():
        out.append((p, cmd, None, sent, None))
    return out


def _reply_value(reply):
    return (reply or "").lstrip("=").strip()


def games(xs):
    """The games in the exchanges, each a dict: start time, komi, rules, moves ([[color,
    vertex or 'pass'], ...], the played line), bot moves (with timings), dead-stone answers,
    whether cleanup ran, and whether it ended with kgs-game_over."""
    out, g = [], None
    komi, rules = 7.5, "chinese"
    for pid, cmd, reply, sent, answered in xs:
        words = cmd.split()
        if not words:
            continue
        name = words[0]
        if name == "komi" and len(words) > 1:
            komi = float(words[1])  # KGS sends it after clear_board, so set the game's too
            if g is not None:
                g["komi"] = komi
        elif name == "kgs-rules" and len(words) > 1:
            rules = words[1].lower()
            if g is not None:
                g["rules"] = rules
        elif name == "clear_board":
            g = {"start": sent, "komi": komi, "rules": rules, "moves": [], "bot": [],
                 "final_status": [], "cleanup": False, "over": False, "pid": pid}
            out.append(g)
        elif g is None:
            continue
        elif name == "set_free_handicap":
            g["moves"] += [["B", v.upper()] for v in words[1:]]
        elif name == "place_free_handicap":
            g["moves"] += [["B", v.upper()] for v in _reply_value(reply).split()]
        elif name == "play" and len(words) > 2:
            g["moves"].append([words[1][0].upper(), words[2].upper()])
        elif name in ("genmove", "kgs-genmove_cleanup") and len(words) > 1 and reply:
            vertex = _reply_value(reply).upper()
            g["moves"].append([words[1][0].upper(), vertex])
            g["cleanup"] = g["cleanup"] or name == "kgs-genmove_cleanup"
            ms = (answered - sent).total_seconds() * 1000 if answered else None
            g["bot"].append({"command": name, "move_number": len(g["moves"]),
                             "vertex": vertex, "ms": ms})
        elif name == "undo" and g["moves"]:
            g["moves"].pop()
        elif name == "final_status_list":
            g["final_status"].append({"after_move": len(g["moves"]),
                                      "dead": _reply_value(reply).upper().split()})
        elif name == "kgs-game_over":
            g["over"] = True
    for g in out:
        g["moves"] = [[c, "pass" if v == "PASS" else v] for c, v in g["moves"]]
    return out


def board(moves, size=19):
    """{(x, y): color} after the moves, captures applied (x = column, y = row from top)."""
    stones = {}

    def group(p):
        color, grp, libs, stack = stones[p], {p}, False, [p]
        while stack:
            a, b = stack.pop()
            for q in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if 0 <= q[0] < size and 0 <= q[1] < size:
                    if q not in stones:
                        libs = True
                    elif stones[q] == color and q not in grp:
                        grp.add(q)
                        stack.append(q)
        return grp, libs
    for c, v in moves:
        if v == "pass":
            continue
        p = (GTP_COLS.index(v[0]), size - int(v[1:]))
        stones[p] = c
        for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)):
            if q in stones and stones[q] != c:
                grp, libs = group(q)
                if not libs:
                    for s in grp:
                        del stones[s]
        grp, libs = group(p)
        if not libs:  # suicide (illegal in the rule sets KGS uses, but stay consistent)
            for s in grp:
                del stones[s]
    return stones


def kata_dead(moves, reply, size=19):
    """KataGo's dead stones (vertex+color) for the position after the moves."""
    own = reply["ownership"]
    return sorted("{}{}{}".format(GTP_COLS[x], size - y, c)
                  for (x, y), c in board(moves, size).items()
                  if (own[y * size + x] < 0) == (c == "B") and abs(own[y * size + x]) > 0.3)


def print_games(gs):
    for i, g in enumerate(gs, 1):
        bot_color = g["bot"][0]["command"] and g["moves"][g["bot"][0]["move_number"] - 1][0] \
            if g["bot"] else "?"
        print("{:>3}  {}  bot {}  {} moves  komi {}  {}{}{}".format(
            i, g["start"].strftime("%Y-%m-%d %H:%M"), bot_color, len(g["moves"]), g["komi"],
            g["rules"], "  cleanup" if g["cleanup"] else "",
            "" if g["over"] else "  (no kgs-game_over: unfinished or log cut)"))


def print_timing(gs):
    by = collections.defaultdict(list)
    for g in gs:
        for b in g["bot"]:
            if b["ms"] is not None:
                kind = b["command"] if b["vertex"] != "PASS" else b["command"] + " (pass)"
                by[kind].append(b["ms"])
    for kind, ms in sorted(by.items()):
        ms = sorted(ms)
        print("{:<28} {:>5} moves  median {:>6.0f} ms  p90 {:>6.0f} ms  max {:>6.0f} ms".format(
            kind, len(ms), statistics.median(ms), ms[int(0.9 * (len(ms) - 1))], ms[-1]))
    plain = [(b["move_number"], b["ms"]) for g in gs for b in g["bot"]
             if b["command"] == "genmove" and b["ms"] is not None]
    if plain:
        print("\ngenmove by game stage:")
        for lo in range(0, max(n for n, _ms in plain) + 1, 50):
            chunk = [ms for n, ms in plain if lo < n <= lo + 50]
            if chunk:
                print("  moves {:>3}-{:<3} {:>4} moves, mean {:>5.0f} ms".format(
                    lo + 1, lo + 50, len(chunk), statistics.mean(chunk)))


def summary(paths, slow_ms=5000.0):
    """One line per log (a bot): games and how many ended normally or with a cleanup phase;
    GTP errors (replies starting '?'), with the commands that drew them; genmove speed and
    slow moves; commands the bot never answered (it stopped or crashed mid-command)."""
    lines = ["{:<24} {:>5} {:>5} {:>7} {:>6} {:>11} {:>9} {:>6} {:>10}".format(
        "log", "games", "ended", "cleanup", "errors", "median ms", "max ms", "slow", "unanswered")]
    notes = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            xs = exchanges(f)
        gs = games(xs)
        errors = collections.Counter(cmd.split()[0] for _p, cmd, reply, _s, _a in xs
                                     if reply is not None and reply.startswith("?"))
        unanswered = [cmd for _p, cmd, reply, _s, _a in xs if reply is None]
        ms = sorted(b["ms"] for g in gs for b in g["bot"]
                    if b["command"] == "genmove" and b["ms"] is not None)
        name = os.path.basename(path)
        lines.append("{:<24} {:>5} {:>5} {:>7} {:>6} {:>11} {:>9} {:>6} {:>10}".format(
            name[:24], len(gs), sum(g["over"] for g in gs), sum(g["cleanup"] for g in gs),
            sum(errors.values()), "{:.0f}".format(statistics.median(ms)) if ms else "-",
            "{:.0f}".format(ms[-1]) if ms else "-", sum(m > slow_ms for m in ms),
            len(unanswered)))
        if errors:
            notes.append("{}: GTP errors from {}".format(name, dict(errors)))
        if unanswered:
            notes.append("{}: no reply to {!r} (the log ends there)".format(name, unanswered[-1]))
    return lines + ([""] + notes if notes else [])


def replay(g, command, visits):
    moves = g["moves"]
    marks = sorted({f["after_move"] for f in g["final_status"]} |
                   {b["move_number"] - 2 for b in g["bot"]
                    if b["command"] == "kgs-genmove_cleanup"} |
                   {len(moves)})
    q = {"id": "g", "moves": moves, "rules": "japanese" if g["rules"] == "japanese" else "chinese",
         "komi": g["komi"], "boardXSize": 19, "boardYSize": 19, "maxVisits": visits,
         "includeOwnership": True, "analyzeTurns": [t for t in marks if 0 <= t <= len(moves)]}
    replies = {turn: r for (_id, turn), r in run(command, [q]).items()}
    print("{} moves, komi {}, {} rules{}".format(len(moves), g["komi"], g["rules"],
                                                 ", cleanup phase" if g["cleanup"] else ""))
    for f in g["final_status"]:
        t = f["after_move"]
        kd = kata_dead(moves[:t], replies[t]) if t in replies else ["?"]
        agree = sorted(v[:-1] for v in kd) == sorted(f["dead"])
        print("  after move {:>3}: bot's dead list {} | KataGo {}  {}".format(
            t, " ".join(f["dead"]) or "(none)", " ".join(kd) or "(none)",
            "agree" if agree else "<- differ"))
    for b in g["bot"]:
        if b["command"] != "kgs-genmove_cleanup":
            continue
        t = b["move_number"] - 2
        opp = moves[t] if t < len(moves) else ["?", "?"]
        kd = kata_dead(moves[:t], replies[t]) if t in replies else ["?"]
        print("  cleanup move {:>3}: opponent {} {:<5} bot {:<5} | dead before the opponent's "
              "move: {}".format(b["move_number"], opp[0], opp[1], b["vertex"],
                                " ".join(kd) or "(none)"))
    end = replies.get(len(moves))
    if end:
        print("final position: KataGo dead {} | Black's lead {:+.1f}".format(
            " ".join(kata_dead(moves, end)) or "(none)", end["rootInfo"]["scoreLead"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("games", "list the games in the log"),
                       ("timing", "the bot's answer times"),
                       ("replay", "judge one game's end with KataGo")):
        p = sub.add_parser(name, help=text)
        p.add_argument("log")
        p.add_argument("--game", type=int, help="game number (from 'games')" +
                       (" - required" if name == "replay" else ""))
        if name == "replay":
            p.add_argument("--visits", type=int, default=200)
            add_katago_args(p, required=True)
            replay_parser = p
    s = sub.add_parser("summary", help="one line per bot log: games, problems, speed")
    s.add_argument("logs", nargs="+")
    s.add_argument("--slow-ms", type=float, default=5000,
                   help="a genmove slower than this counts as slow. Default: 5000")
    args = parser.parse_args(argv)
    if args.command == "summary":
        for line in summary(args.logs, args.slow_ms):
            print(line)
        return
    with open(args.log, encoding="utf-8") as f:
        gs = games(exchanges(f))
    if not gs:
        sys.exit("gtp_log: no games (clear_board) in {}".format(args.log))
    if args.game is not None and not 1 <= args.game <= len(gs):
        sys.exit("gtp_log: --game must be 1-{}".format(len(gs)))
    chosen = [gs[args.game - 1]] if args.game else gs
    if args.command == "games":
        print_games(gs)
    elif args.command == "timing":
        print_timing(chosen)
    else:
        if args.game is None:
            replay_parser.error("--game is required")
        replay(chosen[0], katago_command(args, replay_parser), args.visits)


if __name__ == "__main__":
    main()
