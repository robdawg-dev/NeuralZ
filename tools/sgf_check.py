"""Check SGF game records: is every move legal, does the record end cleanly, and - with
KataGo - how well was it played (points each side lost per move) and does the final
position agree with the recorded result. For records of unknown origin, such as games
copied from the web or written by an LLM: invented records give themselves away with
illegal moves, or with moves losing far more than real players' (~0.2-1 point a move for
professionals and strong amateurs).

    uv run python tools/sgf_check.py <files or folders of .sgf/.txt> [--visits 50]

Without KataGo only the legality checks run. The main line is checked (the first branch
at each variation). Komi, rules (Japanese -> Japanese, Korean -> Korean, else Chinese) and
AB/AW setup stones come from the record. KataGo:
--katago/--katago-model/--katago-config or KATAGO_EXE/KATAGO_MODEL/KATAGO_CONFIG.
"""
import argparse
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_report import parse_tree, played_line, prop  # noqa: E402
from katago_util import COLS, add_katago_args, gtp, katago_command, run  # noqa: E402


def neighbors(p):
    x, y = p
    return [(a, b) for a, b in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if 0 <= a < 19 and 0 <= b < 19]


def group(board, p):
    color, grp, libs, stack = board[p], {p}, set(), [p]
    while stack:
        for q in neighbors(stack.pop()):
            if q not in board:
                libs.add(q)
            elif board[q] == color and q not in grp:
                grp.add(q)
                stack.append(q)
    return grp, libs


def play(board, color, p, ko):
    """Play color at p on board (a {point: color} dict), applying captures. -> (problem
    or None, the next ko point). The board is unchanged when there is a problem."""
    if p in board:
        return "occupied point", ko
    if p == ko:
        return "ko violation", ko
    board[p] = color
    captured = set()
    for q in neighbors(p):
        if q in board and board[q] != color:
            grp, libs = group(board, q)
            if not libs:
                captured |= grp
    for q in captured:
        del board[q]
    grp, libs = group(board, p)
    if not libs:
        del board[p]
        return "suicide", ko
    new_ko = next(iter(captured)) if len(captured) == 1 and len(grp) == 1 and len(libs) == 1 \
        else None
    return None, new_ko


def check(text):
    """Legality of one record's main line. -> dict: moves, setup, the problems found, how
    many moves were out of turn, whether the record looks cut off, and the moves up to the
    first illegal one (what KataGo can analyze)."""
    root = parse_tree(text)
    line = played_line(root) if root else []
    setup = []
    for key, color in (("AB", "B"), ("AW", "W")):
        m = re.search(key + r"((?:\s*\[[a-s]{2}\])+)", root["props"] if root else "")
        setup += [[color, p] for p in re.findall(r"\[([a-s]{2})\]", m.group(1))] if m else []
    board, ko, problems, legal_until = {}, None, [], None
    for c, p in setup:
        board[(COLS.index(p[0]), COLS.index(p[1]))] = c
    has_ab = any(c == "B" for c, _p in setup) and not any(c == "W" for c, _p in setup)
    expect = "W" if (prop(text, "HA") and int(prop(text, "HA") or 0) > 1) or has_ab else "B"
    out_of_turn = 0
    for i, (c, p) in enumerate(line):
        out_of_turn += c != expect
        expect = "W" if c == "B" else "B"
        if not p:
            ko = None
            continue
        pt = (COLS.index(p[0]), COLS.index(p[1]))
        problem, ko = play(board, c, pt, ko)
        if problem:
            problems.append("move {} {}{} {}".format(i + 1, c, gtp(p), problem))
            if legal_until is None:
                legal_until = i
    tail = text.rstrip()
    cut_off = not tail.endswith(")") or bool(re.search(r"[BW]\[[a-s]?$", tail.rstrip(")")))
    return {"line": line, "setup": setup, "problems": problems, "out_of_turn": out_of_turn,
            "cut_off": cut_off,
            "legal": line[:legal_until] if legal_until is not None else line}


def find(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.sgf"), recursive=True) +
                            glob.glob(os.path.join(p, "**", "*.txt"), recursive=True))
        else:
            files.append(p)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--visits", type=int, default=50, help="per position. Default: 50")
    add_katago_args(parser)
    args = parser.parse_args(argv)
    command = katago_command(args)
    records = []
    for path in find(args.paths):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        records.append((path, text, check(text)))
    replies = {}
    if command:
        queries = []
        for k, (path, text, r) in enumerate(records):
            ru = prop(text, "RU").lower()
            queries.append({"id": str(k), "initialStones": [[c, gtp(p)] for c, p in r["setup"]],
                            "moves": [[c, gtp(p)] for c, p in r["legal"]],
                            "rules": "japanese" if ru.startswith("jap") else
                            "korean" if ru.startswith("kor") else "chinese",
                            "komi": float(prop(text, "KM") or 0), "boardXSize": 19,
                            "boardYSize": 19, "maxVisits": args.visits,
                            "analyzeTurns": list(range(len(r["legal"]) + 1))})
        replies = run(command, queries, "positions")
    for k, (path, text, r) in enumerate(records):
        print("\n== {} | {} (B) vs {} (W) | {} | RE[{}]".format(
            os.path.basename(path), prop(text, "PB") or "?", prop(text, "PW") or "?",
            prop(text, "DT") or "no date", prop(text, "RE") or "?"))
        print("   {} moves{}{} | out-of-turn moves: {} | illegal moves: {}{}".format(
            len(r["line"]), " | setup stones {}".format(len(r["setup"])) if r["setup"] else "",
            " | record cut off" if r["cut_off"] else "", r["out_of_turn"], len(r["problems"]),
            " - first: " + "; ".join(r["problems"][:3]) if r["problems"] else ""))
        if not command:
            continue
        leads = [replies.get((str(k), t)) for t in range(len(r["legal"]) + 1)]
        leads = [x["rootInfo"]["scoreLead"] if x else None for x in leads]
        loss = {"B": [], "W": []}
        for i, (c, _p) in enumerate(r["legal"]):
            a, b = leads[i], leads[i + 1]
            if a is not None and b is not None:
                loss[c].append(max((a - b) if c == "B" else (b - a), 0))
        for c in "BW":
            if loss[c]:
                print("   {} points lost per move {:.2f} (median {:.2f}), moves losing 10+: "
                      "{}".format(c, statistics.mean(loss[c]), statistics.median(loss[c]),
                                  sum(x >= 10 for x in loss[c])))
        if leads[-1] is not None:
            print("   KataGo at move {}{}: {}+{:.1f}".format(
                len(r["legal"]), " (last legal move)" if r["problems"] else "",
                "B" if leads[-1] > 0 else "W", abs(leads[-1])))


if __name__ == "__main__":
    main()
