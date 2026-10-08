#!/usr/bin/env python
"""Ownership targets for a shard set: who owns each point at the end of each game,
according to KataGo.

    python -m AlphaGo.preprocessing.add_ownership_targets <shards_dir> \\
        --katago <katago.exe> --model <model.bin.gz> --config <analysis.cfg>

For every game in <shards_dir>/<split>/games.tsv, KataGo's analysis engine evaluates the
game's final position (all of its moves played) and reports the expected ownership of each
point. The result is <split>/ownership.bin: one row of board*board int8 per game, in
games.tsv's game_id order, round(127 * ownership) from BLACK's side (+127: surely Black's,
-127: surely White's). Points are in this project's order, x * size + y (as flatten_idx
and the feature planes), not KataGo's row-major y * size + x. <split>/ownership.json
records the shape, settings and a sanity check.

Every position of a game is trained toward that game's final ownership, as KataGo does.
A trainer turns it to the player to move's side with the value sidecars' black_to_move.

Standard library only (no numpy / h5py), so it runs wherever KataGo does - here, natively
on Windows. Queries are fed from a thread while answers are read: writing them all first
deadlocks once KataGo's answers fill its output pipe.
"""
import argparse
import array
import csv
import io
import json
import os
import re
import subprocess
import sys
import threading
import time

SPLITS = ("train", "val", "test")
GTP_COLS = "ABCDEFGHJKLMNOPQRST"
_RE_MOVE = re.compile(r";\s*([BW])\[([a-s]{0,2})\]")
_RE_SETUP = re.compile(r"A([BW])((?:\[[a-s]{2}\])+)")
_RE_RULES = re.compile(r"RU\[([^\]]*)\]")
_RE_KOMI = re.compile(r"KM\[\s*(-?[\d.]+)\s*\]")
_RE_SIZE = re.compile(r"SZ\[(\d+)\]")
_RE_RESULT = re.compile(r"RE\[([BW])\+")


def gtp_vertex(point, size):
    """SGF point 'ab' -> GTP vertex (column letter without I, row counted from the bottom).
    '' or 'tt' -> 'pass'."""
    if len(point) != 2 or point == "tt":
        return "pass"
    x, y = ord(point[0]) - 97, ord(point[1]) - 97
    return GTP_COLS[x] + str(size - y)


def katago_rules(text):
    """KataGo's own selfplay rules string (e.g. koSITUATIONALscoreTERRITORYtaxSEKIsui1)
    as an analysis-engine rules object, or 'chinese' for anything else."""
    m = _RE_RULES.search(text)
    rules = m.group(1) if m else ""
    parts = re.match(r"ko([A-Z]+)score([A-Z]+)tax([A-Z]+)sui([01])", rules)
    if not parts:
        return "chinese"
    return {"ko": parts.group(1), "scoring": parts.group(2), "tax": parts.group(3),
            "suicide": parts.group(4) == "1"}


def final_position_query(qid, text):
    """The analysis-engine query for a game's final position."""
    m = _RE_SIZE.search(text)
    size = int(m.group(1)) if m else 19
    root = text.split(";", 2)[1] if text.count(";") >= 1 else text
    setup = []
    for color, points in _RE_SETUP.findall(root):
        setup += [[color, gtp_vertex(p, size)] for p in re.findall(r"\[([a-s]{2})\]", points)]
    moves = [[c, gtp_vertex(p, size)] for c, p in _RE_MOVE.findall(text)]
    km = _RE_KOMI.search(text)
    return {"id": qid, "initialStones": setup, "moves": moves, "rules": katago_rules(text),
            "komi": float(km.group(1)) if km else 7.5, "boardXSize": size,
            "boardYSize": size, "analyzeTurns": [len(moves)], "includeOwnership": True}


def to_project_order(ownership, size):
    """KataGo's ownership list (row-major from the top-left, from Black's side - the run
    sets reportAnalysisWinratesAs = BLACK) -> int8 values in x * size + y order."""
    out = array.array("b", bytes(size * size))
    for k, v in enumerate(ownership):
        y, x = divmod(k, size)
        out[x * size + y] = max(-127, min(127, int(round(127 * v))))
    return out


def run_split(split_dir, repo_root, katago, model, config, visits, threads, log_dir,
              quiet=False):
    with io.open(os.path.join(split_dir, "games.tsv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    n_games = max(int(r["game_id"]) for r in rows) + 1
    size = 19  # this project's board; checked per game by the feeder
    results = {}  # game_id -> winner, filled as the feeder reads each game

    proc = subprocess.Popen(
        [katago, "analysis", "-config", config, "-model", model, "-override-config",
         "numAnalysisThreads={},numSearchThreadsPerAnalysisThread=1,"
         "reportAnalysisWinratesAs=BLACK,logDir={}".format(threads, log_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def feed():
        # Each SGF is read and sent as it goes: holding every game's query at once would
        # take several GB.
        for r in rows:
            with open(os.path.join(repo_root, r["path"]), encoding="utf-8",
                      errors="replace") as f:
                text = f.read()
            q = final_position_query(r["game_id"], text)
            if q["boardXSize"] != size:
                raise ValueError("{}: board size {}".format(r["path"], q["boardXSize"]))
            q["maxVisits"] = visits
            m = _RE_RESULT.search(text)
            results[r["game_id"]] = m.group(1) if m else None
            proc.stdin.write(json.dumps(q) + "\n")
        proc.stdin.close()
    threading.Thread(target=feed, daemon=True).start()

    table = array.array("b", bytes(n_games * size * size))
    done, agree, decided, t0 = 0, 0, 0, time.time()
    for line in proc.stdout:
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError("katago: {}".format(resp))
        gid = int(resp["id"])
        own = to_project_order(resp["ownership"], size)
        table[gid * size * size:(gid + 1) * size * size] = own
        # sanity: the side owning more of the board should mostly be the winner
        winner = results.get(resp["id"])
        if winner is not None:
            decided += 1
            black_area = sum(1 for v in own if v > 0) - sum(1 for v in own if v < 0)
            agree += (black_area > 0) == (winner == "B")
        done += 1
        if not quiet and done % 5000 == 0:
            print("  {}/{} games, {:.0f}/s".format(
                done, len(rows), done / (time.time() - t0)), flush=True)
    proc.wait()
    if done != len(rows):
        raise RuntimeError("KataGo answered {} of {} queries".format(done, len(rows)))
    with open(os.path.join(split_dir, "ownership.bin.part"), "wb") as f:
        table.tofile(f)
    os.replace(os.path.join(split_dir, "ownership.bin.part"),
               os.path.join(split_dir, "ownership.bin"))
    meta = {"games": n_games, "points": size * size, "dtype": "int8", "scale": 127,
            "side": "black", "order": "x * size + y", "visits": visits,
            "seconds": round(time.time() - t0),
            "winner_owns_more_share": agree / decided if decided else None}
    with open(os.path.join(split_dir, "ownership.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description="KataGo final-position ownership per game.")
    p.add_argument("shards_dir")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--katago", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True, help="An analysis-engine config")
    p.add_argument("--visits", type=int, default=10,
                   help="Search visits per final position. Default: 10 (a finished game's "
                        "ownership needs little search)")
    p.add_argument("--threads", type=int, default=32, help="Positions analyzed in parallel")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    log_dir = os.path.join(args.shards_dir, "katago_logs")
    for split in args.splits:
        meta = run_split(os.path.join(args.shards_dir, split), args.repo_root, args.katago,
                         args.model, args.config, args.visits, args.threads, log_dir,
                         args.quiet)
        print("{}: {} games in {}s; winner owns more of the board in {:.1%}".format(
            split, meta["games"], meta["seconds"], meta["winner_owns_more_share"] or 0),
            flush=True)


if __name__ == "__main__":
    sys.exit(main())
