#!/usr/bin/env python
"""KataGo's raw-network judgment of every sibling position written by
build_sibling_positions.py (SCORE_NET_PLAN.md).

    python -m AlphaGo.preprocessing.label_sibling_positions <out_dir> --split train \\
        --katago <katago.exe> --model <model.bin.gz> --config <analysis.cfg>

For each row of <out_dir>/<split>/queries.tsv: the source game's moves up to the source
node, plus the candidate move, evaluated by KataGo's analysis engine at --visits (default
1: the raw network, no search - what the score head is to imitate). Writes
<out_dir>/<split>/labels.bin: float32 pairs (score lead, win rate) per row, in queries.tsv
order, **for the sibling's player to move**, and labels.json (row count, settings).

Standard library only (runs natively next to KataGo). Queries are fed from a thread while
answers are read: writing them all first deadlocks once KataGo's output pipe fills.
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

from AlphaGo.preprocessing.add_ownership_targets import gtp_vertex, katago_rules

_RE_MOVE = re.compile(r";\s*([BW])\[([a-s]{0,2})\]")
_RE_SETUP = re.compile(r"A([BW])((?:\[[a-s]{2}\])+)")
_RE_KOMI = re.compile(r"KM\[\s*(-?[\d.]+)\s*\]")


def game_setup(text, size=19):
    """(base query fields, [move nodes as [color, vertex]]) for one SGF."""
    root = text.split(";", 2)[1] if text.count(";") >= 1 else text
    setup = []
    for color, points in _RE_SETUP.findall(root):
        setup += [[color, gtp_vertex(p, size)] for p in re.findall(r"\[([a-s]{2})\]", points)]
    km = _RE_KOMI.search(text)
    base = {"initialStones": setup, "rules": katago_rules(text),
            "komi": float(km.group(1)) if km else 7.5, "boardXSize": size, "boardYSize": size}
    return base, [[c, gtp_vertex(p, size)] for c, p in _RE_MOVE.findall(text)]


def sibling_query(qid, base, moves, node, move, black_to_move, visits):
    """The query for one sibling: the source's moves (node move nodes), then the candidate
    move by the source's mover (the sibling's player to move is the other color)."""
    mover = "W" if black_to_move else "B"
    seq = moves[:node] + [[mover, gtp_vertex(move, base["boardXSize"])]]
    return dict(base, id=qid, moves=seq, analyzeTurns=[len(seq)], maxVisits=visits)


def to_player_to_move(score_black, winrate_black, black_to_move):
    """KataGo's Black-side lead and win rate -> the sibling's player to move's."""
    if black_to_move:
        return score_black, winrate_black
    return -score_black, 1.0 - winrate_black


def label_split(split_dir, repo_root, katago, model, config, visits, threads, log_dir,
                quiet=False):
    with io.open(os.path.join(split_dir, "queries.tsv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    n = len(rows)
    labels = array.array("f", bytes(8 * n))
    black = [r["black_to_move"] == "1" for r in rows]

    proc = subprocess.Popen(
        [katago, "analysis", "-config", config, "-model", model, "-override-config",
         "numAnalysisThreads={},numSearchThreadsPerAnalysisThread=1,"
         "reportAnalysisWinratesAs=BLACK,logDir={}".format(threads, log_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def feed():
        cache_path, base, moves = None, None, None
        for i, r in enumerate(rows):
            if r["path"] != cache_path:  # siblings of a source are contiguous
                with open(os.path.join(repo_root, r["path"]), encoding="utf-8",
                          errors="replace") as f:
                    base, moves = game_setup(f.read())
                cache_path = r["path"]
            proc.stdin.write(json.dumps(sibling_query(
                str(i), base, moves, int(r["node"]), r["move"], black[i], visits)) + "\n")
        proc.stdin.close()
    threading.Thread(target=feed, daemon=True).start()

    done, t0 = 0, time.time()
    for line in proc.stdout:
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError("katago: {}".format(resp))
        i = int(resp["id"])
        info = resp["rootInfo"]
        score, win = to_player_to_move(info["scoreLead"], info["winrate"], black[i])
        labels[2 * i], labels[2 * i + 1] = score, win
        done += 1
        if not quiet and done % 100000 == 0:
            print("  {}/{} siblings, {:.0f}/s".format(done, n, done / (time.time() - t0)),
                  flush=True)
    proc.wait()
    if done != n:
        raise RuntimeError("KataGo answered {} of {} queries".format(done, n))
    with open(os.path.join(split_dir, "labels.bin.part"), "wb") as f:
        labels.tofile(f)
    os.replace(os.path.join(split_dir, "labels.bin.part"), os.path.join(split_dir, "labels.bin"))
    meta = {"rows": n, "fields": ["score", "winrate"], "dtype": "float32",
            "side": "player to move", "visits": visits, "seconds": round(time.time() - t0)}
    with open(os.path.join(split_dir, "labels.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description="KataGo labels for sibling positions.")
    p.add_argument("out_dir")
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument("--katago", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--visits", type=int, default=1)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    split_dir = os.path.join(args.out_dir, args.split)
    meta = label_split(split_dir, args.repo_root, args.katago, args.model, args.config,
                       args.visits, args.threads, os.path.join(args.out_dir, "katago_logs"),
                       args.quiet)
    print("{}: {} siblings labeled in {}s".format(args.split, meta["rows"], meta["seconds"]))


if __name__ == "__main__":
    sys.exit(main())
