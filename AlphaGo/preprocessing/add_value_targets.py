#!/usr/bin/env python
"""Add value-head targets to a converted shard set, without touching the shards.

    python -m AlphaGo.preprocessing.add_value_targets <shards_dir> [--splits train val test]

For every <shards_dir>/<split>/shard_NNNNN.h5 this writes <split>/value_NNNNN.h5 with one
row per shard record, in the same order (named value_*, not shard_*.value.h5, so that
find_split_shards' shard_*.h5 glob never picks them up):

  value       KataGo's win rate for the player to move, win / (win + loss)    float16
  score       KataGo's expected score lead for the player to move, in points  float16
  komi        the game's komi from the player to move's side (+ for White)    float16
  has_target  1 where the position has a KataGo annotation, else 0            uint8
  black_to_move  1 where Black is the player to move (the record's mover)     uint8

Where the numbers come from: KataGo writes, on each move node from startTurnIdx on, its
search result from White's perspective for the position BEFORE that move - the same
search that chose the move (KataGo cpp/program/play.cpp pushes whiteValueTargetsByTurn
during the turn's search; cpp/dataio/sgf.cpp writes entry i on move node i before playing
it). A shard record is (position before move node `move`, that move), so its targets are
that node's comment. Nodes without one (the first startTurnIdx moves, "Pass for ko"
nodes) get has_target 0.

Records are matched to nodes by the shard's (game_id, move): game_id is the row of
<split>/games.tsv, move the move-node index convert_shuffled took from sgf_iter_states.
Every record is checked: the move parsed from that node must equal the record's stored
action. Any mismatch stops the run - it would mean labels attached to the wrong positions.
"""
import argparse
import csv
import glob
import io
import json
import multiprocessing
import os
import re
import sys
import time

import h5py as h5
import numpy as np

from AlphaGo.preprocessing.convert_shuffled import _RE_MOVE

SPLITS = ("train", "val", "test")
_RE_KOMI = re.compile(r"KM\[\s*(-?[\d.]+)\s*\]")
NO_MOVE = -1


def parse_game(text, board_size=19):
    """KataGo annotations of one SGF, indexed like sgf_iter_states' move nodes.

    Returns (komi, black_to_move, move_index, white_win, white_score): per move node,
    whether Black made it, its point as x * board_size + y (NO_MOVE for a pass), and
    KataGo's White win rate (win / (win + loss)) and White score lead - NaN where the
    node has no annotation."""
    m = _RE_KOMI.search(text)
    komi = float(m.group(1)) if m else 0.0
    black, index, win, score = [], [], [], []
    for node in _RE_MOVE.finditer(text):
        color, point = node.group(1), node.group(2)
        black.append(color == "B")
        if len(point) == 2 and point != "tt":
            index.append((ord(point[0]) - 97) * board_size + (ord(point[1]) - 97))
        else:
            index.append(NO_MOVE)
        if node.group(3) is None:
            win.append(np.nan)
            score.append(np.nan)
        else:
            w, lo = float(node.group(3)), float(node.group(4))
            win.append(w / (w + lo) if w + lo > 0 else np.nan)
            score.append(float(node.group(6)))
    return (komi, np.array(black, bool), np.array(index, np.int16),
            np.array(win, np.float32), np.array(score, np.float32))


def _parse_job(job):
    game_id, path = job
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return game_id, parse_game(f.read()), None
    except OSError as e:
        return game_id, None, str(e)


def load_games(split_dir, repo_root, workers):
    """game_id -> parse_game() result for every game in <split_dir>/games.tsv."""
    with io.open(os.path.join(split_dir, "games.tsv"), encoding="utf-8") as f:
        jobs = [(int(row["game_id"]), os.path.join(repo_root, row["path"]))
                for row in csv.DictReader(f, delimiter="\t")]
    games, errors = {}, []
    with multiprocessing.Pool(workers) as pool:
        for game_id, parsed, err in pool.imap_unordered(_parse_job, jobs, chunksize=64):
            if err:
                errors.append((game_id, err))
            else:
                games[game_id] = parsed
    if errors:
        raise RuntimeError("{} games unreadable, e.g. {}".format(len(errors), errors[:3]))
    return games


def targets_for_shard(game_ids, moves, actions, games, board_size=19):
    """Per-record (value, score, komi, has_target, black_to_move) arrays, plus the
    records whose node move doesn't match the stored action (should be none)."""
    n = len(game_ids)
    value = np.zeros(n, np.float32)
    score = np.zeros(n, np.float32)
    komi = np.zeros(n, np.float32)
    has = np.zeros(n, np.uint8)
    black_to_move = np.zeros(n, np.uint8)
    mismatches = []
    for r in range(n):
        g_komi, black, index, win, w_score = games[int(game_ids[r])]
        i = int(moves[r])
        stored = int(actions[r][0]) * board_size + int(actions[r][1])
        if i >= len(index) or index[i] != stored:
            mismatches.append(r)
            continue
        sign = -1.0 if black[i] else 1.0  # the mover's side: White +, Black -
        black_to_move[r] = black[i]
        komi[r] = sign * g_komi
        if not np.isnan(win[i]):
            value[r] = win[i] if sign > 0 else 1.0 - win[i]
            score[r] = sign * w_score[i]
            has[r] = 1
    return value, score, komi, has, black_to_move, mismatches


def write_sidecar(path, shard_name, value, score, komi, has, black_to_move):
    with h5.File(path + ".part", "w") as f:
        for name, data, dtype in (("value", value, np.float16), ("score", score, np.float16),
                                  ("komi", komi, np.float16), ("has_target", has, np.uint8),
                                  ("black_to_move", black_to_move, np.uint8)):
            f.create_dataset(name, data=data.astype(dtype), compression="gzip",
                             compression_opts=1)
        f.attrs["shard"] = shard_name
        f.attrs["rows"] = len(value)
    os.replace(path + ".part", path)


def sidecar_path(shard_path):
    d, name = os.path.split(shard_path)
    return os.path.join(d, name.replace("shard_", "value_", 1))


def process_split(shards_dir, split, repo_root, workers, quiet=False):
    split_dir = os.path.join(shards_dir, split)
    shards = sorted(glob.glob(os.path.join(split_dir, "shard_*.h5")))
    if not shards:
        raise ValueError("no shard_*.h5 files in {}".format(split_dir))
    t0 = time.time()
    games = load_games(split_dir, repo_root, workers)
    if not quiet:
        print("{}: parsed {} games in {:.0f}s".format(split, len(games), time.time() - t0),
              flush=True)
    stats = {"records": 0, "with_target": 0, "shards": len(shards)}
    for shard in shards:
        with h5.File(shard, "r") as f:
            game_ids, moves, actions = f["game_id"][()], f["move"][()], f["actions"][()]
        value, score, komi, has, black, bad = targets_for_shard(game_ids, moves, actions, games)
        if bad:
            r = bad[0]
            raise RuntimeError(
                "{}: {} records whose SGF move node does not match the stored action, e.g. "
                "record {} (game_id {}, move {}, action {}) - labels would attach to the "
                "wrong positions".format(shard, len(bad), r, int(game_ids[r]), int(moves[r]),
                                         tuple(int(a) for a in actions[r])))
        write_sidecar(sidecar_path(shard), os.path.basename(shard), value, score, komi, has,
                      black)
        stats["records"] += len(value)
        stats["with_target"] += int(has.sum())
        if not quiet:
            print("  {} -> {}: {} records, {:.2%} with a target".format(
                os.path.basename(shard), os.path.basename(sidecar_path(shard)), len(value),
                has.mean()), flush=True)
    stats["seconds"] = round(time.time() - t0)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description="Add value-head targets beside a shard set.")
    p.add_argument("shards_dir", help="convert_shuffled output directory (one subdir per split)")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--repo-root", default=".",
                   help="Directory games.tsv paths are relative to. Default: current directory")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    summary = {}
    for split in args.splits:
        summary[split] = process_split(args.shards_dir, split, args.repo_root, args.workers,
                                       args.quiet)
    with open(os.path.join(args.shards_dir, "value_targets.json"), "w") as f:
        json.dump({"args": vars(args), "splits": summary}, f, indent=2)
    for split, s in summary.items():
        print("{}: {} records, {} with a target ({:.2%}), {}s".format(
            split, s["records"], s["with_target"], s["with_target"] / max(s["records"], 1),
            s["seconds"]))


if __name__ == "__main__":
    sys.exit(main())
