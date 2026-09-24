#!/usr/bin/env python
"""Convert selected games into globally shuffled training shards.

    python -m AlphaGo.preprocessing.convert_shuffled <selection_dir> <out_dir>

<selection_dir> holds train.txt / val.txt / test.txt from select_games.py. Each split is
converted on its own into <out_dir>/<split>/shard_NNNNN.h5.

The shards of a split, read in file order, are ONE uniformly random permutation of every
position in that split - not a game-order shuffle, and not a local mix. A trainer can
stream them start to finish with no shuffle buffer. This is done in two passes:

  pass 1  convert every game, sending each position to one of K bucket files chosen at
          random. Buckets live on disk and grow until the last game is done, so every
          bucket ends up with a random ~1/K of positions drawn from ALL games.
  pass 2  load each bucket, shuffle it in memory, write it out as a shard, and delete
          the bucket. Buckets are independent, so several are done in parallel.

Buckets are raw files of fixed-size records rather than HDF5. Every feature plane is 0/1,
so workers bit-pack each position (17,328 bytes -> 2,166 at 48 planes) and the main
process only appends bytes. Compressing into HDF5 there was the pass-1 bottleneck: HDF5
calls cannot run in parallel from Python, so every position went through one thread.

Random bucket assignment + a uniform shuffle inside each bucket + concatenation gives a
uniformly random order: given the bucket sizes, every assignment and every within-bucket
order is equally likely, and each such pair maps to exactly one final order.

This file only builds feature planes. Which games go in is decided upstream. What it
does on its own:
  - passes are never positions (the policy has no pass output)
  - a move the engine rejects (in practice multi-stone suicide under sui1 rules) is not
    emitted; the replay stops there and the positions before it are kept
  - --max-winrate-loss X drops a position whose move KataGo's own search says gave up
    more than X winrate. The replay continues, so the position after a blunder is kept
    with the move that punishes it. The share dropped is reported.

Each shard holds: states (N,19,19,F) uint8, actions (N,2) uint8, game_id (N,) int32,
move (N,) int16, plus `features` and `conversion_args`. game_id indexes
<split>/games.tsv, and move is the move number in that game, so any position traces back
to its SGF.
"""
import argparse
import collections
import concurrent.futures
import io
import json
import math
import os
import re
import sys
import time

import h5py as h5
import numpy as np
import sgf

import AlphaGo.go as go
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import sgf_iter_states

ALL_FEATURES = [
    "board", "ones", "turns_since", "liberties", "capture_size", "self_atari_size",
    "liberties_after", "ladder_capture", "ladder_escape", "sensibleness", "zeros"]
SPLITS = ("train", "val", "test")
POSITIONS_PER_MOVE = 0.97
CHUNK_ROWS = 64

# A move node with its KataGo annotation, if any. win/loss are from White's perspective.
_RE_MOVE = re.compile(
    r";([BW])\[([^\]]*)\](?:C\[\s*(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) v=(\d+))?")


# ----------------------------------------------------------------------- worker side

def _move_losses(text):
    """Winrate the mover gave up with each move node, or None where unknown.

    Indexes line up with sgf_iter_states, which yields one tuple per move node. A move's
    loss is measured against the next ANNOTATED move."""
    raw = []
    for m in _RE_MOVE.finditer(text):
        if m.group(3) is None:
            raw.append(None)
        else:
            win, loss = float(m.group(3)), float(m.group(4))
            total = win + loss
            raw.append((m.group(1), win / total) if total > 0 else None)
    out = []
    for i, entry in enumerate(raw):
        lost = None
        if entry is not None:
            color, white_w = entry
            for nxt in raw[i + 1:]:
                if nxt is not None:
                    lost = (white_w - nxt[1]) if color == "W" else (nxt[1] - white_w)
                    break
        out.append(lost)
    return out


def _truncation_reason(exc, state, move):
    if "board-altering" in str(exc):
        return "setup_node"
    try:
        if state is not None and move not in (None, go.PASS):
            if state.get_board()[move[0]][move[1]] != go.EMPTY:
                return "illegal_occupied"
            if state.get_ko_location() == move:
                return "illegal_ko"
            return "illegal_suicide"
    except Exception:                                      # noqa: BLE001
        pass
    return "illegal_other"


_worker = {}


def record_dtype(board_size, n_features):
    """One bucket record: a position bit-packed, plus what traces it back to its game."""
    packed = (board_size * board_size * n_features + 7) // 8
    return np.dtype([("game_id", "<i4"), ("move", "<i2"), ("action", "u1", (2,)),
                     ("state", "u1", (packed,))])


def _init_worker(features, board_size, max_winrate_loss, keep_unpacked=False):
    proc = Preprocess(features, size=board_size)
    _worker["proc"] = proc
    _worker["board_size"] = board_size
    _worker["max_winrate_loss"] = max_winrate_loss
    _worker["record_dtype"] = record_dtype(board_size, proc.get_output_dimension())
    _worker["keep_unpacked"] = keep_unpacked


def convert_game(job):
    """(game_id, path) -> dict with the game's positions and counters. Never raises."""
    game_id, path = job
    proc = _worker["proc"]
    max_loss = _worker["max_winrate_loss"]
    res = {"game_id": game_id, "records": None, "states": None, "actions": None,
           "moves": None,
           "declared": 0, "passes": 0, "dropped_winrate": 0, "truncation": None,
           "error": None}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        res["truncation"], res["error"] = "unreadable", str(e)
        return res

    losses = _move_losses(text) if max_loss else []
    states, actions, moves = [], [], []
    state = move = None
    try:
        for i, (state, move, player) in enumerate(sgf_iter_states(text, include_end=False)):
            res["declared"] += 1
            if state.get_size() != _worker["board_size"]:
                res["truncation"] = "size_mismatch"
                states = []
                break
            if move is go.PASS:
                res["passes"] += 1
                continue
            if state.get_current_player() == player and not state.is_legal(move):
                continue
            if max_loss and i < len(losses) and losses[i] is not None and losses[i] > max_loss:
                res["dropped_winrate"] += 1
                continue
            states.append(proc.state_to_tensor(state)[0])
            actions.append(move)
            moves.append(i)
    except go.IllegalMove as e:
        res["truncation"] = _truncation_reason(e, state, move)
    except sgf.ParseException:
        res["truncation"] = "parse_error"
    except Exception as e:                                 # noqa: BLE001
        res["truncation"], res["error"] = "other", "{}: {}".format(type(e).__name__, e)

    if states:
        states = np.asarray(states, dtype=np.uint8)
        rec = np.empty(len(states), dtype=_worker["record_dtype"])
        rec["game_id"] = game_id
        rec["move"] = moves
        rec["action"] = np.asarray(actions, dtype=np.uint8)
        rec["state"] = np.packbits(states.reshape(len(states), -1), axis=1)
        res["records"] = rec
        if _worker.get("keep_unpacked"):
            res["states"] = states
            res["actions"] = rec["action"].copy()
            res["moves"] = rec["move"].copy()
    return res


# ----------------------------------------------------------------------- main side

def _create(h5f, name, shape, dtype, rows=0):
    return h5f.create_dataset(
        name, shape=(rows,) + shape, maxshape=(None,) + shape, dtype=dtype,
        chunks=(CHUNK_ROWS,) + shape, compression="lzf")


def _read_keeplist(path):
    games = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                games.append((parts[0], int(parts[1]) if len(parts) > 1 else 0,
                              parts[2] if len(parts) > 2 else ""))
    return games


def _write_shard(job, block=8192):
    """Shuffle one bucket into a shard. Runs in a pass-2 worker. Written under a temp
    name and renamed, so a shard that exists is always complete."""
    (bucket_path, shard_path, seed, board_size, n_features, features,
     conversion_args) = job
    rec = np.fromfile(bucket_path, dtype=record_dtype(board_size, n_features))
    n = len(rec)
    rec = rec[np.random.default_rng(seed).permutation(n)]
    shape = (board_size, board_size, n_features)
    bits = board_size * board_size * n_features
    tmp = shard_path + ".partial"
    with h5.File(tmp, "w") as s:
        states = _create(s, "states", shape, np.uint8, n)
        actions = _create(s, "actions", (2,), np.uint8, n)
        game_id = _create(s, "game_id", (), np.int32, n)
        move = _create(s, "move", (), np.int16, n)
        for start in range(0, n, block):
            part = rec[start:start + block]
            states[start:start + len(part)] = np.unpackbits(
                part["state"], axis=1, count=bits).reshape((len(part),) + shape)
        actions[:] = rec["action"]
        game_id[:] = rec["game_id"]
        move[:] = rec["move"]
        s["features"] = np.bytes_(",".join(features))
        s["conversion_args"] = np.bytes_(conversion_args)
    os.replace(tmp, shard_path)
    os.remove(bucket_path)
    return n


def convert_split(split, games, out_dir, args, features, conversion_args):
    split_dir = os.path.join(out_dir, split)
    bucket_dir = os.path.join(split_dir, "_buckets")
    os.makedirs(bucket_dir, exist_ok=True)

    est = sum(g[1] for g in games) * POSITIONS_PER_MOVE
    k = max(1, int(math.ceil(est / args.positions_per_shard)))
    split_index = SPLITS.index(split)
    rng = np.random.default_rng([args.seed, split_index])
    n_features = Preprocess(features, size=args.size).get_output_dimension()
    bucket_paths = [os.path.join(bucket_dir, "bucket_{:05d}.bin".format(i)) for i in range(k)]
    handles = [open(p, "wb", buffering=1 << 20) for p in bucket_paths]

    stats = collections.Counter()
    truncations = collections.Counter()
    written = [0] * len(games)
    start = time.time()
    print("[{}] pass 1: {:,} games -> {} buckets (~{:,.0f} positions)".format(
        split, len(games), k, est), flush=True)

    jobs = [(i, g[0]) for i, g in enumerate(games)]
    try:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker,
                initargs=(features, args.size, args.max_winrate_loss)) as pool:
            for n_done, res in enumerate(pool.map(convert_game, jobs, chunksize=8), 1):
                stats["games"] += 1
                stats["declared"] += res["declared"]
                stats["passes"] += res["passes"]
                stats["dropped_winrate"] += res["dropped_winrate"]
                if res["truncation"]:
                    truncations[res["truncation"]] += 1
                rec = res["records"]
                if rec is None:
                    stats["games_without_positions"] += 1
                else:
                    n = len(rec)
                    written[res["game_id"]] = n
                    stats["positions"] += n
                    assign = rng.integers(0, k, size=n)
                    order = np.argsort(assign, kind="stable")
                    bounds = np.searchsorted(assign[order], np.arange(k + 1))
                    for b in np.flatnonzero(np.diff(bounds)):
                        handles[b].write(rec[order[bounds[b]:bounds[b + 1]]].tobytes())
                if not args.quiet and n_done % 2000 == 0:
                    rate = n_done / max(time.time() - start, 1e-9)
                    print("  ... {:,}/{:,} games, {:,} positions ({:.1f} games/sec)".format(
                        n_done, len(games), stats["positions"], rate), flush=True)
    finally:
        for h in handles:
            h.close()
    pass1 = time.time() - start

    print("[{}] pass 2: shuffling {} buckets into shards ({} workers)".format(
        split, k, args.pass2_workers), flush=True)
    t2 = time.time()
    shard_jobs = [(bucket_paths[i], os.path.join(split_dir, "shard_{:05d}.h5".format(i)),
                   [args.seed, split_index, i], args.size, n_features, features,
                   conversion_args) for i in range(k)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.pass2_workers) as pool:
        sizes = list(pool.map(_write_shard, shard_jobs))
    os.rmdir(bucket_dir)
    pass2 = time.time() - t2

    with io.open(os.path.join(split_dir, "games.tsv"), "w", encoding="utf-8",
                 newline="\n") as f:
        f.write("game_id\tpath\tgtype\tpositions\n")
        for i, (path, _moves, gtype) in enumerate(games):
            f.write("{}\t{}\t{}\t{}\n".format(i, path, gtype, written[i]))

    eligible = stats["positions"] + stats["dropped_winrate"]
    summary = {
        "games": stats["games"],
        "games_without_positions": stats["games_without_positions"],
        "positions": stats["positions"],
        "moves_declared": stats["declared"],
        "passes_skipped": stats["passes"],
        "dropped_max_winrate_loss": stats["dropped_winrate"],
        "dropped_max_winrate_loss_pct": 100.0 * stats["dropped_winrate"] / max(eligible, 1),
        "truncated_games": dict(truncations),
        "shards": k,
        "positions_per_shard_min_max": [min(sizes), max(sizes)] if sizes else [0, 0],
        "pass1_seconds": round(pass1, 1),
        "pass2_seconds": round(pass2, 1),
    }
    _print_split_summary(split, summary)
    return summary


def _print_split_summary(split, s):
    print("[{}] {:,} games, {:,} positions in {} shards ({:,}-{:,} per shard)".format(
        split, s["games"], s["positions"], s["shards"], *s["positions_per_shard_min_max"]))
    print("       winrate-loss drops: {:,} ({:.3f}% of eligible positions)".format(
        s["dropped_max_winrate_loss"], s["dropped_max_winrate_loss_pct"]))
    print("       passes skipped: {:,}   games with no positions: {:,}".format(
        s["passes_skipped"], s["games_without_positions"]))
    if s["truncated_games"]:
        print("       truncated games: {}".format(s["truncated_games"]))
    print("       pass 1 {:.0f}s, pass 2 {:.0f}s".format(s["pass1_seconds"], s["pass2_seconds"]))


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert selected games into globally shuffled shards.")
    p.add_argument("selection_dir", help="Directory with train.txt / val.txt / test.txt")
    p.add_argument("out_dir", help="Output directory; one subdirectory per split")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--features", default="all", help="Comma-separated feature list, or 'all'")
    p.add_argument("--size", type=int, default=19)
    p.add_argument("--max-winrate-loss", type=float, default=None,
                   help="Drop positions whose move gave up more than this much winrate "
                        "(production runs use 0.10). Off by default.")
    p.add_argument("--positions-per-shard", type=int, default=100000,
                   help="Target shard size. Each pass-2 worker holds one bucket in memory, "
                        "bit-packed: ~2.2KB per position, so 100k is ~220MB.")
    p.add_argument("--pass2-workers", type=int, default=8,
                   help="Buckets shuffled into shards in parallel")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--limit", type=int, default=None, help="Only the first N games per split")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    features = ALL_FEATURES if args.features == "all" else args.features.split(",")
    if os.path.exists(args.out_dir) and any(
            os.path.exists(os.path.join(args.out_dir, s)) for s in args.splits):
        sys.exit("{} already contains output for these splits - use a new directory".format(
            args.out_dir))
    os.makedirs(args.out_dir, exist_ok=True)

    conversion_args = json.dumps({k: v for k, v in vars(args).items()}, sort_keys=True)
    report = {"args": vars(args), "features": features, "splits": {}}
    for split in args.splits:
        games = _read_keeplist(os.path.join(args.selection_dir, split + ".txt"))
        if args.limit:
            games = games[:args.limit]
        report["splits"][split] = convert_split(split, games, args.out_dir, args, features,
                                                conversion_args)
        with io.open(os.path.join(args.out_dir, "conversion.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
