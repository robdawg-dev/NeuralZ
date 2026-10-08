#!/usr/bin/env python
"""Positions for training a score head to rank a policy's candidate moves
(SCORE_NET_PLAN.md): for sampled source positions, the policy's top-k moves, each played
out - k "siblings" per source.

    python -m AlphaGo.preprocessing.build_sibling_positions <shards_dir> <out_dir> \\
        --policy-model <model.json> --policy-weights <weights.h5> --split train \\
        --sources 300000 --top 8

Source positions are records sampled uniformly from <shards_dir>/<split>/shard_*.h5 (KataGo
self-play, as converted). Each is rebuilt by replaying its SGF (games.tsv path, the
record's move-node index), the policy ranks its sensible moves (legal, not filling own
eyes - what the bot considers), and each of its top k is played to make a sibling.

Written to <out_dir>/<split>/:
  sib_NNNNN.h5  per sibling: packed_states (bit-packed feature planes, as shard_stream
                reads them), group (source number), rank (0 = the policy's first choice),
                prior (the policy's probability, renormalized over the k), komi (for the
                sibling's player to move - the source mover's opponent)
  queries.tsv   per sibling, same order: what label_sibling_positions.py asks KataGo -
                SGF path, move-node index of the source, the candidate move, whether
                Black is to move in the sibling
Siblings of one source are contiguous, k per source (a source with fewer than k sensible
moves is skipped); sources come in a random game order. --exclude takes earlier builds'
queries.tsv files: their source positions are not sampled again, so a build can extend
an earlier one without duplicates. Games are
processed in chunks (replay in worker processes, policy on the GPU), so memory stays
bounded.
"""
import argparse
import csv
import glob
import io
import json
import multiprocessing
import os
import time

import h5py as h5
import numpy as np

from AlphaGo.preprocessing.add_value_targets import sidecar_path
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.training.shard_stream import PACKED_STATES, PLANES_SHAPE
from AlphaGo.util import sgf_iter_states

COLS = "abcdefghijklmnopqrs"
CHUNK_GAMES = 4000
FILE_ROWS = 500000

_worker = {}


def _init(features, size):
    _worker["proc"] = Preprocess(features, size=size)
    _worker["size"] = size


def _positions(path, nodes):
    """Yield (node index, GameState) for each wanted move-node index, the state set to the
    mover (as the bot does on genmove)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    wanted = set(nodes)
    for i, (state, move, player) in enumerate(sgf_iter_states(text, include_end=False)):
        if i in wanted:
            state.set_current_player(player)
            yield i, state
            wanted.discard(i)
            if not wanted:
                return


def _source_job(job):
    """Pass 1: (path, [node]) -> [(node, planes uint8, sensible flat indices)]."""
    path, nodes = job
    proc, size = _worker["proc"], _worker["size"]
    out = []
    for i, state in _positions(path, nodes):
        sensible = state.get_legal_moves(include_eyes=False)
        out.append((i, proc.state_to_tensor(state)[0],
                    np.array([x * size + y for x, y in sensible], np.int32)))
    return path, out


def _sibling_job(job):
    """Pass 2: (path, {node: [flat move]}) -> {node: packed sibling planes, in order}."""
    path, cands = job
    proc, size = _worker["proc"], _worker["size"]
    out = {}
    for i, state in _positions(path, list(cands)):
        planes = []
        for k in cands[i]:
            child = state.copy()
            child.do_move((k // size, k % size))
            planes.append(proc.state_to_tensor(child)[0])
        out[i] = np.packbits(np.asarray(planes, np.uint8).reshape(len(planes), -1), axis=1)
    return path, out


def used_sources(query_files, game_of_path):
    """The (game_id, node index) source positions named in earlier builds' queries.tsv."""
    used = set()
    for q in query_files:
        with io.open(q, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="	"):
                used.add((game_of_path[r["path"]], int(r["node"])))
    return used


def sample_sources(shards, n, seed, exclude=()):
    """Uniformly sampled records -> {game_id: [(node index, komi, black_to_move)]}, none of
    them an (game_id, node index) in exclude."""
    sizes, game_ids, moves, komis, blacks = [], [], [], [], []
    for path in shards:
        with h5.File(path, "r") as f:
            game_ids.append(f["game_id"][()])
            moves.append(f["move"][()])
        with h5.File(sidecar_path(path), "r") as f:
            komis.append(f["komi"][()].astype(np.float32))
            blacks.append(f["black_to_move"][()])
        sizes.append(len(game_ids[-1]))
    game_ids, moves = np.concatenate(game_ids), np.concatenate(moves)
    komis, blacks = np.concatenate(komis), np.concatenate(blacks)
    eligible = np.arange(len(game_ids))
    if exclude:
        key = game_ids.astype(np.int64) * 100000 + moves
        used = np.array([g * 100000 + node for g, node in exclude], np.int64)
        eligible = eligible[~np.isin(key, used)]
    pick = eligible[np.random.default_rng(seed).choice(len(eligible),
                                                       size=min(n, len(eligible)),
                                                       replace=False)]
    by_game = {}
    for r in pick:
        by_game.setdefault(int(game_ids[r]), []).append(
            (int(moves[r]), float(komis[r]), int(blacks[r])))
    return by_game


def top_k(policy, planes, sensible, k, batch=128):
    """The policy's top k sensible moves and their renormalized priors, per position."""
    out = []
    for s in range(0, len(planes), batch):
        probs = policy.forward(np.stack(planes[s:s + batch]).astype(np.float32))
        for p, legal in zip(probs, sensible[s:s + batch]):
            order = legal[np.argsort(-p[legal])][:k]
            prior = p[order] / p[order].sum()
            out.append((order, prior))
    return out


class _Writer(object):
    def __init__(self, out_dir, shape, k):
        self.dir, self.shape, self.file_no, self.rows, self.buf = out_dir, shape, 0, 0, None
        self.k = k  # rows per group: one group per packed_states chunk (cheap random reads)
        self.queries = io.open(os.path.join(out_dir, "queries.tsv"), "w", encoding="utf-8",
                               newline="")
        self.q = csv.writer(self.queries, delimiter="\t", lineterminator="\n")
        self.q.writerow(["path", "node", "move", "black_to_move"])
        self._new()

    def _new(self):
        self.buf = {k: [] for k in ("packed", "group", "rank", "prior", "komi")}

    def add_group(self, group, packed, priors, komi, path, node, moves, black_next):
        n = len(packed)
        self.buf["packed"].append(packed)
        self.buf["group"].append(np.full(n, group, np.int32))
        self.buf["rank"].append(np.arange(n, dtype=np.uint8))
        self.buf["prior"].append(priors.astype(np.float16))
        self.buf["komi"].append(np.full(n, komi, np.float16))
        for m in moves:
            self.q.writerow([path, node, m, black_next])
        self.rows += n
        if self.rows >= FILE_ROWS:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        path = os.path.join(self.dir, "sib_{:05d}.h5".format(self.file_no))
        with h5.File(path + ".part", "w") as f:
            for key, data in self.buf.items():
                data = np.concatenate(data)
                # packed rows: one sibling group per chunk, so reading a group at random
                # decompresses just that group (default chunks span thousands of rows)
                chunks = (self.k, data.shape[1]) if key == "packed" else (min(4096, len(data)),)
                d = f.create_dataset(key if key != "packed" else PACKED_STATES, data=data,
                                     chunks=chunks, compression="gzip", compression_opts=1)
                if key == "packed":
                    d.attrs[PLANES_SHAPE] = self.shape
        os.replace(path + ".part", path)
        self.file_no += 1
        self.rows = 0
        self._new()

    def close(self):
        self.flush()
        self.queries.close()


def build(shards_dir, out_dir, policy, split, sources, k, seed, workers, quiet=False,
          exclude=()):
    shards = sorted(glob.glob(os.path.join(shards_dir, split, "shard_*.h5")))
    with h5.File(shards[0], "r") as f:
        shape = tuple(int(d) for d in f[PACKED_STATES].attrs[PLANES_SHAPE])
        features = f["features"][()]
    features = (features.decode() if isinstance(features, bytes) else features).split(",")
    if features != policy.preprocessor.get_feature_list():
        raise ValueError("shards' features differ from the policy's")
    with io.open(os.path.join(shards_dir, split, "games.tsv"), encoding="utf-8") as f:
        paths = {int(r["game_id"]): r["path"] for r in csv.DictReader(f, delimiter="\t")}

    t0 = time.time()
    used = used_sources(exclude, {p: g for g, p in paths.items()})
    by_game = sample_sources(shards, sources, seed, used)
    games = sorted(by_game)
    np.random.default_rng(seed + 1).shuffle(games)
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    writer = _Writer(os.path.join(out_dir, split), shape, k)
    group = short = 0
    with multiprocessing.Pool(workers, initializer=_init,
                              initargs=(features, shape[0])) as pool:
        for c in range(0, len(games), CHUNK_GAMES):
            chunk = games[c:c + CHUNK_GAMES]
            meta = {paths[g]: {node: (komi, black) for node, komi, black in by_game[g]}
                    for g in chunk}
            jobs = [(p, sorted(m)) for p, m in meta.items()]
            src = []
            for path, rows in pool.imap_unordered(_source_job, jobs, chunksize=16):
                src += [(path, node, planes, sensible) for node, planes, sensible in rows]
            ranked = top_k(policy, [s[2] for s in src], [s[3] for s in src], k)
            cands = {}
            for (path, node, _p, _s), (order, prior) in zip(src, ranked):
                if len(order) < k:  # too few sensible moves (late endgame): groups stay whole
                    short += 1
                    continue
                cands.setdefault(path, {})[node] = (order, prior)
            sib_jobs = [(p, {n: list(v[0]) for n, v in nodes.items()})
                        for p, nodes in cands.items()]
            for path, packed in pool.imap_unordered(_sibling_job, sib_jobs, chunksize=16):
                for node, planes in sorted(packed.items()):
                    order, prior = cands[path][node]
                    komi, black = meta[path][node]
                    moves = [COLS[m // shape[0]] + COLS[m % shape[0]] for m in order]
                    # the sibling's player to move is the source mover's opponent
                    writer.add_group(group, planes, prior, -komi, path, node, moves,
                                     1 - black)
                    group += 1
            if not quiet:
                print("  {}/{} games, {} sources, {:.0f}s".format(
                    min(c + CHUNK_GAMES, len(games)), len(games), group, time.time() - t0),
                    flush=True)
    writer.close()
    meta = {"split": split, "sources": group, "skipped_short": short, "top": k, "seed": seed,
            "excluded": len(used), "exclude_files": list(exclude),
            "features": features, "seconds": round(time.time() - t0)}
    with open(os.path.join(out_dir, split, "siblings.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description="Sibling positions: a policy's top-k moves "
                                            "played out from sampled shard positions.")
    p.add_argument("shards_dir")
    p.add_argument("out_dir")
    p.add_argument("--policy-model", required=True)
    p.add_argument("--policy-weights", required=True)
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument("--sources", type=int, default=300000)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--exclude", action="append", default=[], metavar="QUERIES_TSV",
                   help="an earlier build's queries.tsv whose source positions are not "
                        "sampled again (repeatable)")
    args = p.parse_args(argv)
    from AlphaGo.models.nn_util import NeuralNetBase
    policy = NeuralNetBase.load_model(args.policy_model)
    policy.model.load_weights(args.policy_weights)
    meta = build(args.shards_dir, args.out_dir, policy, args.split, args.sources, args.top,
                 args.seed, args.workers, args.quiet, args.exclude)
    print("{}: {} sources x top {} in {}s".format(meta["split"], meta["sources"], meta["top"],
                                                  meta["seconds"]))


if __name__ == "__main__":
    main()
