#!/usr/bin/env python
"""Choose the games for one training set and split them into train/val/test.

Reads the manifest written by `sgf_preparation scan` and writes, into <out_dir>:

    train.txt  val.txt  test.txt    one game per line: path<TAB>n_moves<TAB>gtype
    selection_summary.txt           what was chosen and why

Every selection decision for a run lives here. The converter downstream only builds
feature planes from whatever these lists contain.

Pools
-----
normal     gtype=normal, komi in [5, 9], KataGo's opening winrate in [0.3, 0.7]
handicap   gtype=handicap, opening winrate in [0.3, 0.7] - i.e. the COMPENSATED games,
           where KataGo set komi so the game starts even. Uncompensated handicap games
           start with White already lost (median winrate 0.00) and are played out at
           minimum search, so they are left out.

The opening winrate is White's, at KataGo's first searched move. Selecting on it rather
than on komi alone accounts for fair komi differing by rule set, and drops games whose
policy-randomised opening had already decided them.

Sampling and split
------------------
Whole games are drawn at random from each pool until the pool's position target is met,
estimating positions as n_moves * POSITIONS_PER_MOVE (passes are never positions, and the
converter drops a few more). Each pool is then split by GAME into train/val/test with the
same ratios, so every split carries the same normal:handicap mix and no game appears in
more than one split. Everything is seeded.
"""
import argparse
import io
import json
import os
import random
import sys

POSITIONS_PER_MOVE = 0.97


def _pools(manifest, komi_min, komi_max, wr_min, wr_max):
    normal, handicap = [], []
    with io.open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            try:
                komi = float(d["komi"])
                wr = float(d["first_searched_winrate"])
                moves = int(d["n_moves"])
            except (KeyError, TypeError, ValueError):
                continue
            if moves <= 0 or not (wr_min <= wr <= wr_max):
                continue
            path = d["path"].replace("\\", "/")
            if d.get("gtype") == "normal" and komi_min <= komi <= komi_max:
                normal.append((path, moves, "normal"))
            elif d.get("gtype") == "handicap":
                handicap.append((path, moves, "handicap"))
    return normal, handicap


def _take(pool, target_positions, rng):
    pool = list(pool)
    rng.shuffle(pool)
    chosen, positions = [], 0.0
    for game in pool:
        if positions >= target_positions:
            break
        chosen.append(game)
        positions += game[1] * POSITIONS_PER_MOVE
    return chosen, positions


def _split(games, ratios):
    n = len(games)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return games[:n_train], games[n_train:n_train + n_val], games[n_train + n_val:]


def select(args):
    ratios = args.split
    if abs(sum(ratios) - 1.0) > 1e-9:
        sys.exit("--split must sum to 1, got {}".format(ratios))
    os.makedirs(args.out_dir, exist_ok=True)
    for name in ("train.txt", "val.txt", "test.txt"):
        if os.path.exists(os.path.join(args.out_dir, name)) and not args.force:
            sys.exit("{} already exists in {} - pass --force to overwrite".format(
                name, args.out_dir))

    rng = random.Random(args.seed)
    normal_pool, handicap_pool = _pools(
        args.manifest, args.komi_min, args.komi_max, args.wr_min, args.wr_max)
    normal, normal_pos = _take(normal_pool, args.normal_positions, rng)
    handicap, handicap_pos = _take(handicap_pool, args.handicap_positions, rng)
    if normal_pos < args.normal_positions or handicap_pos < args.handicap_positions:
        print("WARNING: a pool ran out before reaching its target", file=sys.stderr)

    splits = {"train": [], "val": [], "test": []}
    for games in (normal, handicap):
        for name, part in zip(("train", "val", "test"), _split(games, ratios)):
            splits[name].extend(part)

    lines = [
        "manifest          : {}".format(args.manifest),
        "seed              : {}".format(args.seed),
        "normal pool       : gtype=normal, komi [{}, {}], opening winrate [{}, {}]".format(
            args.komi_min, args.komi_max, args.wr_min, args.wr_max),
        "handicap pool     : gtype=handicap, opening winrate [{}, {}] (compensated)".format(
            args.wr_min, args.wr_max),
        "pool sizes        : {:,} normal, {:,} handicap games".format(
            len(normal_pool), len(handicap_pool)),
        "chosen            : {:,} normal (~{:,.0f} positions), {:,} handicap (~{:,.0f})".format(
            len(normal), normal_pos, len(handicap), handicap_pos),
        "handicap share    : {:.2%} of estimated positions".format(
            handicap_pos / max(normal_pos + handicap_pos, 1)),
        "",
        "{:<8}{:>10}{:>12}{:>16}{:>12}".format("split", "games", "handicap", "~positions", "hcap pos"),
    ]
    for name, games in splits.items():
        rng.shuffle(games)
        est = sum(g[1] for g in games) * POSITIONS_PER_MOVE
        est_h = sum(g[1] for g in games if g[2] == "handicap") * POSITIONS_PER_MOVE
        lines.append("{:<8}{:>10,}{:>12,}{:>16,.0f}{:>11.2%}".format(
            name, len(games), sum(1 for g in games if g[2] == "handicap"), est,
            est_h / max(est, 1)))
        with io.open(os.path.join(args.out_dir, name + ".txt"), "w", encoding="utf-8",
                     newline="\n") as f:
            for path, moves, gtype in games:
                f.write("{}\t{}\t{}\n".format(path, moves, gtype))

    with io.open(os.path.join(args.out_dir, "selection_summary.txt"), "w", encoding="utf-8",
                 newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description="Select and split games for one training set.")
    p.add_argument("manifest", help="JSONL manifest from `sgf_preparation scan`")
    p.add_argument("out_dir", help="Where to write train.txt / val.txt / test.txt")
    p.add_argument("--normal-positions", type=float, default=38e6)
    p.add_argument("--handicap-positions", type=float, default=2e6)
    p.add_argument("--split", type=float, nargs=3, default=[0.93, 0.05, 0.02],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--komi-min", type=float, default=5.0)
    p.add_argument("--komi-max", type=float, default=9.0)
    p.add_argument("--wr-min", type=float, default=0.3)
    p.add_argument("--wr-max", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--force", action="store_true")
    select(p.parse_args(argv))


if __name__ == "__main__":
    main()
