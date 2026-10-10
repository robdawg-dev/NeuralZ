"""Round robin among policy networks: every pair plays a play_tests/match_networks.py match
(colors alternating), KataGo decides the winners (tools/match_winners.py), and the result
is a crosstable plus Elo from a Bradley-Terry fit over all games.

    # play: every pair of these models (names from play_tests/policy_loading.py)
    uv run python tools/playoff.py run b20c256 b15c192latest b10c128mb1024 --games 30 \\
        [--gpu] [--docker] [--out workspace/playoff]

    # the table again, from matches already played and judged
    uv run python tools/playoff.py table workspace/playoff/playoff.json
    uv run python tools/playoff.py table play_tests/sgf/match_A_vs_B_*/ ...

--docker runs the matches in the project's GPU container (docker compose run --rm gpu);
KataGo always runs here. "run" writes <out>/playoff.json (the match directories) as it
goes, so a stopped run's finished matches can still be tabled. KataGo:
--katago/--katago-model/--katago-config or KATAGO_EXE/KATAGO_MODEL/KATAGO_CONFIG.
"""
import argparse
import collections
import glob
import itertools
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from katago_util import add_katago_args, katago_command  # noqa: E402
from match_winners import decide  # noqa: E402

SGF_DIR = os.path.join(ROOT, "play_tests", "sgf")


def tally(results):
    """{(winner, loser): games} over katago_results dicts."""
    wins = collections.Counter()
    for r in results:
        for g in r["games"]:
            w = g["katago_winner_model"]
            loser = g["white_model"] if w == g["black_model"] else g["black_model"]
            wins[(w, loser)] += 1
    return wins


def elo(wins, models, iterations=2000):
    """Bradley-Terry strengths by MM iterations (+0.5 virtual win and loss per played pair,
    so a sweep stays finite), as Elo with the weakest model at 0."""
    played = {(a, b) for a in models for b in models
              if a != b and wins[(a, b)] + wins[(b, a)] > 0}
    s = {m: 1.0 for m in models}
    for _ in range(iterations):
        for a in models:
            opp = [b for b in models if (a, b) in played]
            if not opp:
                continue
            num = sum(wins[(a, b)] + 0.5 for b in opp)
            den = sum((wins[(a, b)] + wins[(b, a)] + 1) / (s[a] + s[b]) for b in opp)
            s[a] = num / den
        g = math.exp(sum(math.log(v) for v in s.values()) / len(s))
        s = {m: v / g for m, v in s.items()}
    ratings = {m: 400 * math.log10(v) for m, v in s.items()}
    base = min(ratings.values())
    return {m: r - base for m, r in ratings.items()}


def table(results, models=None):
    """The crosstable and Elo as text lines; models in the given order, else by Elo."""
    wins = tally(results)
    names = models or sorted({m for pair in wins for m in pair})
    ratings = elo(wins, names)
    if not models:
        names = sorted(names, key=lambda m: -ratings[m])
    total = sum(wins.values())
    black = sum(1 for r in results for g in r["games"]
                if g["katago_winner_model"] == g["black_model"])
    w = max(len(n) for n in names) + 2
    lines = ["{} games, KataGo-decided (Black won {:.0%}). Row's wins-losses vs column:".format(
        total, black / total if total else 0)]
    lines.append("".ljust(w) + "".join(n[:14].rjust(16) for n in names) + "total".rjust(12))
    for a in names:
        cells, won, n = [], 0, 0
        for b in names:
            if a == b:
                cells.append("-".rjust(16))
                continue
            x, y = wins[(a, b)], wins[(b, a)]
            won, n = won + x, n + x + y
            cells.append(("{}-{}".format(x, y) if x + y else "").rjust(16))
        lines.append(a.ljust(w) + "".join(cells) +
                     "{}/{} {:.0%}".format(won, n, won / n if n else 0).rjust(12))
    lines.append("")
    lines.append("Elo (Bradley-Terry, weakest = 0): " + ", ".join(
        "{} {:+.0f}".format(m, ratings[m]) for m in sorted(names, key=lambda m: -ratings[m])))
    return lines


def load_results(paths):
    """katago_results dicts from match directories or a playoff.json listing them."""
    dirs = []
    for p in paths:
        if p.endswith(".json"):
            with open(p) as f:
                dirs += json.load(f)["matches"]
        else:
            dirs.append(p)
    out = []
    for d in dirs:
        path = os.path.join(d, "katago_results.json")
        if os.path.exists(path):
            with open(path) as f:
                out.append(json.load(f))
        else:
            sys.stderr.write("playoff: {} not judged yet (no katago_results.json)\n".format(d))
    return out


def play_match(a, b, games, gpu, docker):
    """Run one match_networks.py match; returns its new output directory."""
    before = set(glob.glob(os.path.join(SGF_DIR, "match_{}_vs_{}_*".format(a, b))))
    cmd = ["python", "play_tests/match_networks.py", a, b, "--num-games", str(games)]
    if gpu:
        cmd.append("--gpu")
    if docker:
        cmd = ["docker", "compose", "run", "--rm", "gpu"] + cmd
    else:
        cmd[0] = sys.executable
    subprocess.run(cmd, cwd=ROOT, check=True)
    new = set(glob.glob(os.path.join(SGF_DIR, "match_{}_vs_{}_*".format(a, b)))) - before
    if len(new) != 1:
        raise RuntimeError("expected one new match directory for {} vs {}, found {}".format(
            a, b, sorted(new)))
    return new.pop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="play and judge every pair of the given models")
    r.add_argument("models", nargs="+")
    r.add_argument("--games", type=int, default=30, help="per pair. Default: 30")
    r.add_argument("--gpu", action="store_true", help="matches on the GPU")
    r.add_argument("--docker", action="store_true", help="matches in the GPU container")
    r.add_argument("--visits", type=int, default=400, help="KataGo visits per game")
    r.add_argument("--out", default=os.path.join(ROOT, "workspace", "playoff"),
                   help="where playoff.json goes. Default: workspace/playoff")
    add_katago_args(r, required=True)
    t = sub.add_parser("table", help="the table from judged matches")
    t.add_argument("paths", nargs="+", help="match directories, or a playoff.json")
    args = parser.parse_args(argv)

    if args.command == "table":
        print("\n".join(table(load_results(args.paths))))
        return
    command = katago_command(args, r)
    if len(set(args.models)) < 2:
        parser.error("a playoff needs at least two different models")
    os.makedirs(args.out, exist_ok=True)
    record = os.path.join(args.out, "playoff.json")
    matches = []
    for a, b in itertools.combinations(args.models, 2):
        print("=== {} vs {}".format(a, b), flush=True)
        d = play_match(a, b, args.games, args.gpu, args.docker)
        result = decide(d, command, args.visits)
        print("KataGo wins: {}".format(result["wins"]), flush=True)
        matches.append(os.path.relpath(d, ROOT))
        with open(record, "w") as f:
            json.dump({"models": args.models, "games_per_pair": args.games,
                       "matches": matches}, f, indent=1)
    print()
    print("\n".join(table(load_results([os.path.join(ROOT, m) for m in matches]),
                          args.models)))


if __name__ == "__main__":
    main()
