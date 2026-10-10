"""Where a bot move's time goes on the client, by game stage: the input planes (all
together, and each feature on its own), the ladder guard, the legal-move list, and - with
--server - the round trip to a running go_server.py. Fixed sample positions from real KGS
games (benchmarks/sample_positions.json), so runs on different machines and code versions
compare; each run is appended to benchmarks/results/move_time.jsonl.

    uv run python benchmarks/move_time.py [--model b20c256] [--server http://127.0.0.1:5005]
    # profile the same positions instead (cProfile; open the .prof with snakeviz):
    uv run python benchmarks/move_time.py --profile move_time.prof
    # rebuild the sample set from game folders (bot games, several positions per game)
    uv run python benchmarks/move_time.py --build <folders> [--games 40]

For --profile, build the extensions with profiling hooks first
(NEURALZ_CYTHON_PROFILE=1 python setup_cython.py build_ext --inplace --force) so the
Cython functions show up individually - and rebuild normally before timing, since the
hooks slow everything down (a timing run warns if they're on).
"""
import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
POSITIONS = os.path.join(HERE, "sample_positions.json")
RESULTS = os.path.join(HERE, "results", "move_time.jsonl")
STAGES = [(1, 60), (61, 120), (121, 180), (181, 240), (241, 999)]
COLS = "ABCDEFGHJKLMNOPQRST"


def build(paths, games, moves=(30, 80, 130, 180, 230, 280)):
    """Positions from the first `games` bot games with enough moves, at each move number."""
    import game_report as gr
    from katago_util import gtp
    out = []
    for g in gr.load_games(paths, "NeuralZ"):
        if len(g["line"]) < 150:
            continue
        for n in moves:
            if n < len(g["line"]):
                out.append({"id": "{}@{}".format(g["name"], n), "move_number": n,
                            "setup": [gtp(p) for p in g["setup"]],
                            "moves": [[c, gtp(p)] for c, p in g["line"][:n - 1]],
                            "to_move": g["line"][n - 1][0]})
        if len({p["id"].split("@")[0] for p in out}) >= games:
            break
    return out


def state_of(p, superko=True):
    """The position as the bot's client holds it: GTPGameConnector's board enforces
    positional superko, which makes every legality check consult the game's history."""
    from AlphaGo import go
    state = go.GameState(enforce_superko=superko)
    if p["setup"]:
        state.place_handicaps([(COLS.index(v[0]), 19 - int(v[1:])) for v in p["setup"]])
    for c, v in p["moves"]:
        move = None if v == "pass" else (COLS.index(v[0]), 19 - int(v[1:]))
        state.do_move(move, go.BLACK if c == "B" else go.WHITE)
    state.set_current_player(go.BLACK if p["to_move"] == "B" else go.WHITE)
    return state


def short_groups(state):
    """Groups of the player to move and the opponent with 1 or 2 liberties."""
    board, libs = state.get_board(), state.get_liberty()
    seen, count = set(), 0
    for x in range(19):
        for y in range(19):
            if board[x, y] and (x, y) not in seen and 1 <= libs[x, y] <= 2:
                stack = [(x, y)]
                while stack:
                    a = stack.pop()
                    if a in seen:
                        continue
                    seen.add(a)
                    for q in ((a[0] + 1, a[1]), (a[0] - 1, a[1]), (a[0], a[1] + 1),
                              (a[0], a[1] - 1)):
                        if 0 <= q[0] < 19 and 0 <= q[1] < 19 and board[q] == board[x, y]:
                            stack.append(q)
                count += 1
    return count


def clock(fn, repeat):
    """Median wall time of fn() in ms over `repeat` calls."""
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def profiling_build():
    """True if the Cython extensions were built with profiling hooks (timings inflated)."""
    import cProfile
    import pstats
    from AlphaGo import go
    pr = cProfile.Profile()
    pr.enable()
    go.GameState().get_legal_moves()
    pr.disable()
    return any(".pyx" in key[0] for key in pstats.Stats(pr).stats)


def time_positions(positions, features, repeat, server=None):
    from AlphaGo.ai import ProbabilisticPolicyPlayer
    from AlphaGo.preprocessing.preprocessing import Preprocess
    full = Preprocess(features)
    single = {f: Preprocess([f]) for f in features}
    remote = None
    if server:
        from go_client import RemotePolicy
        remote = RemotePolicy(server, timeout=30)
    rows = []
    for p in positions:
        state = state_of(p)
        row = {"id": p["id"], "move_number": p["move_number"], "short_groups": short_groups(state),
               "planes": clock(lambda: full.state_to_tensor(state), repeat),
               "guard": clock(lambda: ProbabilisticPolicyPlayer._failed_ladder_extensions(state),
                              repeat),
               "legal": clock(lambda: state.get_legal_moves(include_eyes=False), repeat),
               "features": {f: clock(lambda: pre.state_to_tensor(state), repeat)
                            for f, pre in single.items()}}
        if remote:
            row["server"] = clock(lambda: remote.move_probabilities(state), repeat)
        rows.append(row)
    return rows


def report(rows):
    lines = []
    keys = ["planes", "guard", "legal"] + (["server"] if "server" in rows[0] else [])
    lines.append("{:<10} {:>4}  ".format("moves", "n") + "".join(
        "{:>16}".format(k + " med/p90") for k in keys))
    for lo, hi in STAGES:
        rs = [r for r in rows if lo <= r["move_number"] <= hi]
        if not rs:
            continue
        cells = []
        for k in keys:
            v = sorted(r[k] for r in rs)
            cells.append("{:>16}".format("{:.1f} / {:.1f}".format(
                statistics.median(v), v[int(0.9 * (len(v) - 1))])))
        lines.append("{:<10} {:>4}  ".format("{}-{}".format(lo, hi if hi < 999 else "+"),
                                             len(rs)) + "".join(cells))
    lines.append("\nper feature, ms (median / p90 / max over all positions):")
    for f in rows[0]["features"]:
        v = sorted(r["features"][f] for r in rows)
        lines.append("  {:<18} {:>6.2f} / {:>6.2f} / {:>7.2f}".format(
            f, statistics.median(v), v[int(0.9 * (len(v) - 1))], v[-1]))
    lines.append("\nslowest positions (planes ms, groups with 1-2 liberties):")
    for r in sorted(rows, key=lambda r: -r["planes"])[:6]:
        lines.append("  {:<44} {:>6.1f} ms  {:>3} short groups".format(
            r["id"][-44:], r["planes"], r["short_groups"]))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="b20c256",
                        help="whose input planes (play_tests/policy_loading.py name)")
    parser.add_argument("--positions", default=POSITIONS)
    parser.add_argument("--repeat", type=int, default=3, help="timings per position (median)")
    parser.add_argument("--server", help="also time the round trip to this go_server")
    parser.add_argument("--profile", metavar="OUT.prof", help="profile instead of timing")
    parser.add_argument("--build", nargs="+", metavar="FOLDER", help="rebuild the sample set")
    parser.add_argument("--games", type=int, default=40)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    if args.build:
        positions = build(args.build, args.games)
        os.makedirs(os.path.dirname(args.positions), exist_ok=True)
        with open(args.positions, "w") as f:
            json.dump({"positions": positions}, f)
        print("{} positions -> {}".format(len(positions), args.positions))
        return
    from play_tests.policy_loading import MODEL_SPECS
    spec = MODEL_SPECS[args.model]
    with open(os.path.join(ROOT, "play_tests", "models", args.model, spec["json"])) as f:
        features = json.load(f)["feature_list"]
    with open(args.positions) as f:
        positions = json.load(f)["positions"]

    if args.profile:
        import cProfile
        import pstats
        if not profiling_build():
            print("note: the extensions have no profiling hooks - Cython shows as single "
                  "calls (see the docstring)")
        pr = cProfile.Profile()
        pr.enable()
        time_positions(positions, features, 1)
        pr.disable()
        pr.dump_stats(args.profile)
        pstats.Stats(pr).sort_stats("tottime").print_stats(20)
        print("wrote {} (view: uv run snakeviz {})".format(args.profile, args.profile))
        return

    if profiling_build():
        print("WARNING: the extensions were built with profiling hooks - these timings are "
              "inflated; rebuild with python setup_cython.py build_ext --inplace --force")
    rows = time_positions(positions, features, args.repeat, args.server)
    lines = report(rows)
    print("{} positions, model {} ({} features)\n".format(len(rows), args.model, len(features)))
    print("\n".join(lines))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                            "commit": commit, "machine": platform.node(),
                            "platform": platform.platform(), "cpus": os.cpu_count(),
                            "model": args.model, "server": args.server,
                            "summary": lines}) + "\n")


if __name__ == "__main__":
    main()
