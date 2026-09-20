"""Time GreedyPolicyPlayer.get_move() across one real self-play game, CPU-only.

Loads one of the three saved policy networks in play_tests/models/ and has it
play a full game against itself, timing every single get_move() call (feature
generation + legal-move computation + the network forward pass - everything
that's real wait time) as the game naturally progresses from the opening
through the midgame to the endgame.

This answers a concrete question: if this were a bot and the opponent just
moved, about how long would you wait for its reply? A self-play game is used
(rather than one artificial fixed position) because it produces the kind of
positions - and the kind of variety across a game's length - that actually
come up in real play, unlike a single repeated position (which only shows
call-to-call consistency) or purely random moves (which tend to produce
oddly dense, capture-heavy positions unlike real games).

Run natively (no Docker/GPU needed) via uv, e.g.:
    uv run python play_tests/time_get_move.py resnet
"""
import os
# Must be set before any keras/tensorflow import (transitively pulled in by
# AlphaGo.models.*) - this is what actually forces CPU-only execution.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

# This script lives in play_tests/, one level below the repo root where the
# AlphaGo package lives - running it directly (python play_tests/x.py) doesn't
# put the repo root on sys.path the way `python -m` would.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlphaGo import go  # noqa: E402
from AlphaGo.ai import GreedyPolicyPlayer  # noqa: E402
from play_tests.policy_loading import MODEL_SPECS, load_policy  # noqa: E402


def percentile(sorted_times, pct):
    idx = min(len(sorted_times) - 1, int(round(pct / 100 * (len(sorted_times) - 1))))
    return sorted_times[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODEL_SPECS), help="Which saved model to time")
    parser.add_argument("--move-limit", type=int, default=500,
                        help="Force-end the game (and stop timing) after this many moves, in "
                             "case self-play never naturally reaches two passes. Default: 500")
    parser.add_argument("--max-moves", type=int, default=None,
                        help="Optional shorter cap on how many moves to actually time, for a "
                             "quicker sample instead of playing the whole game out. Default: "
                             "play (and time) the full game")
    args = parser.parse_args()

    print("Loading '{}' (CPU only, CUDA_VISIBLE_DEVICES={})...".format(
        args.model, os.environ["CUDA_VISIBLE_DEVICES"]))
    policy = load_policy(args.model)
    board_size = policy.model.input_shape[1]
    player = GreedyPolicyPlayer(policy, pass_when_offered=True, move_limit=args.move_limit)

    state = go.GameState(size=board_size)

    # One untimed warm-up call on the empty board: the first forward pass
    # through a freshly-loaded model can carry one-time costs (backend kernel
    # selection, lazy weight materialization) that a real long-running process
    # only pays once, not on every move.
    warmup_start = time.perf_counter()
    player.get_move(state)
    print("warm-up call: {:.1f} ms (not counted below)\n".format(
        (time.perf_counter() - warmup_start) * 1000))

    times = []
    print("{:>5}  {:>14}".format("move", "get_move (ms)"))
    while not state.is_end_of_game():
        if args.max_moves is not None and len(times) >= args.max_moves:
            break
        start = time.perf_counter()
        mv = player.get_move(state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        print("{:>5}  {:>14.1f}".format(len(times), elapsed_ms))
        state.do_move(mv)

    times_sorted = sorted(times)
    n = len(times_sorted)
    mean = sum(times_sorted) / n
    median = percentile(times_sorted, 50)
    print("\n{} moves timed. mean: {:.1f} ms   median: {:.1f} ms   p90: {:.1f} ms   "
          "min: {:.1f} ms   max: {:.1f} ms".format(
              n, mean, median, percentile(times_sorted, 90), times_sorted[0], times_sorted[-1]))

    # The slowest few moves matter for "worst-case wait", not just the average.
    slowest = sorted(enumerate(times, start=1), key=lambda p: p[1], reverse=True)[:5]
    print("slowest moves: " + ", ".join(
        "#{} ({:.0f}ms)".format(move_num, t) for move_num, t in slowest))


if __name__ == "__main__":
    main()
