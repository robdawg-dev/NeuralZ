"""Play one self-play game with a saved policy network and write it out as an SGF,
so the actual game can be viewed in any SGF viewer - CPU-only.

Same self-play setup as time_get_move.py (GreedyPolicyPlayer against itself,
pass_when_offered so the game can actually end), but instead of timing moves this
just plays the game out and saves it, for visually inspecting what kind of
positions a model's own self-play produces (e.g. checking whether a run with a
lot of slow get_move() calls corresponds to a messy, ladder-heavy game).

Run natively (no Docker/GPU needed) via uv, e.g.:
    uv run python play_tests/self_play_to_sgf.py 2016net
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
from AlphaGo.util import save_gamestate_to_sgf  # noqa: E402
from play_tests.policy_loading import MODEL_SPECS, load_policy  # noqa: E402

SGF_DIR = os.path.join(os.path.dirname(__file__), "sgf")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODEL_SPECS), help="Which saved model to play")
    parser.add_argument("--move-limit", type=int, default=500,
                        help="Force-end the game after this many moves, in case self-play "
                             "never naturally reaches two passes. Default: 500")
    parser.add_argument("--max-moves", type=int, default=100,
                        help="Stop (and save whatever's been played so far) after this many "
                             "moves, even if the game hasn't ended. Default: 100")
    args = parser.parse_args()

    print("Loading '{}' (CPU only, CUDA_VISIBLE_DEVICES={})...".format(
        args.model, os.environ["CUDA_VISIBLE_DEVICES"]))
    policy = load_policy(args.model)
    board_size = policy.model.input_shape[1]
    player = GreedyPolicyPlayer(policy, pass_when_offered=True, move_limit=args.move_limit)

    state = go.GameState(size=board_size)

    n_moves = 0
    while not state.is_end_of_game() and n_moves < args.max_moves:
        mv = player.get_move(state)
        state.do_move(mv)
        n_moves += 1
        if n_moves % 10 == 0:
            print("  played {} moves...".format(n_moves))

    ended_naturally = state.is_end_of_game()
    print("Stopped after {} moves ({}).".format(
        n_moves, "reached end of game" if ended_naturally else "hit --max-moves"))

    os.makedirs(SGF_DIR, exist_ok=True)
    filename = "{}_{}moves_{}.sgf".format(args.model, n_moves, time.strftime("%Y%m%d_%H%M%S"))
    save_gamestate_to_sgf(state, SGF_DIR, filename,
                         black_player_name=args.model, white_player_name=args.model,
                         size=board_size)
    print("Saved to {}".format(os.path.join(SGF_DIR, filename)))


if __name__ == "__main__":
    main()
