"""Play two saved policy networks against each other over N games, CPU-only.

Loads two models from play_tests/models/ and plays --num-games games between
them, alternating who plays Black each game (so results aren't skewed by
Black's inherent first-move advantage). Each player uses
AlphaGo.ai.ProbabilisticPolicyPlayer with a greedy_start: it samples moves
proportional to the policy's output probability for the first --greedy-start
moves of the game (so successive games actually differ from each other),
then plays the highest-probability move for the rest (so most of the game
reflects the network's real preferences, not random sampling noise).

Every game is saved as an SGF, and a results.json summarizing every game
(including the winner as determined by GameState.get_score()/
get_winner_color() - untrusted for now, meant to be cross-checked later
against KataGo's own scoring of the same SGFs) is written alongside them, all
under a new directory per run in play_tests/sgf/.

Run natively (no Docker/GPU needed) via uv, e.g.:
    uv run python play_tests/match_networks.py simplecnn resnet --num-games 10
"""
import os
# Must be decided before any keras/tensorflow import (transitively pulled in
# by AlphaGo.models.*) - this is what actually forces CPU-only execution, so
# it can't wait for argparse to run normally. --gpu is checked here, straight
# out of sys.argv, rather than deferred to the real parser below.
if "--gpu" not in __import__("sys").argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

# This script lives in play_tests/, one level below the repo root where the
# AlphaGo package lives - running it directly (python play_tests/x.py) doesn't
# put the repo root on sys.path the way `python -m` would.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlphaGo import go  # noqa: E402
from AlphaGo.ai import ProbabilisticPolicyPlayer  # noqa: E402
from AlphaGo.util import save_gamestate_to_sgf  # noqa: E402
from play_tests.policy_loading import MODEL_SPECS, load_policy  # noqa: E402

SGF_DIR = os.path.join(os.path.dirname(__file__), "sgf")


def play_one_game(black_policy, white_policy, board_size, args, rng_seed):
    np.random.seed(rng_seed)
    black_player = ProbabilisticPolicyPlayer(
        black_policy, temperature=args.temperature, pass_when_offered=True,
        move_limit=args.max_moves, greedy_start=args.greedy_start,
        top_k=args.top_k, top_k_responding=args.top_k_responding)
    white_player = ProbabilisticPolicyPlayer(
        white_policy, temperature=args.temperature, pass_when_offered=True,
        move_limit=args.max_moves, greedy_start=args.greedy_start,
        top_k=args.top_k, top_k_responding=args.top_k_responding)

    state = go.GameState(size=board_size)
    n_moves = 0
    game_start = time.perf_counter()
    while not state.is_end_of_game() and n_moves < args.max_moves:
        active_player = black_player if state.get_current_player() == go.BLACK else white_player
        mv = active_player.get_move(state)
        state.do_move(mv)
        n_moves += 1
        if n_moves % 10 == 0:
            print("    ...{} moves ({:.1f}s elapsed)".format(
                n_moves, time.perf_counter() - game_start), flush=True)

    ended_naturally = state.is_end_of_game()
    score = state.get_score(komi=args.komi)
    winner_color = state.get_winner_color(komi=args.komi)
    return state, n_moves, ended_naturally, score, winner_color


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_a", choices=sorted(MODEL_SPECS))
    parser.add_argument("model_b", choices=sorted(MODEL_SPECS))
    parser.add_argument("--num-games", type=int, default=10,
                        help="Total games to play, alternating who plays Black each game. "
                             "Default: 10 (5 as Black each)")
    parser.add_argument("--max-moves", type=int, default=800,
                        help="Cap each game at this many total moves. Default: 800 - at 300, "
                             "essentially no games reach a natural two-pass end (confirmed "
                             "0/50 in match_b10c128_vs_2016net_20260912_145559), leaving "
                             "uncaptured dead groups on the board that naive area scoring "
                             "miscounts as alive; 800 was confirmed to reach 100/100 natural "
                             "endings (match_b10c128_vs_2016net_20260912_154303).")
    parser.add_argument("--greedy-start", type=int, default=10,
                        help="Play probabilistically (sampled by policy output, restricted to "
                             "--top-k/--top-k-responding) for this many total plies of the "
                             "game (both colors combined - see "
                             "AlphaGo.ai.ProbabilisticPolicyPlayer), then switch to greedy "
                             "(highest-probability move) for the rest. Default: 10 - not the "
                             "2 used for the live GTP bot (run_gtp_player.py), which only "
                             "needs one probabilistic move per color to avoid a human "
                             "trivially replaying the same opening. A match run needs many "
                             "distinct games instead: since every probabilistic ply past the "
                             "first uses --top-k-responding, the number of distinct possible "
                             "games is top_k * top_k_responding^(greedy_start-1) - only 36 "
                             "at greedy_start=2 (12*3), guaranteeing repeats across 100 games "
                             "by pigeonhole, versus ~236K at greedy_start=10 (12*3^9) - under "
                             "~2%% expected chance of even one repeated pair in 100 games.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for the probabilistic opening moves. Default: 1.0")
    parser.add_argument("--top-k", type=int, default=12,
                        help="Restrict probabilistic sampling to this many highest-probability "
                             "legal moves when the board is empty (i.e. this player is moving "
                             "first). Default: 12 - see benchmarks/_second_move_survey.py and "
                             "the discussion that produced it.")
    parser.add_argument("--top-k-responding", type=int, default=3,
                        help="Same as --top-k, but for when a player is responding to "
                             "something already on the board. Default: 3 - these positions "
                             "are typically more concentrated than an empty board.")
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for reproducibility - each game gets seed+game_index. "
                             "Default: unseeded (a different match every run)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use the GPU instead of forcing CPU-only. Checked directly out of "
                             "sys.argv before this parser even runs, since CUDA_VISIBLE_DEVICES "
                             "has to be set before tensorflow is imported - present here mainly "
                             "so --help documents it and args.gpu reflects reality.")
    args = parser.parse_args()

    print("Loading '{}' and '{}' ({})...".format(
        args.model_a, args.model_b, "GPU" if args.gpu else "CPU only"))
    policy_a = load_policy(args.model_a)
    policy_b = load_policy(args.model_b)
    board_size = policy_a.model.input_shape[1]

    run_dir = os.path.join(SGF_DIR, "match_{}_vs_{}_{}".format(
        args.model_a, args.model_b, time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(run_dir, exist_ok=True)

    base_seed = args.seed if args.seed is not None else np.random.randint(0, 2**31 - 1)
    games = []
    wins = {args.model_a: 0, args.model_b: 0}

    for i in range(args.num_games):
        # Alternate who plays Black each game.
        a_plays_black = (i % 2 == 0)
        black_name, white_name = (args.model_a, args.model_b) if a_plays_black \
            else (args.model_b, args.model_a)
        black_policy, white_policy = (policy_a, policy_b) if a_plays_black \
            else (policy_b, policy_a)

        print("Game {}/{}: Black={} White={} ...".format(
            i + 1, args.num_games, black_name, white_name))
        state, n_moves, ended_naturally, score, winner_color = play_one_game(
            black_policy, white_policy, board_size, args, rng_seed=base_seed + i)

        winner_name = black_name if winner_color == go.BLACK else white_name
        wins[winner_name] += 1

        sgf_filename = "game{:02d}_black-{}_white-{}.sgf".format(i + 1, black_name, white_name)
        save_gamestate_to_sgf(state, run_dir, sgf_filename,
                              black_player_name=black_name, white_player_name=white_name,
                              size=board_size, komi=args.komi)

        print("  {} moves ({}), score={:+.1f}, winner={} (playing {})".format(
            n_moves, "ended naturally" if ended_naturally else "hit move cap",
            score, winner_name, "Black" if winner_color == go.BLACK else "White"))

        games.append({
            "game_index": i + 1,
            "black_model": black_name,
            "white_model": white_name,
            "moves_played": n_moves,
            "ended_naturally": ended_naturally,
            "score": score,
            "winner_color": "BLACK" if winner_color == go.BLACK else "WHITE",
            "winner_model": winner_name,
            "sgf_file": sgf_filename,
        })

    results = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "num_games": args.num_games,
        "max_moves": args.max_moves,
        "greedy_start": args.greedy_start,
        "top_k": args.top_k,
        "top_k_responding": args.top_k_responding,
        "temperature": args.temperature,
        "komi": args.komi,
        "base_seed": base_seed,
        "scoring_method": "GameState.get_score()/get_winner_color() (area scoring) - accuracy "
                          "not yet independently verified; cross-check against KataGo scoring "
                          "of these SGFs before trusting these results.",
        "wins": wins,
        "games": games,
    }
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n{} wins: {}   {} wins: {}".format(
        args.model_a, wins[args.model_a], args.model_b, wins[args.model_b]))
    print("Saved {} SGFs and results.json to {}".format(args.num_games, run_dir))


if __name__ == "__main__":
    main()
