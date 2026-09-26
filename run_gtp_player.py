import argparse
import os

# Must happen before TensorFlow is imported (by AlphaGo.models.nn_util below) - this
# is a standalone CPU-only deployment, so no GPU should be touched even if the host
# machine happens to have one visible.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from AlphaGo.models.nn_util import NeuralNetBase
import AlphaGo.models.resnet_tower_policy  # noqa: F401 (registers CNNPolicy + ResTowerPolicy)
from interface.gtp_wrapper import run_gtp
from AlphaGo.ai import ProbabilisticPolicyPlayer

parser = argparse.ArgumentParser(
    description='Run a trained policy network as a GTP bot (CPU-only).')
parser.add_argument("model", help="Path to a JSON model file (from CNNPolicy.save_model())")
parser.add_argument("weights", help="Path to a .weights.h5 weights file matching model")
# Same defaults as play_tests/match_networks.py, so a GTP game plays like our playoffs.
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature - lower is more greedy. Default: 1.0")
parser.add_argument("--greedy-start", type=int, default=2,
                    help="Play probabilistically (sampled by policy output, restricted to "
                         "--top-k/--top-k-responding) for this many total plies of the game "
                         "(both colors combined - see AlphaGo.ai.ProbabilisticPolicyPlayer), "
                         "then switch to greedy (highest-probability move) for the rest. "
                         "Default: 2 - one probabilistic move for whichever color moves "
                         "first, one for whichever moves second, in an even (no-handicap) "
                         "game. Handicap games place multiple stones before White's first "
                         "move, which alone already exceeds this default, so they fall "
                         "straight through to greedy from move 1 with no code change needed.")
parser.add_argument("--top-k", type=int, default=12,
                    help="Restrict probabilistic sampling to this many highest-probability "
                         "legal moves when the board is empty (i.e. this player is moving "
                         "first) - guarantees excluding any move outside the top K, unlike "
                         "temperature alone which only makes a weak move less likely, never "
                         "impossible. Default: 12, from surveying the network's own real "
                         "empty-board move distribution - see benchmarks/"
                         "_second_move_survey.py and the discussion that produced it.")
parser.add_argument("--top-k-responding", type=int, default=3,
                    help="Same as --top-k, but for when this player is responding to "
                         "something already on the board (a normal opponent move, or "
                         "handicap stones that didn't already push play past --greedy-start) "
                         "- these positions are typically more concentrated than an empty "
                         "board, so a narrower K is appropriate. Default: 3. Set to 1 for "
                         "fully deterministic (greedy) responses while keeping --top-k "
                         "probabilistic for this player's own first move.")
parser.add_argument("--max-moves", type=int, default=800,
                    help="Force a pass once this many moves have been played. Default: 800 - "
                         "300 was found to essentially never let a game reach a natural "
                         "two-pass end (0/50 in one measured match), leaving uncaptured dead "
                         "groups that skew naive scoring; 800 reached 100/100 natural endings.")
parser.add_argument("--version", default="0.3", help="Version string reported to the GTP "
                    "controller (the 'version' command). Default: 0.3")
args = parser.parse_args()

policy = NeuralNetBase.load_model(args.model)
policy.model.load_weights(args.weights)

player = ProbabilisticPolicyPlayer(
    policy, temperature=args.temperature, pass_when_offered=True,
    move_limit=args.max_moves, greedy_start=args.greedy_start,
    top_k=args.top_k, top_k_responding=args.top_k_responding)
run_gtp(player, name='NeuralZ', version=args.version)
