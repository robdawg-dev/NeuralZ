import argparse
import os

# Must happen before TensorFlow is imported (by AlphaGo.models.nn_util below) - this
# is a standalone CPU-only deployment, so no GPU should be touched even if the host
# machine happens to have one visible.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from AlphaGo.models.nn_util import NeuralNetBase
from interface.gtp_wrapper import run_gtp
from AlphaGo.ai import ProbabilisticPolicyPlayer

parser = argparse.ArgumentParser(
    description='Run a trained policy network as a GTP bot (CPU-only).')
parser.add_argument("model", help="Path to a JSON model file (from CNNPolicy.save_model())")
parser.add_argument("weights", help="Path to a .weights.h5 weights file matching model")
# Same defaults as play_tests/match_networks.py, so a GTP game plays like our playoffs.
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Temperature for sampled moves - lower leans harder toward the "
                         "top candidate. Default: 1.0")
parser.add_argument("--sample-ratio", type=float, default=0.5,
                    help="Sample (by policy output, with --temperature) only among the moves at "
                         "least this fraction as likely as the top move - so the bot varies "
                         "where the network sees a close call and plays a clearly preferred "
                         "move every time. Default: 0.5")
parser.add_argument("--sample-moves", type=int, default=20,
                    help="Sample for the bot's first N moves of each game, greedy after. "
                         "Counted from the bot's first genmove, so handicap stones and the "
                         "opponent's moves don't count. 0: always greedy. Default: 20 - with "
                         "--sample-ratio 0.5, on b20c256 this took a human's repeated opening "
                         "line away by ply 30-40 in every game while scoring 49.75%% (+/-2.5) "
                         "against its own greedy self over 400 games (workspace/sample_ratio/).")
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
    move_limit=args.max_moves, sample_ratio=args.sample_ratio,
    sample_moves=args.sample_moves)
run_gtp(player, name='NeuralZ', version=args.version,
        board_size=int(policy.model.inputs[0].shape[1]))
