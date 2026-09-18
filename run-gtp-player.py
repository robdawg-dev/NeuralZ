import argparse
from AlphaGo.models.nn_util import NeuralNetBase
import AlphaGo.models.resnet_tower_policy  # noqa: F401 (registers CNNPolicy + ResTowerPolicy)
from interface.gtp_wrapper import run_gtp
from AlphaGo.ai import ProbabilisticPolicyPlayer

parser = argparse.ArgumentParser(description='Run a trained policy network as a GTP bot.')
parser.add_argument("weights", help="Path to a .weights.h5 weights file matching --model")
parser.add_argument("--model", default="Data/New/model.json", help="Path to a JSON model file (default: Data/New/model.json)")  # noqa: E501
# Same defaults as play_tests/match_networks.py, so a GTP game plays like our playoffs.
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature - lower is more greedy. Default: 1.0")
parser.add_argument("--greedy-start", type=int, default=10,
                    help="Play greedily (argmax) for this many opening moves, then switch "
                         "to probabilistic sampling. Default: 10")
parser.add_argument("--max-moves", type=int, default=300,
                    help="Force a pass once this many moves have been played. Default: 300")
args = parser.parse_args()

policy = NeuralNetBase.load_model(args.model)
policy.model.load_weights(args.weights)

player = ProbabilisticPolicyPlayer(
    policy, temperature=args.temperature, pass_when_offered=True,
    move_limit=args.max_moves, greedy_start=args.greedy_start)
run_gtp(player, name='NeuralZ', version='0.2')

