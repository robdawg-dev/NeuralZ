"""Capture a policy network's raw outputs on a fixed set of positions, or compare them with a
previously captured reference - to check that an environment change (Python, TF, Keras, CUDA,
Docker image) leaves a trained model's behaviour unchanged.

Positions: fixed plies from the first games of each type in a selection list (test split, so
the model never trained on them), including handicap games.

    # in the OLD environment (from the repo root)
    python -m benchmarks._capture_policy_reference capture MODEL WEIGHTS ref_cpu.npz --device cpu
    # in the NEW environment
    python -m benchmarks._capture_policy_reference compare MODEL WEIGHTS ref_cpu.npz --device cpu
"""
import argparse
import os
import sys


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["capture", "compare"])
    parser.add_argument("model", help="model JSON (from CNNPolicy.save_model())")
    parser.add_argument("weights", help=".weights.h5 file matching the model")
    parser.add_argument("reference", help=".npz to write (capture) or read (compare)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="cpu: as go_server.py runs the bot. Default: cpu")
    parser.add_argument("--selection", default="workspace/prod_40m/selection/test.txt",
                        help="select_games list to take games from (path<TAB>moves<TAB>gtype)")
    parser.add_argument("--games-per-type", type=int, nargs=2, default=[6, 2],
                        metavar=("NORMAL", "HANDICAP"))
    parser.add_argument("--plies", type=int, nargs="+", default=[0, 20, 50, 90, 140, 200],
                        help="positions (before this many moves) to take from each game")
    parser.add_argument("--max-abs-diff", type=float, default=1e-3,
                        help="compare: largest allowed probability difference. Default: 1e-3")
    return parser.parse_args()


def _games(selection, n_normal, n_handicap):
    wanted = {"normal": n_normal, "handicap": n_handicap}
    games = []
    with open(selection) as f:
        for line in f:
            path, _moves, gtype = line.rstrip("\n").split("\t")
            if wanted.get(gtype, 0) > 0:
                games.append(path)
                wanted[gtype] -= 1
    return games


def _positions(games, plies):
    from AlphaGo.util import sgf_iter_states
    states, labels = [], []
    for path in games:
        with open(path) as f:
            text = f.read()
        for i, (state, _move, _player) in enumerate(sgf_iter_states(text, include_end=False)):
            if i in plies:
                states.append(state.copy())
                labels.append("{}@{}".format(path, i))
    return states, labels


def main():
    args = _parse_args()
    # before TensorFlow is imported, exactly as go_server.py does for the bot
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np
    from AlphaGo.models.nn_util import NeuralNetBase

    policy = NeuralNetBase.load_model(args.model)
    policy.model.load_weights(args.weights)

    if args.mode == "capture":
        games = _games(args.selection, *args.games_per_type)
        states, labels = _positions(games, set(args.plies))
    else:
        with np.load(args.reference) as ref:
            reference, labels = ref["probs"], list(ref["labels"])
        games = sorted({label.rsplit("@", 1)[0] for label in labels})
        by_label = dict(zip(*_positions(games, set(args.plies))[::-1]))
        states = [by_label[label] for label in labels]

    tensors = np.concatenate([policy.preprocessor.state_to_tensor(s) for s in states])
    probs = np.asarray(policy.forward(tensors), dtype=np.float32)
    print("{} positions from {} games, device={}".format(len(states), len(set(
        label.rsplit("@", 1)[0] for label in labels)), args.device))

    if args.mode == "capture":
        np.savez(args.reference, probs=probs, labels=np.array(labels))
        print("wrote", args.reference)
        return 0

    diff = np.abs(probs - reference)
    top1 = np.mean(probs.argmax(axis=1) == reference.argmax(axis=1))
    top5 = np.mean([set(np.argsort(p)[-5:]) == set(np.argsort(r)[-5:])
                    for p, r in zip(probs, reference)])
    print("max abs diff {:.2e}  mean abs diff {:.2e}".format(diff.max(), diff.mean()))
    print("top-1 move agrees in {:.1%} of positions, top-5 set in {:.1%}".format(top1, top5))
    ok = diff.max() <= args.max_abs_diff and top1 == 1.0 and top5 == 1.0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
