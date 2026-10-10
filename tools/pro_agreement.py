"""How often each network's first choice (and top 5) is the move a strong player actually
played, over a folder of real game records - a quick, cheap comparison of networks. Not
a strength measure: a network can match human moves well and still lose games (reading,
life and death), and only play against people gives a rank.

    uv run python tools/pro_agreement.py [folder] [--models b20c256 b15c192latest ...] \\
        [--by-game]

Default folder: workspace/famous/real (professional games, untracked - point it at any
folder of .sgf files). Each network judges every move of every game from the mover's side,
over all legal moves. Models by name from play_tests/policy_loading.py.
"""
import argparse
import glob
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT_MODELS = ["b20c256", "b15c192latest", "b10c128mb1024", "2016net"]


def agreement(policy, sgf_text):
    """[(rank of the played move, its probability), ...] for every move of the game."""
    from AlphaGo.util import sgf_iter_states
    out = []
    for state, move, player in sgf_iter_states(sgf_text, include_end=False):
        if move is None:
            continue
        state.set_current_player(player)
        probs = sorted(policy.eval_state(state, state.get_legal_moves(include_eyes=True)),
                       key=lambda t: -t[1])
        rank = next((k + 1 for k, (m, _p) in enumerate(probs) if m == move), None)
        out.append((rank, next((float(p) for m, p in probs if m == move), 0.0)))
    return out


def summarize(rows):
    n = len(rows)
    return {"moves": n, "top1": sum(r == 1 for r, _p in rows) / n,
            "top5": sum(r is not None and r <= 5 for r, _p in rows) / n,
            "mean_prob": statistics.mean(p for _r, p in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?",
                        default=os.path.join(ROOT, "workspace", "famous", "real"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--by-game", action="store_true", help="also one line per game")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from play_tests.policy_loading import load_policy
    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.sgf"), recursive=True))
    if not files:
        sys.exit("pro_agreement: no .sgf files in {}".format(args.folder))
    texts = {os.path.basename(f): open(f, encoding="utf-8", errors="replace").read()
             for f in files}
    print("{} games from {}".format(len(files), args.folder))
    print("{:<16} {:>6} {:>7} {:>7} {:>10}".format("model", "moves", "top-1", "top-5",
                                                   "mean prob"))
    for name in args.models:
        policy = load_policy(name)
        per_game = {g: agreement(policy, t) for g, t in texts.items()}
        s = summarize([r for rows in per_game.values() for r in rows])
        print("{:<16} {:>6} {:>7.1%} {:>7.1%} {:>10.1%}".format(
            name, s["moves"], s["top1"], s["top5"], s["mean_prob"]), flush=True)
        if args.by_game:
            for g, rows in per_game.items():
                gs = summarize(rows)
                print("    {:<36} {:>4} {:>7.1%} {:>7.1%}".format(g[:36], gs["moves"],
                                                                  gs["top1"], gs["top5"]))


if __name__ == "__main__":
    main()
