"""Draw a policy network's move probabilities over the positions of an SGF game: the board
with its stones, a circle on each candidate point colored by probability and labeled in %,
and a red square on the last move.

    # every position of a game, the deployed network
    uv run python tools/plot_heatmaps.py game.sgf out_dir
    # chosen positions (before moves 37 and 78), another network, every point labeled
    uv run python tools/plot_heatmaps.py game.sgf out_dir --moves 37 78 \\
        --model b15c192latest --all-points --cmap plasma
    # a model by files rather than by name
    uv run python tools/plot_heatmaps.py game.sgf out_dir --json model.json --weights w.h5

Writes before_move_NNN.png per position (the position before move NNN; 001 is the empty
or handicap board). Default style: points of 0.1% and more, color linear in probability.
--all-points labels every legal point - with a log color scale, so the tiny values still
show structure; values under 0.01% print as e.g. 3e-4 (meaning 0.0003%). --cmap takes any
matplotlib colormap; on a Go board prefer ones whose low end looks like neither a black
nor a white stone (plasma, viridis, YlOrRd for the default style). CPU only. Needs
matplotlib (uv sync --extra viz).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

THRESHOLD = 0.001  # default style: points of 0.1% and more


def label(pct):
    """A probability in percent as a short label: 40.4, 0.55, 3e-4."""
    if pct >= 1:
        return "{:.1f}".format(pct)
    if pct >= 0.01:
        return "{:.2f}".format(pct)
    mantissa, exp = "{:.0e}".format(pct).split("e")
    return "{}e{}".format(mantissa, int(exp))


def vertex(move, size=19):
    """An engine move (column, row from the top) as a GTP vertex: (14, 9) -> P10."""
    if move is None:
        return "pass"
    return "ABCDEFGHJKLMNOPQRST"[move[0]] + str(size - move[1])


def draw(state, probs, path, cmap="cool", all_points=False, title=None):
    """One heat map. probs: [(move, probability), ...] as eval_state returns them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm, Normalize

    from AlphaGo import go
    size = state.get_size()
    board = state.get_board()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, size + 1)
    ax.set_ylim(0, size + 1)
    ax.set_facecolor("#fec97b")
    ax.invert_yaxis()
    ax.tick_params(axis="both", length=0, width=0)
    ax.set_xticks(range(1, size + 1), range(1, size + 1))
    ax.set_yticks(range(1, size + 1), reversed(range(1, size + 1)))
    for i in range(size):
        ax.plot([1, size], [i + 1, i + 1], lw=1, color="k", zorder=0)
        ax.plot([i + 1, i + 1], [1, size], lw=1, color="k", zorder=0)
    pts = [(m, p) for m, p in probs if m is not go.PASS and (all_points or p >= THRESHOLD)]
    if pts:
        vals = np.array([p for _m, p in pts])
        if all_points:
            norm = LogNorm(vmin=max(vals.min(), 1e-6), vmax=max(vals.max(), 2e-6))
            colors = plt.get_cmap(cmap)(norm(np.maximum(vals, 1e-6)))
        else:
            colors = plt.get_cmap(cmap)(Normalize(vmin=0, vmax=vals.max())(vals))
        ax.scatter([m[0] + 1 for m, _p in pts], [m[1] + 1 for m, _p in pts], marker="o",
                   s=700, c=colors, edgecolor="k", zorder=1)
        for (m, p), c in zip(pts, colors):
            dark = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 0.5
            ax.annotate(label(p * 100), (m[0] + 1, m[1] + 1), color="white" if dark else "k",
                        ha="center", va="center", size=7.5 if all_points else 10, zorder=3)
    stones = [(x, y, board[x, y]) for x in range(size) for y in range(size)
              if board[x, y] != go.EMPTY]
    if stones:
        ax.scatter([x + 1 for x, _y, _c in stones], [y + 1 for _x, y, _c in stones],
                   marker="o", edgecolors="k", s=700, zorder=4,
                   c=["black" if c == go.BLACK else "white" for _x, _y, c in stones])
    history = state.get_history()
    if history and history[-1] is not go.PASS:
        ax.scatter(history[-1][0] + 1, history[-1][1] + 1, marker="s", color="r",
                   edgecolors="k", s=100, zorder=5)
    if title:
        ax.set_title(title, fontsize=14, loc="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load(args):
    if args.json:
        from AlphaGo.models.nn_util import NeuralNetBase
        policy = NeuralNetBase.load_model(args.json)
        policy.model.load_weights(args.weights)
        return policy
    from play_tests.policy_loading import load_policy
    return load_policy(args.model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sgf")
    parser.add_argument("out_directory")
    parser.add_argument("--moves", type=int, nargs="+",
                        help="positions to draw: before these move numbers. Default: all")
    parser.add_argument("--model", default="b20c256",
                        help="a name from play_tests/policy_loading.py. Default: b20c256")
    parser.add_argument("--json", help="model JSON instead of --model (needs --weights)")
    parser.add_argument("--weights", help="weights .h5 for --json")
    parser.add_argument("--all-points", action="store_true", help="label every legal point")
    parser.add_argument("--cmap", default="cool", help="matplotlib colormap. Default: cool")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # before TensorFlow loads
    if bool(args.json) != bool(args.weights):
        parser.error("--json and --weights go together")

    from AlphaGo.util import sgf_iter_states
    with open(args.sgf, encoding="utf-8", errors="replace") as f:
        text = f.read()
    os.makedirs(args.out_directory, exist_ok=True)
    policy = load(args)
    wanted = set(args.moves) if args.moves else None
    drawn = 0
    for n, (state, move, player) in enumerate(sgf_iter_states(text, include_end=False), 1):
        if wanted is not None and n not in wanted:
            continue
        state.set_current_player(player)
        probs = policy.eval_state(state, state.get_legal_moves(include_eyes=True))
        top = sorted(probs, key=lambda t: -t[1])
        rank = next((k + 1 for k, (m, _p) in enumerate(top) if m == move), None)
        name = "before_move_{:03d}.png".format(n)
        draw(state, probs, os.path.join(args.out_directory, name), args.cmap, args.all_points)
        print("{}: played {} (the network's choice #{}), its top {} at {:.1%}".format(
            name, vertex(move), rank, vertex(top[0][0]) if top else "-",
            top[0][1] if top else 0), flush=True)
        drawn += 1
    if wanted is not None and drawn < len(wanted):
        print("note: the game has fewer moves than some of --moves", file=sys.stderr)
    print("{} heat maps in {}".format(drawn, args.out_directory))


if __name__ == "__main__":
    main()
