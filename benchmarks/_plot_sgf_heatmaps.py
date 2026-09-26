"""Run a trained policy network over every position in an SGF game and save a heat-plot
visualization of its move-probability distribution for each one, using AlphaGo.util's
existing plot_network_output.

Requires matplotlib, an optional dependency (see pyproject.toml's [project.
optional-dependencies] "viz" extra) - not installed by a plain `uv sync`:
    uv sync --extra viz

Usage - model/weights/sgf/out_directory are positional, not flags. Runs fine
natively/CPU-only (confirmed - this script never touches the GPU itself), or inside
the container if you'd rather match the training scripts:
    uv run python -m benchmarks._plot_sgf_heatmaps \\
        play_tests/models/b15c192/model_restower_b15c192_convnorm.json \\
        play_tests/models/b15c192/weights.00074.weights.h5 \\
        play_tests/sgf/<match_dir>/<game>.sgf \\
        benchmarks/_heatmaps_some_game

--model/--weights load separately (not a single combined file) because every checkpoint
this project produces is saved that way (Data/New/*.json architecture + out_directory's
own weights.NNNNN.weights.h5) - matching how every trainer script here loads a model.
"""
import argparse
import os

import numpy as np

from AlphaGo.training.xla_workarounds import ensure_xla_conv_nhwc
# Must run before the first XLA compilation - same reason every trainer script here calls
# this at module load time, before model.compile()/first forward pass.
ensure_xla_conv_nhwc()

from AlphaGo import go  # noqa: E402
from AlphaGo.util import sgf_iter_states, plot_network_output, flatten_idx  # noqa: E402
from AlphaGo.models.policy import CNNPolicy  # noqa: E402
# Unused directly, but importing it registers ResTowerPolicy (via the @neuralnet
# decorator) so CNNPolicy.load_model() can find it by name in a model.json's "class"
# field - same reason every trainer script here imports this and never references it.
import AlphaGo.models.resnet_tower_policy  # noqa: E402,F401


class _BoardView:
    """Adapts a GameState to the (size, indexable-by-[i][j]) interface
    plot_network_output expects for its `board` argument.

    GameState itself exposes neither directly: .size is a Cython-internal short (not a
    plain Python attribute on the compiled build), and there's no __getitem__ - the
    actual public accessors are get_size() and get_board() (the latter returning a real
    (size, size) numpy array of go.EMPTY/BLACK/WHITE values, freshly built from the
    engine's internal board representation each call).
    """

    def __init__(self, state):
        self.size = state.get_size()
        self._board = state.get_board()

    def __getitem__(self, i):
        return self._board[i]


def _dense_scores(action_probs, size):
    """Convert eval_state's sparse [(action, prob), ...] (legal moves only, normalized
    to sum to 1 over just those) into the flat (size*size,) array plot_network_output
    expects, zero-filled at every position eval_state didn't list (illegal moves).
    """
    scores = np.zeros(size * size, dtype=np.float64)
    for action, prob in action_probs:
        if action is go.PASS:
            continue  # not a board position - nothing to plot it at
        scores[flatten_idx(action, size)] = prob
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Plot a trained policy network's move-probability heatmap for every "
                    "position in an SGF game.")
    parser.add_argument("model", help="Path to a model JSON file (from CNNPolicy.save_model())")
    parser.add_argument("weights", help="Path to a .weights.h5 checkpoint to load onto that model")
    parser.add_argument("sgf", help="Path to the SGF file to visualize")
    parser.add_argument("out_directory",
                        help="Directory to save heatmap PNGs into (created if missing)")
    parser.add_argument("--skip-final", action="store_true",
                        help="Skip the final (game-end, no next move to predict) position "
                             "that sgf_iter_states otherwise includes by default")
    parser.add_argument("--western-column-notation", action="store_true", default=True,
                        help="Passed straight through to plot_network_output. Default: True "
                             "(matches plot_network_output's own default)")
    args = parser.parse_args()

    os.makedirs(args.out_directory, exist_ok=True)

    print("loading model from {}".format(args.model))
    policy = CNNPolicy.load_model(args.model)
    print("loading weights from {}".format(args.weights))
    policy.model.load_weights(args.weights)

    with open(args.sgf) as f:
        sgf_string = f.read()

    # Imported here, not at module level: keeps the ensure_xla_conv_nhwc()/model-loading
    # path importable even before matplotlib is installed, and plot_network_output's own
    # import block already gives a clear, actionable error the first time it's actually
    # needed if it's still missing.
    import matplotlib.pyplot as plt

    move_number = 0
    for state, move, player in sgf_iter_states(sgf_string, include_end=not args.skip_final):
        action_probs = policy.eval_state(state)
        scores = _dense_scores(action_probs, state.get_size())
        board_view = _BoardView(state)
        history = state.get_history()
        output_file = "move_{:03d}.png".format(move_number)
        plot_network_output(scores, board_view, history, args.out_directory, output_file,
                            western_column_notation=args.western_column_notation)
        # plot_network_output builds a fresh figure (plt.subplots(...)) every call but
        # never closes it - across a whole game's worth of positions (SGFs commonly run
        # 150-300+ moves) those figures would all stay resident in memory at once
        # otherwise, since matplotlib keeps every open figure alive until explicitly
        # closed. Close it here instead of editing the shared utility function.
        plt.close('all')
        to_play = "black" if player == go.BLACK else "white" if player == go.WHITE else "n/a"
        print("saved {} (move {}, {} to play: {})".format(
            output_file, move_number, to_play, move))
        move_number += 1

    print("done - {} heatmaps saved to {}".format(move_number, args.out_directory))


if __name__ == '__main__':
    main()
