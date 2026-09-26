import os
import itertools
import numpy as np
import sgf
from AlphaGo import go

# for board location indexing
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
REV_LETTERS = 'SRQPONMLKJIHGFEDCBA'

# SGF properties that change the position, or whose turn it is, WITHOUT being a move.
# Legal on any node, not just the root. Everything else that can appear on a moveless
# node (C, N, markup such as CR/LB/MA/SL/SQ/TR/AR/LN/DD, position judgements DM/GB/GW/UC/V,
# timing BL/WL/OB/OW, FG/PM/VW) is annotation with no board effect.
_BOARD_ALTERING_PROPERTIES = frozenset(('AB', 'AW', 'AE', 'PL'))


def flatten_idx(position, size):
    (x, y) = position
    return x * size + y


def _parse_sgf_move(node_value):
    """Given a well-formed move string, return either PASS or the (x, y) position
    """
    if node_value == '' or node_value == 'tt':
        return go.PASS
    else:
        # GameState expects (x, y) where x is column and y is row
        col = LETTERS.index(node_value[0].upper())
        row = LETTERS.index(node_value[1].upper())
        return (col, row)


def _sgf_init_gamestate(sgf_root):
    """Helper function to set up a GameState object from the root node
    of an SGF file
    """
    props = sgf_root.properties
    s_size = props.get('SZ', ['19'])[0]
    s_player = props.get('PL', ['B'])[0]
    # init board with specified size
    gs = go.GameState(int(s_size), enforce_superko=False)
    # handle 'add black' property
    if 'AB' in props:
        for stone in props['AB']:
            gs.place_handicap_stone(_parse_sgf_move(stone), go.BLACK)
    # handle 'add white' property
    if 'AW' in props:
        for stone in props['AW']:
            gs.place_handicap_stone(_parse_sgf_move(stone), go.WHITE)
    # setup done; set player according to 'PL' property
    gs.set_current_player(go.BLACK if s_player == 'B' else go.WHITE)
    return gs


def sgf_to_gamestate(sgf_string):
    """Creates a GameState object from the first game in the given collection
    """
    # Don't Repeat Yourself; parsing handled by sgf_iter_states
    for (gs, move, player) in sgf_iter_states(sgf_string, True):
        pass
    # gs has been updated in-place to the final state by the time
    # sgf_iter_states returns
    return gs


def save_gamestate_to_sgf(gamestate, path, filename, black_player_name='Unknown',
                          white_player_name='Unknown', size=19, komi=7.5):
    """Creates a simplified sgf for viewing playouts or positions
    """
    str_list = []
    # Game info
    str_list.append('(;GM[1]FF[4]CA[UTF-8]')
    str_list.append('SZ[{}]'.format(size))
    str_list.append('KM[{}]'.format(komi))
    str_list.append('PB[{}]'.format(black_player_name))
    str_list.append('PW[{}]'.format(white_player_name))
    cycle_string = 'BW'
    # Handle handicaps
    handicaps = gamestate.get_handicaps()
    if len(handicaps) > 0:
        cycle_string = 'WB'
        str_list.append('HA[{}]'.format(len(handicaps)))
        str_list.append(';AB')
        for handicap in handicaps:
            str_list.append('[{}{}]'.format(LETTERS[handicap[0]].lower(),
                                            REV_LETTERS[handicap[1]].lower()))
    # Move list (skip the leading handicap placements - already written above as AB[] stones)
    for move, color in zip(gamestate.get_history()[len(handicaps):], itertools.cycle(cycle_string)):
        # Move color prefix
        str_list.append(';{}'.format(color))
        # Move coordinates
        if move is None:
            str_list.append('[tt]')
        else:
            str_list.append('[{}{}]'.format(LETTERS[move[0]].lower(), REV_LETTERS[move[1]].lower()))
    str_list.append(')')
    with open(os.path.join(path, filename), "w") as f:
        f.write(''.join(str_list))


def sgf_iter_states(sgf_string, include_end=True):
    """Iterates over (GameState, move, player) tuples in the first game of the given SGF file.

    Ignores variations - only the main line is returned.
    The state object is modified in-place, so don't try to, for example, keep track of it
    through time

    If include_end is False, the final tuple yielded is the penultimate state, but the state
    will still be left in the final position at the end of iteration because 'gs' is modified
    in-place the state. See sgf_to_gamestate
    """
    collection = sgf.parse(sgf_string)
    game = collection[0]
    gs = _sgf_init_gamestate(game.root)
    if game.rest is not None:
        for node in game.rest:
            props = node.properties
            if 'W' in props:
                move = _parse_sgf_move(props['W'][0])
                player = go.WHITE
            elif 'B' in props:
                move = _parse_sgf_move(props['B'][0])
                player = go.BLACK
            else:
                # A node carrying neither W nor B is not a move. Two very different
                # cases hide here, and they must not be treated alike.
                #
                # Falling through used to reuse the PREVIOUS node's move/player and
                # replay that move: an IllegalMove that truncated the game, or an
                # UnboundLocalError that dropped the whole file when such a node came
                # first. Measured at 0/60,136 KataGo selfplay training games but 89/90
                # rating games, and live for KGS/GoGoD records.
                board_altering = _BOARD_ALTERING_PROPERTIES.intersection(props)
                if board_altering:
                    # AB/AW/AE/PL outside the root node change the position (or whose
                    # turn it is) without being a move. We cannot apply them, and
                    # skipping them would leave the board silently out of step with the
                    # record - every later move would then be replayed against a
                    # position that never occurred. That is the same desync that makes
                    # "skip the suicide and carry on" corrupting, so it gets the same
                    # treatment: stop here and let the caller keep the prefix.
                    # go.IllegalMove deliberately: every caller already catches it and
                    # responds by keeping the prefix and dropping the remainder, which
                    # is exactly the desired behaviour. A dedicated exception type
                    # belongs with the truncation-reason classification work.
                    raise go.IllegalMove(
                        "board-altering setup properties {} on a non-root node are not "
                        "supported; the board cannot be kept in sync with the record"
                        .format(sorted(board_altering)))
                # Pure annotation (C, N, markup, timing, ...) - no board effect, safe to
                # skip. This is 100% of the moveless nodes measured across every corpus
                # to hand (81/81 in KataGo rating games are a terminal C[...result=...]).
                continue
            yield (gs, move, player)
            # update state to n+1
            gs.do_move(move, player)
    if include_end:
        yield (gs, None, None)


def plot_network_output(scores, board, history, out_directory, output_file,
                        should_plot=False, western_column_notation=True):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError as e:
        print(
            'Failed to import matplotlib. This is an optional dependency of ' +
            'the RocAlphaGo project, so it is not included in the requirements file. ' +
            'You must install matplotlib yourself to use the plotting functions.')
        raise e

    from distutils.version import StrictVersion
    matplotlib_version = matplotlib.__version__
    if StrictVersion(matplotlib_version) < StrictVersion('1.5.1'):
        print('Your version of matplotlib might not support our use of it')

    # Initial matplotlib setup
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlim([0, board.size + 1])
    plt.ylim([0, board.size + 1])

    # Wooden background color
    # set_axis_bgcolor was renamed to set_facecolor in matplotlib 2.0, then removed in 3.0
    ax.set_facecolor('#fec97b')
    plt.gca().invert_yaxis()

    # Setup ticks
    ax.tick_params(axis='both', length=0, width=0)
    # Western notation has the origin at the lower-left
    if western_column_notation:
        plt.xticks(range(1, board.size + 1), range(1, board.size + 1))
        plt.yticks(range(1, board.size + 1), reversed(range(1, board.size + 1)))
    # Traditional has the origin at the upper-left and uses letters minus 'I' along the top
    else:
        ax.xaxis.tick_top()
        plt.xticks(range(1, board.size + 1), [x for x in LETTERS[:board.size + 1] if x != 'I'])
        plt.yticks(range(1, board.size + 1), range(1, board.size + 1))

    # Draw grid
    for i in range(board.size):
        plt.plot([1, board.size], [i + 1, i + 1], lw=1, color='k', zorder=0)
    for i in range(board.size):
        plt.plot([i + 1, i + 1], [1, board.size], lw=1, color='k', zorder=0)

    # Display network heat plots
    reshaped = np.reshape(scores, (board.size, board.size))
    score_x_coords = []
    score_y_coords = []
    score_values = []
    for i in range(board.size):
        for j in range(board.size):
            if reshaped[i][j] * 100 >= 0.1:
                score_x_coords.append(i + 1)
                score_y_coords.append(j + 1)
                score_values.append(reshaped[i][j])
    min_seen = np.amin(scores)
    max_seen = np.amax(scores)
    norm = matplotlib.colors.Normalize(vmin=min_seen, vmax=max_seen)
    coloring = cm.ScalarMappable(norm=norm, cmap=cm.cool).to_rgba(score_values)
    plt.scatter(score_x_coords, score_y_coords, marker='o', s=700,
                c=coloring, edgecolor='k', zorder=1)

    # Display network scores on heat plots
    for i, txt in enumerate(score_values):
        ax.annotate('{0:.1f}'.format(txt * 100), (score_x_coords[i], score_y_coords[i]),
                    color='k', ha='center',
                    va='center', size=10, zorder=3)

    # Display stones already played
    stone_x_coords = []
    stone_y_coords = []
    stone_colors = []
    for i in range(board.size):
        for j in range(board.size):
            if board[i][j] != go.EMPTY:
                stone_x_coords.append(i + 1)
                stone_y_coords.append(j + 1)
                if board[i][j] == go.BLACK:
                    stone_colors.append(matplotlib.colors.to_rgb('black'))
                else:
                    stone_colors.append(matplotlib.colors.to_rgb('white'))
    plt.scatter(stone_x_coords, stone_y_coords, marker='o', edgecolors='k',
                s=700, c=stone_colors, zorder=4)

    # Place red marker on last move if it exists
    if len(history) != 0:
        # If last move was not pass
        if history[-1] != go.PASS:
            last_move = history[-1]
            x_coord = last_move[0] + 1
            y_coord = last_move[1] + 1
            last_move = (x_coord, y_coord)
            plt.scatter(last_move[0], last_move[1], marker='s', color='r',
                        edgecolors='k', s=100, zorder=5)

    if output_file is not None:
        plt.savefig(os.path.join(out_directory, output_file), bbox_inches='tight')
    if should_plot:
        plt.show()
    plt.close()
