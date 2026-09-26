# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
from cython.operator cimport dereference as d
import numpy as np
cimport numpy as np


cdef class Preprocess:

    ############################################################################
    #   Variables are declared in the .pxd file                                #
    #                                                                          #
    ############################################################################

    """
    # All feature processors
    cdef vector[preprocess_method] processors

    # Flag whether or not any features require 'lookahead' to groups_after
    cdef bool requires_groups_after

    # List with all string names of features used currently
    cdef list feature_list

    # Output tensor size
    cdef int output_dim

    # Board size, same convention as in GameState
    cdef short size, board_size
    """

    ############################################################################
    #   Tensor generating functions                                            #
    #                                                                          #
    ############################################################################

    cdef int get_board(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature encoding WHITE BLACK and EMPTY on separate planes.

           Note:
           - plane 0 always refers to the current player stones
           - plane 1 to the opponent stones
           - plane 2 to empty locations
        """

        cdef location_t location
        cdef group_ptr_t group
        cdef stone_t opponent = state.opponent_player
        cdef int plane

        # Loop over all locations on board
        for location in range(self.board_size):
            # Get color of stone from its group
            group = state.board[location]
            if d(group).color == stone_t.EMPTY:
                plane = 2
            elif d(group).color == opponent:
                plane = 1
            else:
                plane = 0

            tensor[offset + plane, location] = 1

        return offset + 3

    cdef int get_turns_since(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature encoding the age of the stone at each location up to 'maximum'

           Note:
           - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
           - EMPTY locations are all-zero features

           SETUP STONES (SGF AB/AW):
             place_handicap_stone ends in do_move, so setup stones enter moves_history
             exactly as played moves do and are aged like them here.

             For HANDICAP games that is the correct representation, not a defect. The
             stones really were placed in sequence immediately before White's first move,
             so they really are the most recent events on the board, and the live GTP path
             (place_handicaps -> place_handicap_stone -> do_move) builds the identical
             history at play time. Pushing them into plane 7 instead would describe a
             board where stones were played 7+ turns ago followed by 7 turns of nothing -
             a state that cannot occur. One residual difference: KataGo writes handicap AB
             in row-major order while GTP places from a fixed table, so the permutation of
             ages WITHIN the handicap block differs between training and play.

             For sgfpos/fork games it has no faithful representation: their AB block is a
             serialized mid-game board (median 50-69 stones) in raster order, all black
             then all white, with captured stones absent - no move order exists to encode.
             Those game types are therefore excluded at select time (sgf_preparation's
             DEFAULT_INCLUDE_GTYPES = normal, handicap), not handled here.
        """

        cdef location_t location
        cdef int age = offset + 7
        cdef set marked_locations = set()

        # Set all stones to max age
        for location in state.moves_history:
            if location != action_t.PASS and d(state.board[location]).color > stone_t.EMPTY:
                tensor[age, location] = 1

        # Loop over state.moves_history backwards so that recently-captured stones aren't counted.
        for age, location in enumerate(reversed(state.moves_history)):
            # If age has not been set yet (i.e. this is not a stone that was captured and re-placed)
            if location != action_t.PASS and location not in marked_locations and \
                    d(state.board[location]).color > stone_t.EMPTY:
                tensor[offset + age, location] = 1
                tensor[offset + 7, location] = 0
                marked_locations.add(location)

            # Break if reached maximum-1, since all other stones were set to maximum-1 already.
            if age >= 6:
                break

        return offset + 8

    cdef int get_last_moves(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature encoding the locations of each of the last 5 moves played, one plane per
           move (plane 0 = most recent move, plane 4 = 5th-most-recent).

           Note:
           - unlike turns_since, this is not tied to whether the stone is still on the board -
             a later capture does not erase the mark
           - a pass, or a move further back than history extends, leaves its plane all-zero

           NOT IN USE - DO NOT ENABLE FOR TRAINING WITHOUT READING THIS.
             This feature has the same setup-stone defect as get_turns_since (see there):
             SGF AB/AW stones live in moves_history and are marked here as though they were
             the last moves played. It is worse off than turns_since in two ways:

               1. There is no "older" bucket. turns_since has plane 7 to absorb setup
                  stones once 7 real moves exist; here all 5 planes are recency planes, so
                  a setup stone is either marked as a recent move or not represented. Since
                  the AB/AW order is a row-major board scan (see get_turns_since), that
                  means marking whichever stones happen to sort last as "the last 5 moves".
               2. There is no occupancy test, by design (see the first note above), so the
                  guard that lets turns_since reject captured stones cannot be reused to
                  reject setup stones either.

             With the corpus restricted to normal + handicap games (see get_turns_since)
             the serialized-board case no longer arises, but this feature is still untested
             for training use.

             Measured relationship to turns_since, for reference: with no captures, planes
             0-4 here are IDENTICAL to turns_since planes 0-4 (passes and setup stones
             included - both consume an age slot the same way). They diverge only when a
             capture or self-capture lands inside the last 5 plies: 7.30% of positions
             overall, but 0.47% before move 50 and 13.88% after move 200.
        """

        cdef int n_moves = state.moves_history.size()
        cdef int i
        cdef location_t location

        for i in range(min(n_moves, 5)):
            location = state.moves_history[n_moves - 1 - i]
            if location != action_t.PASS:
                tensor[offset + i, location] = 1

        return offset + 5

    cdef int get_liberties(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature encoding the number of liberties of the group connected to the stone at each
           location

           Note:
           - there is no zero-liberties plane; the 0th plane indicates groups in atari
           - the [maximum-1] plane is used for any stone with liberties greater than or equal to
             maximum
           - EMPTY locations are all-zero features
        """

        cdef int plane
        cdef group_ptr_t group
        cdef location_t location

        for location in range(self.board_size):
            # Get liberty count from group structure directly
            group = state.board[location]
            if d(group).color > stone_t.EMPTY:
                plane = min(d(group).count_liberty - 1, 7)
                tensor[offset + plane, location] = 1

        return offset + 8

    cdef int get_capture_size(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature encoding the number of opponent stones that would be captured by playing at
           each location, up to 'maximum'

           Note:
           - we currently *do* treat the 0th plane as "capturing zero stones" (that is, the 0th
             plane is used for legal moves that would not result in capture)
           - illegal move locations are all-zero features
        """

        cdef location_t location
        cdef short capture_size

        # Loop over all legal moves and set get capture size
        for location in state.legal_moves:
            capture_size = min(groups_after[location, 2], 7)
            tensor[offset + capture_size, location] = 1

        return offset + 8

    cdef int get_self_atari_size(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature encoding the size of the own-stone group that is put into atari by playing at
           a location

           Note:
           - the 0th plane is used for groups of size 1
           - the [maximum-1] plane is used for groups of size greater than or equal to maximum
        """

        cdef location_t location
        cdef short liberties, group_size

        # Loop over all groups on board
        for location in state.legal_moves:
            liberties = groups_after[location, 1]
            # This group is in atari if it has exactly 1 liberty
            if liberties == 1:
                group_size = min(groups_after[location, 0] - 1, 7)
                tensor[offset + group_size, location] = 1

        return offset + 8

    cdef int get_liberties_after(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature encoding what the number of liberties *would be* of the group connected to
           the stone *if* played at a location

           Note:
           - there is no zero-liberties plane; the 0th plane indicates groups in atari
           - the [maximum-1] plane is used for any stone with liberties greater than or equal to
             maximum
           - illegal move locations are all-zero features
        """

        cdef location_t location
        cdef short liberties

        # Loop over all legal moves
        for location in state.legal_moves:
            liberties = min(groups_after[location, 1] - 1, 7)
            tensor[offset + liberties, location] = 1

        return offset + 8

    cdef int get_ladder_capture(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature with 1 indicating that playing at a location would play out a ladder that
           results in capturing an opponent group.
        """

        cdef location_t location
        cdef group_ptr_t group

        # Search for any groups of the opponent player that have exactly 2 liberties
        for group in state.groups_set:
            if d(group).color != state.current_player and d(group).count_liberty == 2:
                # Try each "plausible" capture move on a copy of this state
                for location in get_plausible_capture_moves(state, group):
                    if is_ladder_capture_move(state.copy(), group, location):
                        tensor[offset, location] = 1

        return offset + 1

    cdef int get_ladder_escape(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature with 1 indicating that playing at a location would play out a ladder with the
           current player ultimately escaping.
        """

        cdef location_t location
        cdef group_ptr_t group

        # Search for any groups of the current player that are in atari
        for group in state.groups_set:
            if d(group).color == state.current_player and d(group).count_liberty == 1:
                # Try each "plausible" escape move on a copy of this state
                for location in get_plausible_escape_moves(state, group):
                    if is_ladder_escape_move(state.copy(), group, location):
                        tensor[offset, location] = 1

        return offset + 1

    cdef int get_sensibleness(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
        """

        cdef location_t location
        cdef vector[location_t] moves = state.get_sensible_moves()

        for location in moves:
            tensor[offset, location] = 1

        return offset + 1

    cdef int get_legal(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """Zero at all illegal moves, one at all legal moves. Unlike sensibleness, no eyes check.
        """

        cdef location_t location

        for location in state.legal_moves:
            tensor[offset, location] = 1

        return offset + 1

    cdef int get_response(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        # TODO - implement response pattern features as a lookup table ?

        return offset + 1

    cdef int get_save_atari(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """A feature wrapping a shallow GameState.is_ladder_escape() search. Effectively this
           feature encodes whether a group in atari can be saved for at least one more turn.
        """

        # TODO - pass additional args to feature processors, redirect this function to
        # get_ladder_escapes with less depth.

        cdef location_t location
        cdef group_ptr_t group

        # Search for any groups of the current player that are in atari
        for group in state.groups_set:
            if d(group).color != state.current_player and d(group).count_liberty == 2:
                # Try each "plausible" escape move on a copy of this state
                for location in get_plausible_escape_moves(state, group):
                    if is_ladder_escape_move(state.copy(), group, location, 2):
                        tensor[offset, location] = 1

        return offset + 1

    cdef int get_neighbor(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        # TODO - implement response pattern features as a lookup table ?

        return offset + 1

    cdef int get_nakade(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        # TODO - implement response pattern features as a lookup table ?

        return offset + 1

    cdef int get_response_12d(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        # TODO - implement response pattern features as a lookup table ?

        return offset + 1

    cdef int get_non_response_3x3(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        # TODO - implement response pattern features as a lookup table ?

        return offset + 1

    cdef int zeros(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """Plane filled with zeros
        """

        # Nothing to do; the tensor is zero-initialised in state_to_tensor. The return
        # below is what advances the plane offset for whatever feature comes next - it
        # previously sat on this comment line and was therefore never executed, making
        # this function return 0 and silently reset the offset. Harmless only while
        # "zeros" was last in the feature list; anything ordered after it was written
        # starting at plane 0, on top of the board planes. color() delegates here, so it
        # was broken for the value-net path regardless of ordering.
        return offset + 1

    cdef int ones(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """Plane filled with ones
        """

        # Taking advantage of numpy slice indexing
        tensor[offset, :] = 1

        return offset + 1

    cdef int color(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """Value net feature, plane with ones if active_player is black else zeros
        """

        if state.current_player == stone_t.BLACK:
            return self.ones(state, tensor, groups_after, offset)
        else:
            return self.zeros(state, tensor, groups_after, offset)

    cdef int ko(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset):  # noqa: E501
        """Ko feature (note: only aware of one-move-back ko, not superko)
        """

        if state.ko != -1:
            tensor[offset, state.ko] = 1

        return offset + 1

    ############################################################################
    #   init function                                                          #
    #                                                                          #
    ############################################################################

    def __init__(self, list feature_list, char size=19, verbose=False):
        self.size = size
        self.board_size = size * size

        # Recall that preprocess_method is the type of function pointers
        cdef preprocess_method processor
        cdef int i

        # Create a list with function pointers
        self.processors = vector[preprocess_method]()

        self.requires_groups_after = False

        # TODO - load response pattern files

        self.feature_list = feature_list
        self.output_dim = 0

        # Loop over feature_list add the corresponding function
        # and increment output_dim accordingly
        for i in range(len(feature_list)):
            feat = feature_list[i].lower()
            if feat == "board":
                processor = self.get_board
                self.output_dim += 3

            elif feat == "ones":
                processor = self.ones
                self.output_dim += 1

            elif feat == "turns_since":
                processor = self.get_turns_since
                self.output_dim += 8

            elif feat == "last_moves":
                processor = self.get_last_moves
                self.output_dim += 5

            elif feat == "liberties":
                processor = self.get_liberties
                self.output_dim += 8

            elif feat == "capture_size":
                processor = self.get_capture_size
                self.output_dim += 8
                self.requires_groups_after = True

            elif feat == "self_atari_size":
                processor = self.get_self_atari_size
                self.output_dim += 8
                self.requires_groups_after = True

            elif feat == "liberties_after":
                processor = self.get_liberties_after
                self.output_dim += 8
                self.requires_groups_after = True

            elif feat == "ladder_capture":
                processor = self.get_ladder_capture
                self.output_dim += 1

            elif feat == "ladder_escape":
                processor = self.get_ladder_escape
                self.output_dim += 1

            elif feat == "sensibleness":
                processor = self.get_sensibleness
                self.output_dim += 1

            elif feat == "zeros":
                processor = self.zeros
                self.output_dim += 1

            elif feat == "legal":
                processor = self.get_legal
                self.output_dim += 1

            elif feat == "response":
                raise NotImplementedError("response is not yet implemented; requires preprocessing and training redesign")  # noqa: E501

            elif feat == "save_atari":
                processor = self.get_save_atari
                self.output_dim += 1

            elif feat == "neighbor":
                raise NotImplementedError("neighbor is not yet implemented; requires preprocessing and training redesign")  # noqa: E501

            elif feat == "nakade":
                raise NotImplementedError("nakade is not yet implemented; requires preprocessing and training redesign")  # noqa: E501

            elif feat == "response_12d":
                raise NotImplementedError("response_12d is not yet implemented; requires preprocessing and training redesign")  # noqa: E501

            elif feat == "non_response_3x3":
                raise NotImplementedError("non_response_3x3 is not yet implemented; requires preprocessing and training redesign")  # noqa: E501

            elif feat == "color":
                processor = self.color
                self.output_dim += 1

            elif feat == "ko":
                processor = self.ko
                self.output_dim += 1
            else:
                # Incorrect feature name
                raise ValueError("uknown feature: %s" % feat)

            self.processors.push_back(processor)

    ############################################################################
    #   public cpdef function                                                  #
    #                                                                          #
    ############################################################################

    cpdef np.ndarray[onehot_t, ndim=4] state_to_tensor(self, GameState state):
        """Convert a GameState to a tensor of one-hot features, shape
        (1, size, size, n_features) - channels_last, as expected by the network.
        """

        cdef preprocess_method proc
        cdef np.ndarray[lookahead_t, ndim=2] groups_after

        # Create complete array now instead of concatenate later
        cdef np.ndarray[onehot_t, ndim=2] np_tensor = \
            np.zeros((self.output_dim, self.board_size), dtype=np.uint8)

        cdef int offset = 0

        # Get char array with next move information
        if self.requires_groups_after:
            groups_after = get_groups_after(state)
        else:
            groups_after = np.zeros((1, 1), dtype=np.uint16)

        # Loop over all processors and generate tensor
        for proc in self.processors:
            offset = proc(self, state, np_tensor, groups_after, offset)

        # Reshape result from (features, board_size) to (1, features, size, size), i.e. with a
        # 2D board for input to a convolutional network and a singleton 'batch' dimension, then
        # move the channel axis to the end for channels_last (this network's data_format).
        return np_tensor.reshape((1, self.output_dim, self.size, self.size)).transpose((0, 2, 3, 1))

    ############################################################################
    #   public def function (Python)                                           #
    #                                                                          #
    ############################################################################

    def get_output_dimension(self):
        """return output_dim, the amount of planes an output tensor will have
        """
        return self.output_dim

    def get_feature_list(self):
        """return feature list
        """
        return self.feature_list


############################################################################
#   "Groups after" helper functions for lookahead without copying state    #
#                                                                          #
############################################################################

cdef np.ndarray[lookahead_t, ndim=2] get_groups_after(GameState state):
    """Without creating a copy of the state, compute features of the resulting board state
       IF a stone were played at each legal location. Three features are computed and stored in
       a board_size x 3 array. Values are only computed at legal move locations.

       That is, groups_after[loc, STONE] contains the size of the group that would be formed
       if the current_player were to play a stone at 'loc', hence the name groups_after

       Returns a numpy array of size (board_size, 3):
       - groups_after[loc, 0] = resulting group size by playing at loc
       - groups_after[loc, 1] = number of remaining liberties of group by playing at loc
       - groups_after[loc, 2] = number of stones captured by playing at loc
    """

    cdef location_t loc
    cdef np.ndarray[lookahead_t, ndim=2] result = np.zeros((state.board_size, 3), dtype=np.uint16)

    # Call get_groups_after_at() for each legal move
    for loc in state.legal_moves:
        get_groups_after_at(state, loc, result)

    return result


cdef void get_groups_after_at(GameState state, location_t loc,
                              np.ndarray[lookahead_t, ndim=2] result):
    """Compute 'groups_after' results at a single location, which must be a legal move, writing
       directly into result[loc, :] (see the .pxd docstring for why this doesn't allocate and
       return a fresh array - it runs once per legal move).
    """

    cdef short capture_before = \
        state.capture_black if state.current_player == stone_t.WHITE else state.capture_white
    cdef short capture_after

    with state.try_stone(loc, False):
        result[loc, 0] = d(state.board[loc]).count_stones
        result[loc, 1] = d(state.board[loc]).count_liberty

        capture_after = \
            state.capture_black if state.current_player == stone_t.WHITE else state.capture_white
        result[loc, 2] = capture_after - capture_before
