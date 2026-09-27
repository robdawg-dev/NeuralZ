from tests import parseboard
import unittest
import AlphaGo.go as go
from AlphaGo.go import GameState
from AlphaGo.util import flatten_idx


class TestKo(unittest.TestCase):

    def test_standard_ko(self):

        gs = GameState(size=9)

        gs.do_move((1, 0))  # B
        gs.do_move((2, 0))  # W
        gs.do_move((0, 1))  # B
        gs.do_move((3, 1))  # W
        gs.do_move((1, 2))  # B
        gs.do_move((2, 2))  # W
        gs.do_move((2, 1))  # B

        gs.do_move((1, 1))  # W trigger capture and ko

        self.assertEqual(gs.get_captures_black(), 1)
        self.assertEqual(gs.get_captures_white(), 0)

        self.assertFalse(gs.is_legal((2, 1)))

        gs.do_move((5, 5))
        gs.do_move((5, 6))

        self.assertTrue(gs.is_legal((2, 1)))

    def test_snapback_is_not_ko(self):

        gs = GameState(size=5)

        # B o W B .
        # W W B . .
        # . . . . .
        # . . . . .
        # . . . . .
        # here, imagine black plays at 'o' capturing
        # the white stone at (2, 0). White may play
        # again at (2, 0) to capture the black stones
        # at (0, 0), (1, 0). this is 'snapback' not 'ko'
        # since it doesn't return the game to a
        # previous position
        B = [(0, 0), (2, 1), (3, 0)]
        W = [(0, 1), (1, 1), (2, 0)]
        for (b, w) in zip(B, W):
            gs.do_move(b)
            gs.do_move(w)
        # do the capture of the single white stone
        gs.do_move((1, 0))
        # there should be no ko
        self.assertIsNone(gs.get_ko_location())
        self.assertTrue(gs.is_legal((2, 0)))
        # now play the snapback
        gs.do_move((2, 0))
        # check that the numbers worked out
        self.assertEqual(gs.get_captures_black(), 2)
        self.assertEqual(gs.get_captures_white(), 1)

    def test_positional_superko(self):

        # test with enforce_superko=False
        gs = GameState(size=9, enforce_superko=False)

        move_list = [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (2, 2), (3, 4), (2, 1), (3, 3),
                     (3, 1), (3, 2), (3, 0), (4, 2), (1, 1), (4, 1), (8, 0), (4, 0), (8, 1), (0, 2),
                     (8, 2), (0, 1), (8, 3), (1, 0), (8, 4), (2, 0), (0, 0)]

        for move in move_list:
            gs.do_move(move)
        self.assertTrue(gs.is_legal((1, 0)))

        # test with enforce_superko=True
        gs = GameState(size=9, enforce_superko=True)
        for move in move_list:
            gs.do_move(move)
        self.assertFalse(gs.is_legal((1, 0)))

    # The same superko position, but reached in a HANDICAP game.
    #
    # Handicap stones are all Black and are pushed onto moves_history, so after N of them
    # WHITE moves first - which flips the parity of every subsequent move relative to an
    # even game. is_positional_superko()'s Part 1 pre-filter ("has the current player ever
    # played here? if not, superko is impossible") slices moves_history by that parity, and
    # used to derive it assuming Black always moves first. In a handicap game that made it
    # scan the OPPONENT's moves, miss the current player's own, and return "not superko"
    # without ever running the Part 2 hash check - letting a genuine superko violation
    # through. Only the GTP player sets enforce_superko=True, so this never affected
    # training data, but it did affect handicap games played on KGS.
    #
    # Colours are inverted relative to test_positional_superko above (White moves first
    # here), so the repeated position is the same shape with the colours swapped and the
    # player facing the superko is Black.
    HANDICAP_SUPERKO_MOVES = [
        (0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (2, 2), (3, 4), (2, 1), (3, 3),
        (3, 1), (3, 2), (3, 0), (4, 2), (1, 1), (4, 1), (8, 0), (4, 0), (8, 1), (0, 2),
        (8, 2), (0, 1), (8, 3), (1, 0), (8, 4), (2, 0), (0, 0)]
    # far from the fight in columns 0-4 and 8, so they never interact with it
    HANDICAP_STONES = [(5, 7), (6, 7)]

    def _handicap_superko_state(self, enforce_superko):
        gs = GameState(size=9, enforce_superko=enforce_superko)
        gs.place_handicaps(self.HANDICAP_STONES)
        # after black handicap stones, White is to move - plain alternation from here
        self.assertEqual(gs.get_current_player(), go.WHITE)
        for move in self.HANDICAP_SUPERKO_MOVES:
            gs.do_move(move)
        return gs

    def test_positional_superko_with_handicap(self):
        # sanity: the handicap stones really are on the board and accounted for
        gs = self._handicap_superko_state(enforce_superko=False)
        self.assertEqual(len(gs.get_handicaps()), len(self.HANDICAP_STONES))
        # Black is the player facing the repeat here (colours are swapped vs. the
        # even-game version above)
        self.assertEqual(gs.get_current_player(), go.BLACK)
        # without enforcement the repeat is allowed
        self.assertTrue(gs.is_legal((1, 0)))

        # with enforcement it must be rejected, exactly as in the even game
        gs = self._handicap_superko_state(enforce_superko=True)
        self.assertFalse(
            gs.is_legal((1, 0)),
            "superko went undetected in a handicap game - is_positional_superko's "
            "move-parity pre-filter is scanning the wrong player's moves")

    # A superko where the history does NOT alternate: Black plays twice in a row (as a GTP
    # controller can send, e.g. handicap placed as ordinary 'play black' moves). A pre-filter
    # that finds the current player's moves by index parity then scans the wrong player's
    # moves; it has to go by each move's recorded color instead.
    #
    #   . B W .
    #   B W . W      White's (1, 1) is captured by Black at (2, 1); after two passes clear
    #   . B W .      the ko, White retaking at (1, 1) would recreate an earlier position.
    NON_ALTERNATING_SUPERKO_MOVES = [
        ((1, 0), go.BLACK), ((2, 0), go.WHITE), ((0, 1), go.BLACK), ((1, 1), go.WHITE),
        ((1, 2), go.BLACK), ((7, 7), go.BLACK),  # Black twice in a row
        ((3, 1), go.WHITE), ((5, 5), go.BLACK), ((2, 2), go.WHITE),
        ((2, 1), go.BLACK),  # captures White's (1, 1): simple ko
        (None, go.WHITE), (None, go.BLACK)]  # passes clear the ko

    def _non_alternating_superko_state(self, enforce_superko):
        gs = GameState(size=9, enforce_superko=enforce_superko)
        for move, color in self.NON_ALTERNATING_SUPERKO_MOVES:
            gs.do_move(move, color)
        self.assertEqual(gs.get_current_player(), go.WHITE)
        return gs

    def test_positional_superko_after_consecutive_same_color_moves(self):
        self.assertTrue(self._non_alternating_superko_state(False).is_legal((1, 1)))
        self.assertFalse(
            self._non_alternating_superko_state(True).is_legal((1, 1)),
            "superko went undetected after two consecutive Black moves")


def _standard_ko(**kwargs):
    """White has just taken a ko at (1, 1), capturing Black's (2, 1): Black to move, and
    Black may not retake at (2, 1) immediately. (Same position as TestKo.test_standard_ko.)"""
    gs = GameState(size=9, **kwargs)
    for move in [(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2), (2, 1), (1, 1)]:
        gs.do_move(move)
    return gs


class TestRejectedMoves(unittest.TestCase):

    def _snapshot(self, gs):
        return (gs.get_current_player(), gs.get_history_with_colors(),
                gs.get_ko_location(), sorted(gs.get_legal_moves()), gs.get_hash())

    def test_rejected_out_of_turn_move_changes_nothing(self):
        gs = _standard_ko()
        before = self._snapshot(gs)
        with self.assertRaises(go.IllegalMove):
            gs.do_move((1, 1), go.WHITE)  # occupied, and not White's turn
        self.assertEqual(self._snapshot(gs), before)

    def test_rejected_move_in_turn_changes_nothing(self):
        gs = _standard_ko()
        before = self._snapshot(gs)
        with self.assertRaises(go.IllegalMove):
            gs.do_move((2, 1))  # Black retaking the ko at once
        self.assertEqual(self._snapshot(gs), before)

    def test_off_board_move_is_rejected_not_wrapped(self):
        gs = GameState(size=9)
        for move in [(9, 0), (0, 9), (-1, 3), (3, -1)]:
            with self.assertRaises(go.IllegalMove):
                gs.do_move(move)
        self.assertEqual(gs.get_history(), [])

    def test_ko_binds_only_the_player_to_move(self):
        gs = _standard_ko()
        self.assertFalse(gs.is_legal((2, 1)))  # Black can't retake yet
        gs.do_move((2, 1), go.WHITE)  # but White, moving out of turn, may fill it
        self.assertEqual(gs.get_board()[2][1], go.WHITE)


class TestRecordMove(unittest.TestCase):

    def test_records_a_ko_retake(self):
        gs = _standard_ko()
        gs.record_move((2, 1))
        self.assertEqual(gs.get_history_with_colors()[-1], ((2, 1), go.BLACK))
        self.assertEqual(gs.get_board()[1][1], go.EMPTY)  # the retake captured
        self.assertEqual(gs.get_current_player(), go.WHITE)

    def test_records_a_positional_superko_repeat(self):
        gs = TestKo()._non_alternating_superko_state(enforce_superko=True)
        self.assertFalse(gs.is_legal((1, 1)))
        gs.record_move((1, 1), go.WHITE)
        self.assertEqual(gs.get_board()[1][1], go.WHITE)
        self.assertEqual(gs.get_board()[2][1], go.EMPTY)

    def test_rejects_what_cannot_be_placed(self):
        gs = GameState(size=9)
        gs.do_move((1, 0))  # B
        gs.do_move((5, 5))  # W
        gs.do_move((0, 1))  # B - (0, 0) is now suicide for White
        before = gs.get_history_with_colors()
        for move in [(1, 0),   # occupied
                     (0, 0),   # suicide
                     (9, 9)]:  # off the board
            with self.assertRaises(go.IllegalMove):
                gs.record_move(move, go.WHITE)
        self.assertEqual(gs.get_history_with_colors(), before)
        self.assertEqual(gs.get_current_player(), go.WHITE)

    def test_records_a_pass_with_its_color(self):
        gs = GameState(size=9)
        gs.record_move(None, go.WHITE)
        self.assertEqual(gs.get_history_with_colors(), [(None, go.WHITE)])


class TestEndOfGame(unittest.TestCase):

    def _after(self, moves, handicap=()):
        gs = GameState(size=9)
        if handicap:
            gs.place_handicaps(list(handicap))
        for move, color in moves:
            gs.do_move(move, color)
        return gs.is_end_of_game()

    def test_not_over_at_the_start(self):
        self.assertFalse(self._after([]))

    def test_black_then_white_pass_ends_the_game(self):
        self.assertTrue(self._after([(None, go.BLACK), (None, go.WHITE)]))
        self.assertTrue(self._after([((4, 4), go.BLACK), ((5, 5), go.WHITE),
                                     (None, go.BLACK), (None, go.WHITE)]))

    def test_white_then_black_pass_ends_the_game(self):
        self.assertTrue(self._after([((4, 4), go.BLACK), (None, go.WHITE), (None, go.BLACK)]))

    def test_passes_in_a_handicap_game(self):
        self.assertTrue(self._after([(None, go.WHITE), (None, go.BLACK)], handicap=[(2, 2)]))

    def test_one_pass_is_not_the_end(self):
        self.assertFalse(self._after([((4, 4), go.BLACK), (None, go.WHITE)]))

    def test_passes_separated_by_a_move_are_not_the_end(self):
        self.assertFalse(self._after([(None, go.BLACK), ((4, 4), go.WHITE), (None, go.BLACK)]))

    def test_two_passes_by_the_same_color_are_not_the_end(self):
        self.assertFalse(self._after([(None, go.BLACK), (None, go.BLACK)]))


class TestMoveColors(unittest.TestCase):

    def test_alternating_play_records_alternating_colors(self):
        gs = GameState(size=9)
        for move in [(0, 0), (1, 1), None, (2, 2)]:
            gs.do_move(move)
        self.assertEqual(gs.get_history_with_colors(), [
            ((0, 0), go.BLACK), ((1, 1), go.WHITE), (None, go.BLACK), ((2, 2), go.WHITE)])
        self.assertEqual([m for m, _c in gs.get_history_with_colors()], gs.get_history())

    def test_explicit_colors_are_recorded(self):
        gs = GameState(size=9)
        gs.do_move((0, 0), go.WHITE)
        gs.do_move((1, 1), go.WHITE)
        gs.do_move((2, 2), go.BLACK)
        self.assertEqual([c for _m, c in gs.get_history_with_colors()],
                         [go.WHITE, go.WHITE, go.BLACK])
        board = gs.get_board()
        self.assertEqual((board[0][0], board[1][1]), (go.WHITE, go.WHITE))

    def test_pass_honors_its_color(self):
        gs = GameState(size=9)
        gs.do_move(None, go.WHITE)  # Black to move, White passes
        self.assertEqual(gs.get_history_with_colors(), [(None, go.WHITE)])
        self.assertEqual(gs.get_current_player(), go.BLACK)

    def test_pass_without_color_is_the_current_players(self):
        gs = GameState(size=9)
        gs.do_move((0, 0))
        gs.do_move(None)
        self.assertEqual(gs.get_history_with_colors()[-1], (None, go.WHITE))
        self.assertEqual(gs.get_current_player(), go.BLACK)

    def test_setup_stones_are_recorded_with_their_color(self):
        gs = GameState(size=9)
        gs.place_handicap_stone((0, 0), go.BLACK)
        gs.place_handicap_stone((1, 1), go.BLACK)
        gs.place_handicap_stone((2, 2), go.WHITE)
        gs.do_move((3, 3))
        self.assertEqual([c for _m, c in gs.get_history_with_colors()],
                         [go.BLACK, go.BLACK, go.WHITE, go.BLACK])

    def test_copy_has_its_own_colors(self):
        gs = GameState(size=9)
        gs.do_move((0, 0))
        copy = gs.copy()
        copy.do_move((1, 1), go.BLACK)
        self.assertEqual(gs.get_history_with_colors(), [((0, 0), go.BLACK)])
        self.assertEqual(copy.get_history_with_colors(),
                         [((0, 0), go.BLACK), ((1, 1), go.BLACK)])

    def test_try_stone_records_and_undoes_its_color(self):
        gs = GameState(size=9)
        gs.do_move((0, 0))
        with gs.try_stone(flatten_idx((1, 1), 9)):
            self.assertEqual(gs.get_history_with_colors()[-1], ((1, 1), go.WHITE))
        self.assertEqual(gs.get_history_with_colors(), [((0, 0), go.BLACK)])


class TestEye(unittest.TestCase):

    def test_true_eye(self):

        gs = GameState(size=7)

        gs.do_move((1, 0), go.BLACK)
        gs.do_move((0, 1), go.BLACK)

        # false eye at 0, 0
        self.assertFalse(gs.is_eye((0, 0), go.BLACK))

        # make it a true eye by turning the corner (1, 1) into an eye itself
        gs.do_move((1, 2), go.BLACK)
        gs.do_move((2, 1), go.BLACK)
        gs.do_move((2, 2), go.BLACK)
        gs.do_move((0, 2), go.BLACK)

        # is eyeish function does not exist
        self.assertTrue(gs.is_eye((0, 0), go.BLACK))
        self.assertTrue(gs.is_eye((1, 1), go.BLACK))

    def test_eye_recursion(self):
        # a checkerboard pattern of black is 'technically' all true eyes
        # mutually supporting each other

        gs = GameState(size=7)

        for x in range(gs.get_size()):
            for y in range(gs.get_size()):
                if (x + y) % 2 == 1:
                    gs.do_move((x, y), color=go.BLACK)
        self.assertTrue(gs.is_eye((0, 0), go.BLACK))


class TestGroups(unittest.TestCase):

    def test_liberties_after_capture(self):
        # creates 3x3 black group in the middle, that is then all captured
        # ...then an assertion is made that the resulting liberties after
        # capture are the same as if the group had never been there

        gs_capture = GameState(size=7)
        gs_reference = GameState(size=7)
        # add in 3x3 black stones
        for x in range(2, 5):
            for y in range(2, 5):
                gs_capture.do_move((x, y), go.BLACK)
        # surround the black group with white stones
        # and set the same white stones in gs_reference
        for x in range(2, 5):
            gs_capture.do_move((x, 1), go.WHITE)
            gs_capture.do_move((x, 5), go.WHITE)
            gs_reference.do_move((x, 1), go.WHITE)
            gs_reference.do_move((x, 5), go.WHITE)
        gs_capture.do_move((1, 1), go.WHITE)
        gs_reference.do_move((1, 1), go.WHITE)
        for y in range(2, 5):
            gs_capture.do_move((1, y), go.WHITE)
            gs_capture.do_move((5, y), go.WHITE)
            gs_reference.do_move((1, y), go.WHITE)
            gs_reference.do_move((5, y), go.WHITE)

        # board configuration and liberties of gs_capture and of gs_reference should be identical
        self.assertTrue(gs_reference.is_board_equal(gs_capture))
        self.assertTrue(gs_reference.is_liberty_equal(gs_capture))

    def test_large_group_neighbors(self):

        gs, _ = parseboard.parse(". . B B B . .|"
                                 ". . B B B . .|"
                                 ". . B B B . .|"
                                 ". . W W W . .|"
                                 ". . W W W . .|"
                                 ". . W W W . .|"
                                 ". . . . . . .|")
        self.assertTrue(gs.sanity_check_groups())


class TestCopy(unittest.TestCase):

    def equality_checks(self, original, copy):
        self.assertListEqual(copy.get_legal_moves(), original.get_legal_moves())
        self.assertListEqual(copy.get_history(), original.get_history())
        self.assertTrue(copy.is_board_equal(original))
        self.assertTrue(copy.is_liberty_equal(original))
        self.assertEqual(copy.get_hash(), original.get_hash())
        self.assertListEqual(copy.get_history(), original.get_history())
        self.assertEqual(copy.get_captures_white(), original.get_captures_white())
        self.assertEqual(copy.get_captures_black(), original.get_captures_black())

    def test_copy(self):
        gs, _ = parseboard.parse(". B . . . . .|"
                                 "B W W . . . .|"
                                 ". B W . B . .|"
                                 ". . . . . . B|"
                                 ". . B . . . .|"
                                 "W . . . W W .|")

        copy = gs.copy()

        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)


class TestTemporaryMove(unittest.TestCase):

    def listNotEqual(self, listA, listB):
        if len(listA) != len(listB):
            return True
        else:
            for (a, b) in zip(listA, listB):
                if a != b:
                    return True
            return False

    def equality_checks(self, original, copy):
        self.assertEqual(copy.get_current_player(), original.get_current_player())
        self.assertListEqual(copy.get_legal_moves(), original.get_legal_moves())
        self.assertListEqual(copy.get_history(), original.get_history())
        self.assertTrue(copy.is_board_equal(original))
        self.assertTrue(copy.is_liberty_equal(original))
        self.assertEqual(copy.get_hash(), original.get_hash())
        self.assertEqual(copy.get_captures_white(), original.get_captures_white())
        self.assertEqual(copy.get_captures_black(), original.get_captures_black())

    def inequality_checks(self, original, copy):
        self.assertTrue(self.listNotEqual(copy.get_legal_moves(), original.get_legal_moves()))
        self.assertTrue(self.listNotEqual(copy.get_history(), original.get_history()))
        self.assertFalse(copy.is_board_equal(original))
        self.assertFalse(copy.is_liberty_equal(original))
        self.assertNotEqual(copy.get_hash(), original.get_hash())

    def test_simple_undo(self):
        gs = GameState(size=7)
        copy = gs.copy()

        # Baseline equality checks between gs and copy
        self.equality_checks(gs, copy)

        with copy.try_stone(0):
            self.assertTrue(gs.sanity_check_groups())
            self.assertTrue(copy.sanity_check_groups())
            self.inequality_checks(gs, copy)

            # (0, 0) is occupied and should currently be illegal
            self.assertFalse(copy.is_legal((0, 0)))

        # Move should now be undone - retry equality checks from above
        self.assertTrue(gs.sanity_check_groups())
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

        # With move undone, it should be legal again
        self.assertTrue(copy.is_legal((0, 0)))

    def test_ko_undo(self):
        gs, moves = parseboard.parse(". B . . . . .|"
                                     "B W B . . . .|"
                                     "W k W . . . .|"
                                     ". W . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . a . .|"
                                     ". . . . . . .|")
        gs.set_current_player(go.BLACK)

        # Trigger ko at (1, 1)
        gs.do_move(moves['k'])
        ko = gs.get_ko_location()
        self.assertIsNotNone(ko)

        copy = gs.copy()

        self.equality_checks(gs, copy)

        with copy.try_stone(flatten_idx(moves['a'], gs.get_size())):
            self.inequality_checks(gs, copy)

            # Doing move at 'a' clears ko
            self.assertIsNone(copy.get_ko_location())

        self.equality_checks(gs, copy)

        # Undoing move at 'a' resets ko
        self.assertEqual(copy.get_ko_location(), ko)

    def test_simple_merge_undo(self):
        gs, moves = parseboard.parse(". . . . . . .|"
                                     ". . . B W . .|"
                                     ". . . B W . .|"
                                     ". . . a W . .|"
                                     ". . . B W . .|"
                                     ". . . B W . .|"
                                     ". . . . . . .|")
        gs.set_current_player(go.BLACK)

        copy = gs.copy()

        # Initial equality checks
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

        with copy.try_stone(flatten_idx(moves['a'], gs.get_size())):
            self.assertTrue(copy.sanity_check_groups())
            self.inequality_checks(gs, copy)

        # Move should now be undone - retry equality checks from above
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

    def test_simple_capture_undo(self):
        gs, moves = parseboard.parse(". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . B . .|"
                                     ". . . B W c .|"
                                     ". . . B W B .|"
                                     ". . . . B . .|"
                                     ". . . . . . .|")
        gs.set_current_player(go.BLACK)

        copy = gs.copy()

        # Initial equality checks
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

        with copy.try_stone(flatten_idx(moves['c'], gs.get_size())):
            self.assertTrue(copy.sanity_check_groups())
            self.inequality_checks(gs, copy)

        # Move should now be undone - retry equality checks from above
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

    def test_merge_and_capture_undo(self):
        gs, moves = parseboard.parse(". . B B B . .|"
                                     ". B W W W B .|"
                                     ". B W B W B .|"
                                     ". B W c W B .|"
                                     ". B W B W B .|"
                                     ". B W W W B .|"
                                     ". . B B B . .|")
        gs.set_current_player(go.BLACK)

        copy = gs.copy()

        # Initial equality checks
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

        with copy.try_stone(flatten_idx(moves['c'], gs.get_size())):
            self.assertTrue(copy.sanity_check_groups())
            self.inequality_checks(gs, copy)

        # Move should now be undone - retry equality checks from above
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

    def test_hash_update_matches_actual_hash(self):
        gs = GameState(size=7)
        gs, moves = parseboard.parse("a x b . . . .|"
                                     "z c d . . . .|"
                                     ". . . . . . .|"
                                     ". . . y . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")

        # a,b,c,d are black, x,y,z,x are white
        move_order = ['a', 'x', 'b', 'y', 'c', 'z', 'd', 'x']
        for m in move_order:
            move_1d = flatten_idx(moves[m], gs.get_size())

            # 'Try' move and get hash
            with gs.try_stone(move_1d):
                hash1 = gs.get_hash()

            # Actually do move and get hash
            gs.do_move(moves[m])
            hash2 = gs.get_hash()

            self.assertEqual(hash1, hash2)


if __name__ == '__main__':
    unittest.main()
