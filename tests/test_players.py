import unittest
import numpy as np
from AlphaGo import go
from AlphaGo.ai import ProbabilisticPolicyPlayer
from AlphaGo.go import GameState


class TestProbabilisticPolicyPlayer(unittest.TestCase):

    def test_temperature_increases_entropy(self):
        # helper function to get the entropy of a distribution
        def entropy(distribution):
            distribution = np.array(distribution).flatten()
            return -np.dot(np.log(distribution), distribution.T)
        player_low = ProbabilisticPolicyPlayer(None, temperature=0.9)
        player_high = ProbabilisticPolicyPlayer(None, temperature=1.1)

        distribution = np.random.random(361)
        distribution = distribution / distribution.sum()

        base_entropy = entropy(distribution)
        high_entropy = entropy(player_high.apply_temperature(distribution))
        low_entropy = entropy(player_low.apply_temperature(distribution))

        self.assertGreater(high_entropy, base_entropy)
        self.assertLess(low_entropy, base_entropy)

    def test_extreme_temperature_is_numerically_stable(self):
        player_low = ProbabilisticPolicyPlayer(None, temperature=1e-12)
        player_high = ProbabilisticPolicyPlayer(None, temperature=1e+12)

        distribution = np.random.random(361)
        distribution = distribution / distribution.sum()

        self.assertFalse(any(np.isnan(player_low.apply_temperature(distribution))))
        self.assertFalse(any(np.isnan(player_high.apply_temperature(distribution))))

    def test_close_candidates_keeps_moves_within_ratio_of_top(self):
        player = ProbabilisticPolicyPlayer(None, sample_ratio=0.5)
        move_probs = [((0, 0), 0.4), ((0, 1), 0.25), ((0, 2), 0.2), ((0, 3), 0.15)]
        self.assertEqual([m for m, _ in player._close_candidates(move_probs)],
                         [(0, 0), (0, 1), (0, 2)])

    def test_close_candidates_is_greedy_when_one_move_dominates(self):
        player = ProbabilisticPolicyPlayer(None, sample_ratio=0.5)
        move_probs = [((0, 0), 0.9), ((0, 1), 0.06), ((0, 2), 0.04)]
        self.assertEqual(player._close_candidates(move_probs), [((0, 0), 0.9)])

    def test_sampling_window_uses_own_moves_when_given(self):
        player = ProbabilisticPolicyPlayer(None, sample_ratio=0.5, sample_moves=3)
        state = GameState()
        self.assertTrue(player._in_sampling_window(state, own_moves=2))
        self.assertFalse(player._in_sampling_window(state, own_moves=3))

    def test_sampling_window_is_whole_game_without_sample_moves(self):
        player = ProbabilisticPolicyPlayer(None, sample_ratio=0.5)
        self.assertTrue(player._in_sampling_window(GameState(), own_moves=500))

    def test_own_moves_from_the_board_skip_handicap_stones(self):
        state = GameState()
        state.place_handicaps([(3, 3), (15, 15)])
        state.do_move((9, 9))   # white
        state.do_move((3, 15))  # black's first real move
        state.do_move((15, 3))  # white
        self.assertEqual(state.get_current_player(), go.BLACK)
        self.assertEqual(ProbabilisticPolicyPlayer._own_moves_played(state), 1)
        player = ProbabilisticPolicyPlayer(None, sample_ratio=0.5, sample_moves=2)
        self.assertTrue(player._in_sampling_window(state, own_moves=None))

    def test_stone_in_atari_is_detected_for_the_player_to_move(self):
        state = GameState()
        state.do_move((0, 1), go.BLACK)
        state.do_move((0, 0), go.WHITE)  # white corner stone, one liberty left at (1, 0)
        state.set_current_player(go.WHITE)
        self.assertTrue(ProbabilisticPolicyPlayer._has_stone_in_atari(state))
        state.set_current_player(go.BLACK)
        self.assertFalse(ProbabilisticPolicyPlayer._has_stone_in_atari(state))

    def test_plays_greedy_while_a_stone_is_in_atari(self):
        class Fixed(object):
            def eval_state(self, state, moves=None):
                return [((1, 0), 0.4), ((10, 10), 0.35), ((15, 15), 0.25)]
        state = GameState()
        state.do_move((0, 1), go.BLACK)
        state.do_move((0, 0), go.WHITE)
        state.set_current_player(go.WHITE)
        # (extending at (1, 0) runs a dead ladder along the edge: the ladder guard, tested
        # separately, would refuse it - off here to test the no-sampling rule alone)
        player = ProbabilisticPolicyPlayer(Fixed(), sample_ratio=0.5, ladder_guard=False)
        np.random.seed(0)
        self.assertEqual({player.get_move(state) for _ in range(30)}, {(1, 0)})



def _ladder(breaker=None):
    """White (5, 5) in atari at (5, 6) from Black (4, 5) (5, 4) (6, 5) (6, 6): extending
    runs a ladder toward the lower-right edge, which a White stone on its path breaks."""
    state = GameState()
    for b in [(4, 5), (5, 4), (6, 5), (6, 6)]:
        state.do_move(b, go.BLACK)
    if breaker:
        state.do_move(breaker, go.WHITE)
    state.do_move((5, 5), go.WHITE)
    state.set_current_player(go.WHITE)
    return state


class PrefersExtension(object):
    def eval_state(self, state, moves=None):
        return [((5, 6), 0.6), ((15, 15), 0.3), ((3, 3), 0.1)]


class TestLadderGuard(unittest.TestCase):

    def test_dead_ladder_extension_is_found(self):
        self.assertEqual(ProbabilisticPolicyPlayer._failed_ladder_extensions(_ladder()),
                         {(5, 6)})

    def test_working_ladder_escape_is_not_flagged(self):
        self.assertEqual(
            ProbabilisticPolicyPlayer._failed_ladder_extensions(_ladder(breaker=(3, 8))), set())

    def test_nothing_flagged_without_a_group_in_atari(self):
        state = GameState()
        state.do_move((3, 3))
        self.assertEqual(ProbabilisticPolicyPlayer._failed_ladder_extensions(state), set())

    def test_guard_skips_the_dead_ladder(self):
        player = ProbabilisticPolicyPlayer(PrefersExtension(), sample_moves=0)
        self.assertEqual(player.get_move(_ladder()), (15, 15))

    def test_guard_lets_a_working_ladder_run(self):
        player = ProbabilisticPolicyPlayer(PrefersExtension(), sample_moves=0)
        self.assertEqual(player.get_move(_ladder(breaker=(3, 8))), (5, 6))

    def test_guard_can_be_turned_off(self):
        player = ProbabilisticPolicyPlayer(PrefersExtension(), sample_moves=0, ladder_guard=False)
        self.assertEqual(player.get_move(_ladder()), (5, 6))

    def test_guard_applies_inside_the_sampling_window(self):
        class Close(object):
            def eval_state(self, state, moves=None):
                return [((5, 6), 0.5), ((15, 15), 0.45), ((3, 3), 0.05)]
        player = ProbabilisticPolicyPlayer(Close(), sample_ratio=0.5, sample_moves=None)
        np.random.seed(0)
        self.assertNotIn((5, 6), {player.get_move(_ladder()) for _ in range(30)})

if __name__ == '__main__':
    unittest.main()
