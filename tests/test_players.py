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


if __name__ == '__main__':
    unittest.main()
