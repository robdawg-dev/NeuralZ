import unittest
import numpy as np
from AlphaGo import go
from AlphaGo.ai import ProbabilisticPolicyPlayer, ScoreLookaheadPlayer
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


class FakeScoreNet(object):
    """Policy of fixed move probabilities; scores each candidate board by its last move,
    from the side of its player to move (the opponent of the player choosing)."""

    def __init__(self, move_probs, opponent_scores):
        self.move_probs = move_probs
        self.opponent_scores = opponent_scores
        self.value_calls = []

    def eval_state(self, state, moves=None):
        return list(self.move_probs)

    def eval_value(self, states, komis):
        self.value_calls.append(([s.get_history()[-1] for s in states], list(komis)))
        scores = [self.opponent_scores[s.get_history()[-1]] for s in states]
        return np.zeros(len(states)), np.array(scores)


class TestScoreLookaheadPlayer(unittest.TestCase):

    move_probs = [((3, 3), 0.5), ((15, 15), 0.3), ((9, 9), 0.2)]

    def test_plays_the_move_leaving_the_opponent_worst_off(self):
        net = FakeScoreNet(self.move_probs, {(3, 3): 2.0, (15, 15): -4.0, (9, 9): 1.0})
        player = ScoreLookaheadPlayer(net, 7.5, top_k=10, sample_moves=0)
        self.assertEqual(player.get_move(GameState()), (15, 15))

    def test_only_scores_the_policy_top_k(self):
        net = FakeScoreNet(self.move_probs, {(3, 3): 2.0, (15, 15): 1.0, (9, 9): -9.0})
        player = ScoreLookaheadPlayer(net, 7.5, top_k=2, sample_moves=0)
        self.assertEqual(player.get_move(GameState()), (15, 15))
        self.assertEqual(net.value_calls[0][0], [(3, 3), (15, 15)])

    def test_keeps_the_top_move_unless_another_beats_it_by_the_margin(self):
        net = FakeScoreNet(self.move_probs, {(3, 3): 2.0, (15, 15): 1.0, (9, 9): 3.0})
        self.assertEqual(ScoreLookaheadPlayer(net, 7.5, margin=1.5, sample_moves=0)
                         .get_move(GameState()), (3, 3))
        self.assertEqual(ScoreLookaheadPlayer(net, 7.5, margin=0.5, sample_moves=0)
                         .get_move(GameState()), (15, 15))

    def test_candidate_boards_get_the_opponents_komi(self):
        scores = {(3, 3): 0.0, (15, 15): 0.0, (9, 9): 0.0}
        net = FakeScoreNet(self.move_probs, scores)
        ScoreLookaheadPlayer(net, 7.5, sample_moves=0).get_move(GameState())
        self.assertEqual(net.value_calls[-1][1], [7.5] * 3)    # Black chose: White to move
        state = GameState()
        state.do_move((0, 0))
        ScoreLookaheadPlayer(net, 7.5, sample_moves=0).get_move(state)
        self.assertEqual(net.value_calls[-1][1], [-7.5] * 3)   # White chose: Black to move

    def test_samples_without_lookahead_inside_the_window(self):
        net = FakeScoreNet(self.move_probs, {})
        player = ScoreLookaheadPlayer(net, 7.5, sample_ratio=0.5, sample_moves=5)
        np.random.seed(0)
        moves = {player.get_move(GameState()) for _ in range(30)}
        self.assertEqual(moves, {(3, 3), (15, 15)})
        self.assertEqual(net.value_calls, [])


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
