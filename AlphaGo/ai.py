"""Policy players"""
import numpy as np
from AlphaGo import go
from operator import itemgetter


class GreedyPolicyPlayer(object):
    """A player that uses a greedy policy (i.e. chooses the highest probability
       move each turn)
    """

    def __init__(self, policy_function, pass_when_offered=False, move_limit=None):
        self.policy = policy_function
        self.pass_when_offered = pass_when_offered
        self.move_limit = move_limit

    def get_move(self, state, own_moves=None):
        # check move limit
        if self.move_limit is not None and len(state.get_history()) > self.move_limit:
            return go.PASS

        # check if pass was offered and we want to pass
        if self.pass_when_offered:
            if len(state.get_history()) > 100 and state.get_history()[-1] == go.PASS:
                return go.PASS

        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]

        # check if there are sensible moves left to do
        if len(sensible_moves) > 0:
            move_probs = self.policy.eval_state(state, sensible_moves)
            max_prob = max(move_probs, key=itemgetter(1))
            return max_prob[0]

        # No 'sensible' moves available, so do pass move
        return go.PASS


class ProbabilisticPolicyPlayer(object):
    """A player that samples a move in proportion to the probability given by the
       policy, for its first sample_moves moves of a game (None: the whole game), and
       plays the most likely move after that.

       sample_ratio limits sampling to the close calls: only moves at least sample_ratio
       times as likely as the most likely one (None: every move). Where one move is
       clearly preferred it is the only candidate, so the move is effectively greedy and
       variation only enters where the network itself sees a close call. Measured on
       b20c256 (workspace/sample_ratio/): ratio 0.5 for the first 20 own moves took a
       human's repeated opening line away by ply 30-40 in every game, and scored 49.75%
       (+/-2.5) against its own greedy self over 400 games.

       By manipulating the 'temperature', sampled moves can be pushed towards totally
       random (high temperature) or towards greedy play (low temperature)
    """

    def __init__(self, policy_function, temperature=1.0, pass_when_offered=False,
                 move_limit=None, sample_ratio=None, sample_moves=None):
        assert temperature > 0.0
        assert sample_ratio is None or 0.0 < sample_ratio <= 1.0
        self.policy = policy_function
        self.move_limit = move_limit
        self.beta = 1.0 / temperature
        self.pass_when_offered = pass_when_offered
        self.sample_ratio = sample_ratio
        self.sample_moves = sample_moves

    def _close_candidates(self, move_probs):
        """The moves at least sample_ratio times as likely as the most likely one."""
        if self.sample_ratio is None:
            return move_probs
        max_prob = max(p for _, p in move_probs)
        return [(m, p) for m, p in move_probs if p >= self.sample_ratio * max_prob]

    def _sample(self, move_probs):
        # zip(*list) is like the 'transpose' of zip;
        # zip(*zip([1,2,3], [4,5,6])) is [(1,2,3), (4,5,6)]
        moves, probabilities = zip(*move_probs)
        # apply 'temperature' to the distribution
        probabilities = self.apply_temperature(probabilities)
        # numpy interprets a list of tuples as 2D, so we must choose an
        # _index_ of moves then apply it in 2 steps
        choice_idx = np.random.choice(len(moves), p=probabilities)
        return moves[choice_idx]

    def apply_temperature(self, distribution):
        log_probabilities = np.log(distribution)
        # apply beta exponent to probabilities (in log space)
        log_probabilities = log_probabilities * self.beta
        # scale probabilities to a more numerically stable range (in log space)
        log_probabilities = log_probabilities - log_probabilities.max()
        # convert back from log space
        probabilities = np.exp(log_probabilities)
        # re-normalize the distribution
        return probabilities / probabilities.sum()

    @staticmethod
    def _own_moves_played(state):
        """How many moves the player to move has already made this game, from the board:
        its color's history entries, less the handicap stones placed as setup. A GTP
        controller that sends handicap as ordinary moves makes them count here - which is
        why GTPGameConnector passes its own genmove count instead."""
        color = state.get_current_player()
        n = sum(1 for _, c in state.get_history_with_colors() if c == color)
        if color == go.BLACK:
            n -= len(state.get_handicaps())
        return n

    def _in_sampling_window(self, state, own_moves):
        if self.sample_moves is None:
            return True
        if own_moves is None:
            own_moves = self._own_moves_played(state)
        return own_moves < self.sample_moves

    def get_move(self, state, own_moves=None):
        """own_moves: how many moves this player has already made in the game, if the
        caller knows (GTP counts its genmoves); otherwise it is counted from the board."""
        # check move limit
        if self.move_limit is not None and len(state.get_history()) > self.move_limit:
            return go.PASS

        # check if pass was offered and we want to pass
        if self.pass_when_offered:
            if len(state.get_history()) > 100 and state.get_history()[-1] == go.PASS:
                return go.PASS

        # list with 'sensible' moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]

        # check if there are 'sensible' moves left to do
        if len(sensible_moves) > 0:

            move_probs = self.policy.eval_state(state, sensible_moves)

            if self._in_sampling_window(state, own_moves):
                # probabilistic, among the close calls
                return self._sample(self._close_candidates(move_probs))

            # greedy
            max_prob = max(move_probs, key=itemgetter(1))
            return max_prob[0]

        # No 'sensible' moves available, so do pass move
        return go.PASS
