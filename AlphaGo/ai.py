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

    def get_move(self, state):
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
       policy.
       By manipulating the 'temperature', moves can be pushed towards totally random
       (high temperature) or towards greedy play (low temperature)
    """

    def __init__(self, policy_function, temperature=1.0, pass_when_offered=False,
                 move_limit=None, greedy_start=None, top_k=None, top_k_responding=None):
        assert temperature > 0.0
        self.policy = policy_function
        self.move_limit = move_limit
        self.beta = 1.0 / temperature
        self.pass_when_offered = pass_when_offered
        self.greedy_start = greedy_start
        self.top_k = top_k
        self.top_k_responding = top_k_responding

    def _restrict_to_top_k(self, move_probs, state):
        """Restricts probabilistic sampling to the top-K highest-probability legal
        moves (renormalized implicitly by apply_temperature, which always
        re-normalizes over whatever it's given) - caps how far into the long tail a
        probabilistic pick can ever reach, regardless of how flat or peaked the
        policy's own distribution happens to be for a given position. Unlike
        temperature alone, this guarantees excluding any move outside the top K, not
        just making it less likely.

        Which K applies is decided per call from the board itself, not fixed at
        construction time: a GTP bot doesn't know which color it'll be asked to play
        until the server actually asks (`genmove black` vs `genmove white`), so a
        single static top_k can't distinguish "I'm moving first" from "I'm responding
        to whatever's already on the board" across different games from the same
        long-running process. An empty board means this is our first move of the
        game (use top_k); anything already placed - a normal opponent's move OR
        handicap stones - means we're responding (use top_k_responding).
        """
        board_is_empty = len(state.get_history()) == 0
        k = self.top_k if board_is_empty else self.top_k_responding
        if k is None or len(move_probs) <= k:
            return move_probs
        return sorted(move_probs, key=itemgetter(1), reverse=True)[:k]

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

    def get_move(self, state):
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

            if self.greedy_start is not None and len(state.get_history()) >= self.greedy_start:
                # greedy

                max_prob = max(move_probs, key=itemgetter(1))
                return max_prob[0]
            else:
                # probabilistic

                move_probs = self._restrict_to_top_k(move_probs, state)
                # zip(*list) is like the 'transpose' of zip;
                # zip(*zip([1,2,3], [4,5,6])) is [(1,2,3), (4,5,6)]
                moves, probabilities = zip(*move_probs)
                # apply 'temperature' to the distribution
                probabilities = self.apply_temperature(probabilities)
                # numpy interprets a list of tuples as 2D, so we must choose an
                # _index_ of moves then apply it in 2 steps
                choice_idx = np.random.choice(len(moves), p=probabilities)
                return moves[choice_idx]

        # No 'sensible' moves available, so do pass move
        return go.PASS
