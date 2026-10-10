"""A GTP bot that gets its move probabilities from go_server.py instead of loading the
network itself - so many bots can share one model, and each bot process stays small:
it never imports TensorFlow or Keras.

    python go_client.py [--server http://127.0.0.1:5005] [--temperature ...]

The GTP layer is interface.gtp_wrapper and the game state and move choice AlphaGo.ai, as
in play_tests/match_networks.py; RemotePolicy stands in for the network: it builds the
position's feature planes here and asks the server for the move probabilities.
"""
import argparse
import http.client
import json
import os
import sys
import time
import urllib.parse

import numpy as np

from AlphaGo.ai import ProbabilisticPolicyPlayer
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import flatten_idx
from interface.gtp_wrapper import run_gtp


class RemotePolicy(object):
    """The policy interface ProbabilisticPolicyPlayer uses (eval_state), answered by
    go_server.py: the feature list comes from the server's /info, so the planes built here
    always match the served model. Keeps one connection open to the server across moves
    (a new one per move cost ~10 ms), so one instance is for one thread."""

    def __init__(self, server, timeout=30.0, server_wait=120.0, retry_wait=1.0):
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.server_wait = server_wait
        self.retry_wait = retry_wait
        url = urllib.parse.urlsplit(self.server)
        self._connection_class = (http.client.HTTPSConnection if url.scheme == "https"
                                  else http.client.HTTPConnection)
        self._netloc, self._base = url.netloc, url.path
        self._connection = None
        self.info = json.loads(self._request("/info"))
        self.board_size = self.info["board_size"]
        self.preprocessor = Preprocess(self.info["features"], size=self.board_size)
        if self.preprocessor.get_output_dimension() != self.info["planes"]:
            raise ValueError("server's model takes {} planes, but its feature list builds {}"
                             .format(self.info["planes"],
                                     self.preprocessor.get_output_dimension()))

    def _request(self, path, body=None):
        """GET (body None) or POST to the server, retrying a refused or failed connection
        for up to server_wait seconds - long enough for go_server to restart (TensorFlow
        import, model load, compiling its calls) without costing a bot its game. A few
        retries over ~6 s were not: the bot's process ended and kgsGtp left the game.

        The connection is kept open between requests. One that fails after serving earlier
        requests - the server restarted, or closed it - is reopened and the request resent
        at once (every request is safe to repeat) unless it timed out; other failures wait
        and retry as above."""
        deadline = time.monotonic() + self.server_wait
        delay, warned = self.retry_wait, False
        while True:
            reused = self._connection is not None
            if not reused:
                self._connection = self._connection_class(self._netloc, timeout=self.timeout)
            try:
                if body is None:
                    self._connection.request("GET", self._base + path)
                else:
                    self._connection.request("POST", self._base + path, body=body, headers={
                        "Content-Type": "application/octet-stream"})
                response = self._connection.getresponse()
                reply = response.read()
            except (http.client.HTTPException, OSError) as e:
                self._connection.close()
                self._connection = None
                if reused and not isinstance(e, TimeoutError):
                    continue
                if time.monotonic() + delay > deadline:
                    raise RuntimeError("go_server at {} unreachable (gave up after {:.0f} s): "
                                       "{}".format(self.server, self.server_wait, e)) from e
                if not warned:
                    sys.stderr.write("go_client: go_server at {} unreachable ({}); retrying "
                                     "for up to {:.0f} s\n".format(self.server, e,
                                                                   self.server_wait))
                    sys.stderr.flush()
                    warned = True
                time.sleep(delay)
                delay = min(delay * 2, 5.0)
                continue
            if response.status != 200:
                raise RuntimeError("go_server {} -> {}: {}".format(
                    path, response.status, reply.decode("utf-8", "replace")))
            return reply

    def move_probabilities(self, state):
        """The served network's probabilities for every board point, (size * size,)."""
        planes = self.preprocessor.state_to_tensor(state)
        reply = self._request("/policy", np.packbits(planes.reshape(-1)).tobytes())
        return np.frombuffer(reply, "<f4")

    def eval_state(self, state, moves=None):
        """(move, probability) for each of moves (default: all legal moves), normalized
        over them - as CNNPolicy.eval_state does."""
        probs = self.move_probabilities(state)
        moves = moves or state.get_legal_moves()
        if len(moves) == 0:
            return []
        distribution = probs[[flatten_idx(m, self.board_size) for m in moves]]
        distribution = distribution / distribution.sum()
        return list(zip(moves, distribution))

    @property
    def judges_games(self):
        """Whether the server runs KataGo for the end of the game (go_server --katago)."""
        return bool(self.info.get("katago"))

    def _judge(self, path, stones, to_move, komi, rules):
        body = json.dumps({"stones": stones, "to_move": to_move, "komi": komi,
                           "rules": rules}).encode("utf-8")
        return json.loads(self._request(path, body))

    def final_status(self, stones, to_move, komi, rules):
        """KataGo's verdict on the position: {"dead": [vertex, ...], "score_lead",
        "contested", "open"} (see go_server)."""
        return self._judge("/final_status", stones, to_move, komi, rules)

    def cleanup_move(self, stones, to_move, komi, rules):
        """KataGo's kgs-genmove_cleanup move for to_move (the bot): a vertex, or "pass"."""
        return self._judge("/cleanup_move", stones, to_move, komi, rules)["move"]

    def border_move(self, stones, to_move, komi, rules):
        """KataGo's move closing to_move's (the bot's) open border: a vertex, or None."""
        return self._judge("/border_move", stones, to_move, komi, rules)["move"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a GTP bot whose network is served by go_server.py.")
    parser.add_argument("--server", default="http://127.0.0.1:5005",
                        help="go_server.py's address. Default: http://127.0.0.1:5005")
    parser.add_argument("--server-wait", type=float, default=120.0,
                        help="Seconds to keep retrying while go_server is unreachable (e.g. "
                             "restarting) before giving up. The bot's clock runs meanwhile. "
                             "Default: 120")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Seconds to wait for one move's probabilities. Default: 30")
    # Move choice: see ProbabilisticPolicyPlayer (AlphaGo/ai.py) for what each default is
    # based on.
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampled moves - lower leans harder toward the "
                             "top candidate. Default: 1.0")
    parser.add_argument("--sample-ratio", type=float, default=0.5,
                        help="Sample only among moves at least this fraction as likely as the "
                             "top move. Default: 0.5")
    parser.add_argument("--sample-moves", type=int, default=20,
                        help="Sample for the bot's first N moves of each game (counted from its "
                             "first genmove, so handicap stones don't count), greedy after. "
                             "0: always greedy. Default: 20")
    parser.add_argument("--no-ladder-guard", dest="ladder_guard", action="store_false",
                        help="Let the bot extend groups in atari into ladders the engine reads "
                             "as dead (the guard is on by default)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Support kgs-genmove_cleanup (needs go_server --katago): when the "
                             "opponent disputes the dead stones in a non-Japanese-rules game, "
                             "KGS lets play resume and the bot captures the stones KataGo "
                             "judges dead before passing. Default: off")
    parser.add_argument("--stop-file", default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "STOP"),
                        help="While this file exists the bot declines new games and exits when "
                             "its game ends - create it before a deployment, remove it after. "
                             "Default: STOP next to go_client.py")
    parser.add_argument("--gtp-log", metavar="PATH",
                        help="Append every GTP command and reply to this file, with times - "
                             "give each bot its own. Default: no log")
    parser.add_argument("--max-moves", type=int, default=800,
                        help="Force a pass once this many moves have been played. Default: 800")
    parser.add_argument("--version", default="0.3",
                        help="Version string reported to the GTP controller. Default: 0.3")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        policy = RemotePolicy(args.server, timeout=args.timeout, server_wait=args.server_wait)
    except (RuntimeError, ValueError) as e:
        sys.exit("go_client: {}".format(e))
    player = ProbabilisticPolicyPlayer(
        policy, temperature=args.temperature, pass_when_offered=True,
        move_limit=args.max_moves, sample_ratio=args.sample_ratio,
        sample_moves=args.sample_moves, ladder_guard=args.ladder_guard)
    scorer = policy if policy.judges_games else None
    if scorer is None:
        sys.stderr.write("go_client: go_server runs without --katago - dead stones from GNU Go"
                         "{}\n".format("; --cleanup ignored" if args.cleanup else ""))
    run_gtp(player, name="NeuralZ", version=args.version, scorer=scorer,
            cleanup=args.cleanup and scorer is not None, stop_file=args.stop_file,
            board_size=policy.board_size, log_path=args.gtp_log)


if __name__ == "__main__":
    main()
