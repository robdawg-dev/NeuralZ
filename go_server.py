"""Policy inference server: one process holds the model and serves move probabilities to
any number of GTP bots (go_client.py), so TensorFlow and the weights are loaded once
rather than once per bot.

    python go_server.py <model.json> <weights.h5> [--port 5005]

Localhost-only HTTP (Python's standard library server):

    GET  /info    -> JSON: the model's feature list, board size and number of input
                     planes, and the model/weights paths - what a client checks at startup.
    POST /policy  -> body: one position's feature planes, (size, size, planes) 0/1 values
                     flattened in C order and np.packbits'ed. Reply: the size*size move
                     probabilities as little-endian float32.

With --katago, the server also runs one KataGo analysis engine (a CPU build and a small
network) that judges finished games for every bot (interface/katago_scorer.py):

    POST /final_status -> body: JSON {"stones": [[color, vertex], ...], "to_move", "komi",
                          "rules"}. Reply: JSON {"dead": [vertex, ...], "score_lead",
                          "contested"} - contested: points whose owner is still open, which
                          the bots check before passing back.
    POST /cleanup_move -> body: the same, with "to_move" the bot's color. Reply: JSON
                          {"move": vertex or "pass"} - a pass only once none of the
                          opponent's stones are dead (kgs-genmove_cleanup).

Requests are answered by one inference thread that gathers whatever positions arrive
within --batch-wait-ms (up to --max-batch) into a single model call: bots moving at the
same moment share one batch instead of queueing for the model one at a time.

--symmetries 8 evaluates each position under all 8 board symmetries (the rotations and
reflections training augments with), maps each answer back to the original orientation
and averages them - about +1 point of top-1 accuracy for b20c256 on held-out positions.
--symmetries 4 uses the 4 rotations - most of that gain (about +0.8) for about half the
cost. The default, 1, evaluates the position as given.

The network is called through compiled TensorFlow functions, one per batch size: a batch
is padded up to the next power of two (or --max-batch) positions, and every size is
compiled at startup, so no move ever waits on a compile. On the CPU that is ~2.5-3x faster
per position than an eager Keras call, which spends most of its time on per-layer overhead
(workspace/profiling/results_dev_summary.md: 57-62 vs 143-173 ms at batch 1). --eager uses
the plain Keras call instead.

CPU-only, like run_gtp_player.py: the GPU is hidden before TensorFlow loads.
"""
import os

# Must happen before TensorFlow is imported (by AlphaGo.models.nn_util below).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

import numpy as np  # noqa: E402

from AlphaGo.models.nn_util import NeuralNetBase  # noqa: E402
from AlphaGo.training.shard_stream import (  # noqa: E402
    BATCH_TRANSFORMATIONS, symmetry_permutations)
from interface.katago_scorer import KataGoScorer  # noqa: E402

# The views averaged for each --symmetries choice: the position as given, the 4
# rotations, or all 8 symmetries.
SYMMETRY_SETS = {
    1: ["noop"],
    4: ["noop", "rot90", "rot180", "rot270"],
    8: list(BATCH_TRANSFORMATIONS),
}
SYMMETRY_CHOICES = tuple(SYMMETRY_SETS)


class _Request(object):
    """One position waiting for inference, and its answer once the batch has run."""

    def __init__(self, planes):
        self.planes = planes
        self.done = threading.Event()
        self.probs = None
        self.error = None


class BatchingPolicy(object):
    """A loaded policy network behind a single inference thread that batches requests."""

    def __init__(self, model_path, weights_path, max_batch=4, batch_wait_ms=10.0,
                 symmetries=1, compiled=True):
        if symmetries not in SYMMETRY_CHOICES:
            raise ValueError("symmetries must be one of {}, got {}".format(
                SYMMETRY_CHOICES, symmetries))
        self.policy = NeuralNetBase.load_model(model_path)
        self.policy.model.load_weights(weights_path)
        input_shape = self.policy.model.inputs[0].shape  # (None, size, size, planes)
        self.board_size, self.planes = int(input_shape[1]), int(input_shape[3])
        self.n_bits = self.board_size * self.board_size * self.planes
        self.packed_bytes = (self.n_bits + 7) // 8
        # Each view's point permutation maps an answer for the transformed board back to
        # the original orientation.
        self.symmetries = SYMMETRY_SETS[symmetries]
        self._perms = symmetry_permutations(self.board_size, self.symmetries)
        self.info = {
            "features": self.policy.preprocessor.get_feature_list(),
            "board_size": self.board_size,
            "planes": self.planes,
            "packed_bytes": self.packed_bytes,
            "symmetries": len(self.symmetries),
            "model": os.path.abspath(model_path),
            "weights": os.path.abspath(weights_path),
        }
        self.max_batch = max_batch
        self.batch_wait = batch_wait_ms / 1000.0
        self._queue = queue.Queue()
        self._compiled = self._compile() if compiled else None
        # Warm up before serving: the first call builds the model's graph.
        self._run([np.zeros((self.board_size, self.board_size, self.planes), np.uint8)])
        threading.Thread(target=self._worker, name="inference", daemon=True).start()

    def decode(self, body):
        """Packed request bytes -> (size, size, planes) uint8 planes. Raises ValueError."""
        if len(body) != self.packed_bytes:
            raise ValueError("expected {} bytes of packed planes, got {}".format(
                self.packed_bytes, len(body)))
        bits = np.unpackbits(np.frombuffer(body, np.uint8), count=self.n_bits)
        return bits.reshape(self.board_size, self.board_size, self.planes)

    def predict(self, planes):
        """Move probabilities for one position (blocks until its batch has run)."""
        request = _Request(planes)
        self._queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.probs

    def _compile(self):
        """A compiled model call for each batch size the worker can form - powers of two up
        to max_batch positions, and max_batch itself, times the symmetry views - each traced
        and run once now, so serving never waits on a compile."""
        import tensorflow as tf
        model = self.policy.model
        sizes = {min(2 ** i, self.max_batch) for i in range(self.max_batch.bit_length() + 1)}
        compiled = {}
        for n in sorted(sizes):
            spec = tf.TensorSpec((n * len(self.symmetries), self.board_size, self.board_size,
                                  self.planes), tf.float32)
            fn = tf.function(lambda views: model(views, training=False), input_signature=[spec])
            fn(tf.zeros(spec.shape, tf.float32))
            compiled[n] = fn
        return compiled

    def _forward(self, views, n):
        """The network's output for views holding n positions' symmetry views."""
        if self._compiled is not None and n in self._compiled:
            return self._compiled[n](views).numpy()
        return self.policy.forward(views)

    def _run(self, planes_list):
        """Probabilities for each position, (N, size * size) little-endian float32 -
        with several symmetries, the mean over them, each mapped back to the position's
        own orientation."""
        x = np.stack(planes_list)
        n = len(x)
        if self._compiled is not None and n <= self.max_batch:
            # pad up to the next compiled batch size; the padding rows' answers are dropped
            size = min(s for s in self._compiled if s >= n)
            if size > n:
                x = np.concatenate([x, np.zeros((size - n,) + x.shape[1:], x.dtype)])
        m = len(x)
        if self.symmetries == ["noop"]:
            return self._forward(x.astype(np.float32), m)[:n].astype("<f4")
        views = np.concatenate([BATCH_TRANSFORMATIONS[name](x) for name in self.symmetries])
        probs = self._forward(views.astype(np.float32), m)
        mean = np.zeros((m, probs.shape[1]), np.float64)
        original = np.empty_like(mean)
        for k, perm in enumerate(self._perms):
            # The transformed board's point j is the original's point perm[j].
            original[:, perm] = probs[k * m:(k + 1) * m]
            mean += original
        return (mean[:n] / len(self.symmetries)).astype("<f4")

    def _worker(self):
        while True:
            batch = [self._queue.get()]
            # Gather whatever else arrives within the wait window, up to max_batch.
            while len(batch) < self.max_batch:
                try:
                    batch.append(self._queue.get(timeout=self.batch_wait))
                except queue.Empty:
                    break
            try:
                probs = self._run([r.planes for r in batch])
                for request, p in zip(batch, probs):
                    request.probs = p
            except Exception as e:  # noqa: BLE001 - reported back to every waiting client
                for request in batch:
                    request.error = e
            for request in batch:
                request.done.set()


def make_handler(policy, scorer=None):
    info = dict(policy.info, katago=scorer is not None)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            self._reply(status, message.encode("utf-8"), "text/plain; charset=utf-8")

        def do_GET(self):
            if self.path == "/info":
                self._reply(200, json.dumps(info).encode("utf-8"), "application/json")
            else:
                self._error(404, "unknown path {}".format(self.path))

        def do_POST(self):
            if self.path in ("/final_status", "/cleanup_move"):
                self._judge()
                return
            if self.path != "/policy":
                self._error(404, "unknown path {}".format(self.path))
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                planes = policy.decode(body)
            except ValueError as e:
                self._error(400, str(e))
                return
            try:
                probs = policy.predict(planes)
            except Exception as e:  # noqa: BLE001
                self._error(500, "inference failed: {}".format(e))
                return
            self._reply(200, probs.tobytes(), "application/octet-stream")

        def _judge(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if scorer is None:
                self._error(503, "this go_server runs without --katago")
                return
            try:
                q = json.loads(body)
                args = (q["stones"], q["to_move"], q["komi"], q.get("rules", "chinese"))
            except (ValueError, KeyError, TypeError) as e:
                self._error(400, "bad request: {}".format(e))
                return
            try:
                if self.path == "/final_status":
                    answer = scorer.final_status(*args)
                else:
                    answer = {"move": scorer.cleanup_move(*args)}
            except Exception as e:  # noqa: BLE001
                self._error(500, "katago failed: {}".format(e))
                return
            self._reply(200, json.dumps(answer).encode("utf-8"), "application/json")

        def log_message(self, fmt, *args):
            pass  # one line per move from every bot would drown the log

    return Handler


def default_katago_config():
    """katago_analysis.cfg next to this file - where a deploy bundle puts it - or, in a
    repo checkout, the template build_deploy.py copies it from."""
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(here, "katago_analysis.cfg")
    if os.path.exists(bundled):
        return bundled
    return os.path.join(here, "deploy", "templates", "katago_analysis.cfg")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="Path to a JSON model file")
    parser.add_argument("weights", help="Path to a .weights.h5 weights file matching model")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Address to listen on. Default: 127.0.0.1 (this machine only)")
    parser.add_argument("--port", type=int, default=5005, help="Default: 5005")
    parser.add_argument("--symmetries", type=int, choices=SYMMETRY_CHOICES, default=1,
                        help="Board symmetries each position is evaluated under, the answers "
                             "averaged: 1 (the position as given), 4 (the 4 rotations) or 8 "
                             "(all rotations and reflections - the most accurate). More views "
                             "cost more time per move. Default: 1")
    parser.add_argument("--max-batch", type=int, default=4,
                        help="Most positions evaluated in one model call (with --symmetries "
                             "4 or 8, the call holds that many inputs per position); more "
                             "requests at once wait for the next call. Each batch size up to "
                             "this is compiled at startup (1, 2, 4, ...). A batch only forms "
                             "when bots ask within --batch-wait-ms of each other, so a few "
                             "suffice. Default: 4")
    parser.add_argument("--batch-wait-ms", type=float, default=10.0,
                        help="How long the inference thread waits for more positions to join "
                             "a batch once one has arrived. Default: 10")
    parser.add_argument("--eager", action="store_true",
                        help="Call the network as a plain eager Keras call instead of the "
                             "compiled functions (slower; for debugging or comparison)")
    parser.add_argument("--katago", default=None, metavar="EXE",
                        help="KataGo executable (a CPU build): also judge finished games for "
                             "the bots - dead stones, cleanup moves. Default: off")
    parser.add_argument("--katago-model", default=None,
                        help="KataGo network for --katago (a small one, e.g. b10c128)")
    parser.add_argument("--katago-config", default=None,
                        help="KataGo analysis config for --katago. Default: "
                             "katago_analysis.cfg next to go_server.py (a deploy bundle), "
                             "else deploy/templates/katago_analysis.cfg (a repo checkout)")
    parser.add_argument("--katago-visits", type=int, default=1,
                        help="KataGo visits per dead-stone query: 1 is the network alone, "
                             "enough on finished positions. Default: 1")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads TensorFlow may use per model call. Default: "
                             "TensorFlow's own (all cores)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.threads:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    started = time.time()
    policy = BatchingPolicy(args.model, args.weights, max_batch=args.max_batch,
                            batch_wait_ms=args.batch_wait_ms, symmetries=args.symmetries,
                            compiled=not args.eager)
    scorer = None
    if args.katago:
        if not args.katago_model:
            sys.exit("go_server: --katago needs --katago-model")
        config = args.katago_config or default_katago_config()
        try:
            scorer = KataGoScorer(args.katago, args.katago_model, config,
                                  visits=args.katago_visits)
        except Exception as e:  # noqa: BLE001
            sys.exit("go_server: KataGo failed to start: {}".format(e))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy, scorer))
    sys.stderr.write("go_server: serving {} on http://{}:{} ({} planes, {}x{}, symmetries "
                     "{}, {}; ready in {:.0f} s)\n".format(
                         os.path.basename(args.model), args.host, args.port, policy.planes,
                         policy.board_size, policy.board_size, len(policy.symmetries),
                         "eager calls" if args.eager else "compiled for batches of {}".format(
                             "/".join(str(n) for n in sorted(policy._compiled))),
                         time.time() - started))
    judging = "off (bots fall back to GNU Go)"
    if scorer is not None:
        judging = "by KataGo ({}, {} visit(s))".format(os.path.basename(args.katago_model),
                                                       args.katago_visits)
    sys.stderr.write("go_server: end-of-game judging {}\n".format(judging))
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if scorer is not None:
            scorer.close()


if __name__ == "__main__":
    main()
