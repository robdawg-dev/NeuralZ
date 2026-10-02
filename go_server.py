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

Requests are answered by one inference thread that gathers whatever positions arrive
within --batch-wait-ms (up to --max-batch) into a single model call: bots moving at the
same moment share one batch instead of queueing for the model one at a time.

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

import numpy as np  # noqa: E402

from AlphaGo.models.nn_util import NeuralNetBase  # noqa: E402


class _Request(object):
    """One position waiting for inference, and its answer once the batch has run."""

    def __init__(self, planes):
        self.planes = planes
        self.done = threading.Event()
        self.probs = None
        self.error = None


class BatchingPolicy(object):
    """A loaded policy network behind a single inference thread that batches requests."""

    def __init__(self, model_path, weights_path, max_batch=32, batch_wait_ms=2.0):
        self.policy = NeuralNetBase.load_model(model_path)
        self.policy.model.load_weights(weights_path)
        input_shape = self.policy.model.inputs[0].shape  # (None, size, size, planes)
        self.board_size, self.planes = int(input_shape[1]), int(input_shape[3])
        self.n_bits = self.board_size * self.board_size * self.planes
        self.packed_bytes = (self.n_bits + 7) // 8
        self.info = {
            "features": self.policy.preprocessor.get_feature_list(),
            "board_size": self.board_size,
            "planes": self.planes,
            "packed_bytes": self.packed_bytes,
            "model": os.path.abspath(model_path),
            "weights": os.path.abspath(weights_path),
        }
        self.max_batch = max_batch
        self.batch_wait = batch_wait_ms / 1000.0
        self._queue = queue.Queue()
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

    def _run(self, planes_list):
        x = np.stack(planes_list).astype(np.float32)
        return self.policy.forward(x).astype("<f4")

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


def make_handler(policy):
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
                self._reply(200, json.dumps(policy.info).encode("utf-8"), "application/json")
            else:
                self._error(404, "unknown path {}".format(self.path))

        def do_POST(self):
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

        def log_message(self, fmt, *args):
            pass  # one line per move from every bot would drown the log

    return Handler


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="Path to a JSON model file")
    parser.add_argument("weights", help="Path to a .weights.h5 weights file matching model")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Address to listen on. Default: 127.0.0.1 (this machine only)")
    parser.add_argument("--port", type=int, default=5005, help="Default: 5005")
    parser.add_argument("--max-batch", type=int, default=32,
                        help="Most positions evaluated in one model call. Default: 32")
    parser.add_argument("--batch-wait-ms", type=float, default=2.0,
                        help="How long the inference thread waits for more positions to join "
                             "a batch once one has arrived. Default: 2")
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
    policy = BatchingPolicy(args.model, args.weights, max_batch=args.max_batch,
                            batch_wait_ms=args.batch_wait_ms)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    sys.stderr.write("go_server: serving {} on http://{}:{} ({} planes, {}x{})\n".format(
        os.path.basename(args.model), args.host, args.port, policy.planes,
        policy.board_size, policy.board_size))
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
