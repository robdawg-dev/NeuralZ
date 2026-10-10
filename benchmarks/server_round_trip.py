"""Where a move's server time goes: send positions to a running go_server.py and split each
round trip, using the server's X-NeuralZ-Timing header, into decoding the planes, waiting
for a batch, the model call, and everything else (HTTP). Each bot keeps one connection
open, as go_client does. With --bots N, N threads send at once, as N bots on one server
do. Appends each run to
benchmarks/results/server_round_trip.jsonl.

    # start a server first, e.g.  python go_server.py <model.json> <weights.h5>
    uv run python benchmarks/server_round_trip.py [--server http://127.0.0.1:5005] \\
        [--bots 4] [--repeat 2] [--game-log some_bot.log --game 1]

Positions: benchmarks/sample_positions.json (KGS games, by stage), or a whole game in move
order from a go_client --gtp-log (--game-log). Run against the KGS server's own go_server
to choose its --threads, --max-batch and --batch-wait-ms.
"""
import argparse
import datetime
import http.client
import json
import os
import platform
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from move_time import POSITIONS, STAGES, state_of  # noqa: E402

RESULTS = os.path.join(HERE, "results", "server_round_trip.jsonl")


def connect(server):
    """A connection to the server, kept open across requests as go_client keeps it."""
    url = urllib.parse.urlsplit(server)
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    return cls(url.netloc, timeout=60), url.path


def timed_request(connection, body):
    """One /policy round trip: (total ms, {"decode", "queue", "model", "batch"})."""
    conn, base = connection
    t = time.perf_counter()
    conn.request("POST", base + "/policy", body=body,
                 headers={"Content-Type": "application/octet-stream"})
    r = conn.getresponse()
    reply = r.read()
    total = (time.perf_counter() - t) * 1000
    if r.status != 200:
        raise RuntimeError("/policy -> {}: {}".format(r.status, reply[:200]))
    header = r.headers.get("X-NeuralZ-Timing", "")
    parts = dict(kv.split("=") for kv in header.split(";") if "=" in kv)
    if not parts:
        raise RuntimeError("the server sent no X-NeuralZ-Timing header - it predates the "
                           "timing; restart it from the current code")
    return total, {k: float(v) for k, v in parts.items()}


def encode(positions, server):
    """Each position's packed planes, built with the features the server's model uses."""
    import numpy as np
    from AlphaGo.preprocessing.preprocessing import Preprocess
    with urllib.request.urlopen(server + "/info", timeout=30) as r:
        info = json.loads(r.read().decode("utf-8"))
    pre = Preprocess(info["features"]) if "features" in info else None
    if pre is None:
        raise RuntimeError("/info has no feature list")
    out = []
    for p in positions:
        planes = pre.state_to_tensor(state_of(p))
        out.append((p["move_number"], np.packbits(planes.reshape(-1)).tobytes()))
    return out, info


def run(server, encoded, bots, repeat):
    """Every position sent `repeat` times by each of `bots` threads; rows of timings."""
    rows, lock = [], threading.Lock()

    def bot():
        connection = connect(server)
        timed_request(connection, encoded[0][1])  # open the connection before timing
        for _ in range(repeat):
            for move_number, body in encoded:
                total, parts = timed_request(connection, body)
                with lock:
                    rows.append(dict(parts, total=total, move_number=move_number))
    threads = [threading.Thread(target=bot) for _ in range(bots)]
    t = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return rows, time.perf_counter() - t


def report(rows, elapsed):
    lines = []
    keys = ["total", "decode", "queue", "model", "other"]
    for r in rows:
        r["other"] = r["total"] - r["decode"] - r["queue"] - r["model"]
    lines.append("{:<10} {:>5}  ".format("moves", "n") +
                 "".join("{:>13}".format(k) for k in keys) + "   mean batch")
    for lo, hi in [(1, 999)] + STAGES:
        rs = [r for r in rows if lo <= r["move_number"] <= hi]
        if not rs:
            continue
        label = "all" if (lo, hi) == (1, 999) else "{}-{}".format(lo, hi if hi < 999 else "+")
        lines.append("{:<10} {:>5}  ".format(label, len(rs)) + "".join(
            "{:>13.1f}".format(statistics.median(r[k] for r in rs)) for k in keys) +
            "   {:>10.2f}".format(statistics.mean(r["batch"] for r in rs)))
    lines.append("(medians, ms) - {:.1f} positions/s overall".format(len(rows) / elapsed))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:5005")
    parser.add_argument("--bots", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--positions", default=POSITIONS)
    parser.add_argument("--game-log", help="a go_client --gtp-log: replay one game in order")
    parser.add_argument("--game", type=int, default=1)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    if args.game_log:
        import gtp_log
        with open(args.game_log, encoding="utf-8") as f:
            g = gtp_log.games(gtp_log.exchanges(f))[args.game - 1]
        positions = [{"move_number": n, "setup": [], "moves": g["moves"][:n - 1],
                      "to_move": g["moves"][n - 1][0]} for n in range(1, len(g["moves"]) + 1)]
    else:
        with open(args.positions) as f:
            positions = json.load(f)["positions"]
    encoded, info = encode(positions, args.server)
    timed_request(connect(args.server), encoded[0][1])  # warm the server's calls
    rows, elapsed = run(args.server, encoded, args.bots, args.repeat)
    lines = report(rows, elapsed)
    print("{} positions x {} bots x {} repeats, server {} ({})\n".format(
        len(encoded), args.bots, args.repeat, args.server, os.path.basename(info.get("model", ""))))
    print("\n".join(lines))
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                            "machine": platform.node(), "server": args.server,
                            "bots": args.bots, "summary": lines}) + "\n")


if __name__ == "__main__":
    main()
