"""KataGo plumbing shared by the scripts in tools/: the command-line options (with
environment defaults), running a batch of analysis queries, and coordinates.

KataGo itself is not part of this project: point the tools at an install with
--katago/--katago-model/--katago-config, or once per shell with the environment variables
KATAGO_EXE, KATAGO_MODEL and KATAGO_CONFIG (an analysis-engine config, e.g. KataGo's own
analysis_example.cfg). All values the tools read are from Black's point of view.
"""
import collections
import json
import os
import subprocess
import sys
import tempfile
import threading

COLS = "abcdefghijklmnopqrs"         # SGF
GTP_COLS = "ABCDEFGHJKLMNOPQRST"     # GTP skips I


def add_katago_args(parser, required=False):
    """--katago, --katago-model, --katago-config (default: the KATAGO_* environment)."""
    parser.add_argument("--katago", default=os.environ.get("KATAGO_EXE"),
                        help="KataGo executable. Default: $KATAGO_EXE")
    parser.add_argument("--katago-model", default=os.environ.get("KATAGO_MODEL"),
                        help="KataGo network (.bin.gz / .txt.gz). Default: $KATAGO_MODEL")
    parser.add_argument("--katago-config", default=os.environ.get("KATAGO_CONFIG"),
                        help="KataGo analysis config. Default: $KATAGO_CONFIG")
    parser.set_defaults(_katago_required=required)


def katago_command(args, parser=None):
    """The analysis-engine command line from add_katago_args' options; None if any is
    missing (an error via parser.error when they were required)."""
    if args.katago and args.katago_model and args.katago_config:
        return [args.katago, "analysis", "-config", args.katago_config,
                "-model", args.katago_model]
    if getattr(args, "_katago_required", False) and parser is not None:
        parser.error("KataGo is needed: --katago, --katago-model and --katago-config "
                     "(or KATAGO_EXE, KATAGO_MODEL, KATAGO_CONFIG)")
    return None


def gtp(point, size=19):
    """An SGF point ("dd", or "" / "tt" for a pass) as a GTP vertex ("D16", "pass")."""
    if not point or point == "tt":
        return "pass"
    return GTP_COLS[COLS.index(point[0])] + str(size - COLS.index(point[1]))


def run(command, queries, label=""):
    """{(query id, turn): KataGo's reply} for every analyzed turn of every query, with
    win rates and leads from Black's side. Errors and warnings are skipped; progress goes
    to stderr every 2,000 positions when label is given."""
    # KataGo's example configs log to ./analysis_logs: keep that out of the working tree
    log_dir = os.path.join(tempfile.gettempdir(), "neuralz_tools_katago_logs")
    # The tools hide the GPU from TensorFlow (CUDA_VISIBLE_DEVICES=-1); a GPU build of
    # KataGo started with that inherited finds no device and exits - it chooses its own
    env = dict(os.environ)
    if env.get("CUDA_VISIBLE_DEVICES") == "-1":
        del env["CUDA_VISIBLE_DEVICES"]
    proc = subprocess.Popen(command + ["-override-config",
                                       "reportAnalysisWinratesAs=BLACK,logDir=" + log_dir],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    stderr = collections.deque(maxlen=10)

    def feed():  # from a thread: writing everything first deadlocks on a full stdout pipe
        try:
            for q in queries:
                proc.stdin.write(json.dumps(q) + "\n")
            proc.stdin.close()
        except OSError:
            pass  # KataGo exited; reported below with its last output

    def drain():  # keeps KataGo from blocking on a full stderr pipe; kept for errors
        for line in proc.stderr:
            if line.strip():
                stderr.append(line.rstrip())
    threading.Thread(target=feed, daemon=True).start()
    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    total = sum(len(q["analyzeTurns"]) for q in queries)
    out = {}
    for line in proc.stdout:
        r = json.loads(line)
        if "error" in r or "warning" in r:
            continue
        out[(r["id"], r["turnNumber"])] = r
        if label and len(out) % 2000 == 0:
            sys.stderr.write("  {} {}/{}\n".format(label, len(out), total))
    proc.wait()
    drainer.join(timeout=5)
    if proc.returncode != 0:
        raise RuntimeError("KataGo exited with code {} after {} of {} positions; its last "
                           "output:\n  {}".format(proc.returncode, len(out), total,
                                                  "\n  ".join(stderr) or "(none)"))
    return out
