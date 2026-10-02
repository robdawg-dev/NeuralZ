"""Start the bot programs as real processes, as kgsGtp does, and talk GTP to them:
run_gtp_player.py (network in-process), and go_server.py + go_client.py (shared network).

Marked slow: each process imports TensorFlow (the server and run_gtp_player.py) or the
game engine, which takes a few seconds.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import re
import socket
import subprocess
import sys
import time

import pytest

import go_server
from AlphaGo.models.policy import CNNPolicy

pytestmark = pytest.mark.slow

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["board", "ones", "liberties", "sensibleness"]
VERTEX = re.compile(r"^= ([A-HJ-T](1[0-9]|[1-9])|pass)$", re.IGNORECASE)
GAME = "boardsize 19\nclear_board\nplay b D4\ngenmove w\ngenmove b\nlist_commands\nquit\n"


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bot_startup")
    policy = CNNPolicy(FEATURES, layers=2, filters_per_layer=8, filter_width_1=3)
    model, weights = str(tmp / "model.json"), str(tmp / "model.weights.h5")
    policy.save_model(model)
    policy.model.save_weights(weights)
    return model, weights


def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3")


def _gtp(cmd, stdin, cwd=REPO):
    """Run a GTP program to completion on stdin's commands: (replies, completed process).
    Replies are the non-empty response lines, in order."""
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, cwd=cwd,
                          env=_env(), timeout=180)
    return [line for line in proc.stdout.splitlines() if line.strip()], proc


def _check_game(replies, proc):
    assert proc.returncode == 0, proc.stderr
    # boardsize, clear_board, play: empty successes
    assert replies[:3] == ["=", "=", "="], replies
    assert VERTEX.match(replies[3]), replies[3]  # genmove w
    assert VERTEX.match(replies[4]), replies[4]  # genmove b
    listed = " ".join(replies[5:])
    assert "genmove" in listed and "play" in listed


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_run_gtp_player_plays_through_gtp(model_files):
    replies, proc = _gtp([sys.executable, "run_gtp_player.py", *model_files], GAME)
    _check_game(replies, proc)


@pytest.fixture(scope="module")
def running_server(model_files):
    """go_server.py as its own process on a free port, with 4-view symmetry averaging."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "go_server.py", *model_files, "--port", str(port),
         "--symmetries", "4", "--threads", "2"],
        cwd=REPO, env=_env(), stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 180
    for line in proc.stderr:
        if "serving" in line:
            break
        if time.time() > deadline:
            break
    assert proc.poll() is None, "go_server.py exited during startup"
    yield "http://127.0.0.1:{}".format(port), line
    proc.terminate()
    proc.wait(timeout=30)


def test_go_server_reports_its_settings_at_startup(running_server):
    _url, banner = running_server
    assert "serving model.json" in banner and "symmetries 4" in banner, banner


def test_go_client_plays_through_gtp_against_a_live_server(running_server):
    url, _banner = running_server
    replies, proc = _gtp([sys.executable, "go_client.py", "--server", url,
                          "--version", "9.9"], GAME + "version\n")
    _check_game(replies, proc)


def test_go_client_reports_its_version(running_server):
    url, _banner = running_server
    replies, proc = _gtp([sys.executable, "go_client.py", "--server", url,
                          "--version", "9.9"], "version\nquit\n")
    assert proc.returncode == 0 and replies[0] == "= 9.9", (replies, proc.stderr)


def test_go_client_without_a_server_exits_with_a_clear_message():
    port = _free_port()  # nothing listening
    _replies, proc = _gtp([sys.executable, "go_client.py", "--server",
                           "http://127.0.0.1:{}".format(port), "--timeout", "2"], "quit\n")
    assert proc.returncode != 0
    assert "unreachable" in proc.stderr


@pytest.mark.parametrize("args", [
    ["--symmetries", "3"],
    ["--port", "not-a-number"],
    [],
])
def test_go_server_rejects_bad_command_lines(args):
    with pytest.raises(SystemExit):
        go_server.build_parser().parse_args((["m.json", "w.h5"] if args else []) + args)
