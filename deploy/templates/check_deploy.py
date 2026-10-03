"""Check this bot folder works: the engine is compiled, and go_server.py + go_client.py
play a short GTP game together. Run by install.sh; safe to re-run any time:

    .venv/bin/python check_deploy.py

Uses its own free port, so it does not disturb a server already running on 5005.
"""
import os
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(HERE, "@MODEL_JSON@")
WEIGHTS = os.path.join(HERE, "@WEIGHTS@")
GAME = "boardsize 19\nclear_board\nplay b D4\ngenmove w\ngenmove b\nquit\n"
SERVER_START_TIMEOUT = 300


def fail(msg):
    sys.exit("check_deploy: FAILED - " + msg)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    sys.path.insert(0, HERE)
    try:
        import AlphaGo.go  # noqa: F401
        import AlphaGo.preprocessing.preprocessing  # noqa: F401
    except ImportError as e:
        fail("game engine not compiled ({}) - run install.sh".format(e))
    print("engine: ok")

    if shutil.which("gnugo"):
        print("gnugo: ok (final_score / final_status_list available)")
    else:
        print("gnugo: not installed - final_score / final_status_list will answer empty "
              "(optional: sudo apt install gnugo)")

    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    port = free_port()
    print("starting go_server.py on port {} (loading TensorFlow and the model)...".format(port))
    server = subprocess.Popen(
        [sys.executable, "go_server.py", MODEL_JSON, WEIGHTS, "--port", str(port)],
        cwd=HERE, env=env, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + SERVER_START_TIMEOUT
        banner = None
        for line in server.stderr:
            if "serving" in line:
                banner = line.strip()
                break
            if time.time() > deadline:
                break
        if banner is None:
            server.kill()
            fail("go_server.py did not start (exit code {}) - run start_server.sh to see "
                 "its output".format(server.poll()))
        print("server: " + banner)

        proc = subprocess.run(
            [sys.executable, "go_client.py", "--server", "http://127.0.0.1:{}".format(port)],
            input=GAME, capture_output=True, text=True, cwd=HERE, env=env, timeout=180)
        replies = [line for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode != 0 or replies[:3] != ["=", "=", "="] or len(replies) < 5 \
                or not all(r.startswith("= ") for r in replies[3:5]):
            fail("unexpected GTP replies {}\nclient stderr:\n{}".format(replies, proc.stderr))
        print("client: played {} and {} as white and black".format(replies[3][2:], replies[4][2:]))
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    print("check_deploy: OK")


if __name__ == "__main__":
    main()
