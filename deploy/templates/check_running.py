"""Check the RUNNING bots' setup, any time (check_deploy.py checks a fresh install instead):

    .venv/bin/python check_running.py [--server http://127.0.0.1:5005] \\
        [--client-args "--cleanup --gtp-log /path/to/logs/NeuralZ05.log"]

- the server answers, and whether it judges games with KataGo;
- KataGo answers a dead-stone query, and the server has the closed-border check
  (/border_move) - i.e. the server running is the current version;
- whether the STOP file is present (while it is, the bots accept no games);
- what a bot started with these client arguments - copy them from a kgsGtp config's
  engine= line - tells kgsGtp it supports (kgs-genmove_cleanup only with --cleanup and
  KataGo), and that its --gtp-log can be written.

Read-only for the server and the bots: it starts one extra client that only answers
list_commands. Exits 1 if anything is wrong.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STONES = [["B", "D4"], ["W", "Q16"], ["W", "Q17"], ["B", "C3"]]
problems = []


def say(ok, text):
    print("{} {}".format("ok  " if ok else "FAIL", text))
    if not ok:
        problems.append(text)


def request(server, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(server + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:5005")
    parser.add_argument("--client-args", default="",
                        help="the arguments after start_client.sh in a kgsGtp engine= line")
    parser.add_argument("--stop-file", default=os.path.join(HERE, "STOP"))
    args = parser.parse_args(argv)

    try:
        info = request(args.server, "/info")
    except (urllib.error.URLError, OSError) as e:
        say(False, "server at {} not answering ({}) - start_server.sh".format(args.server, e))
        sys.exit(1)
    katago = bool(info.get("katago"))
    say(True, "server answering: {}".format(os.path.basename(info.get("model", "?"))))
    print("     KataGo judging: {}".format("on" if katago else "off (dead stones from GNU Go)"))
    if katago:
        query = {"stones": STONES, "to_move": "B", "komi": 6.5, "rules": "chinese"}
        try:
            verdict = request(args.server, "/final_status", query)
            say("dead" in verdict, "KataGo dead-stone query: {} dead, {} contested".format(
                len(verdict.get("dead", [])), verdict.get("contested", "?")))
            say("open" in verdict, "closed-border data in /final_status" if "open" in verdict
                else "no 'open' in /final_status: the server is older than the border check")
        except urllib.error.HTTPError as e:
            say(False, "/final_status -> {} {}".format(e.code, e.read().decode("utf-8", "replace")))
        try:
            request(args.server, "/border_move", query)
            say(True, "/border_move answers (current server version)")
        except urllib.error.HTTPError as e:
            say(False, "/border_move -> {}: the server is older than the border check - "
                       "restart it from the current folder".format(e.code))

    stop = os.path.exists(args.stop_file)
    say(not stop, "STOP file {}".format(
        "present: bots decline new games ({}) - rm it to resume".format(args.stop_file)
        if stop else "absent: bots accept games"))

    client_args = shlex.split(args.client_args)
    if "--server" not in client_args:
        client_args += ["--server", args.server]
    proc = subprocess.run([sys.executable, os.path.join(HERE, "go_client.py")] + client_args,
                          input="list_commands\nquit\n", capture_output=True, text=True,
                          timeout=120, cwd=HERE)
    commands = [line.lstrip("= ").strip() for line in proc.stdout.splitlines() if line.strip()]
    say(proc.returncode == 0 and "genmove" in commands,
        "a client with these arguments starts and answers GTP" if proc.returncode == 0 else
        "the client failed (exit {}): {}".format(proc.returncode, proc.stderr.strip()[-300:]))
    wants_cleanup = "--cleanup" in client_args
    has_cleanup = "kgs-genmove_cleanup" in commands
    if wants_cleanup:
        say(has_cleanup, "kgs-genmove_cleanup advertised" if has_cleanup else
            "--cleanup given but kgs-genmove_cleanup not advertised (needs KataGo on the server)")
    else:
        print("     kgs-genmove_cleanup: off (add --cleanup to the engine line to enable)")
    if "--gtp-log" in client_args:
        log = client_args[client_args.index("--gtp-log") + 1]
        folder = os.path.dirname(os.path.abspath(log))
        say(os.path.isdir(folder) and os.access(folder, os.W_OK),
            "--gtp-log folder {} {}".format(folder, "writable" if os.access(folder, os.W_OK)
                                            else "missing or not writable - mkdir -p it"))
    print("check_running: {}".format("OK" if not problems else "{} problem(s)".format(
        len(problems))))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
