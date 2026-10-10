# NeuralZ bot

A Go bot for KGS (via kgsGtp), CPU only, for Ubuntu 24.04 (x86-64). It runs as two
processes:

- **`start_server.sh`** runs `go_server.py`, which loads the network once
  (`@MODEL_JSON@`, `@WEIGHTS@`) and serves move probabilities on `127.0.0.1:5005`.
- **`start_client.sh`** runs `go_client.py`, the GTP engine kgsGtp starts. It gets its
  probabilities from the server. Several bots can share one server.

If this folder has a `katago/` subfolder (the Linux CPU build of KataGo and a small
network), the server also runs KataGo to judge finished games for every bot: which stones
are dead when KGS asks (`final_status_list`). GNU Go is then only the fallback.

`VERSION` says which commit and model this folder was built from.

## One-time server setup

These need `sudo` and are not done by `install.sh`:

```bash
sudo apt install build-essential   # C++ compiler for the game engine (required)
sudo apt install gnugo             # recommended: fallback for KGS's dead stones if KataGo fails
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, if not installed yet
```

Python itself is not needed: `uv` installs Python 3.13 into this folder's `.venv`.

## Install

```bash
./install.sh
```

(If the scripts lost their execute permission in the copy, e.g. from a zip, run
`chmod +x *.sh` first.)

This installs the locked packages into `.venv` (`uv sync --frozen`), compiles the game
engine (output in `build_engine.log`), then runs `check_deploy.py`, which starts a
server on a spare port and plays a few GTP moves through the client. It ends with
`check_deploy: OK`. Loading TensorFlow and the model takes a little while on first run.

## Run

Start the server and leave it running (e.g. in `tmux`, or as a systemd service):

```bash
./start_server.sh                    # port 5005
./start_server.sh --symmetries 4     # average over 4 board rotations: a little stronger, ~2x the CPU
```

In the kgsGtp config, set the engine to the client script (absolute path):

```
engine=/path/to/this/folder/start_client.sh
```

### End of the game

KGS asks the bot which stones are dead (`final_status_list`). With `katago/` present the
server answers from KataGo (~30 ms per position on a CPU, after ~4 s at startup); if
KataGo is missing or fails, the client asks GNU Go; if neither answers, it returns an
error and kgsGtp leaves the marking to the opponent. In **ranked** games the bot's list is
binding: kgsGtp won't finish the game until the opponent accepts it. In **free** games it
is not, and the opponent's marking stands: in early October 2026, five free games the bot
was winning on the board were scored as losses this way.

Optional: `start_client.sh --cleanup` also supports `kgs-genmove_cleanup`. When an
opponent disputes the dead stones in a non-Japanese-rules game, KGS then lets play resume
and the bot captures the stones KataGo judges dead before passing. Off by default.

### Client options

The client defaults are: sample among moves at least 0.5 as likely as the top move
(`--sample-ratio 0.5`) for the bot's first 20 moves (`--sample-moves 20`), never while one
of its own stones is in atari, then always the top move. `--sample-moves 0` makes it fully
greedy. Options go after the script name; options use hyphens, e.g. `--sample-moves=20`,
not `--sample_moves=20`. See `.venv/bin/python go_client.py --help`.

To see afterwards exactly what kgsGtp asked and what the bot answered (moves, dead-stone
lists, scores), give each bot its own `--gtp-log`, e.g.
`engine=/path/to/this/folder/start_client.sh --gtp-log /path/to/this/folder/logs/NeuralZ05.log`
(create `logs/` first). Each command and reply is one timestamped line - about 400 lines
per game - appended for as long as the bot runs.

## Stopping the bots cleanly

While a file named `STOP` exists in this folder (`touch STOP`), every bot declines new
challenges and exits as soon as its current game ends - no game is abandoned. An idle bot
just stops taking games and can be stopped any time. Remove the file (`rm STOP`) before
starting the bots again: while it exists they accept no games. (`go_client.py
--stop-file PATH` uses another file.)

## Update

1. `touch STOP` and wait until every bot's kgsGtp has exited (or is idle).
2. Build a new folder on the development machine (`python deploy/build_deploy.py`), copy
   it over this one (keeping the same path, so the kgsGtp configs still point at it), and
   run `./install.sh` again.
3. Restart the server (`start_server.sh`).
4. `rm STOP`, then start each bot's kgsGtp again.
5. Check the running setup: `.venv/bin/python check_running.py --client-args "<the
   arguments after start_client.sh in a bot's engine= line>"`. It confirms the server
   answers (with KataGo and the current version's endpoints), that no `STOP` file is left,
   and what a bot started with those arguments advertises (e.g. `kgs-genmove_cleanup` with
   `--cleanup`), and that its `--gtp-log` folder is writable. Safe to run any time.

Clients wait up to 2 minutes for a restarting server (`--server-wait`), so the server
alone can also be restarted while bots are playing - their clocks keep running meanwhile.

## Measuring speed

To choose `start_server.sh`'s settings (`--threads`, `--max-batch`, `--batch-wait-ms`)
for this machine, time a test server under the bots' load. Do it while the bots are
stopped (`touch STOP`): a second server competes with the live one for the CPU, and both
measurements would suffer.

```
./start_server.sh --port 5006 --batch-wait-ms 0        # the settings to try
.venv/bin/python benchmarks/server_round_trip.py --server http://127.0.0.1:5006 --bots 8
.venv/bin/python benchmarks/server_round_trip.py --server http://127.0.0.1:5006 --bots 1
```

`--bots` is how many bots ask at once: the number of bots is the worst case, 1 the usual
one, since bots mostly wait for their opponents. Each run prints median milliseconds per
move split into batch wait, model call and HTTP, and appends to
`benchmarks/results/server_round_trip.jsonl`. Stop the test server (Ctrl-C) and repeat
with other settings.

How long the bots took in real games, from a bot's `--gtp-log` folder:
`.venv/bin/python tools/gtp_log.py timing <log file>`.
