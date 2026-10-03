# NeuralZ bot

A Go bot for KGS (via kgsGtp), CPU only, for Ubuntu 24.04 (x86-64). It runs as two
processes:

- **`start_server.sh`** runs `go_server.py`, which loads the network once
  (`@MODEL_JSON@`, `@WEIGHTS@`) and serves move probabilities on `127.0.0.1:5005`.
- **`start_client.sh`** runs `go_client.py`, the GTP engine kgsGtp starts. It gets its
  probabilities from the server. Several bots can share one server.

`VERSION` says which commit and model this folder was built from.

## One-time server setup

These need `sudo` and are not done by `install.sh`:

```bash
sudo apt install build-essential   # C++ compiler for the game engine (required)
sudo apt install gnugo             # optional: answers KGS's final_score / final_status_list
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

The client defaults are: sample among moves at least 0.5 as likely as the top move
(`--sample-ratio 0.5`) for the bot's first 20 moves (`--sample-moves 20`), never while one
of its own stones is in atari, then always the top move. `--sample-moves 0` makes it fully
greedy. Options go after the script name; options use hyphens, e.g. `--sample-moves=20`,
not `--sample_moves=20`. See `.venv/bin/python go_client.py --help`.

## Update

Build a new folder on the development machine (`python deploy/build_deploy.py`), copy it
over this one (keeping the same path, so the kgsGtp config still points at it), and run
`./install.sh` again. Then restart the server; kgsGtp starts new clients by itself.
