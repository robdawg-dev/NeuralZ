# Sample run: train the b15c192 policy network and play it

This walks you from a fresh clone to a trained Go policy network you can play against over
GTP, by reproducing the run that produced the project's current bot:
`_restower_b15c192_v4shuf40m_mb1024_lr1p6_seed90001`.

**What you end up with:** a ResNet policy network (15 residual blocks, 192 filters, about
10M parameters) trained on ~37M positions of KataGo self-play. It predicts KataGo's move
**55.6%** of the time on held-out games (top-5: 87.4%), and plays through any GTP program:
a desktop GUI, or a bot account on KGS.

**What it takes:**

| Step | Time on the reference machine | Disk |
|---|---|---|
| Download and extract the SGFs | depends on your connection | ~20 GB (~9 GB after cleanup) |
| Build the training data | ~3 hours | ~12 GB (~30 GB while it runs) |
| Train, 95 epochs | **~39 hours** | ~12 GB of checkpoints |

The reference machine is a single **NVIDIA RTX 4070 SUPER (12 GB)**. Everything runs in
Docker, on Linux or on Windows with Docker Desktop.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Install](#2-install)
3. [Get the KataGo self-play games](#3-get-the-katago-self-play-games)
4. [Build the training data](#4-build-the-training-data)
5. [Create the model](#5-create-the-model)
6. [Train](#6-train)
7. [Play it over GTP](#7-play-it-over-gtp)
8. [Optional extras](#8-optional-extras)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Before you start

- **An NVIDIA GPU with 12 GB of memory or more.** The recipe trains 1,024 positions per
  step in mixed precision, which fits in 12 GB. With less memory you'll need a smaller
  `--minibatch`, and then a different learning rate (see
  [Optional extras](#8-optional-extras)).
- **A recent NVIDIA driver**, plus:
  - **Linux:** [Docker Engine](https://docs.docker.com/engine/install/) and the
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  - **Windows:** [Docker Desktop](https://docs.docker.com/desktop/) with the WSL 2 backend,
    which passes the GPU through by itself.
- **About 60 GB of free disk** and **16 GB of RAM or more**.
- **git.**

You don't need Python on your machine: the Docker image brings its own. Optionally,
[uv](https://docs.astral.sh/uv/) gives you a local environment for editing and linting
(see [Install](#2-install)).

**Shell:** commands below are written for bash (Linux, WSL, or Git Bash on Windows). In
**Git Bash**, prefix commands that contain container paths with `MSYS_NO_PATHCONV=1`, or
Git Bash rewrites paths like `/data` into Windows paths before Docker sees them.

---

## 2. Install

```bash
git clone <this repository's URL> NeuralZ
cd NeuralZ
```

**Point Docker at the training data.** `docker-compose.yml` mounts your training data at
`/data` in the container, and refuses to start until you've said where it is:

```bash
cp .env.example .env
# edit .env so it reads:
#   TRAINING_DATA_DIR=./workspace/prod_40m/shards
mkdir -p workspace/prod_40m/shards
```

`workspace/` is gitignored; everything this guide generates goes there.

**Build the Docker image.** It installs Python 3.13, TensorFlow 2.21 with CUDA, and every
other package at the exact versions pinned in `uv.lock`:

```bash
docker compose build
```

**Build the Go engine.** The board logic and feature planes are Cython extensions, compiled
into the repository so the container can use them:

```bash
docker compose run --rm gpu python setup_cython.py build_ext --inplace
```

**Check the GPU is visible:**

```bash
docker compose run --rm gpu python -c \
    "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

This should print a list with one `PhysicalDevice(... device_type='GPU')`. If it prints
`[]`, see [Troubleshooting](#9-troubleshooting).

**Optionally, run the test suite** (about 30 seconds, CPU only):

```bash
docker compose run --rm -e CUDA_VISIBLE_DEVICES= gpu python -m pytest tests -q
```

**Optionally, a local environment** for your editor and `flake8`: `uv sync` creates `.venv`
with the same packages, on Python 3.13. The Cython engine is only built inside Docker, so
run training and the tests there.

From here on, `docker compose run --rm gpu <command>` runs a command inside the container,
with the repository at `/workspace` (the working directory) and your training data at
`/data`.

---

## 3. Get the KataGo self-play games

The training data is **KataGo's own self-play games**, which KataGo's distributed training
project publishes as daily archives. **They are not part of this repository**; download them
yourself from the KataGo archive's training-games index:

<https://katagoarchive.org/kata1/traininggames/index.html>

Each file there is one day of games, named like `2025-07-08sgfs.tar.bz2`.

**The reference run used the 26 days from 2025-07-08 to 2025-08-03, except 2025-07-31.**
Any recent stretch of days works; using the same days gets you the closest reproduction.

Extract each archive into its own folder under `game_data/` (gitignored):

```
game_data/
  2025-07-08sgfs/
    kata1-b28c512nbt-s9644411648-d4976498967/     one folder per KataGo network
      0007FCF86A5B0D4A5E6866405B5B0A1D.sgf         one game per file
      ...
  2025-07-09sgfs/
  ...
```

The reference download was **1,794,703 games (20 GB)**. Only about a third of them are
usable: KataGo also plays other board sizes and several special game types (see the next
step).

---

## 4. Build the training data

Four stages turn raw games into training data. Each one writes files the next one reads.
[`DATA_PIPELINE.md`](DATA_PIPELINE.md) explains every decision behind them.

### 4a. Clean out unusable games (optional)

Deletes games that will never be used: other board sizes, special game types, exact
duplicates. It isn't needed for correctness, since the later stages skip them anyway, but it
frees about 11 GB and makes the next stage faster.

```bash
docker compose run --rm gpu python -m AlphaGo.preprocessing.sgf_cull scan game_data workspace/cull
# look at workspace/cull/scan_summary.txt, then:
docker compose run --rm gpu python -m AlphaGo.preprocessing.sgf_cull delete workspace/cull
```

`scan` only writes a list (`workspace/cull/to_delete.txt`) and a summary; `delete`
permanently removes the listed files from `game_data/`. Reference: 1,156,408 files
deleted, 638,295 kept. The scan took about 17 minutes.

### 4b. Scan every game into a manifest

Reads each game once and writes one line per game to a manifest: board size, game type,
komi, KataGo's own evaluation of the opening, per-move quality statistics. Nothing is
filtered yet.

```bash
docker compose run --rm gpu python -m AlphaGo.preprocessing.sgf_preparation \
    scan game_data workspace/analysis/manifest.jsonl
```

Reference: 638,295 games in about 8 minutes.

### 4c. Choose the games

Picks the games to train on and splits them by game into train, validation and test sets.
The defaults are the reference run's settings:
- 19x19 `normal` games with komi 5–9 and a balanced opening;
- `handicap` games whose komi compensates for the handicap;
- 38M + 2M positions (95% normal, 5% handicap);
- a 93/5/2 split, seed 20260922.

```bash
docker compose run --rm gpu python -m AlphaGo.preprocessing.select_games \
    workspace/analysis/manifest.jsonl workspace/prod_40m/selection
```

It writes `train.txt`, `val.txt`, `test.txt` and `selection_summary.txt`. The reference
summary:

```
chosen            : 119,047 normal (~38,000,189 positions), 6,317 handicap (~2,000,014)
split        games    handicap      ~positions    hcap pos
train      116,589       5,875      37,205,518      5.00%
val          6,268         316       1,995,179      5.03%
test         2,507         126         799,506      5.06%
```

> **Your games won't be exactly the reference run's.** The selection draws games at random
> with a fixed seed, but in the order the manifest lists them, and the scan's parallel
> workers write that order differently from run to run. You get a sample of the same size,
> from the same pools, with the same composition, but not the same games. Training results
> should come out very close, not identical.

### 4d. Build the feature planes

Replays every chosen game and turns each position into the network's input: **48 feature
planes of 19x19** (stones, liberties, recent moves, ladders, legal moves and more), with the
move KataGo played as the label. The output is written as shuffled shards of up to
1,000,000 positions each (`--positions-per-file`).

```bash
docker compose run --rm gpu python -m AlphaGo.preprocessing.convert_shuffled \
    workspace/prod_40m/selection workspace/prod_40m/shards --max-winrate-loss 0.10
```

`--max-winrate-loss 0.10` drops positions where KataGo's own move lost more than 10% win
rate, i.e. moves that aren't worth imitating (about 0.14% of positions).

Reference result, recorded in `workspace/prod_40m/shards/conversion.json`:

| Split | Games | Positions |
|---|---|---|
| train | 116,589 | 37,200,866 |
| val | 6,268 | 1,994,800 |
| test | 2,507 | 799,552 |

(The reference run's shards held 100,000 positions each; how the data is split into shard
files makes no difference to training.)

That's about 12 GB: positions are stored bit-packed and compressed, about 300 bytes
each. While it runs, the first pass's temporary files need up to ~16 GB more for the
train split. The train split took about 2 hours, most of it the first pass.

---

## 5. Create the model

```bash
docker compose run --rm gpu python -m AlphaGo.models.make_model \
    restower --blocks 15 --filters 192 --head conv_norm
```

```
workspace/models/model_restower_b15c192_convnorm.json: ResTowerPolicy (num_blocks=15, filters=192, head=conv_norm, ...)
  board 19x19, 48 input planes from 11 features: board,ones,turns_since,liberties,...
  10,072,682 parameters
```

The model JSON is the network's architecture plus the list of input features. It holds no
weights: training produces those. `--head conv_norm` is a batch-normalized output head that
keeps deep towers like this one stable (see `ResTowerPolicy` in
[`AlphaGo/models/policy.py`](AlphaGo/models/policy.py)). The feature list defaults to the one
the shards were built with. To be certain the model matches your shards, you can copy the
list from them with `--features-from workspace/prod_40m/shards`.

---

## 6. Train

```bash
docker compose run --rm gpu python -m AlphaGo.training.supervised_policy_trainer \
    workspace/models/model_restower_b15c192_convnorm.json /data \
    workspace/runs/restower_b15c192 \
    --minibatch 1024 --steps-per-epoch 2734 --epochs 95 --validation-length 60000 \
    --lr-schedule plateau --learning-rate 1.6 \
    --warmup-steps 16928 --warmup-start-lr 0.0001 \
    --plateau-factor 0.5 --plateau-patience 5 --plateau-cooldown 5 \
    --plateau-min-lr 1e-5 --plateau-min-delta 0.005 \
    --mixed-precision --seed 90001 --verbose
```

What the options mean:

| Option | Value | Why |
|---|---|---|
| `--minibatch` | 1024 | Positions per training step: as many as 12 GB holds in mixed precision. |
| `--steps-per-epoch` | 2734 | An "epoch" here is 2,734 steps (2.8M positions), not a full pass over the 37M. It sets how often validation runs and a checkpoint is saved: about every 24 minutes. |
| `--epochs` | 95 | The reference run was stopped after epoch 95. |
| `--validation-length` | 60000 | Validation uses the first 60,000 validation positions, the same ones every epoch. |
| `--lr-schedule plateau` | | Hold the learning rate, and halve it whenever validation loss stops improving. |
| `--learning-rate` | 1.6 | The peak learning rate this model was trained at with batch size 1,024. A different model or batch size needs its own (see [Optional extras](#8-optional-extras)). |
| `--warmup-steps` | 16928 | Ramp the learning rate up from 0.0001 over the first ~6 epochs, rather than starting at 1.6. |
| `--plateau-*` | | Halve the LR after 5 epochs without a validation-loss improvement of at least 0.005; wait 5 epochs after each cut; never go below 1e-5. |
| `--mixed-precision` | | Compute in 16-bit floats, keep weights in 32-bit: it's what fits 1,024 positions in 12 GB. |
| `--seed` | 90001 | Makes weight initialization and the data order reproducible. |

**What to expect.** Each epoch takes about 24 minutes, so 95 epochs is about 39 hours. The
reference run's learning rate:
- ramped to 1.6 over the first 6 epochs;
- held there until the end of epoch 37;
- was halved at the end of epochs 37, 54, 63, 72, 81 and 90, ending at 0.025. All of these
  cuts came from the plateau rule, not by hand.

Progress for the reference run:

| Epoch | Validation loss | Validation accuracy | Top-5 accuracy |
|---|---|---|---|
| 1 | 2.481 | 38.2% | 67.8% |
| 10 | 1.791 | 48.9% | 81.8% |
| 20 | 1.697 | 50.7% | 83.6% |
| 50 | 1.550 | 54.0% | 86.4% |
| 94 | **1.484** | **55.6%** | **87.4%** |

Don't expect identical numbers. GPU arithmetic isn't bit-for-bit repeatable over a long
run, and your game selection will differ slightly (see step 4c). The curves should look the
same, though.

**What the run writes,** in `workspace/runs/restower_b15c192/`:
- `weights.00001.weights.h5` … `weights.00095.weights.h5`: one checkpoint per epoch (about
  121 MB each, including the optimizer's state for resuming);
- `metadata.json`: every epoch's losses, accuracies, learning rate and timing, plus the
  command-line arguments of every invocation.

**If the run stops** (crash, reboot, you need the GPU back), resume from the latest
checkpoint with the same command plus `--weights`. It continues exactly where it left off:
learning rate, momentum, data position and plateau counters included.

```bash
docker compose run --rm gpu python -m AlphaGo.training.supervised_policy_trainer \
    ... same arguments as above ... --weights weights.00042.weights.h5
```

It refuses to resume if you change `--minibatch`, `--steps-per-epoch`, `--warmup-steps` or
`--lr-schedule`, or if you point `--weights` at a checkpoint other than the latest.

**Changing the learning rate while it runs:** write a number into
`workspace/runs/restower_b15c192/lr_override.txt`. From the end of the next epoch on, the
learning rate is set to it and held there for as long as the file exists.

---

## 7. Play it over GTP

**Pick a checkpoint.** The project's bot uses `weights.00094.weights.h5`. `metadata.json`
lists `"best_epoch": 94`, but that's a 0-based index, so it means `weights.00095`. The two
are practically tied: 00095 has a hair lower validation loss, 00094 a little higher
accuracy.

**The bot runs on the CPU**, so it doesn't need the GPU and can run next to a training job.
Try it by typing GTP commands:

```bash
docker compose run --rm -T gpu python run_gtp_player.py \
    workspace/models/model_restower_b15c192_convnorm.json \
    workspace/runs/restower_b15c192/weights.00094.weights.h5
```

```
name
= NeuralZ

boardsize 19
=

genmove black
= Q16
```

Your move may differ: the bot deliberately samples its first moves. `-T` keeps Docker from attaching a terminal, so GTP's plain text passes through unchanged;
that's also what a GUI needs. Useful options:
- `--sample-moves 20`: how many of the bot's own moves are sampled rather than always the
  top choice (0 for always the top choice);
- `--sample-ratio 0.5`: sampling only chooses among moves at least this fraction as likely
  as the top one, so a clearly best move is always played;
- `--temperature 1.0`: how evenly the sampled moves are chosen among those candidates;
- `--max-moves 800`.

See `python run_gtp_player.py --help`.

**In a GUI** such as [Sabaki](https://sabaki.yichuanshen.de/) or GoGui, add an engine with:
- the engine program `docker`;
- the arguments `compose run --rm -T gpu python run_gtp_player.py
  workspace/models/model_restower_b15c192_convnorm.json
  workspace/runs/restower_b15c192/weights.00094.weights.h5`;
- the repository folder as the working directory.

**On KGS**, run the same command through KGS's `kgsGtp` client, which connects any GTP
engine to the server as a bot account. See the kgsGtp documentation for its configuration.

---

## 8. Optional extras

**See what the network is thinking.** Render its move probabilities as a heatmap for every
position of a game:

```bash
docker compose run --rm gpu python -m benchmarks._plot_sgf_heatmaps \
    workspace/models/model_restower_b15c192_convnorm.json \
    workspace/runs/restower_b15c192/weights.00094.weights.h5 \
    <some game>.sgf workspace/heatmaps
```

**Train a different size, or on a smaller GPU.** Create another model with `make_model`
(e.g. `--blocks 10 --filters 128` trains faster), or use a smaller `--minibatch`. Either way
the learning rate needs re-tuning. An LR range test sweeps the learning rate upward
exponentially over a short run and logs the loss at each step, on exactly the pipeline
training uses:

```bash
docker compose run --rm gpu python -m AlphaGo.training.lr_range_test \
    <model.json> /data workspace/runs/range_test \
    --minibatch 512 --steps-per-epoch 2000 --epochs 2 --mixed-precision --seed 1
```

Read `workspace/runs/range_test/step_diagnostics.jsonl` for where the loss stops falling
and becomes unstable. Treat that as a ceiling to stay well under, and confirm a candidate
learning rate with a short training run before committing to a long one.

---

## 9. Troubleshooting

**The GPU check prints `[]`.**
- **Linux:** make sure the NVIDIA Container Toolkit is installed and Docker was restarted
  afterwards; `docker run --rm --gpus all ubuntu nvidia-smi` should list your GPU.
- **Windows:** update the NVIDIA driver and check Docker Desktop uses the WSL 2 backend.

**`the following arguments are required: ...`** The trainer has no defaults for the
options that depend on your model, data and GPU (`--minibatch`, `--epochs`,
`--learning-rate`, `--warmup-steps`, `--lr-schedule`), so a forgotten one can't quietly
run a different experiment.

**`Set TRAINING_DATA_DIR in .env`.** Create `.env` from `.env.example` (see
[Install](#2-install)).

**Out of GPU memory at the start of training.** Lower `--minibatch`, then re-tune the
learning rate (see [Optional extras](#8-optional-extras)). Also close other programs
using the GPU: TensorFlow reserves most of the card's memory when it starts.

**`Model JSON file expects features ... But shards contain ...`** The model and the shards
were built with different feature lists. Recreate the model with
`--features-from workspace/prod_40m/shards`.

**`... has no packed_states - a shard from before positions were stored bit-packed`.**
Shards built by an older version of this repo. Rebuild them with `convert_shuffled`
([step 4](#4-build-the-training-data)).

**Paths like `/data` turn into `C:/Program Files/Git/data` (Git Bash).** Prefix the
command with `MSYS_NO_PATHCONV=1`.

**A resume refuses to start.** The message says which setting changed, or which checkpoint
it expected. Resume with exactly the original arguments plus `--weights` pointing at the
latest checkpoint.
