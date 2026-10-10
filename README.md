# NeuralZ

A Go-playing policy network: supervised training on KataGo self-play games, and a bot that
plays on KGS at about 4 dan with no search at all. Each move is one forward pass of the
network.

NeuralZ is a modernization of [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaGo), a
2016 re-implementation of AlphaGo that is no longer maintained. Of AlphaGo's networks, only
the policy network was ever fully built there, and that is what this project trains and
plays.

- **Train** a residual policy network on a single consumer GPU: a two-pass pipeline turns
  KataGo SGFs into globally shuffled, bit-packed shards, and a resumable trainer streams
  them with warmup, a plateau or cosine learning rate schedule, and mixed precision.
- **Play** over GTP: `go_server.py` holds the network on the CPU and batches requests from
  any number of `go_client.py` bots, the GTP engines a controller such as kgsGtp starts.
  A small KataGo network judges finished games (dead stones, cleanup moves, and whether
  the board is settled enough to pass).

## Models

| Model | Architecture | Training data | Held-out move accuracy (top-1 / top-5) | KGS strength |
|---|---|---|---|---|
| **b20c256** (deployed) | pre-activation ResNet with global pooling, 20 blocks × 256 filters | 60M positions of KataGo self-play; 67 epochs, ~43 h | 57.3% / 89.2% | ~4d |
| b15c192 | ResNet, 15 blocks × 192 filters | 37M positions of KataGo self-play; 95 epochs, ~39 h | 55.6% / 87.4% | ~4d |
| b10c128 | ResNet, 10 blocks × 128 filters | KataGo self-play; 125 epochs, ~23 h | 51.8% / 84.2% | ~4d |

All were trained on one RTX 4070 SUPER. Accuracy is how often the network's top move
matches KataGo's on held-out games. Head to head the order is clear: in a 30-games-per-pair
playoff b20c256 beat b15c192 25-5 and b15c192 beat b10c128 23-7 (MEASUREMENTS.md). Their
KGS ranks are closer, since a policy-only bot's limits (reading, life and death, the
endgame) matter there as much as a few points of accuracy.
[SAMPLE_RUN.md](SAMPLE_RUN.md) reproduces the b15c192 run from scratch.

**History.** In 2016, RocAlphaGo's 12-layer, 96-filter CNN trained on GoGoD games reached
about 2 kyu on KGS. In 2017, the same network trained on KGS games, close to the original
AlphaGo policy network, reached about 2 dan; that run took two months on a GTX 1060.
NeuralZ began in 2026: it moved the project to uv, Docker and current TensorFlow/Keras,
then added residual networks and KataGo self-play data.

## Quick start

Training needs Docker and an NVIDIA GPU; everything else also runs natively on the CPU.

**With Docker** (Linux, or Windows with Docker Desktop):

```bash
cp .env.example .env               # then set TRAINING_DATA_DIR (see SAMPLE_RUN.md)
docker compose build
docker compose run --rm gpu python setup_cython.py build_ext --inplace
docker compose run --rm -e CUDA_VISIBLE_DEVICES= gpu python -m pytest tests -q -m "not slow"
```

**Natively**, with [uv](https://docs.astral.sh/uv/) and a C++ compiler (`build-essential`
on Linux, the Visual Studio Build Tools on Windows):

```bash
uv sync --extra gtp
uv run python setup_cython.py build_ext --inplace
uv run pytest tests -q -m "not slow"
```

The board engine and feature planes are Cython extensions; rebuild them after changing a
`.pyx` or `.pxd` file.

## Playing

Start the server with a model, then point any GTP controller at the client:

```bash
python go_server.py <model.json> <weights.h5>   # loads the network; serves on 127.0.0.1:5005
python go_client.py                             # the GTP engine: stdin/stdout
```

- `--symmetries 4` or `8` on the server averages the network over board rotations and
  reflections, which is a little stronger at 2-4x the CPU.
- `--katago <exe> --katago-model <net>` on the server adds KataGo end-of-game judging.
  Without it, the bots fall back to GNU Go for dead stones.
- The client's sampling options (`--sample-moves`, `--sample-ratio`, `--temperature`) vary
  the opening; after that, the bot plays its top move. See `python go_client.py --help`.

For KGS, [deploy/build_deploy.py](deploy/build_deploy.py) assembles a self-contained folder
for a CPU-only Ubuntu server: the server, the client, KataGo, a locked environment and
install scripts. Its [README](deploy/templates/README.md) covers installing it and the
kgsGtp setup.

## Training

The full walkthrough, from downloading KataGo's games to a trained network, is
[SAMPLE_RUN.md](SAMPLE_RUN.md). In outline:

1. **Prepare the SGFs**: `sgf_cull` deletes files that will never be used (duplicates
   among them), `sgf_preparation` builds a manifest of the rest, and `select_games` chooses
   the games and splits them into train, validation and test sets by game.
2. **Build shards**: `convert_shuffled` writes each split as one uniformly random
   permutation of its positions, bit-packed.
3. **Create a model**: `python -m AlphaGo.models.make_model newres --blocks 20 --filters 256`.
4. **Find a learning rate**: `AlphaGo/training/lr_range_test.py`.
5. **Train**: `AlphaGo/training/supervised_policy_trainer.py`. A run resumes from its
   latest checkpoint exactly where it stopped: optimizer state, data position and plateau
   counters included.
6. **Evaluate**: `play_tests/match_networks.py` plays two networks against each other;
   decide the winners with KataGo (`tools/match_winners.py`) rather than the built-in
   count, which misjudges dead stones. `tools/playoff.py` does both for every pair of a
   list of networks and reports a crosstable and Elo.

[DATA_PIPELINE.md](DATA_PIPELINE.md) explains what each data stage decides and why.

## Repository layout

| Path | What's there |
|---|---|
| `AlphaGo/go/` | the board engine (Cython): legal moves, captures, ko and superko, ladders |
| `AlphaGo/preprocessing/` | feature planes (Cython) and the SGF-to-shard pipeline |
| `AlphaGo/models/` | network architectures and `make_model` |
| `AlphaGo/training/` | the trainer, the LR range test and the shard reader |
| `AlphaGo/ai.py` | move choice: sampling, the ladder guard |
| `interface/` | the GTP engine and the KataGo scorer |
| `go_server.py`, `go_client.py` | the bot: inference server and GTP client |
| `play_tests/` | network-vs-network matches, self-play and timing |
| `tools/` | command-line tools: game reports and reviews, network playoffs and benchmarks, GTP logs, model releases ([list](tools/README.md)) |
| `deploy/` | the KGS server bundle |
| `tests/` | the test suite |
| `deprecated/` | superseded code, kept for reference ([README](deprecated/README.md)) |

## Development

- **Tests:** `pytest tests -m "not slow"` runs in under a minute. Tests marked `slow` run
  full training steps and start the bot as separate processes, and take a few minutes.
- **Coverage:** `coverage run -m pytest tests && coverage combine && coverage report`
  (configured in `pyproject.toml`, including worker processes).
- **Lint:** `flake8` (`.flake8`) and `uv run cython-lint AlphaGo` for the Cython files.
- **Dependencies** are pinned in `uv.lock`. The Docker image builds from it, so the
  container and a native environment run the same versions. Keras is pinned exactly:
  check a model's outputs across an upgrade with `tools/policy_reference.py`.

[MEASUREMENTS.md](MEASUREMENTS.md) collects the measurements behind the defaults: the
sampling settings, the KataGo scorer, inference speed and more.

## License

MIT; see [LICENSE](LICENSE). Originally © 2016 University of Rochester (RocAlphaGo).
