# Plan: Python 3.13 and a lockfile-driven Docker image, then TensorFlow 2.22

Two phases, each a separately verified change:

1. **Stay on TF 2.21.0; move to Python 3.13; build the Docker image from `uv.lock`.**
   **Done 2026-09-27** - see [Phase 1 results](#phase-1-results-2026-09-27).
2. **After TF 2.22.0 final is released: upgrade TF (and Keras).** Now mostly a version bump.

### What must keep working
- **No training runs need to continue** across either phase. Checkpoint resume across versions
  is therefore not a requirement; resume *within* an environment is covered by the test suite.
- **The GTP bot's model is used for inference only, and must behave the same after each phase:**
  - model: `workspace/models/model_restower_b15c192_convnorm.json` (`ResTowerPolicy`);
  - weights:
    `benchmarks/_restower_b15c192_v4shuf40m_mb1024_lr1p6_seed90001/weights.00094.weights.h5`.
  - Both paths are gitignored; copies are in `C:\Users\winst\Documents\NeuralZ-backup\2026-09-27\`.
  - **The bot runs CPU-only**: `run_gtp_player.py` hides the GPU before importing TF. So the CPU
    comparison below decides whether the bot is unaffected.
  - The run was trained with `--mixed-precision`. The model JSON records no dtype, so the bot
    loads the model in float32, which is fine: mixed precision stores weights in float32.
  - Note: that run's `metadata.json` has `best_epoch: 94`, a 0-based index meaning
    `weights.00095` (val_loss 1.4836, val_accuracy 0.5548). `weights.00094` was chosen
    deliberately (val_loss 1.4837, val_accuracy 0.5558).
- **Future training should still work** (GPU, XLA, comparable speed).

### Verification tools (reused in phase 2)
- **Bot model check:** `benchmarks/_capture_policy_reference.py`. It evaluates the model on 48
  fixed positions (8 test-split games, 2 of them handicap games, 6 plies each) and either saves
  the raw move probabilities (`capture`) or compares against a saved file (`compare`: same top
  move and top-5 set everywhere, max abs difference ≤ 1e-3):

      python -m benchmarks._capture_policy_reference compare MODEL WEIGHTS REF.npz --device cpu

  References from the TF 2.21 / Python 3.11 image: `workspace/upgrade_refs/ref_cpu.npz` and
  `ref_gpu.npz`. On the same image and device a capture is bit-for-bit repeatable; CPU vs GPU
  differ by up to ~6e-4 with identical top moves.
- **Training check:** a short run with the real run's settings (minibatch 1024, mixed precision,
  plateau schedule, seed 90001), 3 epochs × 150 steps on a small fixed dataset built from three
  former `val` shards, `workspace/upgrade_refs/data/` (`train/` 2 shards, `val/` 1). (The
  original `prod_40m` train shards were deleted as no longer needed.) Results for each image
  are in `workspace/upgrade_refs/train_<image tag>/metadata.json`.
- **Images:** `neuralz-gpu:latest` = `neuralz-gpu:keras315` (current). Rollback:
  `neuralz-gpu:py313` (phase 1, Keras 3.13.2) and `neuralz-gpu:tf2.21-py311` (the old
  official-image build).

---

## Phase 1: Python 3.13 + lockfile-driven Docker, on TF 2.21.0 (done)

### What changed
- **`pyproject.toml`:**
  - `requires-python = ">=3.13"`;
  - `keras==3.13.2`, the version the official image had, so phase 1 changed neither TF nor
    Keras;
  - new Linux-only `gpu` extra: `tensorflow[and-cuda]==2.21.0`.
- **`.python-version`:** 3.13; `uv.lock` re-resolved.
- **`Dockerfile`:**
  - `ubuntu:24.04` + uv 0.12.9;
  - uv-managed Python in `/opt/python`, environment in `/opt/venv`;
  - `uv sync --locked --extra gpu --extra gtp --extra viz --group dev`.
- **`.dockerignore`:** only `pyproject.toml`, `uv.lock` and `.python-version` go into the build
  context. Before, every build sent the whole repo, including `game_data/`.
- **Cython extensions** rebuilt for 3.13. The `cpython-313` files sit beside the `cpython-311`
  ones, which the rollback image still uses.
- **`AlphaGo/util.py`:** a fix for the `sgf` library on Python 3.13 (see below).

### Phase 1 results (2026-09-27)
| Check | Result |
|---|---|
| GPU visible, CUDA/cuDNN | yes, after the linker fix below. The lock picked cuDNN 9.26 (was 9.5.1); no need to pin back. |
| Test suite | 355 passed on the new image, and on the rollback image |
| Bot model, CPU | **bit-for-bit identical** to the old image on all 48 positions |
| Bot model, GPU | max diff 5.95e-4, identical top-1/top-5 moves (cuDNN 9.5 → 9.26) |
| GTP smoke test (`run_gtp_player.py`) | all commands answered, sensible moves |
| Training, epochs 2–3 | 1.96 steps/s vs 1.85 on the old image (+6%) |
| Training, epoch 1 (XLA compile) | 1.26 steps/s vs 0.53 |
| Training loss | matches to ~3 decimals (4.3612/3.0886/2.7979 vs 4.3611/3.0916/2.8002) |
| XLA | no autotuner errors - and the old workaround is no longer needed (see below) |
| Image size | 11.2 GB (was 15 GB) |

### Problems found, and their fixes
- **TF didn't see the GPU at first.** TF 2.21's own library search paths (RUNPATH) cover only
  some of the pip CUDA packages. They omit cusolver, nvjitlink, curand and nvrtc, so loading
  failed with `Could not load dynamic library 'libcusolver.so.11'`. The old image never hit this
  because it had a system-wide CUDA toolkit. **Fix (Dockerfile):** every
  `site-packages/nvidia/*/lib` directory is registered with `ldconfig`. Check again in phase 2
  whether TF 2.22 still needs it.
- **`sgf` 0.5 breaks on Python 3.13.** Its `NodeIterator` has `__next__` but no `__iter__`.
  Plain `for` loops over `game.rest` still work, but comprehensions and generator expressions
  now raise `TypeError`. That hit `sgf_analyze.scan_file_deep` and one test; the main pipeline
  (`sgf_iter_states`) uses a plain loop. **Fix:** `AlphaGo/util.py` completes the iterator
  protocol when imported; `tests/test_util.py` pins this. `sgf` 0.5 is unmaintained, so a
  replacement library is worth considering eventually.

### XLA workaround removed
The old image needed `--xla_gpu_force_conv_nhwc=true` (`AlphaGo/training/xla_workarounds.py`,
plus `XLA_FLAGS` in `docker-compose.yml`). Without it, **float32** training crashed on the first
step with `Autotuner could not find any supported configs`. Tested on both images without it:

| | Old image (cuDNN 9.5) | New image (cuDNN 9.26) |
|---|---|---|
| Mixed precision | trains | trains |
| Float32, minibatch 256 | **crashes** (autotuner) | trains, same speed as with the flag |
| Float32, minibatch 1024 | crashes | out of GPU memory with or without the flag (too big for 12 GB) |

So it was only ever needed for float32 on the older cuDNN. It was removed, along with the
compose variable. Consequence: **the rollback image `neuralz-gpu:tf2.21-py311` now crashes on
float32 training** with the current code (mixed precision is unaffected). One side note from the
test: at float32 minibatch 1024, the run without the flag asked for 17.7 GB and the one with it
12 GB, so the forced layout may use less memory.

### Not done (optional)
- A repo script for running the full test suite in Docker. The one-liner used:

      docker run --rm -e CUDA_VISIBLE_DEVICES= -e MPLBACKEND=Agg \
          -v "<repo>:/workspace" neuralz-gpu python -m pytest tests -q

  (after building the Cython extensions in the repo for the image's Python).
- Removing the `cpython-311` extension files, once the rollback image is no longer wanted.

---

## Keras 3.15.1 and a dependency refresh, still on TF 2.21.0 (done 2026-09-27)

- **Keras 3.13.2 → 3.15.1**, and a full `uv lock --upgrade`: numpy 2.4.6 → 2.5.3, gast 0.4.0 →
  0.7.0, grpcio 1.83.1 → 1.84.0, wrapt 2.4.0 → 2.5.0, and patch updates (protobuf, idna,
  urllib3, pyparsing, fonttools).
- **Not upgraded:**
  - h5py stays at 3.14.0: TF 2.21 caps it below 3.15.
  - pygtp stays at 0.3. 0.4 (2017) only adds a resign constant and lowercases `PASS`, which
    would mean adjusting the GTP wrapper for no benefit.
- **Trainer fix needed first:** from Keras 3.15, `ReduceLROnPlateau.on_train_begin()` also resets
  `best` to `None`. The trainer used to set a resumed `best` before `fit()`, relying on 3.13
  never resetting it, so a resumed plateau run would have treated its first epoch as an
  automatic improvement. `_PlateauStateRestorer` now re-applies `best` along with
  `wait`/`cooldown_counter` (correct on both versions), and a test mimics the 3.15 reset.

| Check | Result |
|---|---|
| Test suite | 356 passed |
| Bot model, CPU | still bit-for-bit identical to the original image |
| Bot model, GPU | unchanged from phase 1 (max diff 5.95e-4, identical top moves) |
| GTP smoke test | clean |
| Training, 3 epochs | loss, accuracy, val_loss, val_accuracy and entropy identical to 4 decimals vs Keras 3.13.2; epochs 2–3 1.97 vs 1.91 steps/s (noise) |

---

## Phase 2: TensorFlow 2.22 (once 2.22.0 final is out)

### Known so far (from 2.22.0rc0, 2026-09-23; recheck on the final)
- **Python builds 3.10–3.14** (2.21 stops at 3.13).
- **`and-cuda` requires cuDNN ≥ 9.10.2.21.** Already satisfied: phase 1 runs 9.26.
- **h5py still `<3.15`.**
- **TensorBoard no longer installed by default.** No impact: nothing in the repo uses it.
- **rc0 depends on `keras-nightly`,** so wait for the final, which should pin a stable Keras.

### Steps
- [ ] Keep the current image for rollback (it's already tagged `neuralz-gpu:keras315`).
- [ ] **Decide the Python version:** stay on 3.13, or move to 3.14 only if every compiled
  dependency in the new lock has 3.14 builds. As of 2026-09-27 h5py 3.14 doesn't, and TF caps
  h5py below 3.15. Fall back to 3.13 rather than loosening that pin.
- [ ] `pyproject.toml`: `tensorflow==2.22.0` (in `dependencies` and the `gpu` extra), and
  `keras` set to the version 2.22.0 depends on.
- [ ] `uv lock` and `uv sync`, `docker build -t neuralz-gpu:tf2.22 .`, rebuild the Cython
  extensions if the Python version changed.
- [ ] GPU visible (and whether the `ldconfig` step is still needed); test suite passes.
- [ ] **Bot model:** `compare` against `workspace/upgrade_refs/ref_cpu.npz` and `ref_gpu.npz`.
  This is the main check for the Keras upgrade.
- [ ] **Training check** on `workspace/upgrade_refs/data/`: speed and loss vs the phase 1
  results.
- [ ] Switch over: `docker tag neuralz-gpu:tf2.22 neuralz-gpu:latest`.

---

## Risks (phase 2)
- **Keras upgrade:** could change how the bot's weights load or behave. The reference
  comparison catches it.
- **3.14 builds:** h5py may not have them when 2.22 ships.
- **Other old pure-Python dependencies** (`sgf` 0.5, `pygtp` 0.3) are unmaintained; a new
  Python version can surface issues like the `NodeIterator` one. The test suite exercises both.

## References
- TF releases: https://github.com/tensorflow/tensorflow/releases
- TF on PyPI (wheels per Python version, `and-cuda` pins): https://pypi.org/project/tensorflow/
- TF Docker tags: https://hub.docker.com/r/tensorflow/tensorflow/tags
- Python version support: https://devguide.python.org/versions/
- uv and Python versions: https://docs.astral.sh/uv/concepts/python-versions/
