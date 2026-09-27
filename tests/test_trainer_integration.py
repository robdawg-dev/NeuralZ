"""End-to-end tests for run_training() (AlphaGo/training/supervised_policy_trainer.py):
real CLI args, real shards (built from synthetic games by convert_shuffled), a tiny
one-layer policy, a few steps per epoch on CPU.

Marked slow: every run_training() call compiles with XLA (jit_compile=True), which takes a
few seconds on CPU. Deselect with `pytest -m "not slow"`.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import glob
import json
import shutil

import pytest
from keras import mixed_precision

import AlphaGo.training.supervised_policy_trainer as trainer
from AlphaGo.models.policy import CNNPolicy
from tests.test_convert_shuffled import FEATURES, _run, _selection

pytestmark = pytest.mark.slow

# 4 steps per epoch, warmup finished inside the first epoch.
MINIBATCH = 16
EPOCH_LENGTH = 64
STEPS_PER_EPOCH = EPOCH_LENGTH // MINIBATCH
COMMON = ["--minibatch", str(MINIBATCH), "--epoch-length", str(EPOCH_LENGTH),
          "--validation-length", "32", "--warmup-steps", "2", "--learning-rate", "0.05",
          "--seed", "1"]


@pytest.fixture(autouse=True)
def _restore_global_mixed_precision_policy():
    original = mixed_precision.global_policy()
    yield
    mixed_precision.set_global_policy(original)


def _model_json(path, features):
    policy = CNNPolicy(features, layers=1, filters_per_layer=4, filter_width_1=3)
    policy.save_model(str(path))
    return str(path)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """(model_json, train_data_dir) shared by every test in this file."""
    tmp = tmp_path_factory.mktemp("trainer")
    sel = _selection(tmp, {"train": [(s, None) for s in range(12)],
                           "val": [(100 + s, None) for s in range(3)]})
    shards = str(tmp / "shards")
    _run(sel, shards, "--splits", "train", "val")
    return _model_json(tmp / "model.json", FEATURES.split(",")), shards


def _train(data, out_dir, *extra):
    model_json, shards = data
    trainer.run_training([model_json, shards, str(out_dir)] + COMMON + list(extra))


def _metadata(out_dir):
    with open(os.path.join(str(out_dir), "metadata.json")) as f:
        return json.load(f)


def _truncate_to(src, dst, n_epochs):
    """Copy a finished run directory as it would have looked had it stopped after
    n_epochs: that epoch's checkpoint, and metadata with only the first n_epochs."""
    shutil.copytree(str(src), str(dst))
    meta = _metadata(dst)
    meta["epochs"] = meta["epochs"][:n_epochs]
    meta["best_epoch"] = min(meta["best_epoch"], n_epochs - 1)
    with open(os.path.join(str(dst), "metadata.json"), "w") as f:
        json.dump(meta, f)
    return "weights.{:05d}.weights.h5".format(n_epochs)


@pytest.fixture(scope="module")
def cosine_run(data, tmp_path_factory):
    """A fresh 3-epoch cosine run - the uninterrupted reference for the resume tests."""
    out = tmp_path_factory.mktemp("cosine") / "out"
    _train(data, out, "--epochs", "3")
    return out


# --- fresh runs ------------------------------------------------------------------------

def test_fresh_run_writes_checkpoints_and_metadata(data, cosine_run):
    weights = sorted(os.path.basename(p) for p in glob.glob(str(cosine_run / "weights.*")))
    assert weights == ["weights.00001.weights.h5", "weights.00002.weights.h5",
                       "weights.00003.weights.h5"]

    meta = _metadata(cosine_run)
    assert len(meta["epochs"]) == 3
    for epoch in meta["epochs"]:
        for key in ("loss", "val_loss", "accuracy", "top5_accuracy", "prediction_entropy",
                    "learning_rate", "epoch_seconds", "steps_per_second"):
            assert key in epoch
    assert meta["training_data"] == data[1]
    assert meta["model_file"] == data[0]
    assert len(meta["cmd_line_args"]) == 1
    assert meta["cmd_line_args"][0]["minibatch"] == MINIBATCH
    # Synthetic shards are well-formed, so the batch sanity checker has nothing to report.
    assert not (cosine_run / "batch_sanity_log.json").exists()


def test_cosine_lr_decays_after_warmup(cosine_run):
    lrs = [e["learning_rate"] for e in _metadata(cosine_run)["epochs"]]
    assert lrs[0] < 0.05
    assert lrs == sorted(lrs, reverse=True)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-6)  # cosine reaches its floor at total_steps


# --- resume ----------------------------------------------------------------------------

def test_cosine_resume_continues_the_lr_curve(data, cosine_run, tmp_path):
    """A run resumed after epoch 2 logs the same epoch-3 learning rate as the
    uninterrupted run - i.e. it did not re-enter warmup or restart the decay."""
    resumed = tmp_path / "resumed"
    weights = _truncate_to(cosine_run, resumed, 2)
    _train(data, resumed, "--epochs", "3", "--weights", weights)

    meta = _metadata(resumed)
    assert len(meta["epochs"]) == 3
    assert len(meta["cmd_line_args"]) == 2
    reference = _metadata(cosine_run)["epochs"][2]["learning_rate"]
    assert meta["epochs"][2]["learning_rate"] == pytest.approx(reference, abs=1e-7)


def test_plateau_fresh_run(data, tmp_path):
    out = tmp_path / "plateau"
    _train(data, out, "--epochs", "2", "--lr-schedule", "plateau")
    assert (out / "optimizer_state.npz").exists()
    lrs = [e["learning_rate"] for e in _metadata(out)["epochs"]]
    assert lrs == pytest.approx([0.05, 0.05])  # warmup finished within epoch 1, no cut yet


def _plateau_run_cut_in_its_last_epoch(data, out):
    """One plateau epoch whose LR is cut (here by the override file) at the epoch's end,
    after TrainingDiagnosticsCallback logged the pre-cut 0.05. metadata.json then holds
    the LR epoch 1 used; optimizer_state.npz, saved after the cut, holds 0.007."""
    out.mkdir()
    (out / "lr_override.txt").write_text("0.007")
    _train(data, out, "--epochs", "1", "--lr-schedule", "plateau")
    (out / "lr_override.txt").unlink()
    assert _metadata(out)["epochs"][0]["learning_rate"] == pytest.approx(0.05)


def test_plateau_resume_keeps_a_cut_made_in_the_last_epoch(data, tmp_path):
    """optimizer_state.npz stores the optimizer's learning_rate variable alongside
    momentum, so restoring it overrides the metadata-derived initial LR with the
    post-cut value - the resume does not undo the cut or re-run warmup."""
    out = tmp_path / "plateau"
    _plateau_run_cut_in_its_last_epoch(data, out)
    _train(data, out, "--epochs", "2", "--lr-schedule", "plateau",
           "--weights", "weights.00001.weights.h5")
    assert _metadata(out)["epochs"][1]["learning_rate"] == pytest.approx(0.007, rel=1e-5)


def test_plateau_resume_without_optimizer_state_uses_last_logged_lr(data, tmp_path):
    """Without optimizer_state.npz the resume starts from the last *logged* LR - the one
    that epoch ran at, before its end-of-epoch cut - so a cut made in the final epoch
    before stopping is lost. Pinned here as current behavior."""
    out = tmp_path / "plateau"
    _plateau_run_cut_in_its_last_epoch(data, out)
    (out / "optimizer_state.npz").unlink()
    _train(data, out, "--epochs", "2", "--lr-schedule", "plateau",
           "--weights", "weights.00001.weights.h5")
    assert _metadata(out)["epochs"][1]["learning_rate"] == pytest.approx(0.05, rel=1e-5)


def test_plateau_resume_honors_lr_override_file(data, tmp_path):
    out = tmp_path / "plateau"
    _train(data, out, "--epochs", "1", "--lr-schedule", "plateau")
    (out / "lr_override.txt").write_text("0.007")
    _train(data, out, "--epochs", "3", "--lr-schedule", "plateau",
           "--weights", "weights.00001.weights.h5")
    lrs = [e["learning_rate"] for e in _metadata(out)["epochs"]]
    # Epoch 2 still runs at the resumed LR; the override applies at its end.
    assert lrs[1:] == pytest.approx([0.05, 0.007], rel=1e-5)


# --- LR range test ---------------------------------------------------------------------

def test_lr_range_test_writes_step_diagnostics(data, tmp_path):
    out = tmp_path / "range"
    _train(data, out, "--epochs", "2", "--lr-range-test", "--range-check-every", "1",
           "--range-warmup-steps", "2", "--range-floor-lr", "1e-3",
           "--range-ceiling-lr", "0.5")
    with open(str(out / "step_diagnostics.jsonl")) as f:
        records = [json.loads(line) for line in f]
    assert [r["step"] for r in records] == list(range(2 * STEPS_PER_EPOCH))
    lrs = [r["lr"] for r in records]
    assert lrs == sorted(lrs)
    # The last step logged is 5 of the 6 sweep steps; the ceiling itself is only reached
    # at total_steps, one past the end.
    assert lrs[-1] == pytest.approx(1e-3 * (0.5 / 1e-3) ** (5 / 6), rel=1e-5)
    # grad_norm only reaches the logs through the monkey-patched train_step.
    assert all(r["grad_norm"] is not None and r["grad_norm"] > 0 for r in records)
    assert all(r["loss_scale"] is None for r in records)  # float32: no loss scaling


# --- argument guards -------------------------------------------------------------------

def test_resume_with_no_epochs_left_raises(data, cosine_run, tmp_path):
    out = tmp_path / "done"
    shutil.copytree(str(cosine_run), str(out))
    with pytest.raises(ValueError, match="already has 3 recorded epochs"):
        _train(data, out, "--epochs", "3", "--weights", "weights.00003.weights.h5")


@pytest.mark.parametrize("flag,value", [("--minibatch", "8"), ("--epoch-length", "32"),
                                        ("--warmup-steps", "3")])
def test_resume_rejects_changed_step_accounting(data, cosine_run, tmp_path, flag, value):
    out = tmp_path / "resumed"
    weights = _truncate_to(cosine_run, out, 2)
    model_json, shards = data
    args = [model_json, shards, str(out)] + COMMON + ["--epochs", "3", "--weights", weights]
    args[args.index(flag) + 1] = value
    with pytest.raises(ValueError, match="changed across resume"):
        trainer.run_training(args)


def test_unknown_symmetry_raises(data, tmp_path):
    with pytest.raises(ValueError, match="unknown symmetries"):
        _train(data, tmp_path / "out", "--epochs", "1", "--symmetries", "noop,twirl")


def test_model_and_shard_feature_mismatch_raises(data, tmp_path):
    other = _model_json(tmp_path / "other.json", ["board", "ones"])
    with pytest.raises(ValueError, match="Model JSON file expects features"):
        trainer.run_training([other, data[1], str(tmp_path / "out"), "--epochs", "1"])
