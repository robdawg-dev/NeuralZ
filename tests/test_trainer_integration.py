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

import numpy as np
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


def _args(data, out_dir, *extra):
    model_json, shards = data
    return [model_json, shards, str(out_dir)] + COMMON + list(extra)


def _train(data, out_dir, *extra):
    trainer.run_training(_args(data, out_dir, *extra))


class _Interrupted(Exception):
    pass


def _train_interrupted(data, out_dir, after_epochs, *extra):
    """Runs training but stops it right after epoch `after_epochs` has been fully
    written (checkpoint, optimizer state, metadata) - a run killed between epochs."""
    original = trainer.MetadataWriterCallback.on_epoch_end

    def on_epoch_end(self, epoch, logs=None):
        original(self, epoch, logs)
        if len(self.metadata["epochs"]) == after_epochs:
            raise _Interrupted()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(trainer.MetadataWriterCallback, "on_epoch_end", on_epoch_end)
        with pytest.raises(_Interrupted):
            _train(data, out_dir, *extra)


def _metadata(out_dir):
    with open(os.path.join(str(out_dir), "metadata.json")) as f:
        return json.load(f)


def _state_path(out_dir):
    return os.path.join(str(out_dir), trainer.OPTIMIZER_STATE_FILE)


@pytest.fixture(scope="module")
def cosine_run(data, tmp_path_factory):
    """A fresh 3-epoch cosine run - the uninterrupted reference for the resume tests."""
    out = tmp_path_factory.mktemp("cosine") / "out"
    _train(data, out, "--epochs", "3")
    return out


@pytest.fixture(scope="module")
def interrupted_cosine_run(data, tmp_path_factory):
    """The same run as cosine_run, stopped after epoch 2. Tests resume from a copy."""
    out = tmp_path_factory.mktemp("interrupted") / "out"
    _train_interrupted(data, out, 2, "--epochs", "3")
    return out


@pytest.fixture
def stopped(interrupted_cosine_run, tmp_path):
    out = tmp_path / "stopped"
    shutil.copytree(str(interrupted_cosine_run), str(out))
    return out


RESUME_COSINE = ("--epochs", "3", "--weights", "weights.00002.weights.h5")


# --- fresh runs ------------------------------------------------------------------------

def test_fresh_run_writes_checkpoints_state_and_metadata(data, cosine_run):
    weights = sorted(os.path.basename(p) for p in glob.glob(str(cosine_run / "weights.*")))
    assert weights == ["weights.00001.weights.h5", "weights.00002.weights.h5",
                       "weights.00003.weights.h5"]
    with np.load(_state_path(cosine_run)) as f:
        assert int(f["completed_epochs"]) == 3

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

def test_cosine_resume_matches_the_uninterrupted_run(data, cosine_run, stopped):
    """Resumed after epoch 2, the run's epoch 3 matches the uninterrupted run's: same
    weights, same data stream position, and the optimizer's momentum and step count
    restored, so the same learning rate and the same loss."""
    _train(data, stopped, *RESUME_COSINE)

    meta = _metadata(stopped)
    assert len(meta["epochs"]) == 3
    assert len(meta["cmd_line_args"]) == 2
    resumed, reference = meta["epochs"][2], _metadata(cosine_run)["epochs"][2]
    assert resumed["learning_rate"] == pytest.approx(reference["learning_rate"], abs=1e-7)
    assert resumed["loss"] == pytest.approx(reference["loss"], rel=1e-4)
    assert resumed["val_loss"] == pytest.approx(reference["val_loss"], rel=1e-4)


def test_plateau_fresh_run(data, tmp_path):
    out = tmp_path / "plateau"
    _train(data, out, "--epochs", "2", "--lr-schedule", "plateau")
    with np.load(_state_path(out)) as f:
        assert int(f["completed_epochs"]) == 2
    lrs = [e["learning_rate"] for e in _metadata(out)["epochs"]]
    assert lrs == pytest.approx([0.05, 0.05])  # warmup finished within epoch 1, no cut yet


def test_plateau_resume_keeps_a_cut_made_in_the_last_epoch(data, tmp_path):
    """The LR is cut (here by the override file) at the end of the run's last epoch,
    after TrainingDiagnosticsCallback logged the pre-cut 0.05. The optimizer state saved
    after the cut carries it into the resume - no undoing the cut, no re-run warmup."""
    out = tmp_path / "plateau"
    out.mkdir()
    (out / "lr_override.txt").write_text("0.007")
    _train(data, out, "--epochs", "1", "--lr-schedule", "plateau")
    (out / "lr_override.txt").unlink()
    assert _metadata(out)["epochs"][0]["learning_rate"] == pytest.approx(0.05)

    _train(data, out, "--epochs", "2", "--lr-schedule", "plateau",
           "--weights", "weights.00001.weights.h5")
    assert _metadata(out)["epochs"][1]["learning_rate"] == pytest.approx(0.007, rel=1e-5)


def test_plateau_resume_honors_lr_override_file(data, tmp_path):
    out = tmp_path / "plateau"
    _train(data, out, "--epochs", "1", "--lr-schedule", "plateau")
    (out / "lr_override.txt").write_text("0.007")
    _train(data, out, "--epochs", "3", "--lr-schedule", "plateau",
           "--weights", "weights.00001.weights.h5")
    lrs = [e["learning_rate"] for e in _metadata(out)["epochs"]]
    # Epoch 2 still runs at the resumed LR; the override applies at its end.
    assert lrs[1:] == pytest.approx([0.05, 0.007], rel=1e-5)


# --- resume guards ---------------------------------------------------------------------

def test_resume_without_optimizer_state_raises(data, stopped):
    os.remove(_state_path(stopped))
    with pytest.raises(ValueError, match="optimizer_state.npz not found"):
        _train(data, stopped, *RESUME_COSINE)


def test_resume_with_stale_optimizer_state_raises(data, stopped):
    with np.load(_state_path(stopped)) as f:
        store = dict(f)
    store["completed_epochs"] = np.int64(1)
    np.savez(_state_path(stopped), **store)
    with pytest.raises(ValueError, match="saved after epoch 1, but metadata.json records 2"):
        _train(data, stopped, *RESUME_COSINE)


def test_resume_from_an_older_checkpoint_raises(data, stopped):
    with pytest.raises(ValueError, match="checkpoint from epoch 1, but .* records 2"):
        _train(data, stopped, "--epochs", "3", "--weights", "weights.00001.weights.h5")


def test_resume_with_no_epochs_left_raises(data, stopped):
    with pytest.raises(ValueError, match="already has 2 recorded epochs"):
        _train(data, stopped, "--epochs", "2", "--weights", "weights.00002.weights.h5")


@pytest.mark.parametrize("flag,value", [("--minibatch", "8"), ("--epoch-length", "32"),
                                        ("--warmup-steps", "3"),
                                        ("--lr-schedule", "plateau")])
def test_resume_rejects_changed_settings(data, stopped, flag, value):
    args = _args(data, stopped, *RESUME_COSINE)
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
    with pytest.raises(ValueError, match="changed across resume"):
        trainer.run_training(args)


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
    # A range test starts every sweep with a fresh optimizer, so it saves no state.
    assert not os.path.exists(_state_path(out))


# --- other argument guards -------------------------------------------------------------

def test_unknown_symmetry_raises(data, tmp_path):
    with pytest.raises(ValueError, match="unknown symmetries"):
        _train(data, tmp_path / "out", "--epochs", "1", "--symmetries", "noop,twirl")


def test_model_and_shard_feature_mismatch_raises(data, tmp_path):
    other = _model_json(tmp_path / "other.json", ["board", "ones"])
    with pytest.raises(ValueError, match="Model JSON file expects features"):
        trainer.run_training([other, data[1], str(tmp_path / "out"), "--epochs", "1"])
