"""End-to-end tests for run_training() (AlphaGo/training/supervised_policy_trainer.py) and
run_range_test() (AlphaGo/training/lr_range_test.py): real CLI args, real shards (built
from synthetic games by convert_shuffled), a tiny one-layer policy, a few steps per epoch
on CPU.

Marked slow: every run_training() call compiles with XLA (jit_compile=True), which takes a
few seconds on CPU. Deselect with `pytest -m "not slow"`.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import glob
import json
import shutil

import h5py
import pytest
from keras import mixed_precision

import AlphaGo.training.lr_range_test as range_test
import AlphaGo.training.supervised_policy_trainer as trainer
from AlphaGo.training.shard_stream import dataset_info, find_split_shards
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
from tests.test_convert_shuffled import FEATURES, _run, _selection

pytestmark = pytest.mark.slow

# 4 steps per epoch, warmup finished inside the first epoch.
MINIBATCH = 16
STEPS_PER_EPOCH = 4
# What every run takes (add_run_arguments), and what only training adds.
RUN_ARGS = ["--minibatch", str(MINIBATCH), "--steps-per-epoch", str(STEPS_PER_EPOCH),
            "--validation-length", "32", "--seed", "1"]
# --lr-schedule cosine unless a test passes another (argparse keeps the last value).
TRAIN_ARGS = RUN_ARGS + ["--warmup-steps", "2", "--learning-rate", "0.05",
                         "--lr-schedule", "cosine"]


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


def _args(data, out_dir, *extra, common=TRAIN_ARGS):
    model_json, shards = data
    return [model_json, shards, str(out_dir)] + common + list(extra)


def _train(data, out_dir, *extra):
    trainer.run_training(_args(data, out_dir, *extra))


def _range_test(data, out_dir, *extra):
    range_test.run_range_test(_args(data, out_dir, *extra, common=RUN_ARGS))


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


def _checkpoint_iterations(out_dir, epoch):
    """The optimizer step count stored inside a weights checkpoint."""
    path = os.path.join(str(out_dir), "weights.{:05d}.weights.h5".format(epoch))
    with h5py.File(path, "r") as f:
        return int(f["optimizer/vars/0"][()])  # the optimizer's first variable: iteration


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

def test_fresh_run_writes_checkpoints_and_metadata(data, cosine_run):
    weights = sorted(os.path.basename(p) for p in glob.glob(str(cosine_run / "weights.*")))
    assert weights == ["weights.00001.weights.h5", "weights.00002.weights.h5",
                       "weights.00003.weights.h5"]
    # Each checkpoint carries the optimizer state as of its own epoch's end.
    assert [_checkpoint_iterations(cosine_run, e) for e in (1, 2, 3)] == [
        STEPS_PER_EPOCH, 2 * STEPS_PER_EPOCH, 3 * STEPS_PER_EPOCH]
    assert not (cosine_run / "optimizer_state.npz").exists()

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
    assert all(e["weight_norm"] > 0 for e in meta["epochs"])
    # Synthetic shards are well-formed, so the batch sanity checker has nothing to report.
    assert not (cosine_run / "batch_sanity_log.json").exists()


def test_a_nan_loss_stops_training(data, tmp_path):
    """A learning rate far too high makes the loss NaN within the first epoch: training
    stops there instead of running on, with that epoch's checkpoint and metadata row
    still written."""
    out = tmp_path / "nan"
    _train(data, out, "--epochs", "5", "--warmup-steps", "0", "--learning-rate", "1e30")
    epochs = _metadata(out)["epochs"]
    assert len(epochs) == 1
    assert (out / "weights.00001.weights.h5").exists()


def test_cosine_lr_decays_after_warmup(cosine_run):
    lrs = [e["learning_rate"] for e in _metadata(cosine_run)["epochs"]]
    assert lrs[0] < 0.05
    assert lrs == sorted(lrs, reverse=True)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-6)  # cosine reaches its floor at total_steps


def test_a_checkpoint_loads_into_the_policy_for_play(data, cosine_run):
    """Training decodes packed batches inside its steps, not in the model, so a checkpoint
    is still the plain policy's weights: it loads into the model JSON and plays."""
    policy = CNNPolicy.load_model(data[0])
    policy.model.load_weights(os.path.join(str(cosine_run), "weights.00003.weights.h5"))
    moves = policy.eval_state(GameState())
    assert len(moves) == 361
    assert sum(p for _m, p in moves) == pytest.approx(1.0, abs=1e-4)


def test_metadata_records_positions_per_epoch(cosine_run):
    assert _metadata(cosine_run)["positions_per_epoch"] == STEPS_PER_EPOCH * MINIBATCH


def test_default_epoch_is_one_pass_over_the_training_set(data, tmp_path):
    args = trainer.build_parser().parse_args(_args(data, tmp_path / "out", "--epochs", "1"))
    args.steps_per_epoch = None  # as if --steps-per-epoch were left out
    run = trainer.set_up_run(args, (), require_latest_checkpoint=False)
    n_train = sum(dataset_info(find_split_shards(data[1], "train"))[3])
    assert run.steps_per_epoch == n_train // MINIBATCH > 1


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
    assert _checkpoint_iterations(out, 2) == 2 * STEPS_PER_EPOCH
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


def test_plateau_resume_mid_warmup_continues_the_ramp(data, tmp_path):
    """Warmup of 6 steps at 4 steps per epoch: a run stopped after epoch 1 is partway
    through it. The resumed epoch 2 must finish the ramp exactly as the uninterrupted run
    does, not freeze at the checkpoint's partly-warmed LR."""
    extra = ("--epochs", "2", "--lr-schedule", "plateau", "--warmup-steps", "6")
    full = tmp_path / "full"
    _train(data, full, *extra)
    split = tmp_path / "split"
    _train_interrupted(data, split, 1, *extra)
    _train(data, split, *extra, "--weights", "weights.00001.weights.h5")

    full_epochs, split_epochs = _metadata(full)["epochs"], _metadata(split)["epochs"]
    assert split_epochs[0]["learning_rate"] < 0.05  # stopped partway through warmup
    assert split_epochs[1]["learning_rate"] == pytest.approx(0.05)  # ramp finished
    assert split_epochs[1]["learning_rate"] == pytest.approx(
        full_epochs[1]["learning_rate"], rel=1e-6)
    assert split_epochs[1]["loss"] == pytest.approx(full_epochs[1]["loss"], rel=1e-4)


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

def test_resume_with_missing_checkpoint_raises(data, stopped):
    os.remove(str(stopped / "weights.00002.weights.h5"))
    with pytest.raises(ValueError, match="weights.00002.weights.h5 not found"):
        _train(data, stopped, *RESUME_COSINE)


def test_resume_from_an_older_checkpoint_raises(data, stopped):
    with pytest.raises(ValueError, match="checkpoint from epoch 1, but .* records 2"):
        _train(data, stopped, "--epochs", "3", "--weights", "weights.00001.weights.h5")


def test_resume_with_no_epochs_left_raises(data, stopped):
    with pytest.raises(ValueError, match="already has 2 recorded epochs"):
        _train(data, stopped, "--epochs", "2", "--weights", "weights.00002.weights.h5")


@pytest.mark.parametrize("flag,value", [("--minibatch", "8"), ("--steps-per-epoch", "2"),
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
    _range_test(data, out, "--epochs", "2", "--range-check-every", "1",
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
    # grad_norm and the step's own loss only reach the logs through the monkey-patched
    # train_step. The step loss is not Keras's running epoch mean (they differ from the
    # second step of each epoch on).
    assert all(r["grad_norm"] is not None and r["grad_norm"] > 0 for r in records)
    assert all(r["loss"] is not None for r in records)
    assert any(r["loss"] != r["loss_epoch_mean"] for r in records)
    assert all(r["loss_scale"] is None for r in records)  # float32: no loss scaling


def test_lr_range_test_warm_start_takes_weights_but_a_fresh_optimizer(data, tmp_path):
    out = tmp_path / "range"
    range_args = ("--range-check-every", "1", "--range-warmup-steps", "2")
    _range_test(data, out, "--epochs", "1", *range_args)
    _range_test(data, out, "--epochs", "2", "--weights", "weights.00001.weights.h5",
                *range_args)
    # The warm-started sweep's optimizer began again at step 0, not at the first run's end.
    assert _checkpoint_iterations(out, 2) == STEPS_PER_EPOCH


def test_range_test_warm_start_from_a_training_run(data, cosine_run, tmp_path):
    """A sweep can warm-start from a training run's checkpoint in the same directory."""
    out = tmp_path / "from_training"
    shutil.copytree(str(cosine_run), str(out))
    _range_test(data, out, "--epochs", "4", "--weights", "weights.00003.weights.h5",
                "--range-check-every", "1")
    assert _checkpoint_iterations(out, 4) == STEPS_PER_EPOCH


def test_range_test_rejects_training_only_options(data, tmp_path):
    with pytest.raises(SystemExit):
        _range_test(data, tmp_path / "out", "--lr-schedule", "plateau")


def test_training_and_range_test_build_identical_runs(data, tmp_path):
    """The point of sharing the pipeline: from the same run arguments, a training run and
    an LR range test get the same model initialization, the same training batches and the
    same validation set - an LR found by one applies to the other."""
    runs = []
    for name, module, common in [("train", trainer, TRAIN_ARGS),
                                 ("range", range_test, RUN_ARGS)]:
        args = module.build_parser().parse_args(
            _args(data, tmp_path / name, "--epochs", "1", common=common))
        runs.append(trainer.set_up_run(args, (), require_latest_checkpoint=False))
    a, b = runs
    assert (a.steps_per_epoch, a.total_steps) == (b.steps_per_epoch, b.total_steps)
    for wa, wb in zip(a.model.get_weights(), b.model.get_weights()):
        assert (wa == wb).all()
    for _ in range(3):
        ((pa, ca), ya), ((pb, cb), yb) = next(a.train_data_generator), next(b.train_data_generator)
        assert (pa == pb).all() and (ca == cb).all() and (ya == yb).all()
    for ((pa, ca), ya), ((pb, cb), yb) in zip(a.val_dataset, b.val_dataset):
        assert (pa.numpy() == pb.numpy()).all() and (ca.numpy() == cb.numpy()).all()
        assert (ya.numpy() == yb.numpy()).all()


# --- other argument guards -------------------------------------------------------------

def test_unknown_symmetry_raises(data, tmp_path):
    with pytest.raises(ValueError, match="unknown symmetries"):
        _train(data, tmp_path / "out", "--epochs", "1", "--symmetries", "noop,twirl")


def test_model_and_shard_feature_mismatch_raises(data, tmp_path):
    other = _model_json(tmp_path / "other.json", ["board", "ones"])
    with pytest.raises(ValueError, match="Model JSON file expects features"):
        trainer.run_training([other, data[1], str(tmp_path / "out"), "--epochs", "1"] +
                             TRAIN_ARGS)


@pytest.mark.parametrize("missing", ["--minibatch", "--epochs", "--learning-rate",
                                     "--warmup-steps", "--lr-schedule"])
def test_training_requires_recipe_options(data, tmp_path, missing, capsys):
    """No defaults for options whose right value depends on the model, data and GPU - a
    forgotten one must fail loudly, not quietly run a different experiment."""
    args = _args(data, tmp_path / "out", "--epochs", "1")
    i = args.index(missing)
    del args[i:i + 2]
    with pytest.raises(SystemExit):
        trainer.run_training(args)
    assert "required: " + missing in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["--minibatch", "--epochs"])
def test_range_test_requires_minibatch_and_epochs(data, tmp_path, missing, capsys):
    args = _args(data, tmp_path / "out", "--epochs", "1", common=RUN_ARGS)
    i = args.index(missing)
    del args[i:i + 2]
    with pytest.raises(SystemExit):
        range_test.run_range_test(args)
    assert "required: " + missing in capsys.readouterr().err
