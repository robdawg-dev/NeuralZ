"""Unit tests for the trainer's helper functions and callbacks
(AlphaGo/training/supervised_policy_trainer.py) - everything except run_training() itself,
which test_trainer_integration.py drives end to end. LROverrideCallback and
OptimizerStateCallback have their own files (test_lr_override.py, test_optimizer_state.py).

CPU-only; most tests use a stand-in model object, the rest a tiny Dense model.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json
import math
import types

import numpy as np
import pytest
import keras
from keras import layers
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.optimizers.schedules import CosineDecay

import AlphaGo.training.supervised_policy_trainer as trainer


def make_model(lr=0.1):
    model = keras.Sequential([layers.Input(shape=(4,)), layers.Dense(3)])
    model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9, nesterov=True), loss="mse")
    return model


def fake_model(learning_rate=0.0, iterations=0):
    """Just enough of a model for callbacks that only touch model.optimizer."""
    return types.SimpleNamespace(
        optimizer=types.SimpleNamespace(learning_rate=learning_rate, iterations=iterations))


def lr_sequence(cb, model, n):
    """Calls on_train_batch_begin n times, returning the learning rate set by each call."""
    seen = []
    for i in range(n):
        cb.on_train_batch_begin(i)
        seen.append(model.optimizer.learning_rate)
    return seen


# --- _replay_plateau_state -------------------------------------------------------------

def _val_loss_walk(seed, n=40):
    """A noisy, slowly improving val_loss curve with flat stretches - enough plateaus to
    trigger several cuts, cooldowns and wait resets."""
    rng = np.random.default_rng(seed)
    losses, v = [], 3.0
    for i in range(n):
        drift = -0.05 if (i // 8) % 2 == 0 else 0.0
        v += drift + rng.normal(0, 0.01)
        losses.append(float(v))
    return losses


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("factor,patience,cooldown,min_lr,min_delta", [
    (0.5, 5, 2, 0.0, 0.005),
    (0.5, 2, 0, 0.0, 1e-4),
    (0.1, 3, 1, 0.0, 0.005),
    (0.5, 2, 1, 0.0125, 0.005),  # floor reached: cuts stop, wait keeps counting
])
def test_replay_plateau_state_matches_keras(seed, factor, patience, cooldown, min_lr,
                                            min_delta):
    """Replaying the logged history must land on exactly the state the real
    ReduceLROnPlateau had after seeing the same epochs - otherwise a resume would be
    closer to (or further from) its next cut than the uninterrupted run."""
    model = make_model(lr=0.1)
    plateau = ReduceLROnPlateau(monitor="val_loss", factor=factor, patience=patience,
                                cooldown=cooldown, min_lr=min_lr, min_delta=min_delta)
    plateau.set_model(model)
    plateau.on_train_begin()

    history = []
    for epoch, val_loss in enumerate(_val_loss_walk(seed)):
        # What TrainingDiagnosticsCallback logs: the LR used this epoch, read before
        # plateau_cb gets a chance to cut it.
        lr_used = float(model.optimizer.learning_rate)
        plateau.on_epoch_end(epoch, {"val_loss": val_loss})
        history.append({"val_loss": val_loss, "learning_rate": lr_used})

        best, wait, cooldown_counter = trainer._replay_plateau_state(
            history, factor, patience, cooldown, min_lr, min_delta=min_delta)
        assert (best, wait, cooldown_counter) == (
            plateau.best, plateau.wait, plateau.cooldown_counter), "diverged at epoch %d" % epoch


def test_replay_plateau_state_skips_epochs_without_val_loss_or_lr():
    history = [{"val_loss": 2.0, "learning_rate": 0.1},
               {"loss": 1.0},
               {"val_loss": 2.5}]
    assert trainer._replay_plateau_state(history, 0.5, 5, 0, 0.0) == (2.0, 0, 0)


def test_replay_plateau_state_empty_history():
    assert trainer._replay_plateau_state([], 0.5, 5, 2, 0.0) == (float("inf"), 0, 0)


# --- _PlateauStateRestorer -------------------------------------------------------------

def test_plateau_state_restorer_survives_on_train_begin_reset():
    """ReduceLROnPlateau.on_train_begin() zeroes wait/cooldown_counter; the restorer,
    placed after it, must put the resumed values back before the first epoch."""
    model = make_model()
    plateau = ReduceLROnPlateau(monitor="val_loss", patience=5, cooldown=2)
    restorer = trainer._PlateauStateRestorer(plateau, wait=3, cooldown_counter=1)

    seen = {}

    class Probe(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 0:
                seen["state"] = (plateau.wait, plateau.cooldown_counter)

    x = np.zeros((8, 4), np.float32)
    y = np.zeros((8, 3), np.float32)
    model.fit(x, y, epochs=1, batch_size=8, validation_data=(x, y), verbose=0,
              callbacks=[plateau, restorer, Probe()])
    assert seen["state"] == (3, 1)


def test_plateau_state_restorer_order_matters():
    """The same restorer placed BEFORE the plateau callback is wiped out - documents why
    run_training appends it after plateau_cb."""
    model = make_model()
    plateau = ReduceLROnPlateau(monitor="val_loss", patience=5, cooldown=2)
    restorer = trainer._PlateauStateRestorer(plateau, wait=3, cooldown_counter=1)
    for cb in (restorer, plateau):
        cb.set_model(model)
        cb.on_train_begin()
    assert (plateau.wait, plateau.cooldown_counter) == (0, 0)


# --- _ResumedLRSchedule ----------------------------------------------------------------

def test_resumed_lr_schedule_shifts_by_offset():
    base = CosineDecay(initial_learning_rate=1e-4, decay_steps=100, warmup_target=0.05,
                       warmup_steps=10)
    resumed = trainer._ResumedLRSchedule(base, offset=40)
    for step in (0, 1, 5, 30, 69, 70, 200):
        assert float(resumed(step)) == pytest.approx(float(base(step + 40)))


def test_resumed_lr_schedule_continues_instead_of_rewarming():
    """Step 0 of a resumed schedule is mid-decay, not back at warmup's start LR."""
    base = CosineDecay(initial_learning_rate=1e-4, decay_steps=100, warmup_target=0.05,
                       warmup_steps=10)
    resumed = trainer._ResumedLRSchedule(base, offset=40)
    assert float(resumed(0)) > 1e-3
    assert float(resumed(0)) < 0.05


def test_resumed_lr_schedule_get_config():
    base = CosineDecay(initial_learning_rate=1e-4, decay_steps=100)
    config = trainer._ResumedLRSchedule(base, offset=7).get_config()
    assert config["offset"] == 7
    assert config["base_schedule"] == base.get_config()


# --- WarmupCallback --------------------------------------------------------------------

def test_warmup_ramps_linearly_to_target():
    model = fake_model()
    cb = trainer.WarmupCallback(warmup_steps=4, start_lr=0.0, target_lr=1.0)
    cb.set_model(model)
    assert lr_sequence(cb, model, 5) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_warmup_stops_touching_lr_when_done():
    model = fake_model()
    cb = trainer.WarmupCallback(warmup_steps=4, start_lr=0.0, target_lr=1.0)
    cb.set_model(model)
    lr_sequence(cb, model, 4)
    assert not cb.is_done
    cb.on_train_batch_begin(4)
    assert cb.is_done
    # Something else (ReduceLROnPlateau, an override) now owns the LR.
    model.optimizer.learning_rate = 0.3
    cb.on_train_batch_begin(5)
    assert model.optimizer.learning_rate == 0.3


def test_warmup_zero_steps_sets_target_once():
    model = fake_model()
    cb = trainer.WarmupCallback(warmup_steps=0, start_lr=0.5, target_lr=1.0)
    cb.set_model(model)
    cb.on_train_batch_begin(0)
    assert model.optimizer.learning_rate == 0.5
    assert cb.is_done


# --- RangeTestLRCallback ---------------------------------------------------------------

def _range_sequence():
    model = fake_model()
    cb = trainer.RangeTestLRCallback(warmup_steps=4, warmup_start_lr=1e-4, floor_lr=1e-3,
                                     ceiling_lr=1.0, total_steps=14)
    cb.set_model(model)
    return lr_sequence(cb, model, 18)


def test_range_test_warmup_is_linear_to_floor():
    lrs = _range_sequence()
    expected = [1e-4 + (1e-3 - 1e-4) * i / 4 for i in range(5)]
    assert lrs[:5] == pytest.approx(expected)


def test_range_test_sweep_is_geometric_and_reaches_ceiling():
    lrs = _range_sequence()
    sweep = lrs[4:15]  # floor at the end of warmup, through the ceiling at total_steps
    assert sweep[0] == pytest.approx(1e-3)
    assert sweep[-1] == pytest.approx(1.0)
    ratios = [b / a for a, b in zip(sweep, sweep[1:])]
    assert ratios == pytest.approx([1000 ** 0.1] * 10)


def test_range_test_clamps_at_ceiling_past_total_steps():
    assert _range_sequence()[15:] == pytest.approx([1.0] * 3)


# --- RangeTestDiagnosticsCallback ------------------------------------------------------

def test_range_diagnostics_logs_every_nth_step(tmp_path):
    model = make_model(lr=0.2)
    out = tmp_path / "step_diagnostics.jsonl"
    cb = trainer.RangeTestDiagnosticsCallback(check_every=3, out_path=str(out))
    cb.set_model(model)
    for i in range(7):
        cb.on_train_batch_end(i, {"loss": 1.5, "grad_norm": 2.0})
    cb.on_train_end()

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["step"] for r in records] == [0, 3, 6]
    assert records[0]["lr"] == pytest.approx(0.2)
    assert records[0]["loss"] == 1.5
    assert records[0]["grad_norm"] == 2.0
    assert records[0]["loss_scale"] is None
    assert records[0]["weight_norm"] > 0


def test_range_diagnostics_tolerates_missing_logs(tmp_path):
    out = tmp_path / "step_diagnostics.jsonl"
    cb = trainer.RangeTestDiagnosticsCallback(check_every=1, out_path=str(out))
    cb.set_model(make_model())
    cb.on_train_batch_end(0, None)
    cb.on_train_end()
    record = json.loads(out.read_text())
    assert (record["loss"], record["grad_norm"], record["loss_scale"]) == (None, None, None)


# --- sanity_checked_generator ----------------------------------------------------------

def _batch(bad=False):
    X = np.zeros((4, 19, 19, 2), np.float32)
    Y = np.zeros((4, 361), np.float32)
    if not bad:
        Y[np.arange(4), [0, 5, 10, 360]] = 1
    return X, Y


def _logged_steps(tmp_path):
    path = tmp_path / "batch_sanity_log.json"
    if not path.exists():
        return []
    return [entry["step"] for entry in json.loads(path.read_text())]


def test_sanity_generator_passes_batches_through_unchanged(tmp_path):
    batches = [_batch() for _ in range(3)]
    out = list(trainer.sanity_checked_generator(iter(batches), str(tmp_path), "train"))
    assert len(out) == 3
    for (X, Y), (X2, Y2) in zip(batches, out):
        assert X is X2 and Y is Y2
    assert _logged_steps(tmp_path) == []


def test_sanity_generator_checks_first_five_then_every_nth(tmp_path):
    batches = [_batch(bad=True) for _ in range(12)]
    list(trainer.sanity_checked_generator(iter(batches), str(tmp_path), "train",
                                          check_every=5))
    assert _logged_steps(tmp_path) == [1, 2, 3, 4, 5, 10]


@pytest.mark.parametrize("corrupt", ["nan", "above_one", "negative", "two_hot"])
def test_sanity_generator_detects_each_kind_of_bad_batch(tmp_path, corrupt):
    X, Y = _batch()
    if corrupt == "nan":
        X[0, 0, 0, 0] = np.nan
    elif corrupt == "above_one":
        X[0, 0, 0, 0] = 2.0
    elif corrupt == "negative":
        X[0, 0, 0, 0] = -1.0
    else:
        Y[0, 1] = 1
    list(trainer.sanity_checked_generator(iter([(X, Y)]), str(tmp_path), "val"))
    entries = json.loads((tmp_path / "batch_sanity_log.json").read_text())
    assert [(e["label"], e["step"]) for e in entries] == [("val", 1)]


def test_sanity_generator_appends_to_existing_log(tmp_path):
    (tmp_path / "batch_sanity_log.json").write_text(json.dumps([{"label": "old", "step": 99}]))
    list(trainer.sanity_checked_generator(iter([_batch(bad=True)]), str(tmp_path), "train"))
    assert _logged_steps(tmp_path) == [99, 1]


# --- MetadataWriterCallback ------------------------------------------------------------

def _write_epochs(cb, logs_list):
    for i, logs in enumerate(logs_list):
        cb.on_epoch_end(i, logs)


def test_metadata_tracks_best_epoch_by_val_loss(tmp_path):
    path = tmp_path / "metadata.json"
    cb = trainer.MetadataWriterCallback(str(path))
    _write_epochs(cb, [{"loss": 1.0, "val_loss": 3.0}, {"loss": 0.1, "val_loss": 2.0},
                       {"loss": 0.01, "val_loss": 2.5}, {"loss": 0.5, "val_loss": 1.0}])
    saved = json.loads(path.read_text())
    assert saved["best_epoch"] == 3
    assert [e["val_loss"] for e in saved["epochs"]] == [3.0, 2.0, 2.5, 1.0]


def test_metadata_falls_back_to_loss_without_validation(tmp_path):
    cb = trainer.MetadataWriterCallback(str(tmp_path / "metadata.json"))
    _write_epochs(cb, [{"loss": 3.0}, {"loss": 1.0}, {"loss": 2.0}])
    assert cb.metadata["best_epoch"] == 1


def test_metadata_ties_keep_the_earlier_epoch(tmp_path):
    cb = trainer.MetadataWriterCallback(str(tmp_path / "metadata.json"))
    _write_epochs(cb, [{"val_loss": 2.0}, {"val_loss": 2.0}])
    assert cb.metadata["best_epoch"] == 0


def test_metadata_appends_after_resume(tmp_path):
    """A resumed run loads the old metadata and keeps numbering epochs from there,
    whatever epoch index Keras passes in."""
    path = tmp_path / "metadata.json"
    cb = trainer.MetadataWriterCallback(str(path))
    cb.metadata = {"epochs": [{"val_loss": 3.0}, {"val_loss": 2.0}], "best_epoch": 1,
                   "cmd_line_args": [{"minibatch": 8}]}
    cb.on_epoch_end(0, {"val_loss": 1.5})
    saved = json.loads(path.read_text())
    assert len(saved["epochs"]) == 3
    assert saved["best_epoch"] == 2
    assert saved["cmd_line_args"] == [{"minibatch": 8}]


# --- TrainingDiagnosticsCallback -------------------------------------------------------

def test_diagnostics_reads_lr_from_schedule_at_current_iteration():
    model = fake_model(learning_rate=999.0, iterations=5)
    cb = trainer.TrainingDiagnosticsCallback(lambda step: step * 0.1, steps_per_epoch=10)
    cb.set_model(model)
    logs = {}
    cb.on_epoch_begin(0)
    cb.on_epoch_end(0, logs)
    assert logs["learning_rate"] == pytest.approx(0.5)


def test_diagnostics_reads_live_optimizer_lr_without_schedule():
    model = make_model(lr=0.03)
    cb = trainer.TrainingDiagnosticsCallback(None, steps_per_epoch=10)
    cb.set_model(model)
    logs = {}
    cb.on_epoch_begin(0)
    cb.on_epoch_end(0, logs)
    assert logs["learning_rate"] == pytest.approx(0.03)


def test_diagnostics_records_throughput(monkeypatch):
    clock = iter([100.0, 104.0])
    monkeypatch.setattr(trainer.time, "time", lambda: next(clock))
    cb = trainer.TrainingDiagnosticsCallback(None, steps_per_epoch=20)
    cb.set_model(fake_model(learning_rate=0.1))
    logs = {}
    cb.on_epoch_begin(0)
    cb.on_epoch_end(0, logs)
    assert logs["epoch_seconds"] == pytest.approx(4.0)
    assert logs["steps_per_second"] == pytest.approx(5.0)


def test_diagnostics_ignores_missing_logs():
    cb = trainer.TrainingDiagnosticsCallback(None, steps_per_epoch=10)
    cb.set_model(fake_model())
    cb.on_epoch_begin(0)
    cb.on_epoch_end(0, None)  # must not raise


# --- prediction_entropy ----------------------------------------------------------------

def _entropy(y_pred):
    y_pred = np.asarray(y_pred, np.float32)
    return keras.ops.convert_to_numpy(trainer.prediction_entropy(np.zeros_like(y_pred), y_pred))


def test_entropy_of_uniform_prediction_is_log_361():
    assert _entropy(np.full((2, 361), 1 / 361)) == pytest.approx([math.log(361)] * 2, rel=1e-4)


def test_entropy_of_one_hot_prediction_is_only_the_clip_floor():
    # The 360 zero entries are clipped to 1e-7 (avoids log(0)), each contributing
    # -1e-7 * log(1e-7) - so a certain prediction scores ~0.00058, not exactly 0.
    y = np.zeros((1, 361))
    y[0, 42] = 1
    assert _entropy(y)[0] == pytest.approx(360 * 1e-7 * -math.log(1e-7), rel=1e-2)


def test_entropy_ignores_y_true():
    y_pred = np.random.default_rng(0).dirichlet(np.ones(361), size=3).astype(np.float32)
    a = keras.ops.convert_to_numpy(trainer.prediction_entropy(np.zeros_like(y_pred), y_pred))
    b = keras.ops.convert_to_numpy(trainer.prediction_entropy(np.ones_like(y_pred), y_pred))
    assert np.allclose(a, b)
