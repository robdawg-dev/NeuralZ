"""Verifies OptimizerStateCallback and the trainer's resume-side read_optimizer_state /
apply_optimizer_state (AlphaGo/training/supervised_policy_trainer.py) round-trip SGD
momentum correctly under both a plain optimizer and the mixed_float16 case
(--mixed-precision wraps the optimizer in a LossScaleOptimizer at compile() time, and
that wrapped case was never live-tested before being wired into the trainer), and refuse
missing, stale or mismatched state.

CPU-only, tiny model, a handful of steps - this is a mechanism check, not a training run.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import warnings

import numpy as np
import pytest
import keras
from keras import layers, mixed_precision
from keras.optimizers import SGD

import AlphaGo.training.supervised_policy_trainer as trainer


@pytest.fixture(autouse=True)
def _restore_global_mixed_precision_policy():
    # mixed_precision.set_global_policy() is process-global keras state - reset it after
    # each test so a mixed_float16 case here can't leak into unrelated tests elsewhere
    # in the same pytest session (e.g. test_policy.py's own model construction).
    original = mixed_precision.global_policy()
    yield
    mixed_precision.set_global_policy(original)


def make_model(learning_rate=0.1):
    model = keras.Sequential([layers.Input(shape=(6,)), layers.Dense(16, activation="relu"),
                              layers.Dense(4)])
    opt = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss="mse")
    return model


def variable_dict(optimizer):
    store = {}
    optimizer.save_own_variables(store)
    return store


@pytest.mark.parametrize("use_mixed_precision", [False, True])
def test_optimizer_state_roundtrip(tmp_path, use_mixed_precision):
    mixed_precision.set_global_policy("mixed_float16" if use_mixed_precision else "float32")

    model = make_model()
    rng = np.random.default_rng(0)
    x = rng.random((32, 6), dtype="float32")
    y = rng.random((32, 4), dtype="float32")
    for _ in range(5):
        model.train_on_batch(x, y)

    if use_mixed_precision:
        assert type(model.optimizer).__name__ == "LossScaleOptimizer"

    saved = variable_dict(model.optimizer)
    assert len(saved) > 0
    # Momentum after 5 real training steps should be non-trivial, not all-zero - otherwise
    # this test would trivially "pass" even if the save/load path silently did nothing.
    assert any(not np.allclose(v, np.zeros_like(v)) for v in saved.values())

    # Save and restore via the real trainer code, not a hand-rolled equivalent.
    state_path = save_state(model, tmp_path, epoch=0)
    store = trainer.read_optimizer_state(str(state_path), expected_epochs=1)

    # Fresh model/optimizer, as a real --weights resume would have (momentum at 0).
    model2 = make_model()

    # Confirm the real hazard apply_optimizer_state guards against: load_own_variables()
    # on an unbuilt optimizer does NOT raise - it silently no-ops (just a UserWarning) and
    # leaves momentum at 0. This is exactly why it calls build() unconditionally before
    # loading - skipping it would be a silent bug, not a loud one.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model2.optimizer.load_own_variables(dict(store))
    assert len(caught) == 1
    assert len(variable_dict(model2.optimizer)) != len(saved)

    trainer.apply_optimizer_state(model2, store)

    restored = variable_dict(model2.optimizer)
    assert set(restored.keys()) == set(saved.keys())
    for k in saved:
        assert np.allclose(saved[k], restored[k])


def save_state(model, out_dir, epoch):
    cb = trainer.OptimizerStateCallback(str(out_dir))
    cb.set_model(model)
    cb.on_epoch_end(epoch)
    path = out_dir / trainer.OPTIMIZER_STATE_FILE
    assert path.exists()
    return path


def trained_model(**kwargs):
    model = make_model(**kwargs)
    x = np.ones((8, 6), np.float32)
    model.train_on_batch(x, np.ones((8, 4), np.float32))
    return model


def test_saved_state_records_completed_epochs(tmp_path):
    path = save_state(trained_model(), tmp_path, epoch=4)
    with np.load(path) as f:
        assert int(f["completed_epochs"]) == 5


def test_read_missing_state_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        trainer.read_optimizer_state(str(tmp_path / "optimizer_state.npz"), 3)


def test_read_state_from_another_epoch_raises(tmp_path):
    path = save_state(trained_model(), tmp_path, epoch=1)
    with pytest.raises(ValueError, match="saved after epoch 2, but metadata.json records 3"):
        trainer.read_optimizer_state(str(path), 3)


def test_read_state_without_epoch_raises(tmp_path):
    path = tmp_path / "optimizer_state.npz"
    store = {}
    trained_model().optimizer.save_own_variables(store)
    np.savez(path, **store)
    with pytest.raises(ValueError, match="saved after epoch None"):
        trainer.read_optimizer_state(str(path), 1)


def test_apply_state_from_a_differently_configured_optimizer_raises(tmp_path):
    """Plateau's optimizer has a learning_rate variable; a schedule-driven (cosine) one
    doesn't. Loading one into the other must fail rather than warn and skip."""
    path = save_state(trained_model(), tmp_path, epoch=0)
    store = trainer.read_optimizer_state(str(path), 1)
    cosine = make_model(learning_rate=keras.optimizers.schedules.CosineDecay(0.1, 100))
    with pytest.raises(ValueError, match="different --lr-schedule"):
        trainer.apply_optimizer_state(cosine, store)
