"""Verifies OptimizerStateCallback and the trainer's resume-side load sequence
(AlphaGo/training/supervised_policy_trainer_v4.py) round-trip SGD momentum correctly,
under both a plain optimizer and the mixed_float16 case - --mixed-precision wraps the
optimizer in a LossScaleOptimizer at compile() time, and that wrapped case was never
live-tested before being wired into the trainer.

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

import AlphaGo.training.supervised_policy_trainer_v4 as v4


@pytest.fixture(autouse=True)
def _restore_global_mixed_precision_policy():
    # mixed_precision.set_global_policy() is process-global keras state - reset it after
    # each test so a mixed_float16 case here can't leak into unrelated tests elsewhere
    # in the same pytest session (e.g. test_policy.py's own model construction).
    original = mixed_precision.global_policy()
    yield
    mixed_precision.set_global_policy(original)


def make_model():
    model = keras.Sequential([layers.Input(shape=(6,)), layers.Dense(16, activation="relu"),
                              layers.Dense(4)])
    opt = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
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

    # Save via the real trainer callback, not a hand-rolled equivalent.
    save_cb = v4.OptimizerStateCallback(tmp_path)
    save_cb.set_model(model)
    save_cb.on_epoch_end(0)
    state_path = tmp_path / "optimizer_state.npz"
    assert state_path.exists()

    # Fresh model/optimizer, as a real --weights resume would have (momentum at 0).
    model2 = make_model()

    # Confirm the real hazard the trainer's comment warns about: load_own_variables() on
    # an unbuilt optimizer does NOT raise - it silently no-ops (just a UserWarning) and
    # leaves momentum at 0. This is exactly why the trainer calls build() unconditionally
    # before load - skipping it would be a silent bug, not a loud one.
    with np.load(state_path) as f, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model2.optimizer.load_own_variables(dict(f))
    assert len(caught) == 1
    assert len(variable_dict(model2.optimizer)) != len(saved)

    # The real sequence: build() then load_own_variables(), exactly as the trainer's
    # resume path does right after model.compile().
    model2.optimizer.build(model2.trainable_variables)
    with np.load(state_path) as f:
        model2.optimizer.load_own_variables(dict(f))

    restored = variable_dict(model2.optimizer)
    assert set(restored.keys()) == set(saved.keys())
    for k in saved:
        assert np.allclose(saved[k], restored[k])
