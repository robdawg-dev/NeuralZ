"""Verifies the trainer's load_checkpoint (AlphaGo/training/supervised_policy_trainer.py)
restores the optimizer's state - SGD momentum, step count, learning rate, and under
--mixed-precision the LossScaleOptimizer's own state - from a weights checkpoint, under both
a plain optimizer and the mixed_float16 case (--mixed-precision wraps the optimizer in a
LossScaleOptimizer at compile() time), and refuses to silently resume without it.

CPU-only, tiny model, a handful of steps - this is a mechanism check, not a training run.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import warnings

import h5py
import numpy as np
import pytest
import keras
from keras import layers, mixed_precision
from keras.optimizers import SGD
from keras.optimizers.schedules import CosineDecay

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


def optimizer_values(model):
    # By position: a fresh optimizer's variables have different names (sgd_1/...)
    return [v.numpy().copy() for v in model.optimizer.variables]


def weight_values(model):
    return [v.numpy().copy() for v in model.weights]


def checkpoint(tmp_path, use_mixed_precision=False, steps=5):
    """A trained model and its checkpoint, saved the way ModelCheckpoint does in fit()."""
    mixed_precision.set_global_policy("mixed_float16" if use_mixed_precision else "float32")
    model = make_model()
    rng = np.random.default_rng(0)
    x = rng.random((32, 6), dtype="float32")
    y = rng.random((32, 4), dtype="float32")
    for _ in range(steps):
        model.train_on_batch(x, y)
    path = str(tmp_path / "weights.00001.weights.h5")
    model.save_weights(path)
    return model, path


@pytest.mark.parametrize("use_mixed_precision", [False, True])
def test_checkpoint_restores_weights_and_optimizer_exactly(tmp_path, use_mixed_precision):
    model, path = checkpoint(tmp_path, use_mixed_precision)
    if use_mixed_precision:
        assert type(model.optimizer).__name__ == "LossScaleOptimizer"
    saved = optimizer_values(model)
    # Momentum after 5 real training steps is non-trivial, not all-zero - otherwise this
    # test would trivially "pass" even if loading silently did nothing.
    assert int(model.optimizer.iterations) == 5
    assert any(v.ndim and not np.allclose(v, 0) for v in saved)

    # A fresh compiled model, as a real --weights resume has (momentum at 0).
    resumed = make_model()
    trainer.load_checkpoint(resumed, path)

    restored = optimizer_values(resumed)
    assert len(restored) == len(saved)
    for a, b in zip(saved, restored):
        assert np.array_equal(a, b)
    for a, b in zip(weight_values(model), weight_values(resumed)):
        assert np.array_equal(a, b)


def test_keras_alone_skips_the_state_of_an_unbuilt_optimizer(tmp_path):
    """The hazard load_checkpoint guards against: model.load_weights() into a compiled
    model whose optimizer isn't built yet does NOT raise - it warns and leaves momentum at
    0. Documented here so a Keras change in this behaviour shows up."""
    _, path = checkpoint(tmp_path)
    resumed = make_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resumed.load_weights(path)
    assert any("Skipping variable loading for optimizer" in str(w.message) for w in caught)
    assert int(resumed.optimizer.iterations) == 0


def test_mismatched_optimizer_raises(tmp_path):
    """Plateau's optimizer has a learning_rate variable; a schedule-driven (cosine) one
    doesn't. Loading one into the other must fail rather than warn and skip."""
    _, path = checkpoint(tmp_path)
    cosine = make_model(learning_rate=CosineDecay(0.1, 100))
    with pytest.raises(ValueError, match="different --lr-schedule or --mixed-precision"):
        trainer.load_checkpoint(cosine, path)


def test_mixed_precision_toggled_raises(tmp_path):
    _, path = checkpoint(tmp_path, use_mixed_precision=True)
    mixed_precision.set_global_policy("float32")
    with pytest.raises(ValueError, match="different --lr-schedule or --mixed-precision"):
        trainer.load_checkpoint(make_model(), path)


def test_checkpoint_without_optimizer_state_raises(tmp_path):
    """Weights saved from an uncompiled model carry no optimizer section at all."""
    model = keras.Sequential([layers.Input(shape=(6,)), layers.Dense(16, activation="relu"),
                              layers.Dense(4)])
    path = str(tmp_path / "bare.weights.h5")
    model.save_weights(path)
    with h5py.File(path, "r") as f:
        assert "optimizer" not in f
    with pytest.raises(ValueError, match="holds no optimizer state"):
        trainer.load_checkpoint(make_model(), path)


def test_weights_only_leaves_the_optimizer_fresh(tmp_path):
    """An LR range test warm start: the checkpoint's weights, but a fresh optimizer."""
    model, path = checkpoint(tmp_path)
    resumed = make_model()
    trainer.load_checkpoint(resumed, path, with_optimizer=False)
    for a, b in zip(weight_values(model), weight_values(resumed)):
        assert np.array_equal(a, b)
    assert int(resumed.optimizer.iterations) == 0
