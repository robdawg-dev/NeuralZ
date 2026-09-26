"""Exercises LROverrideCallback (AlphaGo/training/supervised_policy_trainer.py) in
isolation - pure CPU logic (reads a file, sets model.optimizer.learning_rate), no GPU
needed at all.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD

import AlphaGo.training.supervised_policy_trainer as trainer


def make_model(lr=0.1):
    model = keras.Sequential([layers.Input(shape=(4,)), layers.Dense(3)])
    opt = SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss="mse")
    return model


def test_no_file_present(tmp_path):
    model = make_model(lr=0.5)
    cb = trainer.LROverrideCallback(tmp_path)
    cb.set_model(model)
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.5) < 1e-6


def test_file_differs_applies_override(tmp_path):
    model = make_model(lr=0.5)
    (tmp_path / "lr_override.txt").write_text("0.2")
    cb = trainer.LROverrideCallback(tmp_path)
    cb.set_model(model)
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.2) < 1e-6


def test_file_matches_current_is_noop(tmp_path):
    model = make_model(lr=0.4)
    (tmp_path / "lr_override.txt").write_text("0.4")
    cb = trainer.LROverrideCallback(tmp_path)
    cb.set_model(model)
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.4) < 1e-6


def test_within_tolerance_is_noop(tmp_path):
    model = make_model(lr=0.4)
    (tmp_path / "lr_override.txt").write_text("0.4000001")  # differs by 1e-7, under the 1e-6 tolerance
    cb = trainer.LROverrideCallback(tmp_path)
    cb.set_model(model)
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.4) < 1e-6


def test_beyond_tolerance_applies(tmp_path):
    model = make_model(lr=0.4)
    (tmp_path / "lr_override.txt").write_text("0.40001")  # differs by 1e-5, over the 1e-6 tolerance
    cb = trainer.LROverrideCallback(tmp_path)
    cb.set_model(model)
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.40001) < 1e-6


def test_malformed_file_does_not_raise(tmp_path):
    model = make_model(lr=0.3)
    (tmp_path / "lr_override.txt").write_text("not_a_number")
    cb = trainer.LROverrideCallback(tmp_path)
    cb.set_model(model)
    cb.on_epoch_end(0)  # must not raise
    assert abs(float(model.optimizer.learning_rate) - 0.3) < 1e-6


def test_empty_file_is_noop(tmp_path):
    model = make_model(lr=0.3)
    (tmp_path / "lr_override.txt").write_text("   \n")
    cb = trainer.LROverrideCallback(tmp_path)
    cb.set_model(model)
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.3) < 1e-6


def test_warmup_active_blocks_override(tmp_path):
    model = make_model(lr=0.05)  # simulates mid-warmup LR
    warmup_cb = trainer.WarmupCallback(warmup_steps=100, start_lr=0.0001, target_lr=0.5)
    warmup_cb._step = 10  # 10 < 100 -> not done yet
    assert warmup_cb.is_done is False
    (tmp_path / "lr_override.txt").write_text("0.9")
    cb = trainer.LROverrideCallback(tmp_path, warmup_cb=warmup_cb)
    cb.set_model(model)
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.05) < 1e-6


def test_warmup_done_allows_override(tmp_path):
    model = make_model(lr=0.5)  # simulates the LR warmup ramped to
    warmup_cb = trainer.WarmupCallback(warmup_steps=100, start_lr=0.0001, target_lr=0.5)
    warmup_cb._step = 101  # 101 > 100 -> done
    assert warmup_cb.is_done is True
    (tmp_path / "lr_override.txt").write_text("0.9")
    cb = trainer.LROverrideCallback(tmp_path, warmup_cb=warmup_cb)
    cb.set_model(model)
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.9) < 1e-6


def test_no_warmup_cb_allows_override(tmp_path):
    # Matches a --weights resume: warmup_cb is None (removed from resume path entirely).
    model = make_model(lr=0.4)
    (tmp_path / "lr_override.txt").write_text("0.2")
    cb = trainer.LROverrideCallback(tmp_path, warmup_cb=None)
    cb.set_model(model)
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.2) < 1e-6


def test_persistent_pin_overrides_plateau_cut(tmp_path):
    # Simulates the real callback ordering: plateau_cb (which can cut LR) runs BEFORE
    # lr_override_cb in the trainer's actual callback list - confirm the override still
    # wins even after something else already changed the LR this same epoch boundary.
    model = make_model(lr=0.4)
    (tmp_path / "lr_override.txt").write_text("0.4")  # pin at 0.4
    cb = trainer.LROverrideCallback(tmp_path)
    cb.set_model(model)

    # Epoch 1: plateau cuts LR to 0.2 (simulated), then override fires and pins back to 0.4.
    model.optimizer.learning_rate = 0.2
    cb.on_epoch_end(0)
    assert abs(float(model.optimizer.learning_rate) - 0.4) < 1e-6

    # Epoch 2: same story - override keeps re-asserting as long as the file exists.
    model.optimizer.learning_rate = 0.1
    cb.on_epoch_end(1)
    assert abs(float(model.optimizer.learning_rate) - 0.4) < 1e-6


def test_integration_real_fit_with_plateau_and_override(tmp_path):
    # End-to-end: real model.fit() with the actual callback ordering used in the
    # trainer (plateau_cb before lr_override_cb), a real lr_override.txt on disk,
    # confirming the wiring works, not just the callback in isolation.
    model = make_model(lr=0.5)
    rng = np.random.default_rng(0)
    x = rng.random((16, 4), dtype="float32")
    y = rng.random((16, 3), dtype="float32")
    x_val = rng.random((8, 4), dtype="float32")
    y_val = rng.random((8, 3), dtype="float32")

    (tmp_path / "lr_override.txt").write_text("0.33")

    plateau_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=0)
    override_cb = trainer.LROverrideCallback(tmp_path)
    model.fit(x, y, validation_data=(x_val, y_val), epochs=2, verbose=0,
             callbacks=[plateau_cb, override_cb])
    assert abs(float(model.optimizer.learning_rate) - 0.33) < 1e-6
