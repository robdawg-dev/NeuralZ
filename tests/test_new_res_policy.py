"""Tests for NewResPolicy (AlphaGo/models/policy.py)."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np
import pytest

from AlphaGo.go import GameState
from AlphaGo.models.nn_util import NeuralNetBase
from AlphaGo.models.policy import NewResPolicy

FEATURES = ["board", "ones", "liberties"]
SMALL = {"num_blocks": 2, "filters": 8, "gpool_channels": 4, "head_channels": 4}


def _layers(policy, name):
    return [layer for layer in policy.model.layers if type(layer).__name__ == name]


def test_outputs_a_distribution_over_the_board():
    policy = NewResPolicy(FEATURES, **SMALL)
    out = policy.forward(policy.preprocessor.state_to_tensor(GameState()))
    assert out.shape == (1, 361)
    assert np.isclose(out.sum(), 1.0, atol=1e-5)


def test_default_is_b15c192_with_three_pooling_blocks_and_a_pooled_head():
    policy = NewResPolicy(FEATURES)
    assert len(_layers(policy, "GlobalAveragePooling2D")) == 3 + 1
    convs = _layers(policy, "Conv2D")
    # stem + 15 blocks x 2 convs + 3 pooling blocks' extra conv + head's 3 convs
    assert len(convs) == 1 + 30 + 3 + 3
    assert {c.filters for c in convs[1:-3]} == {192, 128, 64}


@pytest.mark.parametrize("gpool_every,head_gpool,expected", [
    (1, True, 3), (2, True, 2), (2, False, 1), (0, True, 1), (0, False, 0),
])
def test_pooling_blocks_follow_gpool_every_and_head_gpool(gpool_every, head_gpool,
                                                          expected):
    policy = NewResPolicy(FEATURES, **dict(SMALL, gpool_every=gpool_every,
                                           head_gpool=head_gpool))
    assert len(_layers(policy, "GlobalAveragePooling2D")) == expected


def test_only_the_output_conv_has_a_bias():
    policy = NewResPolicy(FEATURES, **SMALL)
    convs = _layers(policy, "Conv2D")
    assert [c.use_bias for c in convs] == [False] * (len(convs) - 1) + [True]


@pytest.mark.parametrize("pooling,far_changes", [
    ({"gpool_every": 0, "head_gpool": False}, False),
    ({"gpool_every": 2, "head_gpool": False}, True),
    ({"gpool_every": 0, "head_gpool": True}, True),
])
def test_global_pooling_reaches_across_the_board(pooling, far_changes):
    """2 blocks see 5 points away without pooling (3x3 stem + 4 3x3 convs) - a stone in
    one corner can't reach the far corner unless something pools the board.

    Probed at the network's last residual or head sum (its last Add), not at the output:
    a pooled bias shifts every point alike, which the ReLU and softmax after it can hide."""
    keras.utils.set_random_seed(0)
    policy = NewResPolicy(FEATURES, **dict(SMALL, **pooling))
    last_add = [layer for layer in policy.model.layers if type(layer).__name__ == "Add"][-1]
    probe = keras.Model(policy.model.inputs, last_add.output)
    x = policy.preprocessor.state_to_tensor(GameState())
    changed = x.copy()
    changed[0, 0, 0, :] = 1.0 - changed[0, 0, 0, :]
    before, after = probe(x).numpy()[0, 18, 18], probe(changed).numpy()[0, 18, 18]
    assert (not np.allclose(after, before, rtol=0, atol=1e-6)) == far_changes


def test_gpool_channels_must_leave_regular_channels():
    with pytest.raises(ValueError, match="gpool_channels"):
        NewResPolicy(FEATURES, **dict(SMALL, gpool_channels=8))


def test_save_load_round_trip(tmp_path):
    policy = NewResPolicy(FEATURES, **SMALL)
    model_file, weights_file = str(tmp_path / "m.json"), str(tmp_path / "w.weights.h5")
    policy.save_model(model_file)
    policy.model.save_weights(weights_file)
    loaded = NeuralNetBase.load_model(model_file)
    loaded.model.load_weights(weights_file)
    assert type(loaded) is NewResPolicy
    x = policy.preprocessor.state_to_tensor(GameState())
    np.testing.assert_allclose(loaded.forward(x), policy.forward(x), rtol=1e-6)
