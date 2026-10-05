"""Tests for PolicyValueNet (AlphaGo/models/value.py): the value head added to a policy
network leaves the policy exactly as it was, trains only itself, and saves and loads."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
from keras import mixed_precision

from AlphaGo.go import GameState, WHITE
from AlphaGo.models.nn_util import NeuralNetBase
from AlphaGo.models.policy import NewResPolicy
from AlphaGo.models.value import (PolicyValueNet, residual_adds, set_trainable_blocks,
                                  trunk_output)

FEATURES = ["board", "ones", "liberties"]
SMALL = {"num_blocks": 2, "filters": 8, "gpool_channels": 4, "head_channels": 4}


def _planes(policy, n=3, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n,) + tuple(int(d) for d in policy.model.inputs[0].shape[1:])
    return (rng.random(shape) < 0.3).astype(np.float32)


def test_trunk_output_is_the_policy_heads_input():
    policy = NewResPolicy(FEATURES, **SMALL)
    assert tuple(trunk_output(policy.model).shape[1:]) == (19, 19, SMALL["filters"])


def test_policy_output_is_unchanged():
    policy = NewResPolicy(FEATURES, **SMALL)
    planes = _planes(policy)
    before = policy.forward(planes)
    net = PolicyValueNet.from_policy(policy)
    assert np.array_equal(net.forward(planes), before)
    p, value, score = net.forward_all(planes, [7.5, -7.5, 0.5])
    assert np.array_equal(p, before)
    assert value.shape == (3,) and ((value > 0) & (value < 1)).all()
    assert score.shape == (3,)


def test_only_the_value_head_trains():
    policy = NewResPolicy(FEATURES, **SMALL)
    n_policy_weights = len(policy.model.weights)
    net = PolicyValueNet.from_policy(policy)
    trainable = {w.path.split("/")[0] for w in net.model.trainable_weights}
    assert trainable and all(name.startswith(("value_", "score_")) for name in trainable)
    assert len(net.model.weights) > n_policy_weights


def test_residual_adds_are_one_per_block():
    policy = NewResPolicy(FEATURES, num_blocks=3, filters=8, gpool_every=2, gpool_channels=4,
                          head_channels=4)
    adds = residual_adds(policy.model)
    assert len(adds) == 3  # not the pooling block's internal add or the head's


def _trainable_names(net):
    return {layer.name for layer in net.model.layers if layer.trainable and layer.weights}


def test_trainable_blocks_unfreeze_from_the_top():
    net = PolicyValueNet.from_policy(NewResPolicy(FEATURES, num_blocks=3, filters=8,
                                                  gpool_every=0, head_channels=4))
    value_only = _trainable_names(net)
    assert value_only and all(n.startswith(("value", "score")) for n in value_only)
    counts = []
    for n in (1, 2, 3):
        set_trainable_blocks(net, n)
        convs = [layer for layer in net.model.layers
                 if layer.trainable and type(layer).__name__ == "Conv2D"
                 and not layer.name.startswith("value")]
        counts.append(len(convs))
    # each block adds its 2 convs; the policy head's 3 convs (with its pooling conv) train
    # from n = 1; all blocks (n = 3) also unfreeze the stem conv
    assert counts == [2 + 3, 4 + 3, 6 + 3 + 1]
    set_trainable_blocks(net, 0)
    assert _trainable_names(net) == value_only


def test_komi_reaches_the_value():
    net = PolicyValueNet.from_policy(NewResPolicy(FEATURES, **SMALL))
    planes = np.repeat(_planes(net, n=1), 2, axis=0)
    _p, value, score = net.forward_all(planes, [-30.0, 30.0])
    assert value[0] != value[1] or score[0] != score[1]


def test_save_and_load_round_trip(tmp_path):
    net = PolicyValueNet.from_policy(NewResPolicy(FEATURES, **SMALL))
    planes = _planes(net)
    expected = net.forward_all(planes, [6.5, -6.5, 0.0])
    model_json, weights = str(tmp_path / "model.json"), str(tmp_path / "w.weights.h5")
    net.save_model(model_json, weights)
    loaded = NeuralNetBase.load_model(model_json)
    assert isinstance(loaded, PolicyValueNet)
    for a, b in zip(loaded.forward_all(planes, [6.5, -6.5, 0.0]), expected):
        assert np.allclose(a, b, atol=1e-6)
    state = GameState()
    assert len(loaded.eval_state(state)) == 361
    state.set_current_player(WHITE)
    value, score = loaded.eval_value([state], [7.5])
    assert value.shape == (1,) and score.shape == (1,)


JOINT = {"num_blocks": 4, "filters": 8, "gpool_blocks": [2, 3], "gpool_channels": 4,
         "head_channels": 4, "value_channels": 4, "value_hidden": 6}


def test_joint_network_outputs_and_pooling_blocks():
    net = PolicyValueNet(FEATURES, **JOINT)
    names = [type(layer).__name__ for layer in net.model.layers]
    # pooling blocks 2 and 3, the policy head's pooled bias, the value head's mean pool
    assert names.count("GlobalAveragePooling2D") == 2 + 1 + 1
    planes = _planes(net, n=2)
    outputs = net.model([planes, np.array([[7.5], [-7.5]], np.float32)])
    policy, value, score, own = (o.numpy() for o in outputs)
    assert policy.shape == (2, 361) and np.allclose(policy.sum(axis=1), 1.0, atol=1e-5)
    assert value.shape == score.shape == (2, 1)
    assert own.shape == (2, 361) and np.abs(own).max() <= 1.0


def test_joint_network_policy_sees_komi():
    net = PolicyValueNet(FEATURES, **JOINT)
    planes = np.repeat(_planes(net, n=1), 2, axis=0)
    policy = net.forward(planes, komi=[30.0, -30.0])
    assert not np.allclose(policy[0], policy[1])


def test_joint_network_save_and_load(tmp_path):
    net = PolicyValueNet(FEATURES, **JOINT)
    planes = _planes(net)
    expected = net.forward_all(planes, [6.5, -6.5, 0.5])
    model_json, weights = str(tmp_path / "joint.json"), str(tmp_path / "joint.weights.h5")
    net.save_model(model_json, weights)
    loaded = NeuralNetBase.load_model(model_json)
    assert isinstance(loaded, PolicyValueNet) and len(loaded.model.outputs) == 4
    for a, b in zip(loaded.forward_all(planes, [6.5, -6.5, 0.5]), expected):
        assert np.allclose(a, b, atol=1e-6)


def test_gpool_blocks_must_exist():
    with pytest.raises(ValueError, match="gpool_blocks"):
        PolicyValueNet(FEATURES, **dict(JOINT, gpool_blocks=[5]))


def test_outputs_stay_float32_when_loaded_under_mixed_precision(tmp_path):
    net = PolicyValueNet.from_policy(NewResPolicy(FEATURES, **SMALL))
    model_json = str(tmp_path / "model.json")
    net.save_model(model_json, str(tmp_path / "w.weights.h5"))
    mixed_precision.set_global_policy("mixed_float16")
    try:
        loaded = NeuralNetBase.load_model(model_json)
        dtypes = {layer.name: layer.compute_dtype for layer in loaded.model.layers}
        assert dtypes["value"] == dtypes["score"] == "float32"
        assert dtypes["value_conv"] == "float16"
    finally:
        mixed_precision.set_global_policy("float32")
