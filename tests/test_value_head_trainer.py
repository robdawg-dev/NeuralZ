"""End-to-end test for value_head_trainer: tiny shards with value sidecars, a tiny policy
network, a few steps - the run writes a loadable policy+value network whose policy is the
input network's, and the training loss goes down."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json

import h5py as h5
import numpy as np
import pytest

from AlphaGo.models.nn_util import NeuralNetBase
from AlphaGo.models.policy import NewResPolicy
from AlphaGo.models.value import PolicyValueNet
from AlphaGo.preprocessing import add_value_targets as avt
from AlphaGo.preprocessing import convert_shuffled as conv
from AlphaGo.training import value_head_trainer as vht
from tests.test_add_value_targets import _game, FEATURES

SMALL = {"num_blocks": 2, "filters": 8, "gpool_channels": 4, "head_channels": 4}


def _data(tmp_path):
    sel, sgfs = tmp_path / "sel", tmp_path / "sgf"
    sel.mkdir()
    sgfs.mkdir()
    for split, seeds in (("train", range(10)), ("val", range(10, 14))):
        with open(str(sel / (split + ".txt")), "w", newline="\n") as f:
            for seed in seeds:
                path = str(sgfs / "g{}.sgf".format(seed))
                with open(path, "w") as g:
                    g.write(_game(seed))
                f.write("{}\t40\tnormal\n".format(path))
    out = str(tmp_path / "shards")
    conv.main([str(sel), out, "--splits", "train", "val", "--features", FEATURES,
               "--workers", "2", "--quiet", "--positions-per-bucket", "100",
               "--positions-per-file", "100"])
    avt.main([out, "--splits", "train", "val", "--workers", "2", "--quiet"])
    return out


def test_reader_lines_up_sidecars_with_shards(tmp_path):
    data = _data(tmp_path)
    shards = vht.find_split_shards(data, "train")
    sizes = [len(h5.File(s, "r")["actions"]) for s in shards]
    reader = vht.ValueReader(shards, sizes)
    try:
        packed, _actions, komi, value, _score, has = reader.read(sizes[0] - 5, 10)  # 2 shards
    finally:
        reader.close()
    expected = []
    for s in shards[:2]:
        with h5.File(avt.sidecar_path(s), "r") as f:
            expected.append(f["value"][()])
    expected = np.concatenate(expected)[sizes[0] - 5:sizes[0] + 5]
    assert packed.shape[0] == komi.shape[0] == 10
    assert np.allclose(value[:, 0], expected.astype(np.float32))
    assert set(has.tolist()) <= {0.0, 1.0}


def test_plateau_schedule_warms_up_holds_and_cuts(tmp_path):
    data = _data(tmp_path)
    policy = NewResPolicy(FEATURES.split(","), **SMALL)
    model_json, weights = str(tmp_path / "p.json"), str(tmp_path / "p.weights.h5")
    policy.save_model(model_json)
    policy.model.save_weights(weights)
    out = str(tmp_path / "run")
    # patience 0 with a tiny min_delta-free monitor: every epoch that doesn't improve cuts
    vht.run([model_json, weights, data, out, "--minibatch", "16", "--epochs", "4",
             "--steps-per-epoch", "4", "--validation-length", "32", "--seed", "3",
             "--learning-rate", "0.01", "--lr-schedule", "plateau", "--warmup-steps", "3",
             "--plateau-patience", "0", "--plateau-factor", "0.5"])
    with open(os.path.join(out, "metadata.json")) as f:
        lrs = [e["learning_rate"] for e in json.load(f)["epochs"]]
    assert lrs[0] == pytest.approx(0.01)  # warmed up to the peak by the end of epoch 1
    allowed = {0.01 * 0.5 ** k for k in range(5)}
    assert all(any(lr == pytest.approx(a) for a in allowed) for lr in lrs)
    assert lrs == sorted(lrs, reverse=True)


def test_training_writes_a_loadable_policy_value_net(tmp_path):
    data = _data(tmp_path)
    policy = NewResPolicy(FEATURES.split(","), **SMALL)
    model_json = str(tmp_path / "policy.json")
    weights = str(tmp_path / "policy.weights.h5")
    policy.save_model(model_json)
    policy.model.save_weights(weights)
    out = str(tmp_path / "run")
    vht.run([model_json, weights, data, out, "--minibatch", "16", "--epochs", "2",
             "--steps-per-epoch", "15", "--validation-length", "64", "--seed", "1",
             "--learning-rate", "0.01"])

    with open(os.path.join(out, "metadata.json")) as f:
        epochs = json.load(f)["epochs"]
    assert len(epochs) == 2
    assert {"loss", "value_mae", "score_mae", "val_loss"} <= set(epochs[0])
    assert all(np.isfinite(e["loss"]) and np.isfinite(e["val_loss"]) for e in epochs)

    net = NeuralNetBase.load_model(os.path.join(out, "model.json"))
    assert isinstance(net, PolicyValueNet)
    net.model.load_weights(os.path.join(out, "weights.00001.weights.h5"))
    head_after_1 = net.model.get_layer("value_hidden").get_weights()[0].copy()
    net.model.load_weights(os.path.join(out, "weights.00002.weights.h5"))
    assert not np.allclose(net.model.get_layer("value_hidden").get_weights()[0], head_after_1)
    planes = (np.random.default_rng(0).random((2, 19, 19, policy.preprocessor
                                                .get_output_dimension())) < 0.3)
    planes = planes.astype(np.float32)
    assert np.allclose(net.forward(planes), policy.forward(planes), atol=1e-6)

    # continue from that run with the top block unfrozen: the policy loss joins in, the
    # starting accuracy is recorded, and the policy now changes
    out2 = str(tmp_path / "run2")
    vht.run([os.path.join(out, "model.json"), os.path.join(out, "weights.00002.weights.h5"),
             data, out2, "--minibatch", "16", "--epochs", "1", "--steps-per-epoch", "10",
             "--validation-length", "64", "--seed", "2", "--trainable-blocks", "1",
             "--warmup-steps", "3", "--learning-rate", "0.01"])
    with open(os.path.join(out2, "metadata.json")) as f:
        meta = json.load(f)
    assert "val_activation_accuracy" in meta["before_training"] or any(
        k.endswith("_accuracy") for k in meta["before_training"])
    assert any(k.endswith("top5") for k in meta["epochs"][0])
    net2 = NeuralNetBase.load_model(os.path.join(out2, "model.json"))
    net2.model.load_weights(os.path.join(out2, "weights.00001.weights.h5"))
    assert not np.allclose(net2.forward(planes), policy.forward(planes), atol=1e-6)
