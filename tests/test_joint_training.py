"""Joint policy + value + score + ownership training (JOINT_TRAINING_PLAN.md): the data
stream (value sidecars, ownership table, symmetries) and run_training() / run_range_test()
on a tiny joint network.

Marked slow: run_training() compiles with XLA on CPU.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json

import numpy as np
import pytest
from keras import mixed_precision

import AlphaGo.training.lr_range_test as range_test
import AlphaGo.training.supervised_policy_trainer as trainer
from AlphaGo.models.nn_util import NeuralNetBase
from AlphaGo.models.value import PolicyValueNet
from AlphaGo.preprocessing import add_value_targets as avt
from AlphaGo.preprocessing import convert_shuffled as conv
from AlphaGo.training import joint_data as jd
from AlphaGo.training.shard_stream import BATCH_TRANSFORMATIONS, find_split_shards
from tests.test_add_value_targets import FEATURES, _game

pytestmark = pytest.mark.slow

SIZE = 19
JOINT = {"num_blocks": 2, "filters": 8, "gpool_blocks": [2], "gpool_channels": 4,
         "head_channels": 4, "value_channels": 4, "value_hidden": 6}
RUN_ARGS = ["--minibatch", "16", "--steps-per-epoch", "4", "--validation-length", "32",
            "--seed", "1"]


@pytest.fixture(autouse=True)
def _restore_global_mixed_precision_policy():
    original = mixed_precision.global_policy()
    yield
    mixed_precision.set_global_policy(original)


def _ownership_row(game_id):
    """A distinct, recognizable final ownership per game: Black owns x < game_id % 19."""
    row = np.full((SIZE, SIZE), -127, np.int8)
    row[:game_id % SIZE + 1, :] = 127
    return row.reshape(-1)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("joint")
    sel, sgfs = tmp / "sel", tmp / "sgf"
    sel.mkdir()
    sgfs.mkdir()
    for split, seeds in (("train", range(10)), ("val", range(10, 13))):
        with open(str(sel / (split + ".txt")), "w", newline="\n") as f:
            for seed in seeds:
                path = str(sgfs / "g{}.sgf".format(seed))
                with open(path, "w") as g:
                    g.write(_game(seed))
                f.write("{}\t40\tnormal\n".format(path))
    shards = str(tmp / "shards")
    conv.main([str(sel), shards, "--splits", "train", "val", "--features", FEATURES,
               "--workers", "2", "--quiet", "--positions-per-bucket", "100",
               "--positions-per-file", "100"])
    avt.main([shards, "--splits", "train", "val", "--workers", "2", "--quiet"])
    for split, n in (("train", 10), ("val", 3)):
        table = np.stack([_ownership_row(g) for g in range(n)])
        table.tofile(os.path.join(shards, split, "ownership.bin"))
        with open(os.path.join(shards, split, "ownership.json"), "w") as f:
            json.dump({"games": n, "points": SIZE * SIZE}, f)
    model_json = str(tmp / "joint.json")
    PolicyValueNet(FEATURES.split(","), **JOINT).save_model(model_json)
    return model_json, shards


def test_reader_gives_ownership_from_the_player_to_moves_side(data):
    _model, shards = data
    paths = find_split_shards(shards, "train")
    import h5py
    sizes = [len(h5py.File(p, "r")["actions"]) for p in paths]
    reader = jd.JointReader(paths, sizes, jd.load_ownership(os.path.join(shards, "train")))
    try:
        d = reader.read(0, 40)
    finally:
        reader.close()
    for r in range(40):
        black = _ownership_row(int(d["game_id"][r])).astype(np.float32) / 127
        expected = black if d["black"][r] == 1 else -black
        assert np.allclose(d["ownership"][r], expected)


def test_ownership_gets_the_planes_symmetry():
    import tensorflow as tf
    symmetries = list(BATCH_TRANSFORMATIONS)
    n = len(symmetries)
    rng = np.random.default_rng(0)
    own = rng.uniform(-1, 1, (n, SIZE * SIZE)).astype(np.float32)
    packed = np.zeros((n, (SIZE * SIZE * 3 + 7) // 8), np.uint8)
    choices = np.arange(n, dtype=np.int32)
    _x, (_p, _v, _s, got), _w = jd.decode_joint_on_device(
        (tf.constant(packed), tf.constant(choices), tf.zeros((n, 1))),
        (None, None, None, tf.constant(own)), None, SIZE, 3, symmetries, tf.float32)
    for i, name in enumerate(symmetries):
        expected = BATCH_TRANSFORMATIONS[name](own[i:i + 1].reshape(1, SIZE, SIZE))
        assert np.allclose(got.numpy()[i], expected.reshape(-1)), name


def test_joint_training_runs_resumes_and_loads(data, tmp_path):
    model_json, shards = data
    out = str(tmp_path / "run")
    common = [model_json, shards, out] + RUN_ARGS + [
        "--warmup-steps", "2", "--learning-rate", "0.05", "--lr-schedule", "plateau",
        "--plateau-patience", "1"]
    trainer.run_training(common + ["--epochs", "2"])
    trainer.run_training(common + ["--epochs", "3", "--weights", "weights.00002.weights.h5"])
    with open(os.path.join(out, "metadata.json")) as f:
        epochs = json.load(f)["epochs"]
    assert len(epochs) == 3
    keys = set(epochs[-1])
    assert {"val_loss", "learning_rate"} <= keys
    assert any(k.endswith("accuracy") for k in keys)
    assert any("ownership" in k for k in keys) and any("side_agreement" in k for k in keys)
    assert all(np.isfinite(e["val_loss"]) for e in epochs)

    net = NeuralNetBase.load_model(model_json)
    net.model.load_weights(os.path.join(out, "weights.00003.weights.h5"))
    planes = np.zeros((1, SIZE, SIZE, net.preprocessor.get_output_dimension()), np.float32)
    policy, value, score = net.forward_all(planes, [7.5])
    assert policy.shape == (1, SIZE * SIZE) and 0 < value[0] < 1


def test_range_test_runs_on_a_joint_network(data, tmp_path):
    model_json, shards = data
    out = str(tmp_path / "range")
    range_test.run_range_test([model_json, shards, out] + RUN_ARGS + [
        "--epochs", "1", "--range-warmup-steps", "1", "--range-check-every", "1"])
    with open(os.path.join(out, "step_diagnostics.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    assert rows and all(np.isfinite(r["loss"]) for r in rows)
