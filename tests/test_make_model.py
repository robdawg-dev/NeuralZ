"""Tests for AlphaGo/models/make_model.py, the model JSON generator."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from AlphaGo.models import make_model as mm
from AlphaGo.models.nn_util import NeuralNetBase
from AlphaGo.models.policy import CNNPolicy, NewResPolicy, ResTowerPolicy
from AlphaGo.preprocessing.convert_shuffled import ALL_FEATURES
from tests.test_convert_shuffled import FEATURES, _run, _selection

RESTOWER = ["restower", "--blocks", "2", "--filters", "8", "--head", "conv_norm"]
CNN = ["cnn", "--layers", "2", "--filters", "8"]
NEWRES = ["newres", "--blocks", "2", "--filters", "8", "--gpool-every", "2",
          "--gpool-channels", "4"]


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    """Shards built with a reduced feature list (not ALL_FEATURES)."""
    tmp = tmp_path_factory.mktemp("make_model")
    sel = _selection(tmp, {"train": [(s, None) for s in range(2)]})
    out = str(tmp / "shards")
    _run(sel, out, "--splits", "train")
    return out


def _load(path):
    return NeuralNetBase.load_model(str(path))


@pytest.mark.parametrize("args,name,cls", [
    (RESTOWER, "model_restower_b2c8_convnorm.json", ResTowerPolicy),
    (CNN, "model_cnn_l2c8.json", CNNPolicy),
    (NEWRES, "model_newres_b2c8_g2.json", NewResPolicy),
    (NEWRES + ["--no-head-gpool"], "model_newres_b2c8_g2_nohg.json", NewResPolicy),
])
def test_default_output_is_named_from_the_options(in_tmp, args, name, cls):
    out = mm.make_model(args)
    assert out == os.path.join("workspace", "models", name)
    policy = _load(in_tmp / out)
    assert type(policy) is cls
    assert policy.preprocessor.get_feature_list() == ALL_FEATURES
    assert policy.preprocessor.get_output_dimension() == 48


@pytest.mark.parametrize("args,cls,kwargs", [
    (RESTOWER + ["--head-channels", "16", "--stem-filter-width", "5"], ResTowerPolicy,
     {"num_blocks": 2, "filters": 8, "head": "conv_norm", "head_channels": 16,
      "stem_filter_width": 5}),
    (CNN + ["--first-filter-width", "3", "--kernel-initializer", "he_normal"], CNNPolicy,
     {"layers": 2, "filters_per_layer": 8, "filter_width_1": 3,
      "kernel_initializer": "he_normal"}),
    (NEWRES + ["--head-channels", "16", "--no-head-gpool", "--stem-filter-width", "5"],
     NewResPolicy,
     {"num_blocks": 2, "filters": 8, "gpool_every": 2, "gpool_channels": 4,
      "head_channels": 16, "head_gpool": False, "stem_filter_width": 5}),
])
def test_builds_the_same_network_as_the_class_directly(in_tmp, args, cls, kwargs):
    made = _load(in_tmp / mm.make_model(args)).model
    direct = cls(list(ALL_FEATURES), **kwargs).model
    assert made.count_params() == direct.count_params()
    assert [type(layer).__name__ for layer in made.layers] == [
        type(layer).__name__ for layer in direct.layers]


def test_explicit_feature_list(in_tmp):
    out = mm.make_model(RESTOWER + ["--features", "board,ones,sensibleness"])
    assert _load(in_tmp / out).preprocessor.get_feature_list() == [
        "board", "ones", "sensibleness"]


def test_features_from_shards(in_tmp, shards):
    out = mm.make_model(CNN + ["--features-from", shards, "--out", "m.json"])
    assert _load(in_tmp / out).preprocessor.get_feature_list() == FEATURES.split(",")


def test_board_must_match_the_shards(in_tmp, shards):
    with pytest.raises(ValueError, match="doesn't match the shards' board size 19"):
        mm.make_model(CNN + ["--features-from", shards, "--board", "9"])


def test_features_and_features_from_are_exclusive(in_tmp, shards):
    with pytest.raises(SystemExit):
        mm.make_model(CNN + ["--features", "board", "--features-from", shards])


def test_unknown_feature_raises(in_tmp):
    with pytest.raises(ValueError, match="bogus"):
        mm.make_model(CNN + ["--features", "board,bogus"])


def test_board_size(in_tmp):
    out = mm.make_model(CNN + ["--board", "9", "--out", "m9.json"])
    assert _load(in_tmp / out).model.output_shape == (None, 81)


def test_refuses_to_overwrite_without_force(in_tmp):
    out = mm.make_model(CNN)
    first = (in_tmp / out).read_text()
    with pytest.raises(ValueError, match="already exists"):
        mm.make_model(CNN + ["--filters", "4", "--out", out])
    assert (in_tmp / out).read_text() == first
    mm.make_model(CNN + ["--filters", "4", "--out", out, "--force"])
    assert (in_tmp / out).read_text() != first


@pytest.mark.parametrize("args", [
    ["restower", "--filters", "8", "--head", "conv"],
    ["restower", "--blocks", "2", "--filters", "8"],
    ["cnn", "--layers", "2"],
    ["newres", "--blocks", "2"],
    [],
])
def test_architecture_options_are_required(in_tmp, args):
    with pytest.raises(SystemExit):
        mm.make_model(args)
