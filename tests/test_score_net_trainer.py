"""End-to-end test for score_net_trainer: tiny shards with value sidecars, sibling groups
built by a tiny policy, placeholder KataGo labels, a few steps - the run trains, logs the
sibling ranking metrics and policy accuracy, and saves a loadable policy+score network.
Plus the ranking term of the score loss on its own."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json

import h5py as h5
import keras
import numpy as np

from AlphaGo.models.nn_util import NeuralNetBase
from AlphaGo.models.policy import NewResPolicy
from AlphaGo.models.value import PolicyValueNet
from AlphaGo.preprocessing import add_value_targets as avt
from AlphaGo.preprocessing import build_sibling_positions as bsp
from AlphaGo.preprocessing import convert_shuffled as conv
from AlphaGo.training import score_net_trainer as snt
from AlphaGo.training.shard_stream import PACKED_STATES, PLANES_SHAPE
from tests.test_add_value_targets import FEATURES, _game

K = 4


def _data(tmp_path):
    sel, sgfs = tmp_path / "sel", tmp_path / "sgf"
    sel.mkdir()
    sgfs.mkdir()
    for split, seeds in (("train", range(8)), ("val", range(8, 12))):
        with open(str(sel / (split + ".txt")), "w", newline="\n") as f:
            for seed in seeds:
                path = str(sgfs / "g{}.sgf".format(seed))
                with open(path, "w") as g:
                    g.write(_game(seed))
                f.write("{}\t40\tnormal\n".format(path))
    shards = str(tmp_path / "shards")
    conv.main([str(sel), shards, "--splits", "train", "val", "--features", FEATURES,
               "--workers", "2", "--quiet", "--positions-per-bucket", "100",
               "--positions-per-file", "100"])
    avt.main([shards, "--splits", "train", "val", "--workers", "2", "--quiet"])
    policy = NewResPolicy(FEATURES.split(","), num_blocks=1, filters=4, gpool_every=0,
                          head_channels=2)
    model_json, weights = str(tmp_path / "policy.json"), str(tmp_path / "policy.weights.h5")
    policy.save_model(model_json)
    policy.model.save_weights(weights)
    sib = str(tmp_path / "sib")
    rng = np.random.default_rng(0)
    for split, n in (("train", 30), ("val", 12)):
        bsp.build(shards, sib, policy, split, n, K, seed=1, workers=2, quiet=True)
        labels = np.stack([rng.normal(0, 10, n * K), rng.uniform(0, 1, n * K)], axis=1)
        labels.astype(np.float32).tofile(os.path.join(sib, split, "labels.bin"))
    return model_json, weights, shards, sib, policy


def test_ranking_term_only_sees_differences_within_groups():
    loss = snt.make_score_loss(n_sibling_rows=4, k=2, rank_weight=1.0)
    # two groups of 2 sibling rows + 1 policy row; absolute weight 0 everywhere
    y_true = np.array([[10, 0], [0, 0], [5, 0], [5, 0], [3, 0]], np.float32)
    shifted = np.array([[30], [20], [40], [40], [99]], np.float32)  # right gaps, wrong level
    wrong = np.array([[0], [10], [40], [40], [99]], np.float32)     # gap reversed in group 1
    assert np.allclose(np.asarray(loss(y_true, shifted)), 0)
    assert np.asarray(loss(y_true, wrong))[:2].min() > 0


def _lookahead_cache(sib, path, positions=3):
    """A stand-in for the lookahead test: the first val sibling groups as candidate boards."""
    rng = np.random.default_rng(2)
    with h5.File(os.path.join(sib, "val", "sib_00000.h5"), "r") as f:
        packed = f[PACKED_STATES][:positions * K]
        shape = f[PACKED_STATES].attrs[PLANES_SHAPE]
    with h5.File(path, "w") as f:
        f.create_dataset(PACKED_STATES, data=packed)
        f[PACKED_STATES].attrs[PLANES_SHAPE] = shape
        f.create_dataset("komi", data=np.full(len(packed), -7.5, np.float32))
        f.create_dataset("position", data=np.repeat(np.arange(positions), K).astype(np.int32))
        f.create_dataset("kata_loss", data=rng.uniform(0, 3, len(packed)).astype(np.float32))
    return path


def test_training_runs_and_saves_a_policy_score_net(tmp_path):
    model_json, weights, shards, sib, policy = _data(tmp_path)
    out = str(tmp_path / "run")
    cache = _lookahead_cache(sib, str(tmp_path / "cache.h5"))
    snt.run([model_json, weights, shards, sib, out, "--top", str(K), "--sibling-groups", "3",
             "--policy-rows", "8", "--epochs", "2", "--steps-per-epoch", "3",
             "--warmup-steps", "2", "--validation-groups", "4", "--validation-batches", "2",
             "--value-channels", "4", "--value-hidden", "6", "--seed", "1",
             "--oversample", "1,2,4,8", "--lookahead-cache", cache, "--lookahead-every", "2",
             "--bn-refresh-steps", "2", "--keep-optimizer", "2"])
    with open(os.path.join(out, "metadata.json")) as f:
        meta = json.load(f)
    epochs = meta["epochs"]
    assert len(epochs) == 2
    keys = set(epochs[-1])
    assert {"val_sib_best_agree", "val_sib_centered_mae", "val_sib_spearman"} <= keys
    assert any(k.endswith("accuracy") for k in keys)
    assert all(np.isfinite(e["loss"]) for e in epochs)
    # the lookahead test only on every 2nd epoch
    assert "lookahead_score_top10" not in epochs[0]
    assert {"lookahead_greedy", "lookahead_score_top5", "lookahead_score_top10",
            "lookahead_value_top10"} <= keys
    assert len(meta["oversampling"]["pass_shares"]) == 4
    bn = meta["bn_refresh"]
    assert bn["epoch"] == 2 and np.isfinite(bn["before"]) and np.isfinite(bn["after"])
    assert os.path.exists(os.path.join(out, "weights.00002.bn.weights.h5")) == bool(bn["kept"])
    assert [os.path.exists(snt.optimizer_path(out, e)) for e in (1, 2)] == [True, True]

    # a learning rate set while "running" is applied at the next epoch
    with open(os.path.join(out, "lr_override.txt"), "w") as f:
        f.write("0.00123\n")

    # resume from epoch 2 for one more epoch: numbering and metadata carry on
    snt.run([model_json, weights, shards, sib, out, "--top", str(K), "--sibling-groups", "3",
             "--policy-rows", "8", "--epochs", "3", "--steps-per-epoch", "3",
             "--warmup-steps", "2", "--validation-groups", "4", "--validation-batches", "2",
             "--value-channels", "4", "--value-hidden", "6", "--seed", "1",
             "--resume-weights", "weights.00002.weights.h5", "--bn-refresh-steps", "0",
             "--keep-optimizer", "2"])
    with open(os.path.join(out, "metadata.json")) as f:
        meta = json.load(f)
    assert [e["epoch"] for e in meta["epochs"]] == [1, 2, 3]
    assert meta["resumed"][0]["from"] == "weights.00002.weights.h5"
    assert os.path.exists(os.path.join(out, "weights.00003.weights.h5"))
    assert np.isclose(meta["epochs"][-1]["learning_rate"], 0.00123)
    assert os.path.exists(os.path.join(out, "lr_override.applied.00003.txt"))
    assert not os.path.exists(os.path.join(out, "lr_override.txt"))
    # only the latest --keep-optimizer optimizer states are kept
    assert [os.path.exists(snt.optimizer_path(out, e)) for e in (1, 2, 3)] == [False, True,
                                                                              True]

    net = NeuralNetBase.load_model(os.path.join(out, "model.json"))
    assert isinstance(net, PolicyValueNet)
    net.model.load_weights(os.path.join(out, "weights.00002.weights.h5"))
    planes = np.zeros((2, 19, 19, policy.preprocessor.get_output_dimension()), np.float32)
    p, v, s = net.forward_all(planes, [7.5, -7.5])
    assert p.shape == (2, 361) and v.shape == s.shape == (2,)


def test_reader_joins_builds_and_reads_groups_by_number(tmp_path):
    _m, _w, _s, sib, _p = _data(tmp_path)
    one = snt.SiblingReader(os.path.join(sib, "train"), K)
    two = snt.SiblingReader([os.path.join(sib, "train")] * 2, K)
    n = one.total // K
    assert two.total == 2 * one.total and len(two.sources()[0]) == 2 * n
    packed, komi, labels = two.read_groups([n + 3, 1])
    p3, k3, l3 = one.read(3 * K, K)
    p1, k1, l1 = one.read(1 * K, K)
    assert np.array_equal(packed, np.concatenate([p3, p1]))
    assert np.array_equal(labels, np.concatenate([l3, l1]))
    assert np.array_equal(komi, np.concatenate([k3, k1]))


class _Groups(object):
    """A SiblingReader stand-in for group_strata: k = 2."""
    k = 2

    def __init__(self, nodes, paths, labels):
        self.nodes, self.paths = np.array(nodes), paths
        self.labels = np.array(labels, np.float32)

    def sources(self):
        return self.paths, self.nodes


def test_group_strata_from_phase_handicap_and_gap():
    # the labels are the opponent's scores: the gap is first sibling minus the lowest
    reader = _Groups([10, 150, 300], ["a", "h", "a"],
                     [[3, 0], [2.5, 0], [7, 0], [0, 0], [9, 0], [1, 0]])
    strata, tiers = snt.group_strata(reader, {"h"})
    assert list(strata) == [0 * 2, 2 * 2 + 1, 4 * 2]
    assert list(tiers) == [0, 3, 3]
    reader.labels[:, 0] = [4, 2.5, 0, 0, 3, 1]
    assert list(snt.group_strata(reader, set())[1]) == [1, 0, 2]


def test_oversampling_keeps_strata_shares_and_boosts_big_gaps():
    rng = np.random.default_rng(0)
    n = 4000
    strata = rng.integers(0, 4, n)
    tiers = rng.choice(4, n, p=[0.875, 0.064, 0.041, 0.02])
    order = snt.OversampledOrder(strata, tiers, [1, 2, 4, 8], seed=3)
    one = order.pass_order(1)
    assert len(one) == n and np.array_equal(one, order.pass_order(1))
    assert not np.array_equal(one, order.pass_order(2))
    # each stratum keeps its share of the pass (within rounding)
    assert np.abs(np.bincount(strata[one], minlength=4) - np.bincount(strata, minlength=4)
                  ).max() <= 30
    sampled, natural = order.shares()
    assert sampled[3] > 6 * natural[3] and sampled[0] < natural[0]
    # a big-gap group is in every pass, about 8 times
    big = np.flatnonzero(tiers == 3)[0]
    assert all(6 <= (order.pass_order(p) == big).sum() <= 10 for p in range(3))
    # the calm groups rotate: all of them seen within a few passes
    calm = np.flatnonzero(tiers == 0)
    seen = np.concatenate([order.pass_order(p) for p in range(3)])
    assert np.isin(calm, seen).all()
    # the stream continues across passes
    assert np.array_equal(order.groups_at(n - 2, 4), np.concatenate([
        order.pass_order(0)[-2:], order.pass_order(1)[:2]]))


def test_lookahead_rules_pick_by_the_heads():
    class Net(object):  # rates each board for the opponent: lower is better for the bot
        def __init__(self, scores, values):
            self.scores, self.values, self.at = np.array(scores), np.array(values), 0

        def forward_all(self, planes, komi):
            n = len(planes)
            s, v = self.scores[self.at:self.at + n], self.values[self.at:self.at + n]
            self.at += n
            return None, v, s

    cache = {"packed": np.zeros((5, 1), np.uint8), "shape": (2, 2, 2),
             "komi": np.zeros(5, np.float32), "position": np.array([0, 0, 0, 1, 1]),
             "loss": np.array([2.0, 0.5, 4.0, 1.0, np.nan])}
    # position 0: the score head prefers the 2nd candidate, the win rate the 3rd;
    # position 1: one scored candidate, so it stands
    net = Net([3.0, -1.0, 0.0, 5.0, -9.0], [0.5, 0.6, 0.2, 0.5, 0.0])
    m = snt.lookahead_metrics(net, cache, batch=2)
    assert m["lookahead_greedy"] == 3.0
    assert m["lookahead_score_top10"] == m["lookahead_score_top5"] == 1.5
    assert m["lookahead_value_top10"] == 5.0


def test_optimizer_state_round_trips_with_mixed_precision(tmp_path):
    def compiled():
        keras.utils.set_random_seed(0)
        inp = keras.Input((3,))
        model = keras.Model(inp, keras.layers.Dense(1, dtype="float32")(
            keras.layers.Dense(4)(inp)))
        model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
        return model

    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        x, y = np.ones((8, 3), np.float32), np.zeros((8, 1), np.float32)
        a = compiled()
        a.fit(x, y, epochs=2, verbose=0)
        a.optimizer.learning_rate.assign(3e-3)
        path = str(tmp_path / "opt.npz")
        snt.save_optimizer(a.optimizer, path, {"plateau": {"best": 1.0}})
        b = compiled()
        assert snt.restore_optimizer(b, path) == {"plateau": {"best": 1.0}}
        for va, vb in zip(a.optimizer.variables, b.optimizer.variables):
            assert np.array_equal(np.asarray(va), np.asarray(vb))
        assert np.isclose(float(np.asarray(b.optimizer.learning_rate)), 3e-3)
        assert int(np.asarray(b.optimizer.iterations)) == 2
    finally:
        keras.mixed_precision.set_global_policy("float32")


def test_guard_rules_switch_only_past_the_threshold(tmp_path):
    class Net(object):
        def forward_all(self, planes, komi):
            # boards rated for the opponent: the bot's view is minus these
            s = np.array([0.0, -1.5, 0.0, -3.0], np.float32)[:len(planes)]
            return None, np.zeros(len(planes)), s

    # one position, 4 candidates: the 2nd is 1.5 points better, the 4th 3 points better
    cache = {"packed": np.zeros((4, 1), np.uint8), "shape": (2, 2, 2),
             "komi": np.zeros(4, np.float32), "position": np.zeros(4, np.int32),
             "loss": np.array([5.0, 2.0, 1.0, 0.5])}
    m = snt.lookahead_metrics(Net(), cache)
    assert m["lookahead_greedy"] == 5.0
    assert m["lookahead_score_top5"] == 0.5        # T 0: the best predicted
    assert m["lookahead_score_top5_t1"] == 0.5     # 3 > 1: switches
    assert m["lookahead_score_top5_t2"] == 0.5     # 3 > 2: switches
    cache["loss"] = np.array([5.0, 2.0, 1.0, np.nan])  # the 4th unscored: best gain 1.5
    m = snt.lookahead_metrics(Net(), cache)
    assert m["lookahead_score_top5_t1"] == 2.0 and m["lookahead_score_top5_t2"] == 5.0


def test_cache_with_a_sampling_window_keeps_moves_after_it(tmp_path):
    path = str(tmp_path / "c.h5")
    with h5.File(path, "w") as f:
        f.create_dataset(PACKED_STATES, data=np.arange(6, dtype=np.uint8)[:, None])
        f[PACKED_STATES].attrs[PLANES_SHAPE] = np.array([2, 2, 2])
        f.create_dataset("komi", data=np.zeros(6, np.float32))
        f.create_dataset("position", data=np.array([0, 0, 1, 1, 2, 2], np.int32))
        f.create_dataset("kata_loss", data=np.arange(6, dtype=np.float32))
        f.create_dataset("pos_window", data=np.array([True, False, False]))
    c = snt.load_lookahead_cache(path)
    assert list(c["position"]) == [1, 1, 2, 2] and list(c["packed"][:, 0]) == [2, 3, 4, 5]
