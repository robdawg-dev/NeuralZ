"""Fine-tune a policy network into a policy + score network for one-ply lookahead
(SCORE_NET_PLAN.md, option A).

    python -m AlphaGo.training.score_net_trainer <policy model.json> <policy weights.h5> \\
        <shards_dir> <siblings_dir> <out_directory> [options]

Starts from the policy network (b20c256 epoch 67) plus a fresh value/score head, every
layer trainable. Each batch has a fixed layout:

- the first --sibling-groups x k rows: whole sibling groups from build_sibling_positions.py
  (the policy's top k moves played out from one source position), labeled by
  label_sibling_positions.py with KataGo's raw-network score and win rate for the
  sibling's player to move. They train the score head - an absolute Huber loss plus a
  ranking term: within each group, predicted minus group mean should match KataGo's
  minus its group mean (what the lookahead uses) - and the win rate (BCE).
- the remaining --policy-rows rows: ordinary records from <shards_dir>/train with the move
  played, training the policy so it stays where it was, plus their KataGo annotations
  (value sidecars) as further absolute score / win-rate targets where present.

After every epoch, SiblingEval measures the ranking directly on the held-out val sibling
groups: how often the network's best sibling is KataGo's best, the error of the centered
scores, and the rank correlation within groups (logged as val_sib_*). Each epoch saves the
whole network (model.json + weights.NNNNN.weights.h5) and its metrics (metadata.json).
"""
import argparse
import csv
import io
import json
import os
import re

import h5py as h5
import numpy as np
import tensorflow as tf
import keras
from keras import mixed_precision, ops
from keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN
from keras.models import Model
from keras.optimizers import Adam

from AlphaGo.models.nn_util import NeuralNetBase
from AlphaGo.models.value import KOMI_SCALE, SCORE, VALUE, PolicyValueNet
from AlphaGo.training.shard_stream import (
    BATCH_TRANSFORMATIONS, PACKED_STATES, PLANES_SHAPE, _resolve_seed, _symmetry_choices,
    dataset_info, decode_on_device, encode_labels, find_split_shards)
from AlphaGo.training.value_head_trainer import LinearWarmup, SaveEpoch, ValueReader


class SiblingReader(object):
    """Sibling groups from one or more <siblings_dir>/<split> directories (e.g. a build and
    its extension), as one array: packed planes, komi and KataGo's (score, win rate) per
    sibling, read group-aligned - in order, wrapping endlessly, or by group number."""

    def __init__(self, split_dirs, k):
        if isinstance(split_dirs, str):
            split_dirs = [split_dirs]
        self.k = k
        self.dirs = list(split_dirs)
        self.files, self.sizes, labels = [], [], []
        for d in self.dirs:
            names = sorted(f for f in os.listdir(d) if f.startswith("sib_"))
            lab = np.fromfile(os.path.join(d, "labels.bin"), dtype=np.float32).reshape(-1, 2)
            rows = 0
            for name in names:
                with h5.File(os.path.join(d, name), "r") as f:
                    self.sizes.append(len(f["group"]))
                    self.shape = tuple(int(x) for x in f[PACKED_STATES].attrs[PLANES_SHAPE])
                rows += self.sizes[-1]
            if rows != len(lab) or rows % k:
                raise ValueError("{}: {} siblings, {} labels, k {}".format(d, rows, len(lab), k))
            self.files += [os.path.join(d, n) for n in names]
            labels.append(lab)
        self.labels = np.concatenate(labels)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.starts[-1])
        self.handles = {}

    def _file(self, i):
        if i not in self.handles:
            self.handles[i] = h5.File(self.files[i], "r")
        return self.handles[i]

    def read_groups(self, groups):
        """Whole groups by number, in the order given -> packed, komi, labels."""
        packed, komi, labels = [], [], []
        for g in groups:
            p = int(g) * self.k
            i = int(np.searchsorted(self.starts, p, side="right") - 1)
            offset = p - int(self.starts[i])
            f = self._file(i)
            packed.append(f[PACKED_STATES][offset:offset + self.k])
            komi.append(f["komi"][offset:offset + self.k].astype(np.float32))
            labels.append(self.labels[p:p + self.k])
        return np.concatenate(packed), np.concatenate(komi), np.concatenate(labels)

    def sources(self):
        """Per group: its source position's SGF path and move-node index (queries.tsv)."""
        paths, nodes = [], []
        for d in self.dirs:
            with io.open(os.path.join(d, "queries.tsv"), encoding="utf-8") as f:
                for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
                    if i % self.k == 0:
                        paths.append(r["path"])
                        nodes.append(int(r["node"]))
        return paths, np.array(nodes)

    def read(self, position, n):
        """n rows from row position (both multiples of k) -> packed, komi, labels."""
        packed, komi, labels = [], [], []
        while n > 0:
            p = position % self.total
            i = int(np.searchsorted(self.starts, p, side="right") - 1)
            offset = p - int(self.starts[i])
            take = min(n, int(self.starts[i + 1]) - p)
            f = self._file(i)
            packed.append(f[PACKED_STATES][offset:offset + take])
            komi.append(f["komi"][offset:offset + take].astype(np.float32))
            labels.append(self.labels[p:p + take])
            position += take
            n -= take
        return np.concatenate(packed), np.concatenate(komi), np.concatenate(labels)

    def close(self):
        for f in self.handles.values():
            f.close()
        self.handles = {}


PHASE_BOUNDS = (30, 100, 180, 260)   # move-node index: 0-29, 30-99, 100-179, 180-259, 260+
GAP_BOUNDS = (1.0, 2.0, 5.0)         # points: <1, 1-2, 2-5, >5


def group_strata(reader, handicap_paths):
    """Per group: its stratum (game phase x handicap/even) and gap tier - how much better
    KataGo's best sibling is than the policy's first choice, for the source mover."""
    paths, nodes = reader.sources()
    phase = np.searchsorted(PHASE_BOUNDS, nodes, side="right")
    handicap = np.array([p in handicap_paths for p in paths], np.int64)
    opponent = reader.labels[:, 0].reshape(-1, reader.k)  # the sibling's player to move
    gap = opponent[:, 0] - opponent.min(axis=1)
    return phase * 2 + handicap, np.searchsorted(GAP_BOUNDS, gap, side="right")


class OversampledOrder(object):
    """Group order for stratified oversampling. Each pass is as long as the data and every
    stratum keeps its share of it; within a stratum a group of gap tier t appears weights[t]
    times (scaled down, stochastically rounded, where that would leave under a tenth of the
    stratum calm), and the calm groups (tier 0) fill the rest - each pass the next ones in a
    fixed shuffled rotation, so all of them are seen across passes. Pass p depends only on
    (seed, p), so a resumed run continues the same stream."""

    def __init__(self, strata, tiers, weights, seed):
        self.strata, self.tiers = np.asarray(strata), np.asarray(tiers)
        self.weights = np.asarray(weights, np.float64)
        self.seed = 0 if seed is None else seed
        self.n = len(self.strata)
        rng = np.random.default_rng(self.seed)
        self.calm = {int(s): rng.permutation(np.flatnonzero((self.strata == s) & (self.tiers == 0)))
                     for s in np.unique(self.strata)}
        self._cached = (None, None)

    def pass_order(self, p):
        rng = np.random.default_rng([self.seed, p])
        slots = []
        for s, calm in self.calm.items():
            in_s = self.strata == s
            members = np.flatnonzero(in_s & (self.tiers > 0))
            n_s = int(in_s.sum())
            w = self.weights[self.tiers[members]]
            if w.sum() > 0.9 * n_s:
                w = w * (0.9 * n_s / w.sum())
            reps = np.floor(w).astype(np.int64) + (rng.random(len(w)) < w - np.floor(w))
            slots.append(np.repeat(members, reps))
            m = min(max(n_s - int(reps.sum()), 0), len(calm))
            if m:
                slots.append(np.take(calm, np.arange(p * m, (p + 1) * m), mode="wrap"))
        order = np.concatenate(slots)
        rng.shuffle(order)
        if len(order) < self.n:  # rounding: top up so every pass has exactly n groups
            order = np.concatenate([order, rng.choice(order, self.n - len(order))])
        return order[:self.n]

    def groups_at(self, position, count):
        """count group numbers from stream position (in groups) on, across passes."""
        out = []
        while count > 0:
            p, offset = divmod(position, self.n)
            if self._cached[0] != p:
                self._cached = (p, self.pass_order(p))
            take = min(count, self.n - offset)
            out.append(self._cached[1][offset:offset + take])
            position += take
            count -= take
        return np.concatenate(out)

    def shares(self):
        """Fraction of pass 0 per gap tier (vs the data's natural shares)."""
        order = self.pass_order(0)
        tiers = len(self.weights)
        return (np.bincount(self.tiers[order], minlength=tiers) / len(order),
                np.bincount(self.tiers, minlength=tiers) / self.n)


def make_score_loss(n_sibling_rows, k, rank_weight):
    """Score loss over a batch laid out as [sibling rows | policy rows]. y_true columns:
    target, absolute-loss weight. Absolute Huber (score / KOMI_SCALE) on every row with
    weight > 0; on the sibling rows also the ranking term - centered prediction vs centered
    target within each group of k."""
    def score_loss(y_true, y_pred):
        target, weight = y_true[:, 0], y_true[:, 1]
        pred = ops.cast(y_pred[:, 0], "float32")
        absolute = keras.losses.huber(target[:, None] / KOMI_SCALE, pred[:, None] / KOMI_SCALE,
                                      delta=1.0) * weight
        t = ops.reshape(target[:n_sibling_rows], (-1, k))
        p = ops.reshape(pred[:n_sibling_rows], (-1, k))
        t = t - ops.mean(t, axis=1, keepdims=True)
        p = p - ops.mean(p, axis=1, keepdims=True)
        rank = keras.losses.huber(ops.reshape(t, (-1, 1)) / KOMI_SCALE,
                                  ops.reshape(p, (-1, 1)) / KOMI_SCALE, delta=1.0)
        rank = ops.concatenate([rank, ops.zeros_like(absolute[n_sibling_rows:])], axis=0)
        return absolute + rank_weight * rank
    return score_loss


def make_batches(sib, val_reader, groups, policy_rows, k, symmetries, board_size, seed,
                 sib_start=0, pol_start=0, order=None):
    """Endless batches: ((packed, choices, komi), (policy, value, score), weights). The
    sibling groups come in file order, or from order (an OversampledOrder)."""
    seed = _resolve_seed(seed)
    n_sib = groups * k
    sp, pp = sib_start, pol_start
    while True:
        if order is None:
            s_packed, s_komi, s_lab = sib.read(sp, n_sib)
        else:
            s_packed, s_komi, s_lab = sib.read_groups(order.groups_at(sp // k, groups))
        p_packed, p_actions, p_komi, p_value, p_score, p_has = val_reader.read(pp, policy_rows)
        n = n_sib + policy_rows
        choices = np.concatenate([
            _symmetry_choices(seed, sp, n_sib, len(symmetries)),
            _symmetry_choices(seed + 7, pp, policy_rows, len(symmetries))]).astype(np.int32)
        policy = np.zeros((n, board_size * board_size), np.float32)
        policy[n_sib:] = encode_labels(p_actions, choices[n_sib:], symmetries, board_size)
        value = np.concatenate([s_lab[:, 1], p_value[:, 0]])[:, None]
        score = np.stack([np.concatenate([s_lab[:, 0], p_score[:, 0]]),
                          np.concatenate([np.ones(n_sib, np.float32), p_has])], axis=1)
        x = (np.concatenate([s_packed, p_packed]), choices,
             np.concatenate([s_komi, p_komi[:, 0]])[:, None])
        w = (np.concatenate([np.zeros(n_sib, np.float32), np.ones(policy_rows, np.float32)]),
             np.concatenate([np.ones(n_sib, np.float32), p_has]),
             np.ones(n, np.float32))
        yield x, (policy, value, score), w
        sp += n_sib
        pp += policy_rows


def decode_in_steps(model, board_size, n_features, symmetries):
    dtype = model.inputs[0].dtype
    train_step, test_step = model.train_step, model.test_step

    def decoded(data):
        (packed, choices, komi), y, w = data
        planes = decode_on_device(packed, choices, board_size, n_features, symmetries, dtype)
        return (planes, komi), y, w

    model.train_step = lambda data: train_step(decoded(data))
    model.test_step = lambda data: test_step(decoded(data))


def sibling_metrics(net, reader, k, n_groups, batch_groups=16):
    """Ranking metrics on the first n_groups groups (batch_groups at a time: an eager call,
    which needs more memory than the compiled training step): agreement with KataGo on the best move
    (the sibling with the lowest score for the opponent, its player to move),
    mean absolute error of group-centered scores, mean Spearman correlation."""
    n_groups = min(n_groups, reader.total // k)
    agree, err, rho, done = 0, 0.0, 0.0, 0
    for g in range(0, n_groups, batch_groups):
        take = min(batch_groups, n_groups - g)
        packed, komi, lab = reader.read(g * k, take * k)
        planes = np.unpackbits(packed, axis=1, count=int(np.prod(reader.shape))).reshape(
            (-1,) + reader.shape).astype(np.float32)
        _p, _v, score = net.forward_all(planes, komi)
        pred, target = score.reshape(take, k), lab[:, 0].reshape(take, k)
        # the labels are for the sibling's player to move - the opponent - so the move best
        # for the source mover is the sibling with the LOWEST score
        agree += int((pred.argmin(axis=1) == target.argmin(axis=1)).sum())
        err += float(np.abs((pred - pred.mean(1, keepdims=True))
                            - (target - target.mean(1, keepdims=True))).mean(1).sum())
        rp, rt = pred.argsort(1).argsort(1), target.argsort(1).argsort(1)
        rho += float(np.nansum([np.corrcoef(a, b)[0, 1] for a, b in zip(rp, rt)]))
        done += take
    return {"val_sib_best_agree": agree / done, "val_sib_centered_mae": err / done,
            "val_sib_spearman": rho / done}


class SiblingEval(Callback):
    """After each epoch, the ranking metrics on held-out sibling groups, into the logs
    (and so metadata.json and the plateau monitor)."""

    def __init__(self, net, reader, k, n_groups):
        super().__init__()
        self.net, self.reader, self.k, self.n_groups = net, reader, k, n_groups

    def on_epoch_end(self, epoch, logs=None):
        metrics = sibling_metrics(self.net, self.reader, self.k, self.n_groups)
        if logs is not None:
            logs.update(metrics)
        print(" - " + " - ".join("{}: {:.4f}".format(k, v) for k, v in metrics.items()),
              flush=True)


def _cycle(batches):
    """The fixed validation batches, endlessly (Keras takes a generator, not an iterator)."""
    while True:
        for b in batches:
            yield b


def load_lookahead_cache(path):
    """The lookahead test's candidate boards (workspace/kgs50/build_cache.py, or the older
    7-game workspace/score_net/build_lookahead_cache.py): per candidate, in rank order within
    each bot move - packed planes, komi for the board's player to move, the move's index,
    KataGo's points lost (NaN: never scored). A cache that marks the moves inside the bot's
    sampling window (pos_window) keeps only the moves after it - where the lookahead acts."""
    with h5.File(path, "r") as f:
        cache = {"packed": f[PACKED_STATES][()],
                 "shape": tuple(int(d) for d in f[PACKED_STATES].attrs[PLANES_SHAPE]),
                 "komi": f["komi"][()].astype(np.float32), "position": f["position"][()],
                 "loss": f["kata_loss"][()].astype(np.float64)}
        if "pos_window" in f:
            keep = ~f["pos_window"][()][cache["position"]]
            cache = dict(cache, packed=cache["packed"][keep], komi=cache["komi"][keep],
                         position=cache["position"][keep], loss=cache["loss"][keep])
    return cache


LOOKAHEAD_RULES = {  # name -> (head, top k, threshold): None is greedy
    "lookahead_greedy": None,
    "lookahead_score_top5": ("score", 5, 0.0), "lookahead_score_top10": ("score", 10, 0.0),
    "lookahead_value_top10": ("value", 10, 0.0),
    "lookahead_score_top5_t1": ("score", 5, 1.0), "lookahead_score_top5_t2": ("score", 5, 2.0)}


def lookahead_metrics(net, cache, batch=128):
    """KataGo points lost over the cached bot moves, for greedy (the policy's top move) and
    for one-ply lookahead picking by the score head or the win rate among the policy's top
    k: the policy's move stands unless another is predicted more than the threshold (points
    for the score head) better - T 0 is lookahead_report.py's rule, T 1 / T 2 the blunder
    guard. The heads rate each board for its player to move, the bot's opponent."""
    scores, values = [], []
    for i in range(0, len(cache["komi"]), batch):
        planes = np.unpackbits(cache["packed"][i:i + batch], axis=1,
                               count=int(np.prod(cache["shape"]))).reshape(
            (-1,) + cache["shape"]).astype(np.float32)
        _p, value, score = net.forward_all(planes, cache["komi"][i:i + batch])
        scores.append(-np.asarray(score, np.float64).reshape(-1))
        values.append(1 - np.asarray(value, np.float64).reshape(-1))
    ours = {"score": np.concatenate(scores), "value": np.concatenate(values)}
    totals = dict.fromkeys(LOOKAHEAD_RULES, 0.0)
    scored = np.flatnonzero(~np.isnan(cache["loss"]))
    # candidates come grouped by position, in rank order: split the scored ones per position
    bounds = np.flatnonzero(np.diff(cache["position"][scored])) + 1
    for rows in np.split(scored, bounds):
        if not len(rows):
            continue
        for name, rule in LOOKAHEAD_RULES.items():
            pick = rows[0]
            if rule is not None:
                key, top = ours[rule[0]], rows[:rule[1]]
                best = top[int(np.argmax(key[top]))]
                pick = best if key[best] > key[rows[0]] + rule[2] else rows[0]
            totals[name] += cache["loss"][pick]
    return totals


class LookaheadEval(Callback):
    """Every `every` epochs, the lookahead test on the cached candidate boards, into the logs
    (and so metadata.json)."""

    def __init__(self, net, cache, every):
        super().__init__()
        self.net, self.cache, self.every = net, cache, every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every:
            return
        metrics = lookahead_metrics(self.net, self.cache)
        if logs is not None:
            logs.update(metrics)
        print(" - " + " - ".join("{}: {:.0f}".format(k, v) for k, v in metrics.items()),
              flush=True)


class PlateauWithState(ReduceLROnPlateau):
    """ReduceLROnPlateau whose best / wait / cooldown can be carried over a resume (Keras
    resets them when training begins)."""

    def __init__(self, restore=None, **kwargs):
        super().__init__(**kwargs)
        self.restore = restore

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if self.restore:
            self.best, self.wait = self.restore["best"], self.restore["wait"]
            self.cooldown_counter = self.restore["cooldown_counter"]

    def state(self):
        return {"best": float(self.best), "wait": int(self.wait),
                "cooldown_counter": int(self.cooldown_counter)}


class LrOverride(Callback):
    """At the start of each epoch, a learning rate written to <out>/lr_override.txt replaces
    the current one (and restarts the plateau's patience); the file is then renamed
    lr_override.applied.NNNNN.txt. Steers a running job without a restart."""

    def __init__(self, out_directory, plateau=None):
        super().__init__()
        self.path = os.path.join(out_directory, "lr_override.txt")
        self.plateau = plateau

    def on_epoch_begin(self, epoch, logs=None):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            text = f.read().strip()
        os.replace(self.path, os.path.join(os.path.dirname(self.path),
                                           "lr_override.applied.{:05d}.txt".format(epoch + 1)))
        try:
            lr = float(text)
        except ValueError:
            print("lr_override.txt: not a number: {!r}".format(text), flush=True)
            return
        self.model.optimizer.learning_rate.assign(lr)
        if self.plateau is not None:
            self.plateau.wait = 0
        print("epoch {}: learning rate set to {:g} (lr_override.txt)".format(epoch + 1, lr),
              flush=True)


def optimizer_path(out_directory, epoch):
    return os.path.join(out_directory, "optimizer.{:05d}.npz".format(epoch))


def save_optimizer(optimizer, path, state):
    """The optimizer's variables (Adam's moments, step count, learning rate; with mixed
    precision also the loss scale) plus a JSON state dict."""
    arrays = {"var_{:05d}".format(i): ops.convert_to_numpy(v)
              for i, v in enumerate(optimizer.variables)}
    tmp = path + ".tmp.npz"
    np.savez(tmp, state=np.array(json.dumps(state)), **arrays)
    os.replace(tmp, path)


def restore_optimizer(model, path):
    """Builds the compiled model's optimizer and loads saved variables into it -> state."""
    model.optimizer.build(model.trainable_variables)
    with np.load(path) as f:
        values = [f["var_{:05d}".format(i)] for i in range(len(f.files) - 1)]
        state = json.loads(str(f["state"]))
    variables = model.optimizer.variables
    if len(values) != len(variables):
        raise ValueError("{}: {} optimizer variables, the model has {}".format(
            path, len(values), len(variables)))
    for v, x in zip(variables, values):
        v.assign(x)
    return state


class SaveOptimizer(Callback):
    """After each epoch (after the plateau's update): optimizer.NNNNN.npz beside the weights,
    so a resume continues Adam's moments, the learning rate and the plateau's state. Only
    the latest `keep` are kept (each is about twice the weights' size)."""

    def __init__(self, out_directory, plateau, keep=3):
        super().__init__()
        self.out, self.plateau, self.keep = out_directory, plateau, keep

    def on_epoch_end(self, epoch, logs=None):
        save_optimizer(self.model.optimizer, optimizer_path(self.out, epoch + 1),
                       {"epoch": epoch + 1, "plateau": self.plateau.state()})
        old = optimizer_path(self.out, epoch + 1 - self.keep)
        if os.path.exists(old):
            os.remove(old)


class BnRefreshAtEnd(Callback):
    """When training ends: refresh the batch-norm running averages of the final weights
    (`steps` training batches through the network in training mode, no weight updates, so
    they settle on the final weights rather than lagging them), and keep the refreshed copy
    (weights.NNNNN.bn.weights.h5) only if it scores better - `evaluate` returns a number,
    lower is better."""

    def __init__(self, net, batches, steps, evaluate, saver):
        super().__init__()
        self.net, self.batches, self.steps = net, batches, steps
        self.evaluate, self.saver = evaluate, saver

    def on_train_end(self, logs=None):
        if not self.saver.meta["epochs"] or not self.steps:
            return
        epoch = self.saver.meta["epochs"][-1]["epoch"]
        before = self.evaluate()
        for _ in range(self.steps):
            planes, komi = next(self.batches)
            self.net.model([planes, komi], training=True)
        after = self.evaluate()
        path = os.path.join(self.saver.out, "weights.{:05d}.bn.weights.h5".format(epoch))
        kept = after < before
        if kept:
            self.net.model.save_weights(path)
        self.saver.meta["bn_refresh"] = {"epoch": epoch, "steps": self.steps,
                                         "before": before, "after": after,
                                         "kept": os.path.basename(path) if kept else None}
        with open(self.saver.meta_path, "w") as f:
            json.dump(self.saver.meta, f, indent=2)
        print("BN refresh of epoch {}: {:.4f} -> {:.4f}, {}".format(
            epoch, before, after, "kept " + path if kept else "not kept"), flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="Policy network JSON (b20c256's)")
    p.add_argument("weights", help="Its weights (epoch 67)")
    p.add_argument("shards_dir", help="Shards with value sidecars (policy rows, validation)")
    p.add_argument("siblings_dir", help="build_sibling_positions.py output, labeled")
    p.add_argument("out_directory")
    p.add_argument("--top", type=int, default=8, help="Siblings per group. Default: 8")
    p.add_argument("--sibling-groups", type=int, default=32,
                   help="Sibling groups per batch. Default: 32 (256 rows at --top 8)")
    p.add_argument("--policy-rows", type=int, default=256, help="Policy rows per batch")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--plateau-min-delta", type=float, default=0.01,
                   help="Smallest drop in val_sib_centered_mae (points) that counts as an "
                        "improvement. Default: 0.01")
    p.add_argument("--extra-siblings", action="append", default=[], metavar="SIBLINGS_DIR",
                   help="Another labeled build (e.g. an extension made with "
                        "build_sibling_positions --exclude) whose train/ groups are added to "
                        "siblings_dir's (repeatable)")
    p.add_argument("--oversample", default=None, metavar="W0,W1,W2,W3",
                   help="Stratified oversampling: within each game phase x handicap/even "
                        "stratum, groups whose best sibling beats the policy's first choice "
                        "by <1, 1-2, 2-5, >5 points appear this many times per pass, calm "
                        "ones drawn fresh each pass (e.g. 1,2,4,8). Default: off, file order")
    p.add_argument("--lookahead-cache", default=None,
                   help="Candidate boards + KataGo losses for the lookahead test "
                        "(workspace/score_net/build_lookahead_cache.py)")
    p.add_argument("--lookahead-every", type=int, default=5,
                   help="Run the lookahead test every this many epochs. Default: 5")
    p.add_argument("--bn-refresh-steps", type=int, default=800,
                   help="When training ends, refresh the final weights' batch-norm averages "
                        "with this many batches of 128 rows and keep the copy if it scores "
                        "better (lookahead score top 5, T 1 with --lookahead-cache, else "
                        "val_sib_centered_mae). 0: off. Default: 800")
    p.add_argument("--keep-optimizer", type=int, default=3,
                   help="Latest optimizer.NNNNN.npz files kept. Default: 3")
    p.add_argument("--rank-weight", type=float, default=1.0,
                   help="Weight of the sibling ranking term in the score loss. Default: 1.0")
    p.add_argument("--score-weight", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--value-channels", type=int, default=48)
    p.add_argument("--value-hidden", type=int, default=112)
    p.add_argument("--validation-groups", type=int, default=5000,
                   help="Held-out sibling groups for the ranking metrics. Default: 5000")
    p.add_argument("--validation-batches", type=int, default=40,
                   help="Fixed validation batches (same layout as training). Default: 40")
    p.add_argument("--symmetries", default=",".join(BATCH_TRANSFORMATIONS))
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume-weights", default=None,
                   help="Continue a run: weights.NNNNN.weights.h5 in out_directory (with its "
                        "model.json). Epoch numbering, metadata.json and the data stream carry "
                        "on from epoch NNNNN. With its optimizer.NNNNN.npz, Adam's state, the "
                        "learning rate and the plateau's state carry on too (--learning-rate "
                        "and the warmup are then unused: steer with lr_override.txt); without "
                        "it the optimizer starts fresh, with --warmup-steps again")
    return p


def run(argv=None):
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        keras.utils.set_random_seed(args.seed)
    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
    os.makedirs(args.out_directory, exist_ok=True)

    initial_epoch = 0
    if args.resume_weights:
        initial_epoch = int(re.fullmatch(r"weights\.(\d+)\.weights\.h5",
                                         args.resume_weights).group(1))
        net = NeuralNetBase.load_model(os.path.join(args.out_directory, "model.json"))
        net.model.load_weights(os.path.join(args.out_directory, args.resume_weights))
        for layer in net.model.layers:
            layer.trainable = True
    else:
        policy = NeuralNetBase.load_model(args.model)
        policy.model.load_weights(args.weights)
        net = PolicyValueNet.from_policy(policy, value_channels=args.value_channels,
                                         value_hidden=args.value_hidden, freeze=False)
    model = Model(net.model.inputs, [net.model.outputs[0], net.model.get_layer(VALUE).output,
                                     net.model.get_layer(SCORE).output])

    train_shards = find_split_shards(args.shards_dir, "train")
    val_shards = find_split_shards(args.shards_dir, "val")
    _f, board_size, n_features, train_sizes = dataset_info(train_shards)
    _vf, _vb, _vn, val_sizes = dataset_info(val_shards)
    symmetries = args.symmetries.split(",")
    k = args.top
    sib_train = SiblingReader([os.path.join(d, "train")
                               for d in [args.siblings_dir] + args.extra_siblings], k)
    sib_val = SiblingReader(os.path.join(args.siblings_dir, "val"), k)
    n_sib = args.sibling_groups * k

    model.compile(
        optimizer=Adam(args.learning_rate),
        loss=["categorical_crossentropy", keras.losses.BinaryCrossentropy(),
              make_score_loss(n_sib, k, args.rank_weight)],
        loss_weights=[1.0, args.value_weight, args.score_weight],
        weighted_metrics=[["accuracy"], [], []],
        jit_compile=True)
    decode_in_steps(model, board_size, n_features, symmetries)

    plateau_state, warmup_steps = None, args.warmup_steps
    if args.resume_weights and os.path.exists(optimizer_path(args.out_directory,
                                                             initial_epoch)):
        state = restore_optimizer(model, optimizer_path(args.out_directory, initial_epoch))
        plateau_state, warmup_steps = state["plateau"], 0
        print("optimizer state restored: learning rate {:g}, plateau {}".format(
            float(ops.convert_to_numpy(model.optimizer.learning_rate)), plateau_state),
            flush=True)

    order = None
    if args.oversample:
        with io.open(os.path.join(args.shards_dir, "train", "games.tsv"), encoding="utf-8") as f:
            handicap = {r["path"] for r in csv.DictReader(f, delimiter="\t")
                        if r["gtype"] == "handicap"}
        strata, tiers = group_strata(sib_train, handicap)
        weights = [float(w) for w in args.oversample.split(",")]
        if len(weights) != len(GAP_BOUNDS) + 1:
            raise ValueError("--oversample takes {} weights".format(len(GAP_BOUNDS) + 1))
        order = OversampledOrder(strata, tiers, weights, args.seed)
        sampled, natural = order.shares()
        print("oversampling: gap tier shares {} -> {} per pass".format(
            np.round(natural, 3).tolist(), np.round(sampled, 3).tolist()), flush=True)

    done_steps = initial_epoch * args.steps_per_epoch  # a resume continues the streams
    train = make_batches(sib_train, ValueReader(train_shards, train_sizes), args.sibling_groups,
                         args.policy_rows, k, symmetries, board_size, args.seed,
                         sib_start=done_steps * n_sib, pol_start=done_steps * args.policy_rows,
                         order=order)
    # validation: fixed batches of the same layout (held-out siblings + val shard records),
    # past the groups SiblingEval ranks, so the two don't overlap
    val_gen = make_batches(sib_val, ValueReader(val_shards, val_sizes), args.sibling_groups,
                           args.policy_rows, k, symmetries, board_size,
                           None if args.seed is None else args.seed + 1,
                           sib_start=args.validation_groups * k)
    val_batches = [next(val_gen) for _ in range(args.validation_batches)]

    cache = load_lookahead_cache(args.lookahead_cache) if args.lookahead_cache else None
    if cache is not None:
        # the deployed rule (top 5, switch only if > 1 point better)
        evaluate = lambda: lookahead_metrics(net, cache)["lookahead_score_top5_t1"]  # noqa: E731
    else:
        evaluate = lambda: sibling_metrics(  # noqa: E731
            net, sib_val, k, args.validation_groups)["val_sib_centered_mae"]
    dtype = model.inputs[0].dtype

    def refresh_batches():  # 128 rows (an eager call), past the stream the run trained on
        for (packed, choices, komi), _y, _w in make_batches(
                sib_train, ValueReader(train_shards, train_sizes), 8, 64, k, symmetries,
                board_size, args.seed, sib_start=(done_steps + 1) * n_sib,
                pol_start=(done_steps + 1) * args.policy_rows):
            yield decode_on_device(packed, choices, board_size, n_features, symmetries,
                                   dtype), ops.convert_to_tensor(komi)

    saver = SaveEpoch(net, args.out_directory, args)
    if args.resume_weights:
        with open(saver.meta_path) as f:
            previous = json.load(f)
        saver.meta["epochs"] = previous["epochs"][:initial_epoch]
        saver.meta["resumed"] = previous.get("resumed", []) + [
            {"from": args.resume_weights, "args": vars(args)}]
    print("training {} weights; {} train / {} val sibling groups".format(
        sum(int(np.prod(w.shape)) for w in model.trainable_weights), sib_train.total // k,
        sib_val.total // k), flush=True)
    if order is not None:
        saver.meta["oversampling"] = {"weights": args.oversample,
                                      "natural_shares": natural.tolist(),
                                      "pass_shares": sampled.tolist()}
    plateau = PlateauWithState(restore=plateau_state, monitor="val_sib_centered_mae",
                               mode="min", factor=0.5, patience=args.plateau_patience,
                               min_delta=args.plateau_min_delta, verbose=1)
    callbacks = [LinearWarmup(args.learning_rate, warmup_steps),
                 LrOverride(args.out_directory, plateau), TerminateOnNaN(),
                 SiblingEval(net, sib_val, k, args.validation_groups)]
    if cache is not None:
        callbacks.append(LookaheadEval(net, cache, args.lookahead_every))
    callbacks += [saver, plateau, SaveOptimizer(args.out_directory, plateau,
                                                args.keep_optimizer),
                  BnRefreshAtEnd(net, refresh_batches(), args.bn_refresh_steps, evaluate,
                                 saver)]
    model.fit(
        train, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
        initial_epoch=initial_epoch,
        validation_data=_cycle(val_batches), validation_steps=args.validation_batches,
        callbacks=callbacks)


if __name__ == "__main__":
    run()
