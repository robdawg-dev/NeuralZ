"""LR range test: a coarse, cheap localizer for a new architecture/batch-size combination's
viable learning-rate region.

    python -m AlphaGo.training.lr_range_test MODEL TRAIN_DATA OUT_DIRECTORY [options]

Ramps the LR through a gentle linear warmup (--range-warmup-start-lr -> --range-floor-lr
over --range-warmup-steps) and then an EXPONENTIAL sweep from --range-floor-lr up to
--range-ceiling-lr over the rest of the run (--epochs x --steps-per-epoch steps), logging
step/lr/loss/weight_norm/grad_norm/loss_scale to <out_directory>/step_diagnostics.jsonl
every --range-check-every steps. A NaN loss stops the run.

This is a coarse localizer only, NOT a verdict on a safe LR: this mode only ramps, it never
holds, and ramping tolerance is not the same as genuine stability at a held LR - treat its
output as candidates for follow-up held-constant training runs, not an answer.

The run itself - model, data stream, validation set, precision, compilation, checkpoints,
metadata - is built by the trainer's own pipeline functions (supervised_policy_trainer.py),
so a sweep sees exactly what training will. Only the learning rate handling is this
module's own.
"""
import os
import json
import types

import keras
import tensorflow as tf
from keras.callbacks import Callback
from keras.optimizers import SGD

from AlphaGo.training.supervised_policy_trainer import (
    add_run_arguments, compile_model, fit_run, load_checkpoint, set_up_run)


class RangeTestLRCallback(Callback):
    """Gentle linear warmup up to a conservative floor, then an EXPONENTIAL sweep from that
    floor to a ceiling over the rest of the run.

    The warmup phase exists because the model's raw gradient magnitude at
    initialization is very large on real data (observed ~100K at step 0 in an earlier
    ramp test on this project's data/architecture) - hitting a real candidate LR
    immediately, before that settles, conflates "unstable at this LR" with "unstable
    because weights are still fresh". Warming up to a floor low enough that it
    obviously isn't itself the target LR gives the optimizer a chance to leave that
    initial regime before the sweep starts probing.

    The sweep itself is exponential rather than linear so it gets even resolution per
    decade on a log-LR axis (the standard shape for this kind of test - Smith,
    "Cyclical Learning Rates for Training Neural Networks"). This is a coarse
    localizer only, not a verdict: a ramp's apparent tolerance for a given LR is not
    the same as genuine stability at that LR held constant (confirmed on this exact
    project - a prior linear ramp climbed to LR=0.8 with no visible break, while later
    held-constant tests failed well below that) - candidate LRs from this sweep must
    still be verified by a separate held-constant run before being trusted.
    """

    def __init__(self, warmup_steps, warmup_start_lr, floor_lr, ceiling_lr, total_steps):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.floor_lr = floor_lr
        self.ceiling_lr = ceiling_lr
        self.sweep_steps = max(1, total_steps - warmup_steps)
        self._step = 0

    def on_train_batch_begin(self, batch, logs=None):
        if self._step <= self.warmup_steps:
            frac = self._step / max(1, self.warmup_steps)
            lr = self.warmup_start_lr + (self.floor_lr - self.warmup_start_lr) * frac
        else:
            frac = min(1.0, (self._step - self.warmup_steps) / self.sweep_steps)
            lr = self.floor_lr * (self.ceiling_lr / self.floor_lr) ** frac
        self.model.optimizer.learning_rate = lr
        self._step += 1


class RangeTestDiagnosticsCallback(Callback):
    """Step-granularity visibility into the sweep.

    Logs step/lr/loss/loss_epoch_mean/weight_norm/grad_norm/loss_scale to
    <out_directory>/step_diagnostics.jsonl every check_every steps. "loss" is that step's
    own loss; "loss_epoch_mean" is Keras's running mean of the loss since the epoch began
    (what logs["loss"] holds, and what earlier versions of this file recorded as "loss" -
    it resets every epoch and averages over a wide stretch of the sweep's LRs, so it is
    not a loss-vs-LR measurement). Deliberately has NO automatic
    stop-on-weight_norm-ratio (an earlier version of this diagnostic, in
    lr_testing_trainer.py, stopped training once weight_norm exceeded a fixed
    multiple of its starting value) - that heuristic produced misleading verdicts in
    this project's past LR investigations (weight_norm grew smoothly through LRs that
    later turned out to be well past the real stability boundary, and looked alarming
    at LRs that turned out fine). This callback only records; judging the sweep is a
    manual/offline read of the full curve afterward, primarily via the loss trend, not
    a live threshold. TerminateOnNaN (fit_run adds it to every run) is the only
    automatic stop, as a backstop against a genuine runaway wasting the rest of the
    sweep's step budget.

    The step's loss, grad_norm and loss_scale are read from logs["batch_loss"]/
    logs["grad_norm"]/logs["loss_scale"], only present when the model's train_step has
    been monkey-patched (see _grad_norm_and_loss_scale_train_step) - the logs.get(...)
    guards keep this callback harmless if that patch isn't present.
    """

    def __init__(self, check_every, out_path):
        super().__init__()
        self.check_every = check_every
        self._step = 0
        self._f = open(out_path, "w")

    def on_train_batch_end(self, batch, logs=None):
        if self._step % self.check_every == 0:
            weight_norm = float(tf.linalg.global_norm(self.model.trainable_variables))
            lr = float(self.model.optimizer.learning_rate)
            logs = logs or {}

            def logged(key):
                return float(logs[key]) if logs.get(key) is not None else None
            record = {"step": self._step, "lr": lr, "loss": logged("batch_loss"),
                      "loss_epoch_mean": logged("loss"), "weight_norm": weight_norm,
                      "grad_norm": logged("grad_norm"), "loss_scale": logged("loss_scale")}
            self._f.write(json.dumps(record) + "\n")
            self._f.flush()
        self._step += 1

    def on_train_end(self, logs=None):
        self._f.close()


def _grad_norm_and_loss_scale_train_step(self, data):
    """Replaces model.train_step so the step's own loss, grad_norm and (under mixed
    precision) the optimizer's current dynamic loss scale reach
    RangeTestDiagnosticsCallback via logs["batch_loss"]/logs["grad_norm"]/
    logs["loss_scale"] every step - model.fit() doesn't expose any of them to callbacks
    otherwise (its logs["loss"] is the running mean since the epoch began).

    Mirrors Model.train_step (keras/src/models/model.py): under mixed_float16, gradients
    must be computed w.r.t. the SCALED loss (keeps small gradient values representable
    through the fp16 backward pass; apply_gradients then unscales internally before
    actually updating weights) - computing them against the raw loss instead would starve
    the effective step size, since apply_gradients always assumes its input was
    pre-scaled.

    loss_scale: under --mixed-precision, model.compile() wraps the optimizer in
    keras.src.optimizers.loss_scale_optimizer.LossScaleOptimizer, whose current dynamic
    scale isn't a plain attribute - it's one of the optimizer's own tracked variables
    (named "dynamic_scale", created lazily on the optimizer's first apply_gradients
    call - i.e. by the line below, on this very first traced step). It halves
    automatically whenever an inf/nan gradient triggers a skipped step, then grows back
    over time - logging it directly lets an occasional Infinity grad_norm reading during
    a sweep be told apart from genuine instability (a self-correcting mixed-precision
    artifact vs a real blowup) instead of guessed at. Looked up AFTER apply_gradients
    (not before) so the lookup - itself plain Python running once at trace time, not
    per-call - finds the variable already built; under plain float32 (no mixed
    precision) the optimizer has no such variable and loss_scale is left out of results
    entirely.
    """
    # (x, y) for a policy network; (x, y, sample_weight) for a joint one, whose value and
    # score weigh 0 where KataGo left no annotation
    x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        scaled_loss = self.optimizer.scale_loss(loss) if self.optimizer is not None else loss
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(scaled_loss, trainable_vars)
    grad_norm = tf.linalg.global_norm(gradients)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # As Model.train_step: the loss tracker by hand, everything compiled through
    # compute_metrics (a joint network's per-output loss trackers were already updated by
    # compute_loss).
    for metric in self.metrics:
        if metric.name == "loss":
            metric.update_state(loss)
    results = dict(self.compute_metrics(x, y, y_pred, sample_weight=sample_weight))
    results["batch_loss"] = loss
    results["grad_norm"] = grad_norm
    for v in self.optimizer.variables:
        if v.name == "dynamic_scale":
            results["loss_scale"] = tf.convert_to_tensor(v)
            break
    return results


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_run_arguments(parser)
    parser.add_argument("--weights", default=None,
                        help="Name of a .h5 weights file (in the output directory) to "
                             "warm-start a NEW sweep from - e.g. to continue an earlier "
                             "sweep's LR curve from where it left off without re-running "
                             "the cheap-to-skip low end of the range. Only the weights are "
                             "loaded: the optimizer (SGD momentum) starts fresh, so the "
                             "sweep has a brief (~10-50 step, matching momentum's "
                             "1/(1-momentum) memory) transient while momentum rebuilds.")
    parser.add_argument("--range-warmup-steps", type=int, default=None,
                        help="Steps to linearly ramp from --range-warmup-start-lr to "
                             "--range-floor-lr before the exponential sweep begins. "
                             "Default: one epoch's worth of steps (--steps-per-epoch) - a "
                             "minimum gentle warmup so the sweep doesn't start while the "
                             "model's initial (very large) raw gradient magnitude is still "
                             "settling.")
    parser.add_argument("--range-warmup-start-lr", type=float, default=1e-4,
                        help="LR at step 0. Default: .0001")
    parser.add_argument("--range-floor-lr", type=float, default=1e-3,
                        help="LR reached at the end of warmup, and the starting point of the "
                             "exponential sweep. Should be low enough that it obviously "
                             "isn't itself the target LR. Default: .001")
    parser.add_argument("--range-ceiling-lr", type=float, default=2.0,
                        help="LR reached at the end of the run (the last step of the last "
                             "epoch). The exponential sweep moves from --range-floor-lr to "
                             "this value over the steps remaining after warmup. Default: 2.0")
    parser.add_argument("--range-check-every", type=int, default=50,
                        help="Log step/lr/loss/weight_norm/grad_norm/loss_scale to "
                             "<out_directory>/step_diagnostics.jsonl every this many steps. "
                             "Default: 50")
    return parser


# Settings a --weights warm start must keep (see set_up_run): minibatch and
# steps_per_epoch place the data stream's start position and number the epochs. Nothing
# about the LR carries over - RangeTestLRCallback's step counter always starts fresh at 0,
# so a warm start is a NEW, independent sweep rather than the SAME sweep continued
# mid-step-count (which would need offset-aware accounting it doesn't have).
RANGE_TEST_RESUME_SETTINGS = ("minibatch", "steps_per_epoch")


def run_range_test(cmd_line_args=None):
    args = build_parser().parse_args(cmd_line_args)
    run = set_up_run(args, RANGE_TEST_RESUME_SETTINGS, require_latest_checkpoint=False)

    # No LearningRateSchedule object - RangeTestLRCallback owns the optimizer's
    # learning_rate directly for the whole run, the same way the trainer's WarmupCallback
    # does for --lr-schedule plateau (see there for why a callback, not a schedule, is
    # needed for a plain mutable LR).
    optimizer = SGD(learning_rate=args.range_warmup_start_lr, momentum=args.momentum,
                    nesterov=True)
    range_warmup_steps = (args.range_warmup_steps if args.range_warmup_steps is not None
                          else run.steps_per_epoch)
    range_lr_cb = RangeTestLRCallback(
        range_warmup_steps, args.range_warmup_start_lr, args.range_floor_lr,
        args.range_ceiling_lr, run.total_steps)
    if args.verbose:
        print("LR range test: warmup {} -> {} over {} steps, then exponential sweep "
              "{} -> {} over the remaining {} steps".format(
                  args.range_warmup_start_lr, args.range_floor_lr, range_warmup_steps,
                  args.range_floor_lr, args.range_ceiling_lr,
                  max(1, run.total_steps - range_warmup_steps)))

    compile_model(run.model, optimizer, args)
    if run.resume:
        # Weights only: a warm-started sweep starts with a fresh optimizer (see --weights).
        load_checkpoint(run.model, run.weights_path, with_optimizer=False)
        if args.verbose:
            print("loaded weights from {}".format(run.weights_path))

    # Monkey-patch AFTER compile() (so the patched step still picks up jit_compile=True
    # via make_train_function()) and BEFORE fit() - see
    # _grad_norm_and_loss_scale_train_step's docstring for why this is a monkey-patch
    # rather than a model_class kwarg.
    run.model.train_step = types.MethodType(_grad_norm_and_loss_scale_train_step, run.model)
    range_diagnostics = RangeTestDiagnosticsCallback(
        args.range_check_every, os.path.join(args.out_directory, "step_diagnostics.jsonl"))

    fit_run(run, args, lr_schedule=None, lr_callbacks=[range_lr_cb],
            epoch_callbacks=[range_diagnostics])


if __name__ == "__main__":
    run_range_test()
