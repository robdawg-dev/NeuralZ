"""Debug/instrumented variant of supervised_policy_trainer_v2.py, kept as a separate file
rather than folded permanently into the production trainer. Carries the diagnostic
tooling built to investigate the intermittent resnet validation-collapse bug
(CollapseDiagnosticCallback, SyncBeforeValidationCallback, --sync-before-validation,
--lr-schedule-epochs for kill-after-N-epoch reproduction runs) - reach for this file
again if a similar anomaly needs the same kind of live diagnostic capture, rather than
re-deriving it from scratch.
"""
import os
import json
import math
import time
import numpy as np
from AlphaGo.training.xla_workarounds import ensure_xla_conv_nhwc

# Must run before the first XLA compilation, i.e. before model.compile(jit_compile=True)
# below is ever exercised - module level, ahead of everything else, is the safest place.
ensure_xla_conv_nhwc()

from keras import mixed_precision, ops, utils as keras_utils  # noqa: E402
from keras.metrics import TopKCategoricalAccuracy  # noqa: E402
from keras.optimizers import SGD  # noqa: E402
from keras.optimizers.schedules import CosineDecay, LearningRateSchedule  # noqa: E402
from keras.callbacks import ModelCheckpoint, Callback  # noqa: E402
from AlphaGo.models.policy import CNNPolicy  # noqa: E402
# Unused directly, but importing it registers ResTowerPolicy (via the @neuralnet
# decorator) so CNNPolicy.load_model() can find it by name in a model.json's "class"
# field - registration only happens when a class's defining module is actually imported
# somewhere in the process, and nothing else here pulls this one in.
import AlphaGo.models.resnet_tower_policy  # noqa: E402,F401
from shuffle_buffer import (  # noqa: E402
    BOARD_TRANSFORMATIONS, find_shard_files, build_game_index, get_or_create_game_split,
    shuffle_buffer_batch_generator, build_validation_arrays)


def prediction_entropy(y_true, y_pred):
    """Shannon entropy of the predicted move distribution, averaged over the batch.

    Ignores y_true - a diagnostic on the model's own confidence, not its correctness.
    Starts near log(361) (uniform, untrained) and should fall as training progresses;
    comparing the train vs. val curves can also surface overconfidence before loss alone
    would show it.
    """
    p = ops.clip(y_pred, 1e-7, 1.0)
    return -ops.sum(p * ops.log(p), axis=-1)


def sanity_checked_generator(base_generator, out_directory, label, check_every=50):
    """Wraps a batch generator, periodically checking yielded batches for basic sanity
    (proper one-hot Y rows, valid X range, no NaN) and logging loudly the moment
    something looks wrong - catches data corruption at the exact step it starts, rather
    than only showing up in a loss/accuracy number at the next epoch boundary. This
    project has seen a training run stay stuck at the random-guessing baseline (loss ==
    entropy) from step 0 - epoch-boundary checks alone missed that entirely, since
    nothing ever *changed*, it just never looked right from the start. Checks every
    Nth step plus the first few unconditionally (to catch a from-the-start problem too)
    rather than every step, to keep this cheap.
    """
    log_path = os.path.join(out_directory, "batch_sanity_log.json")
    step = 0
    for X, Y in base_generator:
        step += 1
        if step % check_every == 0 or step <= 5:
            row_sums = Y.sum(axis=1)
            x_nan = int(np.isnan(X).sum())
            x_min, x_max = float(X.min()), float(X.max())
            y_min, y_max = float(row_sums.min()), float(row_sums.max())
            bad = x_nan > 0 or x_min < -1e-3 or x_max > 1 + 1e-3 or abs(y_min - 1.0) > 1e-3 or abs(y_max - 1.0) > 1e-3
            if bad:
                print("\n*** BAD BATCH detected: {} step {}: X in [{:.4f}, {:.4f}] "
                      "({} NaN), Y row-sums in [{:.4f}, {:.4f}] (should be exactly 1.0) "
                      "***".format(label, step, x_min, x_max, x_nan, y_min, y_max))
                entry = {"label": label, "step": step, "X_min": x_min, "X_max": x_max,
                        "X_nan_count": x_nan, "Y_row_sum_min": y_min, "Y_row_sum_max": y_max}
                existing = []
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        existing = json.load(f)
                existing.append(entry)
                with open(log_path, "w") as f:
                    json.dump(existing, f, indent=2)
        yield X, Y


class CollapseDiagnosticCallback(Callback):
    """When val_loss jumps well past what a healthy model should ever produce, capture
    a full diagnostic snapshot instead of letting the number silently ride by - aimed at
    an intermittent validation collapse this project has seen (val_loss ~3 -> ~11-12,
    reproduced with the exact same command on some attempts but not others). Does NOT
    stop training - this is for data collection during an open investigation, not a
    production safety net, so it must not otherwise change training behavior/timing.
    """

    def __init__(self, X_val, Y_val, minibatch, board_size, out_directory):
        super().__init__()
        # The exact same fixed, finite validation arrays passed to model.fit() (via the
        # tf.data.Dataset built from build_validation_arrays) - not a freshly-streamed
        # generator. There's no more "live vs fresh generator" distinction to test now
        # that validation isn't backed by a long-lived stateful generator at all: every
        # check below draws from this same fixed pool Keras's own validation pass reads.
        self.X_val = X_val
        self.Y_val = Y_val
        self.minibatch = minibatch
        self.log_path = os.path.join(out_directory, "collapse_diagnostics.json")
        self.best_val_loss = None
        # A healthy model should never do meaningfully worse than random guessing
        # (ln(board*board)) once training has progressed past the first few steps.
        self.sanity_ceiling = 1.5 * math.log(board_size * board_size)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return

        is_collapse = val_loss > self.sanity_ceiling
        if self.best_val_loss is not None and val_loss > 2 * self.best_val_loss:
            is_collapse = True
        prior_best = self.best_val_loss
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        if not is_collapse:
            return

        print("\n*** COLLAPSE DETECTED at epoch {}: val_loss={:.4f} (best so far: {}, "
              "ceiling: {:.4f}) ***".format(epoch, val_loss, prior_best, self.sanity_ceiling))

        snapshot = {"epoch": epoch, "val_loss": val_loss, "best_val_loss_before": prior_best}

        # 1. Are the LIVE model's weights actually NaN/Inf, or just the reported metric?
        weights = self.model.get_weights()
        snapshot["weights_nan_count"] = sum(int(np.isnan(w).sum()) for w in weights)
        snapshot["weights_inf_count"] = sum(int(np.isinf(w).sum()) for w in weights)
        snapshot["weights_total_params"] = sum(int(w.size) for w in weights)
        print("    weights: {} NaN, {} Inf, out of {} total params".format(
            snapshot["weights_nan_count"], snapshot["weights_inf_count"],
            snapshot["weights_total_params"]))

        # 2. Pull several random batches straight from the fixed validation arrays (not
        # just one - a single batch could happen to look fine by chance) and, for each,
        # both inspect the raw data AND run the LIVE model forward on it right now,
        # in-process - a direct eager call (self.model(...), not model.predict()),
        # which bypasses Keras's jit_compile=True compiled test_step entirely. This is
        # the direct test of "is the live model's own predictive behavior already bad
        # at this moment" versus "does even a plain uncompiled forward pass look
        # completely normal" - the latter would mean the collapse is somehow an
        # artifact of Keras's own compiled validation computation specifically (a
        # stale/incorrect compiled graph, a genuine XLA bug, something not part of the
        # model's actual weights/predictions at all), not a real predictive failure -
        # which would explain why a saved checkpoint from right after a collapse always
        # replays clean in a fresh process (fresh XLA compilation from scratch) even
        # though the *live* run kept reporting bad val_loss for every subsequent epoch
        # (nothing in that process ever recompiles/resets a poisoned graph, if that's
        # what this turns out to be).
        #
        # Now that validation is a fixed finite array (not a stateful streaming
        # generator), there's no "fresh vs. live generator" distinction left to probe -
        # every batch here is drawn from the exact same pool model.fit()'s own
        # validation pass reads, just at randomly sampled indices rather than the
        # dataset's own iteration order.
        last_X, last_Y = None, None
        try:
            n_val = self.X_val.shape[0]
            batch_n = min(self.minibatch, n_val)
            rand_rng = np.random.default_rng()
            eps = 1e-7
            batch_checks = []
            for _ in range(5):
                idx = rand_rng.choice(n_val, size=batch_n, replace=False)
                X, Y = self.X_val[idx], self.Y_val[idx]
                last_X, last_Y = X, Y
                row_sums = Y.sum(axis=1)
                preds = np.asarray(self.model(X, training=False))
                per_example_loss = -np.sum(Y * np.log(np.clip(preds, eps, 1.0)), axis=1)
                batch_checks.append({
                    "X_min": float(X.min()), "X_max": float(X.max()),
                    "X_nan_count": int(np.isnan(X).sum()),
                    "Y_row_sum_min": float(row_sums.min()), "Y_row_sum_max": float(row_sums.max()),
                    "eager_loss": float(per_example_loss.mean()),
                    "eager_accuracy": float((preds.argmax(axis=1) == Y.argmax(axis=1)).mean()),
                    "pred_min": float(preds.min()), "pred_max": float(preds.max()),
                    "pred_nan_count": int(np.isnan(preds).sum()),
                })
            snapshot["val_batch_checks"] = batch_checks
            print("    5 random val batches, live eager forward pass on each: losses={} "
                  "(reported val_loss this epoch was {:.4f})".format(
                      ["{:.3f}".format(b["eager_loss"]) for b in batch_checks], val_loss))
        except Exception as e:
            snapshot["val_batch_check_error"] = str(e)
            print("    val batch check failed:", e)

        # 2b. Direct comparison, on the exact same final batch: eager (above) vs.
        # Keras's own compiled evaluate() path (the same test_step machinery its
        # internal fit()-validation uses). If eager looks fine but compiled also comes
        # back bad on this identical data, that's strong evidence the problem lives in
        # the compiled graph/test_step specifically, not the model's real predictions.
        if last_X is not None:
            try:
                compiled_results = self.model.evaluate(x=last_X, y=last_Y, verbose=0)
                names = self.model.metrics_names
                snapshot["compiled_evaluate"] = dict(zip(names, [float(v) for v in compiled_results]))
                print("    compiled model.evaluate() on that same final batch: {} "
                      "(eager loss on it was {:.4f})".format(
                          snapshot["compiled_evaluate"], batch_checks[-1]["eager_loss"]))
            except Exception as e:
                snapshot["compiled_evaluate_error"] = str(e)
                print("    compiled evaluate check failed:", e)

        # 2c. BatchNormalization running-stats summary for every BN layer - not just a
        # NaN/Inf check (already covered by the whole-model weights check above), but
        # whether these specifically look plausible. A moving_mean/moving_variance that
        # drifted to an extreme-but-finite value wouldn't show up as NaN/Inf yet could
        # still produce exactly this kind of validation-only (eval-mode-only) failure -
        # BatchNorm is the one thing in this model whose state updates via a moving
        # average during training rather than the optimizer's gradient step, and it's
        # used in a completely different way (accumulated running stats vs. per-batch
        # stats) between training mode and eval mode.
        try:
            bn_stats = []
            for layer in self.model.layers:
                if type(layer).__name__ != "BatchNormalization":
                    continue
                entry = {"layer": layer.name}
                for attr in ("moving_mean", "moving_variance", "gamma", "beta"):
                    if hasattr(layer, attr):
                        w = np.asarray(getattr(layer, attr))
                        entry[attr] = {
                            "min": float(w.min()), "max": float(w.max()),
                            "mean": float(w.mean()), "std": float(w.std()),
                            "nan": int(np.isnan(w).sum()), "inf": int(np.isinf(w).sum()),
                        }
                bn_stats.append(entry)
            snapshot["batchnorm_stats"] = bn_stats
            extreme = [b["layer"] for b in bn_stats
                      if b.get("moving_variance", {}).get("max", 0) > 1e4
                      or abs(b.get("moving_mean", {}).get("max", 0)) > 1e4]
            print("    {} BatchNorm layers checked{}".format(
                len(bn_stats), " - extreme values in: " + ", ".join(extreme) if extreme else ""))
        except Exception as e:
            snapshot["batchnorm_stats_error"] = str(e)
            print("    batchnorm stats check failed:", e)

        # 3. Memory snapshots at the exact moment of collapse - host/WSL2 AND GPU (the
        # original version of this diagnostic only captured host memory).
        try:
            with open("/proc/meminfo") as f:
                lines = f.read().splitlines()[:6]
            snapshot["meminfo"] = dict(
                (p[0].strip(), p[1].strip()) for p in (ln.split(":", 1) for ln in lines) if len(p) == 2)
            print("    meminfo:", snapshot["meminfo"])
        except Exception as e:
            snapshot["meminfo_error"] = str(e)

        try:
            import tensorflow as tf
            gpu_mem = tf.config.experimental.get_memory_info("GPU:0")
            snapshot["gpu_memory"] = {k: int(v) for k, v in gpu_mem.items()}
            print("    GPU memory:", snapshot["gpu_memory"])
        except Exception as e:
            snapshot["gpu_memory_error"] = str(e)

        existing = []
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                existing = json.load(f)
        existing.append(snapshot)
        with open(self.log_path, "w") as f:
            json.dump(existing, f, indent=2)


class SyncBeforeValidationCallback(Callback):
    """Forces all pending asynchronously-dispatched GPU ops to complete before each
    epoch's validation pass starts reading the model's state.

    Investigating the intermittent resnet validation-only collapse (val_loss jumping
    from ~3 to ~11-12 out of nowhere while training metrics stay fine, non-
    deterministic - same command both failing and succeeding): a saved checkpoint's
    weights (including BatchNormalization's running mean/variance, itself updated via
    a moving average during training - a different mechanism from the gradient-based
    weight updates) always evaluate cleanly on replay, even replaying the exact same
    seeded validation data that originally produced the collapse. That rules out bad
    data and rules out the saved weights themselves being corrupted, and points at
    something specific to the live run's timing - a plausible mechanism being that TF
    dispatches ops asynchronously, and if the last training step's BatchNorm
    moving-average update hadn't actually finished landing on the GPU before
    validation's forward pass started reading it (nothing in Keras's fit() loop
    otherwise guarantees that ordering), validation could transiently read
    inconsistent BatchNorm statistics - while the checkpoint, saved *after* validation
    completes for that epoch, would already reflect the fully-settled correct state,
    exactly matching what's been observed. This callback is the direct test of that:
    if forcing a sync here measurably changes the collapse rate, that's real evidence
    for (or against) this specific mechanism.
    """

    def on_test_begin(self, logs=None):
        import tensorflow as tf
        tf.test.experimental.sync_devices()


class _ResumedLRSchedule(LearningRateSchedule):
    """Wraps a base schedule, shifting its step input by a fixed offset.

    Resuming only reloads model weights (save_weights_only=True, no optimizer state), so a
    resumed run's freshly-created SGD optimizer always starts its own step counter at 0.
    Without this, a resumed run would re-enter warmup and restart the cosine decay curve
    from its beginning every time instead of continuing the single, whole-run schedule
    total_steps was originally shaped for.
    """

    def __init__(self, base_schedule, offset):
        super().__init__()
        self.base_schedule = base_schedule
        self.offset = offset

    def __call__(self, step):
        return self.base_schedule(step + self.offset)

    def get_config(self):
        return {"base_schedule": self.base_schedule.get_config(), "offset": self.offset}


class TrainingDiagnosticsCallback(Callback):
    """Adds a few per-epoch numbers to logs that Keras doesn't track on its own, so they
    end up in metadata.json (via MetadataWriterCallback, which must run AFTER this one in
    the callbacks list - Keras passes the same logs dict through every callback for a
    given event, in list order):
    - learning_rate: the actual LR used this epoch, read from the schedule directly rather
      than guessed from the args - useful to plot against loss given the warmup+cosine
      schedule.
    - epoch_seconds / steps_per_second: wall-clock throughput, so a long run degrading
      partway through (as happened once already this session, on spinning-disk hardware)
      shows up in the data instead of only being caught by watching it live.
    """

    def __init__(self, lr_schedule, steps_per_epoch):
        super().__init__()
        self.lr_schedule = lr_schedule
        self.steps_per_epoch = steps_per_epoch
        self._epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        logs["learning_rate"] = float(self.lr_schedule(self.model.optimizer.iterations))
        elapsed = time.time() - self._epoch_start
        logs["epoch_seconds"] = elapsed
        logs["steps_per_second"] = self.steps_per_epoch / elapsed if elapsed > 0 else 0.0


class _LastBatchHolder:
    """Shared mutable holder a training-data generator wrapper writes into on every
    yield, so a callback (which has no direct access to the generator) can read
    whatever batch was most recently fed to the model."""

    def __init__(self):
        self.X = None
        self.Y = None


def _capturing_generator(base_generator, holder):
    """Passes batches through unchanged, side-effect-storing each one into `holder`
    first. Keras pulls the next batch (running this generator's __next__) before
    calling a callback's on_train_batch_begin for that step, so by the time any
    callback runs, `holder` already holds the exact batch about to be (or just was)
    trained on - the only way to recover a batch's contents after the fact, since
    Keras's callback API exposes computed metrics (`logs`), not raw batch data.
    """
    for X, Y in base_generator:
        holder.X, holder.Y = X, Y
        yield X, Y


class GradientExplosionDiagnosticCallback(Callback):
    """Investigating a gradient-explosion NaN seen training plain CNN (simplecnn,
    no BatchNorm) on kgs-ugo-highdan: loss looked normal batch-to-batch, then one
    batch's own loss spiked far above baseline (derived from the jump in Keras's
    running-average loss metric - the actual per-batch value, not an average, since
    that's what on_train_batch_end's `logs` reports). The input data at explosion
    batches has been independently verified clean (no NaN/Inf, X in [0,1], Y
    properly one-hot) - so whatever's happening is in the model's own
    forward/backward computation, not corrupted input. Two explosions caught so far
    at different steps (320 in one run, 829 in another, same seed) confirm this
    isn't tied to one specific unlucky batch - it's a recurring instability, so this
    version tracks every occurrence across a whole epoch rather than just the first.

    Deliberately lightweight: no per-batch weight snapshotting (that cost a real,
    measured slowdown - GPU util dropped from ~90% to ~70%, RAM climbed, from
    get_weights() forcing a host sync every single step) and no set_weights()+
    GradientTape replay (that combination produced a highly suspicious
    exactly-zero gradient on every layer in an earlier version of this callback,
    almost certainly a jit_compile=True graph-caching artifact rather than a real
    finding - not trusted, not repeated here). Each detected spike gets only a
    cheap forward pass (no gradient computation) against whatever the model's
    CURRENT weights happen to be at that moment - not the exact pre-explosion
    state, but enough to see which example in the batch the model was most
    confidently wrong about and what its raw predictions looked like.

    Also samples a real per-layer gradient norm (forward+backward via GradientTape,
    current weights, no set_weights() involved so no risk of the replay bug above)
    every `sample_every` batches, building an actual measured "healthy" baseline
    instead of guessing a clip threshold - this is what answers "how big is a normal
    gradient here" concretely, from this exact model/data/step-in-training, rather
    than a number picked out of the air (which is exactly what went wrong with the
    first clipnorm=1.0 attempt: it throttled ordinary gradients, not just outliers).

    Once NaN is confirmed for a few consecutive steps, stops logging/printing
    further - after that point every subsequent step is identical (permanently
    poisoned weights), so continuing to record adds nothing but log/disk spam (a
    prior version of this callback learned this the hard way: an unbounded per-step
    write after the model went NaN flooded stdout for the rest of the epoch).
    """

    def __init__(self, batch_holder, out_directory, anomaly_threshold=50.0, sample_every=10,
                max_nan_events=3):
        super().__init__()
        self.batch_holder = batch_holder
        self.log_path = os.path.join(out_directory, "gradient_explosion_diagnostics.json")
        self.norm_log_path = os.path.join(out_directory, "gradient_norm_samples.json")
        self.anomaly_threshold = anomaly_threshold
        self.sample_every = sample_every
        self.max_nan_events = max_nan_events
        self._step = 0
        self._prev_avg_loss = None
        self._events = []
        self._norm_samples = []
        self._nan_event_count = 0
        self._stopped_logging = False

    def _gradient_norm(self, X, Y):
        import tensorflow as tf
        eps = 1e-7
        with tf.GradientTape() as tape:
            preds = self.model(X, training=True)
            preds_clipped = tf.clip_by_value(preds, eps, 1.0)
            loss_value = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(preds_clipped), axis=1))
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        norms = [float(tf.norm(g).numpy()) for g in gradients if g is not None]
        global_norm = sum(n * n for n in norms) ** 0.5
        return global_norm, norms, float(loss_value.numpy())

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self._step += 1
        cur_avg = logs.get("loss")
        if cur_avg is None:
            return

        prev_avg = self._prev_avg_loss
        is_nan = cur_avg != cur_avg
        if not is_nan:
            this_batch_loss = (self._step * cur_avg - (self._step - 1) * prev_avg
                               if prev_avg is not None else cur_avg)
            self._prev_avg_loss = cur_avg
        else:
            this_batch_loss = float("nan")

        is_anomaly = is_nan or this_batch_loss > self.anomaly_threshold

        # Healthy-gradient-norm sampling - only while nothing's on fire, so the
        # baseline reflects ordinary training, not the aftermath of an explosion.
        if not is_anomaly and not self._stopped_logging and self._step % self.sample_every == 0:
            X, Y = self.batch_holder.X, self.batch_holder.Y
            if X is not None:
                try:
                    global_norm, layer_norms, loss_now = self._gradient_norm(X, Y)
                    self._norm_samples.append({"step": self._step, "global_norm": global_norm,
                                              "max_layer_norm": max(layer_norms) if layer_norms else None,
                                              "loss": loss_now})
                    if self._step % (self.sample_every * 10) == 0:
                        with open(self.norm_log_path, "w") as f:
                            json.dump(self._norm_samples, f, indent=2)
                except Exception as e:
                    print("    gradient norm sampling failed at step {}: {}".format(self._step, e))

        if not is_anomaly or self._stopped_logging:
            return

        if is_nan:
            self._nan_event_count += 1
            if self._nan_event_count > self.max_nan_events:
                if self._nan_event_count == self.max_nan_events + 1:
                    print("\n*** NaN confirmed persistent for {} consecutive steps - "
                          "no further diagnostic logging this run ***".format(self.max_nan_events))
                    with open(self.norm_log_path, "w") as f:
                        json.dump(self._norm_samples, f, indent=2)
                    with open(self.log_path, "w") as f:
                        json.dump(self._events, f, indent=2)
                self._stopped_logging = True
                return

        print("\n*** LOSS SPIKE at step {}: this batch's own loss ~{:.2f} (running avg "
              "{} -> {}) ***".format(
                  self._step, this_batch_loss,
                  "{:.4f}".format(prev_avg) if prev_avg is not None else "n/a",
                  "nan" if is_nan else "{:.4f}".format(cur_avg)))

        event = {"step": self._step, "this_batch_loss": this_batch_loss,
                 "running_avg_loss_after": cur_avg}

        X, Y = self.batch_holder.X, self.batch_holder.Y
        if X is not None:
            try:
                preds = np.asarray(self.model(X, training=True))
                eps = 1e-7
                per_example_loss = -np.sum(Y * np.log(np.clip(preds, eps, 1.0)), axis=1)
                worst_idx = int(np.argmax(per_example_loss))
                event["current_weights_nan_count"] = sum(
                    int(np.isnan(w).sum()) for w in self.model.get_weights())
                event["preds_min"] = float(preds.min())
                event["preds_max"] = float(preds.max())
                event["preds_nan_count"] = int(np.isnan(preds).sum())
                event["forward_pass_loss_now"] = float(per_example_loss.mean())
                event["worst_example_index_in_batch"] = worst_idx
                event["worst_example_loss"] = float(per_example_loss[worst_idx])
                event["worst_example_X_min_max"] = [float(X[worst_idx].min()), float(X[worst_idx].max())]
                event["worst_example_Y_argmax"] = int(Y[worst_idx].argmax())
                event["worst_example_pred_at_true_class"] = float(preds[worst_idx, Y[worst_idx].argmax()])
                print("    forward pass now: preds in [{:.4g},{:.4g}], worst example idx={} "
                      "loss={:.4f} (true-class pred={:.2e})".format(
                          event["preds_min"], event["preds_max"], worst_idx,
                          event["worst_example_loss"], event["worst_example_pred_at_true_class"]))
            except Exception as e:
                event["forward_pass_error"] = str(e)
                print("    forward pass check failed:", e)

        self._events.append(event)
        with open(self.log_path, "w") as f:
            json.dump(self._events, f, indent=2)

    def on_epoch_end(self, epoch, logs=None):
        with open(self.norm_log_path, "w") as f:
            json.dump(self._norm_samples, f, indent=2)
        with open(self.log_path, "w") as f:
            json.dump(self._events, f, indent=2)


class MetadataWriterCallback(Callback):

    def __init__(self, path):
        self.file = path
        self.metadata = {
            "epochs": [],
            "best_epoch": 0
        }

    def on_epoch_end(self, epoch, logs={}):
        # in case appending to logs (resuming training), get epoch number ourselves
        epoch = len(self.metadata["epochs"])

        self.metadata["epochs"].append(logs)

        if "val_loss" in logs:
            key = "val_loss"
        else:
            key = "loss"

        best_loss = self.metadata["epochs"][self.metadata["best_epoch"]][key]
        if logs.get(key) < best_loss:
            self.metadata["best_epoch"] = epoch

        with open(self.file, "w") as f:
            json.dump(self.metadata, f, indent=2)


def run_training_v2(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list

    Tuned large-batch recipe, kept as a separate file from supervised_policy_trainer.py
    (which is left untouched) rather than changing it in place. Differences from that
    file:
    - momentum+Nesterov SGD, and a warmup + cosine-decay learning rate schedule in place
      of InverseTimeDecay.
    - train_data is a DIRECTORY of sharded .h5 files (see game_converter_parallel.py),
      not a single .h5 file - a single-file dataset large enough for real training
      degrades badly on spinning-disk hardware past roughly 100-150GB (confirmed the
      hard way), so data now lives in several smaller shard files instead.
    - reads via a streaming shuffle buffer (AlphaGo/training/shuffle_buffer.py) instead
      of random single-row HDF5 access: the old generator's random access pattern reads
      one HDF5 chunk (compressed as a unit) to serve one row, and measured 7x slower than
      pure GPU compute for this reason. The shuffle buffer reads whole games (each
      already one contiguous slice) and shuffles at the individual-position level from an
      in-memory buffer, giving comparable randomness with far less disk overhead. See
      that module's docstrings for the full design.
    - train/val/test split is at the GAME level, not the position level - positions
      within one game are highly correlated, so a position-level split (what both the
      original trainer and the position-permutation approach this replaces actually do)
      risks the same game appearing in both train and validation.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network (tuned large-batch recipe: momentum+Nesterov, warmup+cosine LR, shuffle-buffer data loading).')  # noqa: E501
    # required args
    parser.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicy.save_model())")  # noqa: E501
    parser.add_argument("train_data", help="Directory containing sharded .h5 training data files (see game_converter_parallel.py)")  # noqa: E501
    parser.add_argument("out_directory", help="directory where metadata and weights will be saved")
    # frequently used args
    parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 256", type=int, default=256)  # noqa: E501
    parser.add_argument("--epochs", "-E", help="Total number of iterations on the data across the WHOLE training run, including any epochs already completed in a prior --weights resume (not additional epochs on top of those) - this also shapes the LR schedule's decay horizon. Default: 20", type=int, default=20)  # noqa: E501
    parser.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)  # noqa: E501
    parser.add_argument("--validation-length", help="Number of validation examples to check per epoch. Default: # validation data (full validation set every epoch). Validation reads whole games too, so this only needs to be much smaller than the full validation set if --epoch-length is also set much smaller than the full training set - otherwise a fast per-epoch training pass ends up dominated by a fixed-cost full validation pass every time", type=int, default=None)  # noqa: E501
    parser.add_argument("--learning-rate", "-r", help="Peak learning rate, reached at the end of warmup. Default: .055 (from an LR range test on this model/data/optimizer - see benchmarks/lr_range_test.py; loss was stable through .167, unstable by .183, so .055 is ~1/3 of the instability threshold)", type=float, default=.055)  # noqa: E501
    parser.add_argument("--warmup-steps", help="Number of steps to linearly warm up the learning rate over before cosine decay begins. Default: 1500", type=int, default=1500)  # noqa: E501
    parser.add_argument("--warmup-start-lr", help="Learning rate at step 0, before warmup begins. Default: .0001", type=float, default=.0001)  # noqa: E501
    parser.add_argument("--momentum", help="SGD momentum, used with Nesterov. Default: .9", type=float, default=.9)  # noqa: E501
    parser.add_argument("--global-clip-norm", help="Clip the combined L2 norm across ALL trainable variables together (Keras's global_clipnorm) before the optimizer step. See the same flag in supervised_policy_trainer_v2.py for the full rationale. Default: unset (disabled)", type=float, default=None)  # noqa: E501
    parser.add_argument("--no-jit", help="Disable jit_compile (run eager instead of XLA-compiled) - a direct A/B test of whether XLA's fused/compiled math itself is implicated in a numerical-instability investigation, rather than guessing. Default: jit_compile=True (matches production)", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--buffer-size", help="Number of positions held in the shuffle buffer at once. Default: 400000 (~7GB at 19x19x48 planes)", type=int, default=400000)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # slightly fancier args
    parser.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training", default=None)  # noqa: E501
    parser.add_argument("--mixed-precision", help="Enable the mixed_float16 policy (fp16 compute, fp32 weights). Off by default: measured no benefit on this GPU/driver/model combo - 103.5ms/step with XLA+mixed precision together vs 103ms/step for XLA alone (statistically the same), despite this being an Ada GPU with Tensor Cores. XLA's fusion is apparently already capturing the available speedup here, leaving mixed precision nothing to add while still carrying its own numerical-stability surface (see the forced float32 softmax in policy.py). Only enable to re-test under different conditions (e.g. a larger batch size)", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--train-val-test", help="Fraction of games to use for training/val/test. Must sum to 1. Only used the first time (see game_split.json in out_directory)", nargs=3, type=float, default=[0.93, .05, .02])  # noqa: E501
    parser.add_argument("--symmetries", help="Comma-separated list of transforms, subset of noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2", default='noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2')  # noqa: E501
    parser.add_argument("--seed", help="Seed for the shuffle buffer's game order, buffer sampling, and symmetry choice (both train and val, offset by 1 from each other since they read disjoint game pools). Default: unseeded (a fresh, unrecoverable draw from OS entropy every run) - set this to make the exact position stream fed to the network reproducible, e.g. to replay a run that hit an anomaly.", type=int, default=None)  # noqa: E501
    parser.add_argument("--lr-schedule-epochs", help="Shape the cosine LR schedule's total step budget as if training were running this many epochs, even though --epochs controls how many actually run. Default: same as --epochs. Use this to stop a run early (e.g. after 3 of what would normally be 7 epochs) while keeping the LR trajectory at those epochs identical to a full run - without it, --epochs 3 alone compresses the whole cosine decay into 3 epochs, so epoch 3 ends up far more decayed than epoch 3 of an --epochs 7 run, confounding any comparison between them.", type=int, default=None)  # noqa: E501
    parser.add_argument("--sync-before-validation", help="Force all pending async GPU ops to complete before each epoch's validation pass starts (see SyncBeforeValidationCallback) - a candidate fix for the intermittent resnet validation collapse, still unconfirmed. Off by default so the default command reproduces the bug's original, untouched conditions; opt in explicitly once actually testing this fix.", default=False, action="store_true")  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    resume = args.weights is not None

    if args.verbose:
        if resume:
            print("trying to resume from %s with weights %s" %
                  (args.out_directory, os.path.join(args.out_directory, args.weights)))
        else:
            if os.path.exists(args.out_directory):
                print("directory %s exists. any previous data will be overwritten" %
                      args.out_directory)
            else:
                print("starting fresh output directory %s" % args.out_directory)

    # Must be set before the model is constructed - controls the actual random draw
    # used for weight initialization. Without this, --seed only ever controlled the
    # data stream (shuffle_buffer_batch_generator's game order/sampling/symmetry
    # choice) - weight init drew from Keras's own global RNG regardless of --seed,
    # silently breaking the "same --seed -> same run" assumption for anything
    # sensitive to initial weights (e.g. comparing two seeds' explosion behavior
    # while actually comparing two uncontrolled random inits at the same time).
    if args.seed is not None:
        keras_utils.set_random_seed(args.seed)

    # Must be set before the model is constructed - it controls the dtype policy new
    # layers are built with. The final softmax layer overrides back to float32 regardless
    # (see policy.py) - the standard mixed-precision recommendation to keep the
    # loss-adjacent computation at full precision.
    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
        if args.verbose:
            policy_obj = mixed_precision.global_policy()
            print("mixed precision enabled: compute dtype=%s, variable dtype=%s" %
                  (policy_obj.compute_dtype, policy_obj.variable_dtype))
    elif args.verbose:
        print("mixed precision disabled (default - see --help)")

    # load model from json spec
    policy = CNNPolicy.load_model(args.model)
    model_features = policy.preprocessor.get_feature_list()
    model = policy.model
    if resume:
        model.load_weights(os.path.join(args.out_directory, args.weights))

    # discover shards and build the combined game index
    shard_files = find_shard_files(args.train_data)
    if args.verbose:
        print("found {} shard files in {}".format(len(shard_files), args.train_data))
    games, dataset_features, board_size, n_features = build_game_index(
        shard_files, verbose=args.verbose)

    if dataset_features != model_features:
        raise ValueError("Model JSON file expects features \n\t%s\n"
                         "But shards contain \n\t%s" % ("\n\t".join(model_features),
                                                        "\n\t".join(dataset_features)))
    elif args.verbose:
        print("Verified that shard features and model features exactly match.")

    # ensure output directory is available
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    # Game-level train/val/test split, persisted so a resumed run uses the exact same
    # split rather than a freshly (and differently) drawn one.
    train_games, val_games, test_games = get_or_create_game_split(
        games, args.out_directory, args.train_val_test, verbose=args.verbose)

    n_train_data = sum(g["length"] for g in train_games)
    n_val_data = sum(g["length"] for g in val_games)
    n_test_data = sum(g["length"] for g in test_games)

    if args.verbose:
        print("dataset loaded")
        print("\t%d games, %d total positions" % (
            len(games), n_train_data + n_val_data + n_test_data))
        print("\t%d training games, %d training positions" % (len(train_games), n_train_data))
        print("\t%d validation games, %d validation positions" % (len(val_games), n_val_data))
        print("\t%d test games, %d test positions" % (len(test_games), n_test_data))

    # create metadata file and the callback object that will write to it
    meta_file = os.path.join(args.out_directory, "metadata.json")
    meta_writer = MetadataWriterCallback(meta_file)
    # load prior data if it already exists
    if os.path.exists(meta_file) and resume:
        with open(meta_file, "r") as f:
            meta_writer.metadata = json.load(f)
        if args.verbose:
            print("previous metadata loaded: %d epochs. new epochs will be appended." %
                  len(meta_writer.metadata["epochs"]))
    elif args.verbose:
        print("starting with empty metadata")

    epochs_already_trained = len(meta_writer.metadata["epochs"]) if resume else 0
    if resume and epochs_already_trained >= args.epochs:
        raise ValueError(
            "{} already has {} recorded epochs, which is >= --epochs {}. Increase --epochs "
            "to train further.".format(meta_file, epochs_already_trained, args.epochs))

    if resume and meta_writer.metadata.get("cmd_line_args"):
        # _ResumedLRSchedule's step offset (epochs_already_trained * steps_per_epoch,
        # below) is only correct if steps_per_epoch is the same value it was in every
        # prior invocation - which depends on --minibatch and --epoch-length, and the
        # schedule's own decay_steps also depends on --warmup-steps. A silent change to
        # any of these would silently point the resumed run at the wrong spot on the LR
        # curve with no error, so this is checked explicitly rather than trusted.
        prev_args = meta_writer.metadata["cmd_line_args"][-1]
        for key in ("minibatch", "epoch_length", "warmup_steps"):
            prev_value = prev_args.get(key)
            cur_value = getattr(args, key)
            if prev_value != cur_value:
                raise ValueError(
                    "--{} changed across resume ({} -> {}): this would silently break the "
                    "learning-rate schedule's step accounting, since warmup/decay timing "
                    "depends on steps_per_epoch (minibatch, epoch-length) and the decay "
                    "horizon (warmup-steps). Keep these identical across a resume, or "
                    "start a fresh out_directory.".format(
                        key.replace('_', '-'), prev_value, cur_value))

    meta_writer.metadata["training_data"] = args.train_data
    meta_writer.metadata["model_file"] = args.model
    # Record all command line args in a list so that all args are recorded even
    # when training is stopped and resumed.
    meta_writer.metadata["cmd_line_args"] = meta_writer.metadata.get("cmd_line_args", [])
    meta_writer.metadata["cmd_line_args"].append(vars(args))

    # create ModelCheckpoint to save weights every epoch
    checkpoint_template = os.path.join(args.out_directory, "weights.{epoch:05d}.weights.h5")
    checkpointer = ModelCheckpoint(checkpoint_template, save_weights_only=True)

    symmetries = [BOARD_TRANSFORMATIONS[name] for name in args.symmetries.strip().split(",")]

    # If --validation-length caps validation below the full set, pick a STABLE subset of
    # val_games up front rather than letting the shuffle buffer's per-epoch reshuffle pick
    # a different random slice each time (which it would, if the generator's output were
    # simply truncated after N steps every epoch) - otherwise val_loss becomes noisy
    # epoch-to-epoch for reasons unrelated to the model actually improving, and
    # MetadataWriterCallback's best_epoch (which just tracks lowest val_loss seen) could
    # end up picking a noise-driven "best" epoch rather than a genuinely better one. The
    # existing order of val_games (from game_split.json) is already randomized once at
    # split time, so taking a prefix of it needs no further shuffling.
    if args.validation_length is not None and args.validation_length < n_val_data:
        val_games_for_eval = []
        n_val_eval = 0
        for g in val_games:
            if n_val_eval >= args.validation_length:
                break
            val_games_for_eval.append(g)
            n_val_eval += g["length"]
    else:
        val_games_for_eval = val_games
        n_val_eval = n_val_data

    # create dataset generators. Unlike the position-permutation generator this replaces
    # (which shuffles once at the very start of training and then cycles that same fixed
    # order forever), the shuffle buffer re-shuffles at the start of every full pass over
    # the data, so every epoch gets its own independent shuffle.
    # Offset the val seed by 1 from train's: they read disjoint game pools (per
    # game_split.json) so there's no real correlation risk either way, but distinct
    # seeds rule it out entirely rather than relying on that reasoning holding up.
    train_seed = args.seed
    val_seed = None if args.seed is None else args.seed + 1
    train_batch_holder = _LastBatchHolder()
    train_data_generator = _capturing_generator(
        sanity_checked_generator(
            shuffle_buffer_batch_generator(
                train_games, args.buffer_size, args.minibatch, board_size, n_features, symmetries,
                seed=train_seed),
            args.out_directory, "train"),
        train_batch_holder)

    # Validation, unlike training, is small enough (tens of thousands of positions, not
    # tens of millions) to fully materialize in memory once rather than stream through
    # the infinite shuffle-buffer generator. That infinite-generator approach had a real
    # correctness bug: validation_steps = n_val_eval // minibatch (floor division) never
    # evenly divides n_val_eval, so Keras stops pulling batches partway through a full
    # pass over the shuffle buffer every single epoch, leaving a rolling handful of
    # unconsumed positions to bleed into the next epoch's validation - contradicting the
    # "fresh independent shuffle every epoch" this generator was designed to give. A
    # finite tf.data.Dataset sidesteps this entirely: every epoch sees every validation
    # position exactly once, with TF's own dataset machinery handling the reset.
    print("materializing {} validation positions into fixed arrays...".format(n_val_eval))
    X_val, Y_val = build_validation_arrays(
        val_games_for_eval, board_size, n_features, symmetries, seed=val_seed)

    import tensorflow as tf
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(args.minibatch)

    # Computed here (before building the LR schedule) rather than after model.compile():
    # CosineDecay needs the total step budget up front to shape its decay curve.
    samples_per_epoch = args.epoch_length or n_train_data
    steps_per_epoch = samples_per_epoch // args.minibatch
    total_steps = steps_per_epoch * (args.lr_schedule_epochs or args.epochs)

    # Warmup (linear, warmup_start_lr -> learning_rate over warmup_steps) then cosine
    # decay to zero over the rest of total_steps, replacing InverseTimeDecay's
    # lr/(1+decay*step) - InverseTimeDecay's decay_rate was tuned for a step count this
    # recipe doesn't use, whereas cosine-to-a-known-horizon needs no per-run retuning.
    #
    # CosineDecay's decay_steps is the length of the decay phase AFTER warmup ends, not
    # the total schedule length - confirmed empirically (a schedule with warmup_steps=10,
    # decay_steps=100 reaches its floor at step 110, not step 100). So decay_steps here is
    # total_steps minus the warmup already spent, not total_steps itself - otherwise the
    # schedule would run warmup_steps longer than intended and never reach its floor
    # within the actual training run.
    lr_schedule = CosineDecay(
        initial_learning_rate=args.warmup_start_lr,
        decay_steps=max(1, total_steps - args.warmup_steps),
        warmup_target=args.learning_rate,
        warmup_steps=args.warmup_steps)
    if epochs_already_trained:
        # See _ResumedLRSchedule: the resumed optimizer's own step counter restarts at 0,
        # so shift the schedule's input by however many steps were already trained in
        # prior invocations, keeping this one continuous warmup+decay curve across resumes.
        lr_schedule = _ResumedLRSchedule(lr_schedule, epochs_already_trained * steps_per_epoch)
    sgd = SGD(learning_rate=lr_schedule, momentum=args.momentum, nesterov=True,
              global_clipnorm=args.global_clip_norm)
    # jit_compile=True (XLA): safe here because of the ensure_xla_conv_nhwc() call at
    # module load time above - see xla_workarounds.py for why it's needed. Measured
    # 102.5ms/step vs 113ms/step without XLA.
    model.compile(
        loss='categorical_crossentropy', optimizer=sgd,
        metrics=["accuracy", TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
                 prediction_entropy],
        jit_compile=not args.no_jit)

    diagnostics = TrainingDiagnosticsCallback(lr_schedule, steps_per_epoch)
    collapse_diagnostics = CollapseDiagnosticCallback(
        X_val, Y_val, args.minibatch, board_size, args.out_directory)
    gradient_diagnostics = GradientExplosionDiagnosticCallback(
        train_batch_holder, args.out_directory)

    if args.verbose:
        print("STARTING TRAINING")

    fit_callbacks = [checkpointer, diagnostics, collapse_diagnostics, gradient_diagnostics, meta_writer]
    if args.sync_before_validation:
        # Position doesn't matter for correctness (only hooks on_test_begin, which none
        # of the others use) - inserted first just to read as "happens before anything
        # else touches validation".
        fit_callbacks.insert(0, SyncBeforeValidationCallback())

    model.fit(
        x=train_data_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        initial_epoch=epochs_already_trained,
        # diagnostics must run before meta_writer: Keras passes the same logs dict
        # through every callback for a given event in list order, and meta_writer just
        # persists whatever's in logs by the time it sees it.
        callbacks=fit_callbacks,
        validation_data=val_dataset)


if __name__ == '__main__':
    run_training_v2()
