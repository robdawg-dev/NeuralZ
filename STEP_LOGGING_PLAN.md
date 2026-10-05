# Plan: optional step-level logging in the trainer

Proposed 2026-09-28, following the LR study ([LR_STUDY_PLAN.md](LR_STUDY_PLAN.md)). Not yet
implemented.

## Why

The LR study's case for a safety margin rests on rare events: a minibatch (or a moment in
training) whose gradient is far larger than usual, so a single step at high LR throws the
weights far enough to spike the loss, overflow fp16, or go NaN. A 39-hour run takes ~20x
the steps of a Phase 2 run, so an event that never showed up in 7,000 steps may still
happen there.

The only evidence so far is indirect, from the Phase 1 range tests (which log every 10th
step):

- `p1_fast_s90002` looks like a sudden event. At step 1730 (LR 8.8), the loss was still
  falling, but the gradient norm was 7-10x its recent level. Ten steps later the weight
  norm had doubled (205 -> 424), the gradient was infinite and the loss scale had
  collapsed (32,768 -> 512). This record doesn't show whether one batch caused it or the
  model was about to tip over at that LR anyway.
- `p1_fast_s90001` looks like a gradual runaway instead. Its gradient norms stayed in the
  usual range, the loss climbed over ~30 steps, then it went NaN at LR 2.75 - an LR its
  twin run passed without trouble.

The trainer records only per-epoch values, so held runs can't show any of this. Logging
steps in held runs would measure how large the largest steps get, how often they happen,
and whether they line up with loss-scale drops or validation bounces. That turns the
safety-margin argument into a measurement.

In general training, per-step logging is unwanted, so it is **off by default**.

## Changes

### 1. A trainer flag: `--step-log-every N`
- **Default: off.** Without it the trainer behaves exactly as now: no replaced training
  step, no extra callback, no extra file.
- **When set,** the trainer writes `<out_directory>/step_diagnostics.jsonl` every N steps.
  Each record has the fields the range test logs today: `step`, `lr`, `loss` (that step's
  own loss), `loss_epoch_mean` (Keras's running mean), `weight_norm`, `grad_norm` and
  `loss_scale`.
- `--step-log-every 1` logs every step. Phase 1 showed why that matters: `p1_fast_s90002`
  went from normal to destroyed in under 10 steps.

### 2. Share the code between the trainer and the range test
- Move `_grad_norm_and_loss_scale_train_step` and `RangeTestDiagnosticsCallback` from
  `AlphaGo/training/lr_range_test.py` into `AlphaGo/training/supervised_policy_trainer.py`,
  with general names (e.g. `_step_diagnostics_train_step`, `StepDiagnosticsCallback`). The
  range test already imports from that module.
- Put `--step-log-every` in the shared `add_run_arguments`, replacing the range test's
  `--range-check-every`. **The range test defaults to 50** (as now) and always logs; the
  trainer defaults to off.
- Update `benchmarks/lr_study/phase1.queue` to the new flag name. It has already run, so
  this only keeps it re-runnable.

*Open decision:* replacing `--range-check-every` with the shared name. Recommended: the
range test was split into its own entry point this week, and nothing else depends on the
old name.

### 3. Fix: log the unscaled gradient norm
Under mixed precision, the training step measures the gradient norm of the *scaled*
gradients (x32,768 at the start). That is why Phase 1's values ran in the thousands to
tens of thousands (medians of ~63k-75k per run). Log the true norm instead, divided by the current loss scale.

Within a Phase 1 range test this didn't matter, because the scale stayed constant until
the blow-up. In a held run the scale doubles every few thousand steps, and each doubling
would look like a false 2x jump in gradient norm.

Earlier `step_diagnostics.jsonl` files keep the scaled values, and that difference should
be noted wherever they are read.

### 4. Fix: resume
The callback's step counter starts at 0 and the file is opened fresh. On a trainer resume,
the counter should start at the steps already trained, and records should be appended to
the existing file. The range test's `--weights` warm start begins a new sweep, so it keeps
starting fresh.

### 5. Cost
The weight norm is computed outside the compiled training step, on every logged step. At
`--step-log-every 1` that may slow training slightly. Measure it on the first run (steps/s
against Phase 2's ~2.0). If it matters, compute the weight norm inside the training step
instead.

### Not included
No per-step panels in `benchmarks/lr_study/report.py` yet. Read the first run's file
directly, then decide what is worth plotting. Candidates:
- the gradient norm's distribution and its largest values relative to the median;
- the relative update size (roughly LR x grad_norm / weight_norm - with batch norm, raw
  gradients shrink as the weight norm grows, so this is the better measure of how hard one
  step pushes the model);
- how the largest steps line up with loss-scale drops and loss jumps.

## Tests
- Flag off: no `step_diagnostics.jsonl` is written, and the model's training step is
  unchanged.
- `--step-log-every N` writes records N steps apart.
- On a resume, the step count continues and records are appended.
- Under mixed precision, the recorded `grad_norm` is the unscaled norm.
- The range test still writes its log at its default interval, under the new flag name.
- An integration run of the trainer with the flag set.

## The first run
Queued once the changes are in: `p2_lr0.8_w1000_s90002`, identical to `p2_lr0.8_w1000`
(LR 0.8, 1,000-step warmup, 7,000 steps; see Phase 2 in the study plan) except
`--seed 90002` and `--step-log-every 1`. About an hour. It serves two purposes:

- **Second seed** for the recommended LR 0.8 / 1,000-step warmup. Does its smooth
  validation curve and 1.979 training loss hold up?
- **First measurement** of the per-step gradient tail at the setting we'd actually train
  with.
