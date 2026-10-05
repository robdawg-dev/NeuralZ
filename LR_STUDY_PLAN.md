# Plan: find a good learning rate and warmup for the b15c192 recipe

Started 2026-09-28. The recipe is the one in [SAMPLE_RUN.md](SAMPLE_RUN.md): ResTowerPolicy
b15c192 `conv_norm`, minibatch 1024, mixed precision, SGD momentum 0.9 (Nesterov), plateau
schedule. The reference run used **LR 1.6 with a 16,928-step warmup**; this study checks
whether that is a good choice and how much warmup it needs.

## Why this needs a careful method

Earlier LR work on this project gave confusing results: going from minibatch 256 to 1024
moved the chosen LR from roughly 0.01–0.1 up to 1.6; LR 2.0 once looked about as stable
as 1.6 while 1.2 diverged; and short warmups became unstable early while long ones settled.
Several things likely explain this, and the method is designed around them:

- **Larger batches usually allow a larger LR with SGD, not a smaller one.** The common
  "linear scaling rule" (Goyal et al. 2017) multiplies the LR by k for a k-times larger
  batch, with a warmup (so 256 -> 1024 suggests ~4x); some work argues for sqrt(k).
- **Batch norm plus no weight decay makes the effective step size shrink during training.**
  For weights feeding a batch-norm layer, only their direction matters, and the effective
  step scales roughly with LR / ||w||^2. With no weight decay, ||w|| grows as training
  goes, so a nominal LR that is violent at the start becomes gentle later. This is why a
  high nominal LR can work, why warmup matters (it lets ||w|| grow before the LR is
  high), and why stability depends on the path a run took, not only its LR.
- **One run per LR can't separate a threshold from luck.** GPU arithmetic isn't
  bit-for-bit repeatable, and at high LR small differences grow (seen directly while
  testing on 2026-09-28: two identical runs at LR 1.6 differed after a few epochs).
- **A range test's ramp passes through each LR too briefly to show stability at it.**
  It locates the region; held-constant runs decide.
- **Other things have changed since the earlier results**, notably the `conv_norm` head
  (introduced to fix an output-head collapse), so they aren't directly comparable.

Therefore: **2 seeds per configuration** (more near a boundary if needed), **ramp speed and
warmup length treated as variables**, **held-constant runs to decide**, and the **weight
norm tracked** alongside the loss.

**What a seed varies here:** weight initialization and each position's board orientation
(and the validation set's orientations) - **not the data order**, which convert_shuffled
fixes once when it writes the shards. Both seeds of a configuration meet the same batches
at the same steps. Accepted as is: the study samples sensitivity to initialization and
orientation, not to when a hard stretch of data arrives.

## Budget and data

A run of 1-2 hours is ~7,000-14,000 steps at ~1.96 steps/s, i.e. 7-14M positions. The
reference warmup alone is 2.4 hours, so warmup is studied at shorter lengths.

**Data:** an 8M-position subset built from the existing manifest with the same filters and
composition (95% normal, 5% handicap), train and val only, ~20 GB:

```bash
docker compose run --rm gpu python -m AlphaGo.preprocessing.select_games \
    workspace/analysis/manifest.jsonl workspace/lr_study/selection \
    --normal-positions 7.6e6 --handicap-positions 0.4e6
docker compose run --rm gpu python -m AlphaGo.preprocessing.convert_shuffled \
    workspace/lr_study/selection workspace/lr_study/shards \
    --splits train val --max-winrate-loss 0.10
```

With ~7.4M training positions, a 1-hour run sees each position about once.

**Tooling** (in `benchmarks/lr_study/`):
- `run_queue.sh`: runs a list of experiments one after another through `docker compose`,
  logging each, and deletes every checkpoint but the last when a run ends (a checkpoint is
  121 MB and these runs write many).
- `report.py`: plots and tabulates the runs' logs - loss against LR for range tests, loss
  against steps for held runs - with the weight norm and effective step size alongside.

Outputs go to `workspace/lr_study/runs/<run name>/`.

## Phase 1: coarse range tests - locate the region

All runs: the b15c192 `conv_norm` model, minibatch 1024, mixed precision, validation every
250 steps on 10,240 positions. The LR ramps linearly from 1e-4 to 1e-3 over 250 steps, then
exponentially from 1e-3 to 10 (runs stop early if the loss becomes NaN). Logged every 10
steps: LR, loss, weight norm, gradient norm, loss scale.

| Run | Sweep 1e-3 -> 10 over | Seeds | Steps | ~Time |
|---|---|---|---|---|
| `p1_fast_s90001`, `p1_fast_s90002` | 1,500 steps | 90001, 90002 | 1,750 | 15 min each |
| `p1_slow_s90001`, `p1_slow_s90002` | 4,500 steps | 90001, 90002 | 4,750 | 40 min each |

**Ramp speed is the variable, not warmup-to-floor length:** a warmup that only reaches 1e-3
barely moves the weights, while a fast ramp reaches high LRs while ||w|| is still small -
the situation a short warmup creates. If the slow ramp tolerates noticeably higher LRs than
the fast one, that confirms the ||w|| effect and says warmup length matters.

**Read from each run:** where the (smoothed) loss stops improving, where it turns upward or
spikes, where it goes non-finite, and how the weight norm and loss scale behave around
those points - and how much all of this differs between seeds.

**Pause after Phase 1** to reassess before committing to Phase 2's grid.

## Phase 2: held-constant runs - decide

Revised after Phase 1, which showed stability depends on LR and warmup together: **3 LRs x
2 warmups, one seed each** (a second seed only where results look odd), every run 7,000
steps (~60 min) so losses compare directly at equal steps. Queue:
`benchmarks/lr_study/phase2.queue`.

| Run | Warmup 1e-4 -> LR | Held at | Steps held |
|---|---|---|---|
| `p2_lr0.8_w1000` / `p2_lr0.8_w4000` | 1,000 / 4,000 | 0.8 | 6,000 / 3,000 |
| `p2_lr1.6_w1000` / `p2_lr1.6_w4000` | 1,000 / 4,000 | 1.6 | 6,000 / 3,000 |
| `p2_lr3.2_w1000` / `p2_lr3.2_w4000` | 1,000 / 4,000 | 3.2 | 6,000 / 3,000 |

All: b15c192 `conv_norm`, minibatch 1024, mixed precision, seed 90001; the plateau schedule
with `--plateau-patience 1000` so the LR never cuts after warmup; `--steps-per-epoch 500`
(14 epochs), validation each epoch on 20,000 positions.

**Instrumentation added first (2026-09-28):** the trainer now records `weight_norm` and (with
mixed precision) `loss_scale` in every epoch's metadata, and stops on a NaN loss; the range
test logs each step's own loss (`loss`) alongside Keras's running epoch mean
(`loss_epoch_mean`).

- **Judged on:** non-finite losses or spikes; training and validation loss at steps 4,000
  and 7,000; weight norm, effective step LR/||w||^2 and loss-scale behavior.
- **Oddness that triggers a second seed:** a result breaking the pattern (e.g. a lower LR
  diverging while a higher one survives with the same warmup), a borderline run (recovered
  spikes, a sharply falling loss scale), or two configurations too close for one run each
  to separate.

## Phase 3: choose

The LR with the best loss at a fixed step budget that was stable for every seed, with a
safety margin below the highest stable LR (a 39-hour run meets many more bad batches than a
35-minute one), and the shortest warmup that was reliably stable. Then compare with the
reference's 1.6 / 16,928.

## Results

### Phase 0 - data (2026-09-28)
`workspace/lr_study/shards`, 19 GB, built in ~27 minutes:

| Split | Games | Positions |
|---|---|---|
| train | 23,302 (1,175 handicap) | 7,442,835 |
| val | 1,253 (63 handicap) | 400,035 |

Handicap share 5.0% of positions, as in the reference data.

### Phase 1 - range tests (2026-09-28)
Plot: `workspace/lr_study/phase1.png` (`python -m benchmarks.lr_study.report ...`).

| Run | Ramp 1e-3 -> 10 over | How it ended | Weight norm at the end | Loss scale |
|---|---|---|---|---|
| `p1_fast_s90001` | 1,500 steps | **NaN at LR 2.75** (step 1,540; run stopped) | 134 -> 188 | steady |
| `p1_fast_s90002` | 1,500 steps | **blew up at LR ~9.4** (loss ~100) | 134 -> 424 | collapsed to ~500 at the end (fp16 overflows) |
| `p1_slow_s90001` | 4,500 steps | **survived to LR 9.8**, loss still falling (2.41) | 134 -> 278 | grew (stable) |
| `p1_slow_s90002` | 4,500 steps | **survived to LR 9.8**, loss still falling (2.41) | 134 -> 274 | grew (stable) |

**Findings:**
1. **Ramp speed decides survival, as the ||w|| explanation predicts.** Both slow ramps reached
   LR ~10 without trouble; both fast ramps broke, at very different LRs (2.75 vs ~9.4) - so
   for fast ramps the break point is largely luck. In the slow runs the weight norm had
   roughly doubled by the high LRs, keeping the effective step LR/||w||^2 at about 1.3e-4 at
   LR 10; the fast runs broke with effective steps of ~1-2e-4 at lower nominal LRs. The
   boundary looks like an effective-step size, reached at very different nominal LRs
   depending on how much ||w|| has grown.
2. **The fast ramps failed in two ways:** one NaN outright, the other a slide into fp16
   overflow (the loss scale fell ~64x as mixed precision skipped overflowing steps) - worth
   watching for in Phase 2.
3. **The range test cannot place an LR ceiling for slow ramps** within 1e-3 -> 10: nothing
   broke. The nominal LR alone is not the quantity that decides stability here.
4. **Instrumentation problem found:** the logged "loss" is Keras's running average since the
   start of the epoch (it resets every 250 steps - visible as steps in the curves), not each
   step's loss. That smears the loss-vs-LR curve across a wide LR range and delays how soon
   a blow-up shows. The NaN, loss-scale and weight-norm observations above are unaffected;
   loss-vs-LR shapes and "lowest loss at LR" values are not trustworthy. Every earlier range
   test in this project logged the same way.

### Phase 2 - held-constant runs (2026-09-28)
Plot: `workspace/lr_study/phase2.png`. All six ran the full 7,000 steps: no non-finite loss,
no loss spikes, and the loss scale only grew (32k -> 262k, i.e. no fp16 overflow).

| Run | Train loss @4,000 | Val @4,000 | Train loss @7,000 | Val @7,000 | Best val (step) | ||w|| @7,000 | LR/||w||^2 @7,000 |
|---|---|---|---|---|---|---|---|
| `p2_lr0.8_w1000` | 2.090 | 2.123 | 1.979 | 2.012 | 2.012 (7,000) | 227 | 1.6e-5 |
| `p2_lr1.6_w1000` | 2.080 | 2.095 | **1.975** | 2.036 | 2.006 (6,500) | 294 | 1.9e-5 |
| `p2_lr3.2_w1000` | 2.080 | 2.106 | **1.975** | 2.029 | **1.999** (6,500) | 397 | 2.0e-5 |
| `p2_lr0.8_w4000` | 2.171 | 2.269 | 2.007 | 2.074 | 2.066 (6,500) | 215 | 1.7e-5 |
| `p2_lr1.6_w4000` | 2.147 | 2.266 | 1.990 | 2.017 | 2.017 (7,000) | 277 | 2.1e-5 |
| `p2_lr3.2_w4000` | 2.136 | 2.177 | 1.985 | 2.049 | 2.017 (6,500) | 373 | 2.3e-5 |

**Findings:**
1. **Over 0.8-3.2 the nominal LR barely matters.** With a 1,000-step warmup the training loss
   at 7,000 steps is 1.979 / 1.975 / 1.975 for LR 0.8 / 1.6 / 3.2. The weight norm grows
   faster at higher LR (227 / 294 / 397) and cancels most of it: a 4x range of nominal LR
   ends as a 1.3x range of effective step (1.6-2.0e-5). This is the ||w|| effect from the
   plan, acting as a built-in LR decay whose strength rises with the LR.
2. **A 1,000-step warmup is enough, and faster.** At every LR the 1,000-step warmup is ahead
   of the 4,000-step one at 4,000 and at 7,000 steps (train loss by 0.010-0.028 at the end),
   with no sign of instability. Nothing here supports the reference's 16,928-step warmup.
3. **Validation loss bounces more at higher LR**, by up to ~0.07 between checks (e.g.
   `p2_lr1.6_w1000`: 2.095, 2.137, 2.136, 2.060, 2.078, 2.006, 2.036), while LR 0.8 with
   1,000-step warmup improved at nearly every check. The bounces line up across the LR
   1.6 and 3.2 runs (up at 6,000, down at 6,500, up at 7,000), which suggests a reaction to
   the same stretch of data - all runs see the same batches at the same steps - more than
   random noise. Single val checks therefore can't rank these runs; training loss (a mean
   over 500 steps) is the steadier measure.
4. **Survival at LR 3.2 after a 1,000-step warmup is one sample.** `p1_fast_s90001` hit a NaN
   at LR 2.75 with a similar weight norm (~188 vs 182 here at the end of warmup), so one
   clean run doesn't rule out an occasional failure; a 39-hour run gives more chances.
   Phase 1 showed the break point for fast ramps is largely luck.

### NewResPolicy vs ResTowerPolicy (2026-09-28)
`nr_g5_lr0.8_w1000` (`benchmarks/lr_study/newres.queue`): NewResPolicy b15c192 - pre-activation
blocks, a final trunk norm, global pooling in blocks 5/10/15 and in the policy head - on the
exact `p2_lr0.8_w1000` recipe (LR 0.8, 1,000-step warmup, 7,000 steps, seed 90001).

| Step | ResTowerPolicy train / val | NewResPolicy train / val | Val accuracy |
|---|---|---|---|
| 1,000 | 2.519 / 2.520 | 2.442 / 2.438 | 37.9% -> 38.4% |
| 4,000 | 2.090 / 2.123 | 2.022 / 2.060 | 43.3% -> 44.1% |
| 7,000 | 1.979 / 2.012 | **1.914 / 1.945** | 45.1% -> 46.1% |

- Ahead by ~0.065-0.08 training loss at every check from step 1,000 on, not narrowing. It
  reached the baseline's final training loss (1.979) by step 5,000: ~30% fewer steps.
- ~6% faster per step (2.17 vs 2.04 steps/s; 55 vs 58 minutes), so ~35% less wall time to
  that loss.
- Equally stable: no spikes, the loss scale grew the same way, and the weight norm tracked
  the baseline's within ~2%.
- One seed and 7,000 steps; not yet confirmed over a long run.

**Training memory** (peak over a few steps at mixed precision, `tf.config.experimental.
get_memory_info`): ResTowerPolicy b15c192 9.2 GB and NewResPolicy b15c192 9.1 GB at batch
1024; NewResPolicy b20c256 4.2 GB at batch 256 and 8.1 GB at batch 512 - so ~16 GB at 1024,
matching the 1.78x (20x256 / 15x192) scaling from b15c192. **b20c256 fits a 12 GB card at
batch 512, not 1024.**

### NewResPolicy b20c256 vs b15c192 (2026-09-29)
`nr_b20c256_g5_lr0.8_w1000` (`benchmarks/lr_study/newres.queue`): NewResPolicy b20c256 (pooling
in blocks 5/10/15/20, head width 48, 23.3M parameters), minibatch 1024 as two accumulated
halves of 512 (`--accumulation-steps 2`, a trainer option since removed - a b20c256 run
would use a plain minibatch of 512), otherwise the `nr_g5_lr0.8_w1000` recipe.

| Step | b15c192 train / val | b20c256 train / val | Train lead | b15c192 steps to b20c256's train loss |
|---|---|---|---|---|
| 2,000 | 2.200 / 2.202 | 2.187 / 2.174 | 0.013 | ~2,110 (+6%) |
| 4,000 | 2.022 / 2.060 | 2.006 / 2.055 | 0.016 | ~4,320 (+8%) |
| 6,000 | 1.940 / 1.955 | 1.922 / 1.930 | 0.018 | ~6,670 (+11%) |
| 7,000 | 1.914 / 1.945 | **1.895 / 1.903** | 0.020 | ~7,900 (extrapolated, +13%) |

- Ahead at every check, and the lead grows steadily (0.013 -> 0.020 training loss; +6% ->
  +13% in steps) - the bigger net pulling away, but slowly.
- 2.0x slower per step (1.07 vs 2.17 steps/s; 111 vs 55 minutes), so ~1.8x the wall time
  to the same loss. Break-even needs a ~2x step advantage; at this rate of growth, not
  within anything like a 39-hour run's early phase.
- Stable: one loss-scale halving (at step ~6,500, a single skipped half-batch), no spikes.
- Only ~one pass over 7.4M positions: says nothing about the regime where b15c192 runs
  out of capacity on the full data.

### b20c256 at minibatch 512 on the 60M data (2026-09-29/30)
For the long run: NewResPolicy b20c256 (head width 48) at a plain minibatch of 512 on
`workspace/prod_60m/shards` (55.8M training positions). Queue:
`benchmarks/lr_study/b20_512.queue`. ~2.15 steps/s (~1,100 positions/s).

**Range test** (`b20_512_range_s90001`, LR 1e-3 -> 10 over 5,000 steps, per-step loss): no
non-finite loss, no loss-scale drop, nothing blew up to LR 9.8. Smoothed loss flat from ~0.2
to ~5, a slight upturn above ~6. Plots: `workspace/lr_study/b20_512_range_loss_vs_lr.png`,
`..._vs_time.png`.

**Held runs** (7,000 steps = 3.6M positions, 2,000-step warmup, seed 90001):

| Step | Train loss 0.4 / 0.8 / 1.6 | Val loss 0.4 / 0.8 / 1.6 | Val acc 0.4 / 0.8 / 1.6 |
|---|---|---|---|
| 3,000 | 2.252 / 2.232 / 2.225 | 2.272 / 2.252 / 2.257 | 40.3 / 41.0 / 41.5% |
| 5,000 | 2.117 / 2.096 / 2.089 | 2.149 / 2.113 / 2.099 | 42.4 / 42.7 / 43.4% |
| 7,000 | 2.035 / 2.013 / **2.006** | 2.057 / **2.047** / 2.049 | 44.0 / 44.7 / **44.8%** |

- 0.4 -> 0.8 gained a steady 0.021-0.022 training loss; 0.8 -> 1.6 a steady 0.006-0.010 -
  diminishing returns. Validation agreed (1.6 lowest at 4 of the last 6 checks).
- All three clean: loss scale 32k -> 262k, no spikes. Weight norm at the end 226 / 264 / 328,
  effective step LR/||w||^2 7.8e-6 / 1.15e-5 / 1.49e-5 - doubling the LR raised it only ~30%.
- Unlike batch 1024 in Phase 2 (0.8-3.2 within 0.004), at batch 512 linear scaling's 0.4 is
  clearly too low; the gains flatten from 0.8 up.
