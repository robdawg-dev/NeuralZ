# Joint training log

Running record for `VALUE_HEAD_PLAN.md` (the fine-tuned value heads) and
`JOINT_TRAINING_PLAN.md` (the from-scratch joint run): results, conclusions, and every
decision made while working unattended. Newest entries at the bottom of each section.

## Instructions for the unattended stretch (2026-10-03)

From the user, before ~8 hours away:

1. Continue v3 (all blocks unfrozen) unless it has flattened enough that more training
   won't matter; then test its last checkpoint and record conclusions here.
2. Then move to the from-scratch joint training: a non-ending run trained like the
   original b20c256 run, with the starting learning rate from an LR range test.
   Monitor it and report each epoch.
3. Where a response is needed, pick my own suggestion and record it here.
4. Decided before leaving: **include the ownership target** in the from-scratch run.

## Value head fine-tunes (b20c256 trunk)

Validation numbers on the first 200K val positions. Baselines: always 50% = win-rate loss
0.693, error 0.278; always 0 points = score error 4.51; perfect = loss 0.409.

| Run | Trainable | Best epoch | Win-rate loss | Error | Agrees on who's ahead | Score error | Policy top-1 / top-5 |
|---|---|---|---|---|---|---|---|
| v1 | value head only (12.6K weights) | 3 | 0.571 | 0.213 | 64.8% | 3.56 | 57.21% / 89.11% (frozen) |
| v2 | + top 4 blocks, policy head (4.6M) | 6 (stopped) | 0.550 | 0.198 | 67.6% | 3.11 | 57.24% / 89.13% |
| v3 | everything (23.3M) | 8 (stopped) | 0.522 | 0.174 | 71.2% | 2.66 | 56.98% / 89.08% |

### v3 per epoch

Started from v2 epoch 6. Adam, LR 1e-4 after 500 warmup steps, plateau schedule
(halve after 2 epochs without improvement in val win-rate loss), 2,000 steps x 512 per
epoch, ~17 min per epoch.

| Epoch | LR | Policy top-1 | Top-5 | Win-rate loss | Error | Agreement | Score error |
|---|---|---|---|---|---|---|---|
| start | - | 57.19% | 89.23% | 0.5503 | 0.198 | 67.6% | 3.11 |
| 1 | 1e-4 | 57.04% | 89.05% | 0.5410 | 0.191 | 68.8% | 2.96 |
| 2 | 1e-4 | 57.02% | 89.09% | 0.5359 | 0.187 | 69.3% | 2.87 |
| 3 | 1e-4 | 57.04% | 89.03% | 0.5330 | 0.184 | 69.8% | 2.81 |
| 4 | 1e-4 | 56.97% | 89.06% | 0.5295 | 0.181 | 69.9% | 2.77 |
| 5 | 1e-4 | 57.05% | 89.06% | 0.5298 | 0.180 | 70.1% | 2.80 |
| 6 | 1e-4 | 56.98% | 89.07% | 0.5262 | 0.178 | 70.7% | 2.71 |
| 7 | 1e-4 | 57.01% | 89.08% | 0.5234 | 0.176 | 71.0% | 2.69 |
| 8 | 1e-4 | 56.98% | 89.08% | 0.5215 | 0.174 | 71.2% | 2.66 |

**Stopped after epoch 8 (01:33).** The gain fell to 0.0019 per epoch, below the ~0.002
threshold. The plateau schedule never cut the LR: epoch 5's stall was noise, so no
2-epoch streak occurred. Waiting for a cut and its effect would have cost ~1 more hour
of GPU time, and the from-scratch run is the bigger goal; the epoch-8 checkpoint is good
enough to judge what fine-tuning buys in the lookahead test.

### Lookahead test (7 KGS games, 804 bot moves; KataGo points lost)

Greedy (today's bot) 882; best possible pick in the top 5: 3.

| Rule | v1 epoch 3 | v2 epoch 6 | v3 epoch 4 | v3 epoch 8 |
|---|---|---|---|---|
| (val win-rate loss) | 0.571 | 0.550 | 0.530 | 0.522 |
| value, top 5 (plain best) | 1,079 | 1,179 | 1,004 | 948 |
| ... of which on the 90 big mistakes | 340 | 355 | 315 | 299 |
| value, top 5, switch if +0.06 | 765 | 769 | 837 | 827 |
| score, top 5, switch if +2 pts | 756 | 734 | 771 | 748 |
| value + 0.2 x policy, top 5 | 713 | 784 | 781 | 767 |
| value + 0.5 x policy, top 5 | 799 | 774 | 749 | 723 |

Full tables: `workspace/sample_ratio/lookahead_report.txt` (v1, v2) and
`lookahead_report_v3.txt` (all four).

## Conclusions: fine-tuning a value head onto b20c256

1. **Each step of unfreezing improved the value head on validation,** with the policy
   unchanged (57.0-57.2% top-1 throughout): frozen 0.571, top 4 blocks 0.550, all
   blocks 0.522 (agreement on who's ahead 64.8% -> 67.6% -> 71.2%; score error 3.56 ->
   3.11 -> 2.66 points). All blocks learned fastest; it was still improving (~0.002 per
   epoch) when stopped.
2. **The better heads catch blunders better, but that hasn't yet turned into a better
   deployable rule.** With "pick the best value in the top 5", points lost on the 90 big
   mistakes fell from 340 (v1) to 299 (v3), and the total from 1,079 to 948 - but that
   rule still switches too often and loses more than greedy (882) overall. The
   conservative rules (a switching margin, or blending with the policy) save 15-19% of
   points lost (882 -> ~713-750) for every checkpoint; differences between checkpoints
   are within this 7-game test's noise.
3. **The ceiling is far from reached:** the best of the top 5, picked perfectly, loses 3
   points. The bottleneck is still the value head's accuracy in close positions (it
   agrees with KataGo on who's ahead 71% of the time), which a few more epochs of
   fine-tuning won't change much.
4. **So the from-scratch joint run is the next step,** as planned: a trunk shaped by
   value, score and ownership from the start. The lookahead test is the yardstick for
   its checkpoints (greedy 882; conservative fine-tuned lookahead ~720-750).
5. **If the bot is wanted sooner:** v3 epoch 8 with "value + 0.5 x policy, top 5" (723)
   or "score, top 5, switch if +2 pts" (748) is a usable ~17% improvement in this test,
   pending the CPU timing (`CPU_INFERENCE_PROFILE_PLAN.md`) and real games.

## From-scratch pipeline (built while v3 trained)

- `AlphaGo/preprocessing/add_value_targets.py`: sidecars gain `black_to_move` (needed to
  turn Black-side ownership into the player to move's). Re-run needed before training.
- `AlphaGo/preprocessing/add_ownership_targets.py` (new, standard library only, runs
  natively next to KataGo): KataGo analysis of each game's final position at 10 visits,
  `<split>/ownership.bin` (int8 x127, Black's side, x * 19 + y order) + `ownership.json`,
  including a sanity check (share of games whose winner owns more of the board).
- `AlphaGo/models/policy.py`: NewResPolicy's builder split into `newres_trunk` /
  `newres_policy_head`, plus `gpool_blocks` (explicit pooling-block list) and a global
  bias input after the stem conv. Existing models unaffected (they load from JSON).
- `AlphaGo/models/value.py`: `PolicyValueNet.create_network` - the joint network from
  scratch (komi into the trunk; policy, value, score, ownership outputs).
- `AlphaGo/training/joint_data.py` (new) and `supervised_policy_trainer.py`: a 4-output
  model trains on shards + value sidecars + ownership table (ownership under the same
  board symmetry as the planes, on the GPU); losses policy CE + value BCE + score Huber
  + ownership MSE; new options `--value-weight` (1.0), `--score-weight` (0.5),
  `--ownership-weight` (1.0). `lr_range_test.py` takes the joint batches too.
- Tests: `tests/test_joint_training.py` (perspective, symmetry, plateau run + resume,
  range test), joint-network tests in `tests/test_value_model.py`,
  `tests/test_add_ownership_targets.py`.
- Model: `workspace/models/model_joint_b20c256_g7.json` - 20 blocks x 256, pooling blocks
  7/12/17 (64 pooled channels, as b20c256), policy head 48, value head 48 / 112, komi
  input; 23.43M parameters (b20c256: 23.30M).

### Ownership labels: checked

`val` (9,406 games) took 111 s at 10 visits (80 games/s). The script's own sanity
number (winner owns more of the board: 63.9%) ignores komi, so it understates. Checked
properly in the container: Black's owned area minus komi against the actual result -
**correlation 0.993 with the game's margin, median error 0.5 points, right winner in
94.2% of games** (the rest are close games, where area vs territory counting differs).
Mean |ownership| 0.999: final positions are fully settled, as expected.

### LR range test (02:13-02:56)

`workspace/lr_study/runs/joint_b20_512_range_s90001`, same settings as b20c256's
(`b20_512_range_s90001`). Report: `python workspace/lr_study/joint_range_report.py`.

| | b20c256 (policy only) | Joint b20c256 |
|---|---|---|
| Lowest smoothed loss | LR 6.0 | **LR 2.2** |
| Smoothed loss 2% above its minimum | never (to 9.8) | **LR 3.6** |
| End | stable at 9.8 | **NaN at LR ~7.2** (step 5,076) |

The blow-up came from the value and score heads: just before the NaN, score error 1,020
points and win-rate loss 17.8, while the policy (38% accuracy) and ownership were still
normal. **Chosen peak LR: 0.8** (per the rule: a third of 3.6 is ~1.2, rounded down to the
0.4 / 0.8 / 1.6 ladder). b20c256's held runs found 0.8 within 0.007 training loss of 1.6
after 7,000 steps, so little should be lost, and it keeps a ~4.5x margin below the
upturn.

## From-scratch joint run: `workspace/runs/joint_b20c256`

Launched 02:57 (2026-10-03), Docker container `joint_b20c256`, log
`workspace/runs/joint_b20c256.log`:

```
python -m AlphaGo.training.supervised_policy_trainer \
  workspace/models/model_joint_b20c256_g7.json workspace/prod_60m/shards \
  workspace/runs/joint_b20c256 --minibatch 512 --epochs 1000 --steps-per-epoch 5000 \
  --validation-length 100000 --mixed-precision --seed 90001 --learning-rate 0.8 \
  --warmup-steps 2000 --warmup-start-lr 0.0001 --lr-schedule plateau --plateau-factor 0.5 \
  --plateau-patience 3 --plateau-cooldown 3 --plateau-min-lr 1e-5 --plateau-min-delta 0.005
```

Loss weights: value 1.0, score 0.5, ownership 1.0 (defaults). 477 ms/step, ~41 min per
epoch (b20c256: ~38.5). Resume after an interruption: the same command plus
`--weights weights.NNNNN.weights.h5` (the latest). Manual LR change: write the new LR
into `workspace/runs/joint_b20c256/lr_override.txt`. Per-epoch summary beside b20c256 at
the same epoch: `python workspace/lr_study/joint_epochs.py`.

| Epoch | Hours | LR | Policy top-1 (b20c256 same epoch) | Top-5 | Win-rate loss | Agreement | Score error | Ownership error | val_loss |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7 | 0.8 | 43.48% (43.32%) | 75.87% | 0.6240 | 59.7% | 4.33 | 0.464 | 3.2012 |
| 2 | 1.4 | 0.8 | 46.16% (46.03%) | 78.91% | 0.5674 | 62.8% | 3.35 | 0.450 | 2.9704 |
| 3 | 2.0 | 0.8 | 47.46% (46.80%) | 80.46% | **0.6787** | 59.0% | **4.91** | 0.453 | 3.0119 |

**Epoch 3: validation win rate and score got worse, training didn't.** The training-side
win-rate loss improved smoothly through the epoch (running mean 0.551 -> 0.547, score
error 3.31 -> 3.23), no NaNs, loss scale steady; only the validation numbers jumped.
Most likely a batch-norm mismatch: validation uses the BN layers' running averages,
which lag weights moving fast at LR 0.8, and the value head's pooled outputs are
sensitive to that (policy and ownership unaffected). **Decision: keep training** - it
usually closes as the LR drops. If validation still lags training after the first LR
cuts, recompute the BN statistics on training data before using a checkpoint (no
retraining needed).

| Epoch | Hours | LR | Policy top-1 (b20c256 same epoch) | Top-5 | Win-rate loss | Agreement | Score error | Ownership error | val_loss |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2.7 | 0.8 | 48.46% (48.38%) | 81.44% | 0.5524 | 64.4% | 3.25 | 0.454 | 2.8277 |
| 5 | 3.4 | 0.8 | 49.24% (48.75%) | 82.18% | 0.5660 | 66.0% | 3.23 | 0.446 | 2.8015 |
| 6 | 4.0 | 0.8 | 50.04% (49.52%) | 82.86% | 0.5310 | 68.0% | 2.89 | 0.450 | 2.7185 |
| 7 | 4.7 | 0.8 | 50.28% (49.84%) | 83.25% | **0.6073** | 62.5% | **4.36** | 0.451 | 2.7855 |

Epoch 7: a second validation-only swing in win rate and score (training side still
improving: win-rate loss 0.525, score error 2.88). It's a recurring pattern at LR 0.8, so
the value head's validation numbers are noisy epoch to epoch until the LR comes down;
judge it by the best / recent trend, not single epochs. The combined val_loss rose
(plateau counter 1 of 3). Recommendation for any checkpoint used in play or in the
lookahead test: recompute the BN statistics on training data first (and compare the
validation value metrics before and after).

| Epoch | Hours | LR | Policy top-1 (b20c256 same epoch) | Top-5 | Win-rate loss | Agreement | Score error | Ownership error | val_loss |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 5.4 | 0.8 | 50.51% (50.77%) | 83.22% | 0.6045 | 64.1% | 3.12 | 0.445 | 2.7764 |
| 9 | 6.0 | 0.8 | 51.11% (51.06%) | 83.93% | 0.5441 | 66.3% | 3.18 | 0.449 | 2.6773 |
| 10 | 6.7 | 0.8 | 51.61% (51.02%) | 84.37% | 0.5319 | 68.0% | 2.81 | 0.444 | 2.6430 |

**Paused at 09:41 (2026-10-03) at the user's request,** right after epoch 10 finished
(10 epochs recorded, `weights.00010.weights.h5`; only the first minute of epoch 11 was
lost). To resume - the launch command plus `--weights`, which restores the optimizer
state, the plateau schedule's state and the data-stream position:

```
docker compose run -d --name joint_b20c256 gpu sh -c "python -m AlphaGo.training.supervised_policy_trainer workspace/models/model_joint_b20c256_g7.json workspace/prod_60m/shards workspace/runs/joint_b20c256 --minibatch 512 --epochs 1000 --steps-per-epoch 5000 --validation-length 100000 --mixed-precision --seed 90001 --learning-rate 0.8 --warmup-steps 2000 --warmup-start-lr 0.0001 --lr-schedule plateau --plateau-factor 0.5 --plateau-patience 3 --plateau-cooldown 3 --plateau-min-lr 1e-5 --plateau-min-delta 0.005 --verbose --weights weights.00010.weights.h5 >> workspace/runs/joint_b20c256.log 2>&1"
```

Epoch 8: training side best yet again (win-rate loss 0.522, agreement 70.1%, score 2.83);
validation win-rate loss still high (2nd epoch in a row), score recovered. Combined
val_loss hasn't beaten epoch 6 for 2 epochs: one more, and the plateau schedule cuts the
LR to 0.4.

Epoch 4 recovered (training side: win-rate loss 0.539, score error 3.09), so epoch 3
was a transient. The win-rate loss is already at v2's level (0.550) after 2.7 hours.

## Checks on joint epoch 10 (after the pause)

### Batch-norm refresh: stale averages are (at least part of) the swings

`workspace/sample_ratio/bn_refresh.py`: validate on 50K val positions, re-estimate every
BN layer's running mean/variance from ~100K training positions (training mode, no weight
updates), validate again. Refreshed weights: `workspace/sample_ratio/joint_e10_bnrefresh.weights.h5`.

| Epoch 10 | As saved | BN refreshed |
|---|---|---|
| Agreement on who's ahead | 67.9% | **71.5%** |
| Score error | 2.78 | **2.59** |
| Win-rate loss (true; see below) | 0.571 | **0.554** |
| Policy top-1 | 51.37% | 51.60% |
| Ownership error | 0.445 | 0.442 |

**Refresh every checkpoint's BN statistics before testing or deploying it.** Refreshed,
epoch 10 beats v3 on agreement (71.5% vs 71.2%) and score error (2.59 vs 2.66).

### Correction: logged win-rate losses are understated by 0.932

Keras averages a sample-weighted loss over all positions, including the ~7% without a
KataGo annotation (weight 0), so every logged win-rate loss here (v1-v3 and the joint
run, training and validation) is the true mean over labeled positions x 0.932 (bn_refresh's
own 0.571 vs the trainer's 0.532 for the same checkpoint). The metrics (agreement, MAE)
are proper weighted means and unaffected. **Comparisons between runs stand** (same factor
everywhere); comparisons with the baselines (0.693 always-50%, 0.409 perfect) need the
logged number / 0.932 - e.g. v3's 0.522 is really 0.560, 47% of the way from 0.693 to
0.409, not the 61% the logged number suggested.

### Lookahead tests on joint epoch 10, and KataGo as a value judge

All on the same 7 KGS games (804 bot moves); points lost by KataGo's 100-visit judgment
(greedy b20c256: 882). Scripts in `workspace/sample_ratio/` (`lookahead_eval.py`,
`katago_raw_lookahead.py`, `lookahead_report.py`).

| Rule | v3 e8 | joint e10 | joint e10 BN-refreshed | joint e10 own moves | b20c256 moves + **KataGo raw value** | KataGo raw moves + value |
|---|---|---|---|---|---|---|
| greedy (policy top move) | 882 | 882 | 882 | 1,011 | 882 | -60 |
| win rate, top 5 | 948 | 1,351 | 1,400 | 1,696 | 923 | 491 |
| win rate, top 5, switch if +0.06 | 827 | 946 | 878 | 1,140 | 611 | -14 |
| **score, top 5** | 938 | 1,162 | 1,138 | 1,496 | **136** | -58 |
| **score, top 10** | 1,176 | 1,454 | 1,427 | - | **39** | - |
| score, top 5, switch if +2 pts | 748 | 956 | 974 | 1,237 | 299 | -74 |
| value + 0.5 x policy, top 5 | 723 | 736 | 748 | 914 | 664 | -60 |

- Komi 7.5 instead of the games' 0.5 didn't help the joint network (not tabled), so komi
  isn't why it trails v3. At the 90 blunders, the BN-refreshed joint e10 prefers
  KataGo's move over the blunder 62% of the time (v3: 72%); its own policy repeats 59 of
  the 90 blunders.
- **KataGo's raw network (1 visit, no search) as the judge of b20c256's own top 5: ranking
  by score cuts points lost from 882 to 136 (top 5) or 39 (top 10).** Ranking by win
  rate barely helps (923): in these lopsided handicap games every candidate's win rate is
  near 0 or 1. Caveat: the judge (KataGo 1 visit) and the referee (KataGo 100 visits)
  are the same network family, so this is optimistic.
- **Correction to an earlier conclusion:** one-ply lookahead does not have a low ceiling.
  KataGo's own candidates gain nothing from it because its policy is already near
  perfect; with b20c256's candidates, a strong enough score judgment recovers almost
  everything. **Our value heads' bottleneck is the score output:** v3's score rules are
  no better than greedy despite a 2.66-point average score error - picking between
  candidates needs the differences between them right.
- Decision (user, after this): **continue the joint run, with the score loss weight raised
  from 0.5 to 1.0.** The score output predicts score / 20, so a typical 3-point error
  pulls with ~0.15 - comparable to the win-rate head's ~0.2 at equal weight; at 0.5 it
  was half that. Not higher yet: the range test showed the score head blowing up at high
  LR, and weight 1.0 already halves its margin at LR 0.8 (~4.5x -> ~2.2x). Revisit at the
  first LR cut. Track progress with this test's score rules on BN-refreshed checkpoints.
- **Resumed 15:11** from `weights.00010.weights.h5` with `--score-weight 1.0` (stream at
  position 25.6M, plateau state best 2.6430 / wait 0, optimizer restored). The combined
  val_loss now carries the score loss at weight 1 instead of 0.5 (~+0.02), so it reads
  slightly higher than before the change against the plateau's best of 2.6430 - the next
  cut may come an epoch or so earlier than it otherwise would.
- Epoch 11 (7.4 h, LR 0.8, first with score weight 1.0): policy 51.82% (b20c256: 51.59%),
  top-5 84.39%; validation value swung again (win-rate loss 0.597, agreement 63.4%,
  score error 4.13 - the BN effect); training side 0.517 / 71.2% / score 2.89 (score up
  from 2.76, one epoch after the weight change). val_loss 2.7349 not comparable with
  earlier epochs (score term doubled). **Stopped at 15:52 after epoch 11** (user's
  decision) to move to the score network (`SCORE_NET_PLAN.md`); all checkpoints kept, so
  it can be resumed (same command, `--score-weight 1.0`, `--weights weights.00011.weights.h5`).

## Decisions made while unattended

- **v3 stopping rule:** stop when the val win-rate loss improves by less than ~0.002 per
  epoch for 2 epochs after a learning-rate cut, or after about 3 more hours, whichever
  comes first.
- **No GPU or heavy CPU work alongside v3** (user's instruction): coding and small unit
  tests only until it stops.
- **From-scratch run settings,** following the original b20c256 run
  (`workspace/runs/newres_b20c256/metadata.json`): `--epochs 1000` (non-ending), 5,000
  steps x 512 per epoch, mixed precision, all 8 symmetries, SGD + momentum 0.9, 2,000-step
  warmup from 1e-4, plateau schedule (factor 0.5, patience 3, cooldown 3, min_delta
  0.005, min LR 1e-5). Peak LR from a new LR range test on the combined loss. Plateau
  monitor: the combined validation loss.
- **Left out of the from-scratch run:** pass output, rules input (as in
  `JOINT_TRAINING_PLAN.md`).
- **Checkpoints:** keep every epoch (~270 MB each; 63 GB free at the start).
- **Loss weights for the joint run:** policy 1, value 1, score 0.5, ownership 1 (the
  plan's starting point; KataGo's own weights differ in form, so no direct copy).
- **Ownership labels at 10 KataGo visits** per final position: a finished game's ownership
  needs little search, and it keeps the GPU time short.
- **Joint trainer = the main trainer extended** (not a separate script), so the LR range
  test and resume logic are shared with the policy runs.
- **Peak LR rule** (LR_STUDY_PLAN.md's method, without the 3-hour held-run comparison):
  run the same range test as b20c256's (`b20_512_range_s90001`: 1e-3 -> 10 over 5,000
  steps after a 250-step warmup, minibatch 512, mixed precision). b20c256's was stable to
  ~5, and held runs then picked 1.6 over 0.8 / 0.4. So: if the joint network's sweep is
  stable to at least ~5 (no NaN, no loss-scale collapse, no loss upturn), use **1.6**;
  if it turns up earlier, use about a third of where the smoothed loss starts rising,
  rounded down to 0.4 / 0.8 / 1.6.
