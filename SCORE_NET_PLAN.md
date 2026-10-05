# Score network plan

A **policy + score network** fine-tuned from b20c256 (epoch 67): its score head rates the
positions after its top candidate moves, for one-ply lookahead, while its policy is kept
at epoch 67's level by continuing to train it on the moves played.

## Why

On the 7 KGS test games (804 bot moves, points lost judged by KataGo at 100 visits;
`JOINT_TRAINING_LOG.md`):

- b20c256 greedy: **882** points lost.
- b20c256's top 5 ranked by **KataGo's raw network score** (1 visit, no search): **136**;
  top 10: **39**. By win rate instead of score: 923 (win rates saturate in lopsided
  handicap games).
- b20c256's top 5 ranked by our own value heads' score (v3): 938 - no better than greedy,
  though v3's average score error is only 2.66 points. **Lookahead needs the score
  differences between sibling positions right, and our heads never trained on that.**

So: train a network to reproduce KataGo's raw score on exactly the positions the
lookahead sees.

## Decisions (2026-10-03)

| | |
|---|---|
| Start point | b20c256 epoch 67 (all its weights), plus a **fresh** value/score head (small initial weights) |
| Network role | **one network for both (option A):** all blocks trainable, with the **policy loss kept on** (ordinary shard positions and their real moves) so the policy stays at epoch 67's level - the v3 recipe, under which the policy held 57%. The bot loads this one network. If the policy slips, the same run can drop the policy loss and become a score-only copy beside the original epoch 67 (option B) |
| Source positions | **KataGo self-play only**, sampled **uniformly** from `workspace/prod_60m/shards` (handicap share as is, 5%). KGS games are evaluation-only |
| Candidates | b20c256's **top 8** moves per source position |
| Labels | **KataGo's raw network, 1 visit**: score lead and win rate for the player to move, on each position after a candidate (and on the source position) |
| First batch | **300K** source positions x 8 = **2.4M** labeled positions (~2-3 h of KataGo); a held-out set from the val shards for validation |
| Referee | **a different KataGo network** at 100 visits for the final evaluation (e.g. `kata1-tf3-b11c768` in `.katrain`), so the labeler and the judge aren't the same network; the existing referee numbers kept for comparison |

## Pipeline

1. **Positions (container, GPU + CPU):** sample source records from the train shards (and
   ~10K from val); replay each one's SGF (`games.tsv` + the record's `move`) to rebuild
   the position; b20c256's top 8 moves (batched on the GPU); play each, build its feature
   planes. Written as shards in groups of 8 siblings (packed planes, komi for the player
   to move, group id, candidate rank, b20c256's prior), plus a query list for KataGo
   (SGF path, move index, candidate move). Parallel workers for the replays and features.
2. **Labels (native KataGo):** one 1-visit query per position after a candidate (and per
   source position), queries fed from a thread (as `lookahead_katago.py`); score and win
   rate turned to the player to move's side; written in the shards' order.
3. **Merge:** the labels as sidecars beside the new shards, checked row for row.

## Training

- Model: b20c256 epoch 67 + a score/value head (as `PolicyValueNet.from_policy`'s, fresh),
  all layers trainable. Outputs: policy (kept), score (primary), win rate (secondary).
- **Batches mix two streams:** whole sibling groups for the score/win-rate losses (policy
  weight 0) and ordinary shard positions with their real moves for the policy loss
  (score/win-rate weight 0). Validation tracks policy accuracy against epoch 67's 57.3%.
- **Why all blocks:** in the value fine-tunes, frozen -> top 4 -> all blocks gave win-rate
  loss 0.571 -> 0.550 -> 0.522, all blocks learning fastest: b20c256's move-trained trunk
  lacks "who's ahead by how much" features; the policy loss keeps the policy where it was
  (v3: 57% throughout). **Risk: overfitting** - 2.4M labels but
  only 300K distinct source positions (siblings are near-identical boards) for 23M
  trainable weights. So validation uses held-out *source positions* (val shards), and
  training stops when it stops improving. Fallbacks if it overfits early: unfreeze only
  the top 8 blocks, and/or scale up the labels.
- Batches made of whole sibling groups (e.g. 64 groups x 8 = 512 positions).
- **Loss:** Huber on the score (as now) **plus a sibling-ranking term**: within each group,
  the predicted scores minus their group mean should match KataGo's minus theirs (Huber
  on the centered differences) - what the lookahead uses. A small BCE on the win rate.
- Adam, short warmup, plateau schedule (`value_head_trainer.py`'s); validation on the
  held-out groups.
- Batch-norm statistics refreshed on the final checkpoint before evaluation
  (`workspace/sample_ratio/bn_refresh.py`'s approach).

## Data built (2026-10-03)

`workspace/score_net/b20top8/`: train 300,000 sources x 8 = 2.4M siblings (built in 29
min), val 10,000 sources (held-out source positions, labeled in 103 s at 1 visit).
Perspective checked: the best sibling, seen from the source mover's side, matches
KataGo's own annotated score for the source (correlation 0.983, median difference 0.48
points over 187 groups). Labels are for the sibling's player to move (the opponent), so
the source mover's best move is the sibling with the **lowest** label.

Baselines on the val groups - the bar the network has to clear:

| | |
|---|---|
| b20c256's top move is KataGo's best of the 8 | 44.4% (random 12.5%) |
| KataGo's best minus the top move | mean 0.58 points, median 0.03 |
| groups where the best beats the top move by > 2 / > 5 points | 6.2% / 2.0% |
| centered score error if all 8 are guessed equal | 1.91 points |

So in self-play positions the policy's first choice is usually fine; the useful signal is
the ~6% of groups with a clearly better alternative.

## Training run `workspace/runs/score_b20_v1`

b20c256 epoch 67 + fresh value/score head (48 / 112), all layers trainable; batches of 32
sibling groups (256 rows) + 256 policy rows; Adam 1e-4, 500 warmup steps; mixed
precision; 2,000 steps per epoch (~17 min; one pass over the 300K groups ~ 4.7 epochs).

| Epoch | Best-move agreement (policy 44.4%) | Centered score error (1.91) | Spearman | Policy acc val / train |
|---|---|---|---|---|
| 1 | 31.0% | 1.98 | 0.318 | 57.54% / 57.12% |
| 2 | 31.9% | 1.92 | 0.333 | 57.38% / 56.96% |
| 3 | 30.6% | 1.85 | 0.343 | 57.27% / 56.95% |
| 4 | 32.8% | 1.82 | 0.363 | 57.15% / 56.89% |
| 5 | 32.4% | 1.80 | 0.360 | 57.16% / 56.86% |

**Lookahead test at epoch 5** (7 KGS games, points lost; greedy 882):

| Rule | v3 e8 | score net e5 | e5 BN-refreshed | KataGo raw judge |
|---|---|---|---|---|
| score, top 5 | 938 | 678 | **599** | 136 |
| score, top 10 | 1,176 | 631 | **575** | 39 |
| win rate, top 5 | 948 | 628 | 627 | 923 |
| win rate, top 10 | 1,360 | 607 | **593** | 988 |
| value + 0.2 x policy, top 5 | 767 | 671 | 656 | 585 |

**A 32-35% cut in points lost** (previous best: v1's 713, -19%), and the plain "pick the
best" rules now beat greedy. The sibling metrics understate this: most self-play groups
are near-ties (median gap 0.03 points), so exact best-move agreement is largely noise,
while the KGS blunders have 10-30-point gaps the network now catches. **The lookahead test
is the metric to track.** BN refresh: helps the score rules (678 -> 599) but slightly
worsens validation on ordinary positions (score error 3.10 -> 3.30) - mixed, unlike the
joint run; refresh before deploying. Resumed at 19:45 from epoch 5 (fresh optimizer,
warmup again).

| Epoch | Best-move agreement | Centered score error | Spearman | Policy acc (val) | Val score loss |
|---|---|---|---|---|---|
| 6 | 33.2% | 1.77 | 0.372 | 57.09% | 0.0418 |
| 7 | 33.1% | 1.74 | 0.378 | 57.38% | 0.0421 |
| 8 | 34.2% | 1.74 | 0.382 | 57.46% | 0.0419 |
| 9 | 33.5% | 1.72 | 0.393 | 57.29% | 0.0424 |
| 10 | 33.7% | 1.70 | 0.392 | 57.12% | 0.0429 |

**Lookahead test at epoch 10** (points lost; greedy 882):

| Rule | e5 BN | e10 | e10 BN | KataGo raw judge |
|---|---|---|---|---|
| score, top 10 | 575 | 532 | **502** | 39 |
| win rate, top 10 | 593 | 515 | **512** | 988 |
| score, top 5 | 599 | 586 | **568** | 136 |
| win rate, top 5 | 627 | 551 | **539** | 923 |
| score, top 5, switch if +2 pts | 667 | 617 | 627 | 299 |

**Best: 502 (-43%)**, from 575 at epoch 5; every main rule improved by 30-80 points
although the validation score loss was flat (0.042-0.043) - so the lookahead test, not
that loss, decides. Top-10 rules now clearly best. LR left at 1e-4 (plateau rule never
triggered: the centered error improved nearly every epoch); decided with the user to
keep it until the epoch-15 test and cut to 5e-5 only if the 10 -> 15 gain is much
smaller than 5 -> 10. Resumed 21:23 from epoch 10. For future runs: give the plateau
rule a minimum improvement (e.g. 0.01 points), as the policy runs' min_delta.

| Epoch | Best-move agreement | Centered score error | Spearman | Policy acc (val) | Val score loss |
|---|---|---|---|---|---|
| 11 | 35.2% | 1.71 | 0.388 | 57.67% | 0.0399 |
| 12 | 34.1% | 1.67 | 0.397 | 57.60% | 0.0396 |
| 13 | 34.5% | 1.70 | 0.385 | 57.31% | 0.0396 |
| 14 | 35.2% | 1.66 | 0.397 | 57.34% | 0.0391 |
| 15 | 34.1% | 1.67 | 0.377 | 57.42% | 0.0402 |

**Lookahead test at epoch 15** (BN-refreshed; points lost, greedy 882): score top 10 **501**
(e10: 502), score top 5 **551** (568), win rate top 10 562 (512), win rate top 5 583 (539).
Score rules flat to slightly better, win-rate rules ~45-50 points worse (the win-rate
output has half the loss weight and may be drifting; the score rules are what the bot
would use). Ordinary-position validation (as saved) improved: score error 2.75 (e10:
3.07), agreement 69.6% (66.0%). **The 10 -> 15 gain (1-17 points) is much smaller than
5 -> 10 (33-73), so per the agreed rule the LR was cut to 5e-5:** resumed from epoch 15
at 23:03 with `--learning-rate 5e-5` (the automated test had already resumed at 1e-4 at
23:00; restarted).

| Epoch (LR 5e-5) | Best-move agreement | Centered score error | Spearman | Policy acc (val) | Val score loss |
|---|---|---|---|---|---|
| 16 | 34.0% | 1.65 | 0.392 | 57.56% | 0.0373 |
| 17 | 35.0% | 1.67 | 0.391 | 57.42% | 0.0385 |
| 18 | 34.5% | 1.66 | 0.412 | 57.35% | 0.0398 |

Stopped after epoch 18 (decided with the user, instead of 20), 23:54.

**Lookahead test at epoch 18** (points lost; greedy 882):

| Rule | e10 BN | e15 BN | e18 | e18 BN | KataGo raw judge |
|---|---|---|---|---|---|
| score, top 10 | 502 | 501 | **458** | 461 | 39 |
| score, top 5 | 568 | 551 | **516** | 537 | 136 |
| win rate, top 10 | 512 | 562 | **509** | 513 | 988 |
| win rate, top 5 | 539 | 583 | **526** | 536 | 923 |

**The LR cut helped: 501 -> 458 (-48% vs greedy)**, every main rule better; the sibling
metrics barely moved, again. BN refresh no longer helps (461 vs 458) and hurts
ordinary-position validation (score error 2.90 -> 3.03), so **epoch 18 as saved** is the
match / deployment candidate (`play_tests/models/score_b20_v1`). The trainer's cached
lookahead test on e18 BN gives 457 / 537 / 512 (report: 461 / 537 / 513) - near-ties
under mixed precision; close enough to track a run.

**Match, e18 + lookahead top 10 (T 0) vs b20c256** (2026-10-04, stopped after 7 games):
recorded 0 / 7. Two findings:
1. **Recorded match results are distorted by dead stones.** The bots' sensible-move filter
   excludes their own eye-like points, so a dead group whose last liberties are such points
   is never captured, and Tromp-Taylor counts it as alive. KataGo on the final positions:
   games 5 and 7 were lookahead wins (B+47.0, B+32.9; recorded W+7.5, W+0.5), game 6 a
   smaller loss (5.4 vs 19.5) -> **2 / 7**. On the earlier b15c192 vs b20c256 match, 2 of 100
   winners flip (69-31 -> 71-29). From now on KataGo decides match winners after each match
   (`workspace/score_net/katago_winners.py <match dir>`, katago_results.json).
2. **T 0 overrides the policy too often**: 786 of 1,232 decisions (64%), many to
   near-zero-prior moves for predicted gains under 1 point (below the head's ~1.7-point
   error). KataGo (100 visits, all 10 candidates of every decision;
   `match_decisions.py`, `match_katago.py`, `guard_rules.py`) judges per-move points lost
   (incl. a constant tempo term, so only differences count):

| Rule | Match decisions (greedy 1,103) | Overrides | KGS test (greedy 882) |
|---|---|---|---|
| top 10, T 0 (as played) | 1,123 | 786 | 458 |
| **top 5, T 1** | **955** | 322 | 561 |
| top 10 / top 5, T 1, prior >= 0.05 | 942 | 253 | 587 / 591 |
| top 10, T 1 | 978 | 398 | 483 |
| top 3, T 2 | 1,003 | 141 | 723 |
| top 5-10, T 3 | 1,024-1,060 | 85-136 | 673-719 |
| top 5-10, T 5-8 | 996-1,039 | 14-62 | 769-834 |

A blunder guard (the user's framing: trust the policy unless the score head flags its move
as clearly worse) with **T = 1 point** is best on both: ~150 points better than greedy on
the match decisions while keeping ~3/4 of the KGS gain. Net-vs-net games have few policy
blunders to catch, so the match mainly checks that the guard does no harm. **10-game match, e18 + lookahead top 5, T 1 vs b20c256** (sampling 0.5 / 20 both sides,
komi 7.5; `play_tests/sgf/match_score_b20_v1_vs_b20c256_20261004_075904`): **7-3 for the
lookahead** by KataGo (no winner differs from the recorded count this time); wins by 8-59
points, losses by 7-64. 7 of 10 has p ~ 0.17 under even strength - encouraging, not proof;
the 100-game run is the check.

**100-game match, same setup** (`play_tests/sgf/match_score_b20_v1_vs_b20c256_20261004_081606`,
2026-10-04 01:15-03:24): **59-41 for the lookahead by KataGo** (recorded count 55-45; 4
winners flipped by dead stones, all 4 to the lookahead). As Black 27 / 50, as White
32 / 50. P(>= 59 of 100 | even strength) = 0.044 - a real if modest gain over today's bot
in net-vs-net play, where the guard has few policy blunders to catch; the KGS test suggests
the gain against humans is larger.

**Data extension (2026-10-04):** `workspace/score_net/b20top8_ext/train`: 699,995 new
sources x 8 (built in 50 min, seed 20261004, `--exclude` the original 300K: 0 overlap; 5
short sources skipped), labeled in 1.8 h (5.6M siblings, ~850/s). Train with
`--extra-siblings workspace/score_net/b20top8_ext` for 1M groups.

**Oversampling A/B** (`workspace/runs/score_b20_os_ab`: 5 epochs from b20c256 e67 on the
300K, `--oversample 1,2,4,8`, else as the pilot; pass shares by gap tier 0.55 / 0.13 / 0.16
/ 0.16 vs natural 0.875 / 0.064 / 0.041 / 0.020). Lookahead test, points lost (greedy 882):

| Checkpoint | top 5, T 0 | top 10, T 0 | **top 5, T 1** | **top 10, T 1** | top 5, T 2 | top 5, T 3 |
|---|---|---|---|---|---|---|
| pilot e5 (file order) | 678 | 631 | 586 | 542 | 662 | 740 |
| pilot e5 BN | 598 | 575 | 610 | 576 | 665 | 723 |
| **A/B e5 (oversampled)** | 662 | 606 | **569** | **500** | 657 | 740 |
| A/B e5 BN (kept by the trainer) | 647 | 604 | 594 | 526 | 666 | 701 |
| pilot e18 (match model) | 516 | 458 | 561 | 483 | 645 | 714 |

- **Oversampling helps at equal epochs**: 16-42 points better on the T 0 / T 1 rules; under
  the guard rules the oversampled e5 nearly matches the pilot's e18 (569 vs 561, 500 vs 483).
  Sibling metrics at e5 were flat vs the pilot (centered error 1.81 vs 1.80).
- **BN refresh doesn't help under the guard rules** (worse by 24-34 at T 1), though it can
  help T 0. The trainer's keep/discard criterion (score top 10, T 0) kept it - the criterion
  should be the deployed rule.
- **Cost**: oversampled epochs take ~26 min vs ~18 (730 vs 510 ms/step): 32 random group
  reads per step from HDF5. Fix: hold the sibling planes in RAM (~4 GB for 1M groups).

## Next steps (agreed 2026-10-03)

GPU steps run strictly one at a time (KataGo labeling and training each need the GPU);
code is written while the GPU works.

| # | GPU | Alongside (code) |
|---|---|---|
| 1 | Pilot through epoch 18 at LR 5e-5 (stopped early, decided with the user), lookahead test, **then stop the pilot** | lookahead player for `match_networks` |
| 2 | If still stronger than greedy: **100-game match**, pilot epoch 18 + lookahead (top 10) vs today's bot, same sampling window both sides (ratio 0.5, first 20 moves), then greedy vs score-lookahead; colors alternate, komi 7.5; logged (no KataGo cross-check needed: games run to 800 plies or no legal moves, and Tromp-Taylor scoring is accurate) | builder option to exclude already-used source positions |
| 3 | **Extend the data to 1M sources, top 8**: +700K new sources, excluding the existing 300K (~1.1 h build + ~2.1 h label) | trainer improvements + tests: optimizer state in checkpoints; plateau minimum improvement (0.01 points) + `lr_override.txt`; lookahead test inside training every 5 epochs (cached candidate boards + KataGo losses, ~1-2 min); stratified oversampling option |
| 4 | **Oversampling A/B**: 5 epochs from epoch 67 on the 300K with oversampling, vs the pilot's epoch 5 (599 / 575), incl. "worse" switches (~1.5 h) | - |
| 5 | **Production run**: fresh from epoch 67 on 1M sources, oversampling per step 4; BN-refreshed copy tested at the end, kept only if better | - |

Oversampling design: strata = game phase (0-29, 30-99, 100-179, 180-259, 260+) x
handicap/even, each keeping its original share per pass; within a stratum, gap tiers
(KataGo's best minus the policy's top move: <1, 1-2, 2-5, >5 points) sampled with
weights 1 / 2 / 4 / 8, calm groups drawn fresh each pass. Natural shares: <1 87.5%,
1-2 6.4%, 2-5 4.1%, >5 2.0%; 84% of the >5-point groups are at moves 100-259.

**Done alongside step 1 (2026-10-03, tested on CPU):**
- `ScoreLookaheadPlayer` (AlphaGo/ai.py): sampling window as today, then one-ply score
  lookahead over the policy's top k (one batched call), margin optional;
  `match_networks.py --lookahead-a K [--lookahead-margin M]`; model spec `score_b20_v1`.
- `build_sibling_positions --exclude <queries.tsv>` (repeatable): an earlier build's source
  positions are never sampled again.
- `score_net_trainer`:
  - `--extra-siblings DIR`: an extension's train/ groups are added to the original's.
  - `--oversample 1,2,4,8`: the stratified design above.
  - `--lookahead-cache workspace/score_net/lookahead_cache.h5 --lookahead-every 5`:
    the lookahead test (all 804 bot moves x 10 candidates cached; greedy reproduces 882)
    logged as `lookahead_*` in metadata.json.
  - optimizer.NNNNN.npz each epoch (latest 3): a resume carries on Adam's state, the
    learning rate and the plateau's state, without a warmup.
  - `--plateau-min-delta 0.01` (default); `lr_override.txt` in the run directory sets the
    learning rate at the next epoch start.
  - `--bn-refresh-steps 800` (default): when a run ends, the final weights are
    BN-refreshed, and `weights.NNNNN.bn.weights.h5` is kept only if it scores better
    (lookahead score top 10, else the centered error).

Not now: more KGS test games (the user may provide them later; the 100-game match is the
bigger check meanwhile).

## Evaluation

1. **Sibling metrics** on held-out groups: how often the network's best sibling is
   KataGo's best; rank correlation; centered score error.
2. **The lookahead test** (7 KGS games): score rules over b20c256's top 5 / top 8, against
   greedy 882, v3 938, and KataGo-raw-judged 136 / 39.
3. **The blunder set:** how often the network prefers KataGo's move to the blunder (v3:
   72%).
4. **Independent referee** for the headline numbers (a different KataGo network).

## Cost in play, and a single-evaluation alternative

A score head scores the position it is given, so ranking k candidates needs **k+1
evaluations per move**: the current position (policy) and the position after each
candidate (one batched call). An alternative to train on **the same labels**: a
**per-move score head** - a spatial output on the source position predicting the score
after each move (trained on the 8 labeled moves per source, the rest masked), so the bot
needs **one evaluation per move** and looks up its top-k moves' scores. Cheaper on the
CPU by roughly the batch cost of k positions, but harder to learn (the network must infer
each move's consequence without seeing the resulting board). KataGo's data has related
per-move value targets (`whiteQValueTargets`). Worth training both and comparing on the
lookahead test once the labels exist.

## Then

If it gets much of the way from 882 toward ~136: CPU profiling (`CPU_INFERENCE_PROFILE_PLAN.md`)
to pick k and possibly a smaller network trained on the same labels (b10/b15
initialization - the labels are the expensive part and are reused), then the bot
integration (server returns scores for a candidate batch; `ai.py` picks by score among
the top k). If it doesn't: more labels (scale the batch) before changing the method.

## kgs50: 50 ranked KGS games as the lookahead test (2026-10-04)

`workspace/kgs50/`: 50 ranked games of the deployed bot (b20c256 e67, sampling 0.5 / 20;
started >= 02:30 UTC Oct 3), 35 losses + 15 wins stratified by margin x handicap (<= 6), max
3 per opponent, none of the original 7 (`select_games.py`, `manifest.json`; game types from
the KGS archive pages, `kgs_types.json`). 5,453 bot moves, b20c256's top 10 at each
(`candidates.py`), every game position and all 54,530 candidates scored by KataGo at 100
visits under each game's rules and komi (`katago.py games|candidates`, ~1.5 h);
`kgs50_cache.h5` (`build_cache.py`); per-game summary `report_games.txt`. Fast check of any
checkpoint: `evaluate.py <model.json> <weights>...` (~1-2 min each).

Points lost after the sampling window (4,453 moves; greedy 5,746; of which 4,035 on the 262
positions where the bot's actual move lost >= 5 points):

| Rule | pilot e18 (match model) | A/B e5 (oversampled) | pilot e5 |
|---|---|---|---|
| top 5, T 0 | 5,287 | 5,340 | 5,499 |
| **top 5, T 1** | 5,163 | **4,979** | 5,278 |
| top 5, T 2 | 5,021 | 4,999 | 5,032 |
| top 5, T 3 | **4,979** | 5,035 | 5,042 |
| top 10, T 0 | 5,563 | 5,944 | 6,017 |
| top 10, T 1 | 5,242 | 5,591 | 5,635 |

- Gains are real but much smaller than the 7-game set suggested: **-10 to -13%** of points
  lost after the window (7 games: -40 to -48%).
- **Top 5 beats top 10** here (top 10, T 0 is worse than greedy for the e5 checkpoints) -
  the opposite of the 7-game set; T 1-3 all similar, higher T with half the switches.
- On the blunder positions the guard rules recover ~30-40% of the points (4,035 -> ~2,400-
  2,900), but switches elsewhere are about as often worse as better, giving much of it back:
  the score head's noise, not the rule, is the limit.
- The oversampled 5-epoch A/B matches or beats the 18-epoch pilot under the guard rules -
  further support for oversampling in the production run.

## Production run `workspace/runs/score_b20_prod` (2026-10-04, stopped at epoch 5)

b20c256 e67 + fresh head on 1M groups (300K + 700K extension, rechunked to one group per
HDF5 chunk: random 32-group reads 34 ms/step, epochs ~17 min), `--oversample 1,2,4,8`, LR
1e-4, in-training lookahead test on kgs50 every 5 epochs. Early check at epoch 5 (agreed:
must clearly beat the oversampled A/B's 4,979 under top 5, T 1):

| kgs50, after the window (greedy 5,746) | prod e5 (1M) | A/B e5 (300K) | pilot e18 |
|---|---|---|---|
| top 5, T 1 | 5,090 | 4,973 | 5,163 |
| top 5, T 2 | 5,085 | 5,000 | 5,021 |
| top 5, T 0 | 5,278 | 5,329 | 5,287 |
| top 10, T 0 | 5,710 | 5,941 | 5,563 |

**Failed the check; stopped at 19:31 (user's decision).** 3x the data does not speed up early
learning; whether it raises the ceiling later is untested (at equal steps the 1M run has
seen each group a third as often). Checkpoints e1-e5 and optimizer.00005.npz kept - the run
can be resumed with `--resume-weights weights.00005.weights.h5`.
