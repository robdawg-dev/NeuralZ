# Joint policy + value training plan (from scratch)

Train a b20c256 from scratch with a value head (and score and ownership heads) trained
together with the policy, for about 12 hours as a pilot, instead of fine-tuning a value
head onto the policy-only b20c256 (`VALUE_HEAD_PLAN.md`).

## Why

- **Policy accuracy hasn't moved the bot's KGS rank.** b10c128 (51.8% val accuracy,
  ~23 h), b15c192 (55.5%, ~39 h) and b20c256 (57.3%, ~43 h) all play 4d on KGS (greedy).
  The bot loses games to tactical blunders - urgent points ignored for many moves, groups
  dying - not to slightly worse average moves (`workspace/sample_ratio/`, 7 KGS games).
- **A value head with one-ply lookahead targets those blunders.** On the 7 games,
  conservative lookahead rules cut KataGo-measured points lost from 882 to ~713-734
  (-17-19%); picking the best of the top 5 perfectly would cut them to ~0. The gap is the
  value head's accuracy.
- **Fine-tuned value heads are limited by a trunk trained only for moves:** frozen trunk,
  win-rate loss 0.571 (64.8% agreement with KataGo on who's ahead); top 4 blocks
  unfrozen, 0.550 (67.6%); all blocks unfrozen (v3), running.
- **A from-scratch b20 learns fast early:** 52.8% val accuracy at 12 h (past b10's final at
  8.3 h, b15's final at 24 h). If a 12-hour joint network plays at least as well as
  today's bot greedily, its value head may add a lot on top.

## Decide after v3

The all-blocks fine-tune (v3) shows how good a value the b20c256 architecture reaches
when the whole trunk adapts. Run its checkpoints through the lookahead report
(`workspace/sample_ratio/lookahead_report.py`). If v3 gets close to what the lookahead
needs, a from-scratch run may not be worth it yet; if it plateaus near v2, it is.

## Network changes

| | b20c256 today | Joint network |
|---|---|---|
| Trunk | NewResPolicy, 20 x 256, BN | same |
| Global pooling blocks | 5, 10, 15, 20 (`gpool_every=5`) | **7, 12, 17**, as KataGo's b20c256 (`modelconfigs.py`): none in the first 6 blocks, 3 plain blocks after the last. New option: an explicit list of pooling blocks |
| Global inputs | none | **komi** (player to move's side, / 20), as a small vector, dense-projected and added to the trunk after the stem conv (KataGo's way of feeding global features). Optionally rules later |
| Policy head | conv head, 361 outputs, no pass | same (pass output: see open questions) |
| Value head | - | KataGo-sized for b20: 1x1 conv to 48, BN + ReLU, mean + max pooling, dense 112, then **value** (win probability), **score** (lead in points) |
| Ownership head | - | 1x1 conv from the value head's 48 channels to 1 per point, tanh: who owns each point at the end (+1 player to move, -1 opponent) |

Komi enters the trunk, so the policy can use it too. It comes from the value sidecars
(already from the player to move's side), so the shards need no re-conversion. At play
time it comes from the GTP `komi` command (`GTPGameConnector._komi`), sent to the server
with each position.

## Targets

| Target | Source | Status |
|---|---|---|
| Move played | shards (`actions`) | exists |
| Value, score | KataGo comments, `value_NNNNN.h5` sidecars | exists, verified (0 mismatches over 60M records) |
| **Ownership** | KataGo's ownership estimate for each game's **final position**: one analysis-engine query per game (`includeOwnership`), stored per game as int8 (n_games x 361, ~70 MB for the ~190K games across the splits), looked up by `game_id` at batch time | **new**: a few hours of GPU KataGo |
| Komi (input) | sidecars | exists |

Every position of a game shares that game's final ownership, as in KataGo. Ownership is
spatial, so it gets the same board symmetry as the planes and the move label.

## Training

- Extend the main trainer (`AlphaGo/training/supervised_policy_trainer.py`) rather than
  `value_head_trainer.py`. It already has the SGD+momentum recipe, warmup, schedules,
  resume checks and validation tuned for this data. New: multiple outputs and losses,
  the value sidecars and ownership table read alongside the shards, the komi input.
- **Losses:** policy cross-entropy + value BCE + score Huber + ownership MSE (mean over
  points). The weights are to be tuned; starting point policy 1, value 1, score 0.5,
  ownership 1. Positions without a KataGo annotation get value/score weight 0 (ownership
  still applies).
- **Learning rate:** b20c256 used SGD + momentum, LR 1.6 after a 2,000-step warmup, then
  0.8/0.4/0.2/0.1/0.05 (`LR_STUDY_PLAN.md`). The extra losses add gradient, so run a short
  LR range test (`AlphaGo/training/lr_range_test.py`) with the combined loss first.
- **Pilot:** ~12 hours. Since the from-scratch b20 was still at LR 1.6 at 12 h, the
  pilot needs its own shorter schedule (a step down or two before the end), so the
  12-hour checkpoint is annealed rather than mid-high-LR.

## Evaluation (checkpoints every few hours)

1. Validation: policy accuracy (compare with b20c256's 52.8% at 12 h), value loss and
   agreement, score error, ownership error.
2. The lookahead report on the 7 KGS games: greedy points lost (the policy alone) and the
   best conservative rule (policy + value), against today's bot (882 greedy) and v3.
3. If it beats today's bot on (2): KGS games with lookahead, then the same KataGo
   point-loss analysis.

## Work items

1. Ownership targets: script running KataGo's analysis engine on each game's final
   position (threaded feeding, as `lookahead_katago.py`), writing a per-split ownership
   table; a check on a sample that ownership signs match the game result.
2. Model: NewResPolicy options for explicit pooling-block positions and a global-input
   vector; a joint model class with policy, value, score, ownership outputs. Tests.
3. Trainer: multi-output data stream (sidecars + ownership by `game_id`, symmetry on the
   ownership map), losses, metrics. Tests, including an end-to-end tiny run.
4. LR range test, then the pilot.
5. Play: `go_server.py` returns value/score and takes komi; client sends komi; the
   lookahead in `ai.py` (`VALUE_HEAD_PLAN.md` step 5); deploy bundle.

## Open questions

- **Pass output:** the policy has no pass move; the bot passes by rule. A from-scratch run
  could add it (the data has the passes), but it changes endgame behavior - probably a
  separate step.
- **Rules input:** area vs territory scoring shifts values by about a point; the
  training SGFs carry `RU[...]`, and KGS games have known rules. Low priority.
- **Ownership label quality:** final-position ownership from KataGo's analysis is an
  estimate (seki and unsettled groups at game end); spot-check before trusting it.
- **Not planned:** KataGo's search-policy targets (not in our SGFs), score distribution,
  pass-alive territory input planes (would need new Cython features and re-conversion).
