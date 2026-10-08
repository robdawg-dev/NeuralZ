# Measurements

The numbers behind the project's defaults and design choices, each with where it came from.
The code comments that rely on them point here.

The scripts and raw outputs live in `workspace/`, which is gitignored: they are one-off
studies on local data, not part of the project. So each entry says how far it can be
checked:

- **Checked**: matched against a saved output in `workspace/` (2026-10-07).
- **Not re-checked**: the source script exists, but its result was printed rather than
  saved. The figure is as recorded when the study was run.

Before relying on an unchecked figure for a new decision, re-run its study.

## Playing

### Sampling in the opening (`--sample-ratio 0.5`, `--sample-moves 20`)

The bot samples among the "close calls" (moves at least half as likely as its top move)
for its first 20 own moves, then plays its top move.

- **Cost in move quality** (checked): over 95,111 positions from 300 games, b20c256's top
  move matched KataGo's 56.5% of the time. Sampling at ratio 0.5 matched 54.9% (1.6 points
  lower). Sampling triggered on 38.6% of positions, with 2.6 candidates on average. Lower
  ratios cost more: 0.3 cost 3.8 points, 0.1 cost 8.0.
  Source: `workspace/sample_ratio/step1_b20c256.txt`.
- **Variety against a replaying opponent** (checked): against a fully greedy copy of
  itself, which replays the same moves every game, b20c256 at ratio 0.5 for 20 moves
  produced 31 distinct games out of 100 as Black and 8 out of 100 as White. Sampling at
  ratio 0.5 on every move produced 100 of 100; the earlier top-k sampler (`current` in the
  output) produced 12 and 3. Source: `workspace/sample_ratio/step2_3_b20c256.txt`
  (`replay_attack.py`), run with an earlier version of the player.
- **Strength cost** (checked, re-run 2026-10-07): none. With the current player (exactly
  `go_client.py`'s defaults) against a fully greedy copy of itself, b20c256 won 234 of 400
  games, 58.5% (±2.5): 107 of 200 as Black, 127 of 200 as White. Winners are KataGo's; the
  built-in count disagreed on 2 games. All 400 ended naturally (median 391 moves).
  The ±2.5 understates the uncertainty: the greedy side replays its moves, so the 400
  games hold only 141 distinct ones (75 as Black, 66 as White), and repeats count again.
  Read it as "sampling costs nothing", not as sampling being stronger.
  Source: `workspace/sample_ratio/replay_attack_current.py`, output
  `replay_current_b20c256.json` and `.log`, 10 SGFs per color in `games_current/`.
  The two earlier figures don't hold for today's player: 49.75% over 400 games (recorded
  in `ai.py`, output not saved) and 40% over 200 (`step2_3_b20c256.txt`, an earlier player
  scored by the built-in count).
- **Against a human's repeated line** (not re-checked): sampling took a KGS opponent's
  repeated opening line away by ply 30-40 in every game. Source:
  `workspace/sample_ratio/kgs_spring_analysis.py`.

### No sampling while a stone is in atari

There, a "close call" can be the first step of a failing ladder. Checked against
`workspace/sample_ratio/ladder_check.txt`: in a game against spring on 2026-10-02, White's
20th move was sampled to hn at 21.9% over the top move fl at 25.1%, starting a ladder that
lost the game.

### Ladder guard

The bot never extends a group in atari into a ladder the engine reads as dead. The network
has the ladder feature as an input plane and still ran dead ladders: 15-19 extensions at 3-8
points each in spring's wins of 2026-10-02, 03 and 05 (not re-checked; source
`workspace/sample_ratio/ladder_check.py`, `ladder_features.py`).

### Move limit (`--max-moves 800`)

With a 300-move cap, 0 of 50 games between b10c128 and the 2016 network reached a natural
two-pass end, leaving dead groups that skew the built-in count; with 800, 100 of 100 did
(2026-09-12; not re-checked, as the match directories are no longer on disk). Consistent
with that, every one of the 1,000 games in the sampling study above, all at an 800-move
cap, ended naturally (checked, `step2_3_b20c256.txt`).

## End of game (KataGo judging)

### Dead stones from a small KataGo network

- **Agreement**: on the final positions of 263 scored KGS games, KataGo's b10c128 network at
  1 visit agreed with a large network at 400 visits on 258 dead-stone lists, at ~30 ms per
  query on a CPU. The 263 reference positions are checked
  (`workspace/kgs_scoring/dead_reference.json`); the agreement and timing were printed by
  `workspace/kgs_scoring/dead_stones.py` and not saved.
- **Why not GNU Go**: GNU Go's dead-stone list cost the bot a ranked game it had won
  (checked): NeuralZ05 vs yosh45 on 2026-10-03, recorded B+73.5. Two dead Black stones
  inside White's center were left unmarked, so the center counted for no one; KataGo
  (ownership -1.0 on both) puts White ahead by 3-4. Source:
  `workspace/exploits/spring_yosh.py`.

### When the bot may pass back (`SETTLED_MAX_CONTESTED = 10`)

After the opponent passes, the bot passes back only if at most 10 points are "contested"
(KataGo ownership between 0.3 and 0.9 in absolute value). Over 212 KGS games, finished
boards had 0-2 contested points (median 0) and mid-game positions 12-275, so 10 lets no
mid-game position end, and it stopped exactly the games that had ended on an unsettled
board. Before this check, opponents passing right after move 100 got results off by up to
~90 points. Not re-checked; source `workspace/samples2/contested_dist.py`.

After the deploy (KGS games of 2026-10-06, not re-checked): after an opponent's pass past
move 100 the bot played on once and passed back 167 times, and only 1 of 151 scored
results differed from KataGo's count, marginally. Source: `workspace/kokoyyy/folder_check.py`.

### Closed borders before passing back (`OPEN_MAX = 4`)

The contested check misses one way to lose points: KataGo is sure whose area it is, but the
border isn't closed, so a count gives the area to no one. Opponents passing on the bot's
open border took 23-47 points in 5 of the 916 KGS games of 2026-10-06 to 08 (3 of them by
one opponent, HoreaUrsu); 2 of those were wins turned into losses (checked,
`workspace/exploits/open_borders.py`). So the bot also passes back only with at most 4 of
its own points open, and otherwise plays the KataGo move that best closes the border.

- **Threshold** (checked, `workspace/exploits/open_calibration.py`, the deployed judge: b10
  at 1 visit): on the 540 pass-backs the contested check allowed, the bot had 0-2 open
  points on 531 boards and 10-47 on 9 - the 5 games above, a 4.5-point loss with 10 open
  points, and 3 wins. Nothing fell between 3 and 9.
- **Closing** (checked, `workspace/exploits/border_replay.py`): on those boards, with the
  opponent passing every time, the border moves left 0-2 open points after 1-8 moves.

## Inference speed

From `workspace/profiling/results_dev_summary.md` (checked): b20c256, 48 input planes,
Docker on the 16-CPU Windows dev PC, CPU only, 2026-10-04. Median ms per call:

| TF threads | batch 1, eager | batch 1, compiled | batch 8, compiled |
|---|---|---|---|
| 1 | 247 | 156 | 1,296 |
| 4 | 173 | 62 | 439 |
| 8 | 143 | **57** | 329 |
| 16 (default) | 173 | 79 | 300 |

- A compiled `tf.function` call is 2.5-3x faster at batch 1 than an eager Keras call,
  which is mostly per-layer overhead. That is why `go_server.py` compiles every batch size
  at startup (`--eager` turns it off).
- The client's own work (feature planes, ladder guard, move choice) is ~1.5 ms per move,
  so there is nothing to gain there.
- This was measured on the dev machine, not the KGS server: re-run there before choosing
  `--threads`.

### Symmetry averaging (`--symmetries`)

Averaging b20c256 over all 8 board symmetries gained about 1 point of top-1 accuracy on
held-out positions; the 4 rotations gained about 0.8 for about half the cost. Not
re-checked; source `workspace/symmetry_check.py`, `workspace/symmetry_subsets.py`.

## Training

### Playoff (2026-10-07, checked)

Every pair of four policies played 30 games (colors alternating, komi 7.5, `go_client`'s
sampling defaults, ladder guard on, no search); KataGo decided the winners. Row's wins
against the column:

| | b20c256 | b15c192latest | b10c128mb1024 | 2016net | Elo |
|---|---|---|---|---|---|
| **b20c256** | - | 25-5 | 22-8 | 26-4 | +482 |
| **b15c192latest** | 5-25 | - | 23-7 | 30-0 | +369 |
| **b10c128mb1024** | 8-22 | 7-23 | - | 24-6 | +235 |
| **2016net** | 4-26 | 0-30 | 6-24 | - | 0 |

Elo is a Bradley-Terry fit over all 180 games (2016net = 0), each gap ±60-80. No two games
of a pair shared even their first 40 moves. Source: `workspace/playoff/` (`run.sh`,
`table.py`; match directories in `matches.txt`).

### Model results

Checked against each run's `metadata.json`:

| Run | Epochs | Wall time | val loss | Top-1 / top-5 |
|---|---|---|---|---|
| `workspace/runs/newres_b20c256` (epoch 67) | 67 × 2.56M positions | 42.9 h | 1.391 | 57.3% / 89.2% |
| `benchmarks/_restower_b15c192_v4shuf40m_mb1024_lr1p6_seed90001` (epoch 94) | 95 | 38.6 h | 1.484 | 55.6% / 87.4% |
| `play_tests/models/b10c128mb1024` (epoch 125) | 125 | 23.3 h | 1.669 | 51.8% / 84.2% |

### Trainer settings

Recorded where the setting is defined; not re-checked:

- **XLA** (`jit_compile=True`): 102.5 ms per step against 113 ms without
  (`supervised_policy_trainer.py`).
- **`--plateau-min-delta 0.005`**: Keras's default of 1e-4 is below this project's
  epoch-to-epoch validation-loss noise, so noise alone kept resetting the patience count.
- **`--plateau-factor 0.5`**: Keras's default of 0.1 was harsher than training here
  tolerated well.
