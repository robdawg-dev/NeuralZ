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
- **Cost in KGS games** (not re-checked, 2026-10-09): over the bot's first 20 moves in
  100 post-deploy games, sampling was possible at 39% of positions (2.8 candidates on
  average); each candidate weighted by its chance of being picked and judged by KataGo
  against the top move, sampling costs +0.05 points per sampled position, **+0.35 per
  game**, rising a little later in the window (-0.00 for moves 1-5, +0.08 for 16-20). An
  earlier pass over the moves actually sampled in 916 games agreed: +0.10 per sampled
  move, +0.4 per game, none 10+ points worse than the top move. Sources:
  `tools/sampling_audit.py`, `workspace/sampling_cost/`.

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

Since the deploy, no dead-ladder extension in 916 KGS games (`tools/game_report.py`). One
known false positive: the guard removed the only good move (NeuralZ02 vs tugkan,
2026-10-07, move 166) and the bot lost 66 points - the reader saw that the stones could be
captured, not that capturing them would cost the opponent more. Open; see
[LADDER_ISSUE.md](LADDER_ISSUE.md).

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
- This was measured on the dev machine, not the KGS server: re-run there before choosing
  `--threads`.

### Server settings under load (2026-10-09)

`benchmarks/server_round_trip.py`, native Windows CPU server on the dev PC, b20c256, the
169 sample positions once each; 8 bots means 8 threads sending back to back (the worst
case). Median ms per move, and throughput (one run each, so a few ms is noise):

| go_server settings | 1 bot | 8 bots | 8 bots, positions/s |
|---|---|---|---|
| defaults (`--max-batch 4 --batch-wait-ms 10`, as deployed) | 92 | 223 | 34.5 |
| `--threads 8` | 81 | 244 | 32.5 |
| `--max-batch 8` | 92 | **192** | **40.8** |
| `--threads 8 --max-batch 8` | 81 | 216 | 36.4 |
| `--batch-wait-ms 0` | **69** | 218 | 36.0 |
| `--max-batch 8 --batch-wait-ms 0` | 70 | 261 | 30.1 |

- A lone request waits out the whole batch window: the default costs a bot ~16-23 ms a
  move whenever no other bot is asking at the same moment. `--batch-wait-ms 0` removes
  that and loses almost nothing at full load, since requests that arrive during a model
  call queue up and form a batch anyway.
- `--max-batch 8` helps only when all 8 bots ask at once, and only together with the
  batch window (without it the batches come out uneven, ~5).
- HTTP is now ~0.5 ms a request (1.7 ms under load), with go_client keeping its connection.
- On KGS the bots mostly wait for their opponents, so requests rarely coincide, which
  favors `--batch-wait-ms 0`. But the KGS median was 362 ms a move, above even this
  saturated case, so the KGS machine differs. Measure there before changing
  `start_server.sh`.

### Whole moves (2026-10-08)

Time from `genmove` to the answer, over a 340-move bot-vs-bot game on the dev PC (checked:
`workspace/timing/game.log`, read with `tools/gtp_log.py timing`):

- **Native CPU server: about 96 ms a move; GPU server in the container: about 66 ms**
  (`go_server.py --gpu`, `--batch-wait-ms 0`; TensorFlow has no GPU on native Windows).
- **The client's work grew with the position, unlike the network call.** Building the
  48 input planes took 2.5 ms after move 40 but 33 ms after move 200, and the ladder guard
  0.2 ms and 3.5 ms; the network call stays ~57-60 ms. The cause was not the reading
  itself: every ply of a ladder read (`try_stone`) recomputed the whole legal-move list on
  entry and exit, each point checked against the game's history for superko, which the
  bot's board enforces - so the cost grew with both the reading and the game's length.
  **Fixed 2026-10-09:** the ladder reader skips that recompute (it only uses
  `is_legal_move`); planes, guard decisions and legal moves are byte-identical on all 169
  sample positions. `benchmarks/move_time.py` (sample positions, superko on), before ->
  after: planes median 13.2 -> 2.2 ms at moves 181-240 (p90 35.7 -> 3.4), worst position
  104 -> 5 ms; ladder guard p90 6.6 -> 0.4 ms. The client's work is now ~1-3 ms a move
  throughout the game.
- **The client keeps its connection to the server open (2026-10-09).** A new connection
  per move cost ~12 ms: round trip median 82 -> 70 ms on the native CPU server (169
  positions x 3), i.e. now essentially the model call.
- On the KGS server (NeuralZ02's log, 2026-10-08): median 362 ms a move; KataGo-searched
  moves (border and cleanup moves, 32 visits) about 3 s each.

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

### Other comparisons (2026-10-09, not re-checked)

Neither replaces the playoff - only games, and in the end games against people, measure
strength:

- **Agreement with professional moves** (`tools/pro_agreement.py`, 11 professional games
  from `workspace/famous/real/`, 2,605 moves): top-1 / top-5 - b20c256 51.3% / 82.9%,
  b15c192latest 50.6% / 82.6%, b10c128mb1024 49.1% / 81.2%, **2016net 50.0% / 82.6%**.
  The 2016 network, trained on human games, matches professionals as often as b20c256,
  which beats it 26-4: networks trained on KataGo self-play imitate KataGo, not humans.
- **Tactics benchmark** (`tools/tactics_bench.py`, `tools/data/tactics.json`: 57 positions
  from the bot's 197 post-deploy losses where its move lost 20+ points - 45 blunders, 12
  ignored fights): right move found - b20c256 0 (these are its own mistakes), b15c192latest
  6, b10c128mb1024 7, 2016net 10; ignored fights 0-1 of 12 for every network. With the
  ladder guard off, b20c256 gets 1 (the false positive above).

### Model results

Checked against each run's `metadata.json`:

| Run | Epochs | Wall time | val loss | Top-1 / top-5 |
|---|---|---|---|---|
| `workspace/runs/newres_b20c256` (epoch 67) | 67 × 2.56M positions | 42.9 h | 1.391 | 57.3% / 89.2% |
| `play_tests/models/b15c192latest` (epoch 94; run `_restower_b15c192_v4shuf40m_mb1024_lr1p6_seed90001`) | 95 | 38.6 h | 1.484 | 55.6% / 87.4% |
| `play_tests/models/b10c128mb1024` (epoch 125) | 125 | 23.3 h | 1.669 | 51.8% / 84.2% |

### Trainer settings

Recorded where the setting is defined; not re-checked:

- **XLA** (`jit_compile=True`): 102.5 ms per step against 113 ms without
  (`supervised_policy_trainer.py`).
- **`--plateau-min-delta 0.005`**: Keras's default of 1e-4 is below this project's
  epoch-to-epoch validation-loss noise, so noise alone kept resetting the patience count.
- **`--plateau-factor 0.5`**: Keras's default of 0.1 was harsher than training here
  tolerated well.

## On KGS

### Results by handicap (games of 2026-10-06 to 08, checked)

916 games after the KataGo passing deploy, 890 finished: **693-197 (78%)**, every loss on
points. KataGo's expected lead for the weaker side at the first move, and the bot's
results, over the 864 games where the bot gave the handicap or none (7-9 stones: 14
games, not shown; source: `workspace/presentation/handicap_data.json`, from
`handicap_data.py`):

| Handicap | Games | Weaker side's expected lead | Bot won |
|---|---|---|---|
| even | 129 | +1 | 93% |
| 2 | 32 | +18 | 69% |
| 3 | 141 | +31 | 94% |
| 4 | 130 | +43 | 86% |
| 5 | 226 | +58 | 74% |
| 6 | 192 | +72 | 62% |

Each stone is worth about 13 points. In its 4-6 stone losses the bot wins back most of the
handicap (median start -43 to -72, end -6 to -12) but runs out of board; even and small
handicap losses come from big mistakes instead, 5-6 moves of 10+ points per game
(`tools/game_report.py` on the same games).
