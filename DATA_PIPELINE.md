# Training data pipeline

How KataGo selfplay SGFs become training shards, what each stage decides, and the
measurements behind those decisions.

Supersedes `SGF_FILTER_POLICY.md` and `NEXT_STEPS.md`, which described an earlier design.
`AUDIT_NOTES.md` and `CONVERSATION_LOG.md` are kept as history and are not maintained.

---

## The stages

```
game_data/                     raw KataGo selfplay SGFs, one game per file

sgf_cull.py       scan     ->  to_delete.txt        files we will never use
                  delete   ->  (frees disk)

sgf_preparation.py scan    ->  manifest.jsonl       one row per file, nothing filtered

select_games.py            ->  train.txt val.txt test.txt    which games, and which split

convert_shuffled.py        ->  train/ val/ test/ shard_NNNNN.h5    feature planes

supervised_policy_trainer_v4.py                      streams the shards
```

Each stage decides one thing and nothing else. Selection is entirely in `select_games.py`;
the converter only builds planes; the trainer only trains.

### Commands, as used for the 40M set

```bash
python -m AlphaGo.preprocessing.sgf_cull scan   game_data workspace/cull
python -m AlphaGo.preprocessing.sgf_cull delete workspace/cull

python -m AlphaGo.preprocessing.sgf_preparation scan game_data workspace/analysis/manifest.jsonl

python -m AlphaGo.preprocessing.select_games \
    workspace/analysis/manifest.jsonl workspace/prod_40m/selection

python -m AlphaGo.preprocessing.convert_shuffled \
    workspace/prod_40m/selection workspace/prod_40m/shards --max-winrate-loss 0.10
```

`sgf_analyze.py` is a read-only survey tool for characterising a new download. It is not
part of the pipeline.

---

## What the corpus looks like

From a 1,794,703-file download (2026-07 KataGo runs, networks s9644M–s9974M):

| | files | share |
|---|---|---|
| not 19x19 (sizes 7–18 and rectangles) | 727,200 | 40.5% |
| `sgfpos` | 270,365 | 15.1% |
| `fork`, `asym`, `hintpos`, `cleanuptraining`, `hintfork` | 158,770 | 8.8% |
| duplicates (byte-identical, same gameHash) | 73 | ~0% |
| **kept: 19x19 `normal` + `handicap`** | **638,295** | **35.6%** |

Of what is kept: `normal` 604,636 games (198.6M moves), `handicap` 33,659 (11.0M moves).
Handicap stone counts are uniform over HA 2–6 at about 1% of games each. There are **no
7–9 stone handicap games at all**: KataGo draws `1 + nextUInt(5)` extra stones on 19x19,
and Black's own first move makes the total 2–6.

---

## Selection criteria

### Game types: `normal` and `handicap` only

An allowlist, not a denylist, so a new game type in a future download is excluded until
someone looks at it. The six excluded types and why:

| type | reason |
|---|---|
| `sgfpos`, `fork` | The `AB` block is a serialized mid-game **board** (median 69 and 50 stones), written in row-major raster order, all-black-then-all-white, with captured stones absent. It carries no move order, so `turns_since` cannot represent it faithfully. |
| `asym` | Asymmetric playouts. Measured over 1,200 asym handicap games: the handicap-receiving side runs **142 visits against White's 373** (2.63x) and blunders **3.6x** more (0.361% vs 0.101% for Black in ordinary handicap games). Worst blunder rate of any type. |
| `hintpos`, `hintfork` | Positions carrying a hinted move, deliberately biased toward it. |
| `cleanuptraining` | Endgame/scoring drill, starting ~106 stones in. |

### Normal games: komi in [5, 9] and opening winrate in [0.3, 0.7]

Our network has **no komi input**, so it learns the average best move across whatever komi
a board appeared with. Komi in KataGo selfplay is deliberately varied (492 distinct values
in `normal` alone, range −192 to 230.5), which is training signal for *its* komi input and
label noise for ours.

The winrate check uses KataGo's own evaluation at its first searched move. It is used
instead of komi alone because fair komi depends on the rule set (territory ≈ 6.5, area ≈ 7),
and because it also drops games whose randomised opening had already decided them.

What this removes, and why it matters:

- **15.3% of `normal` games have negative komi**, median −6.5, and White's median opening
  winrate in those is **0.01**. This is KataGo's `flipKomiProbWhenNoCompensate`: fair komi
  is computed as if White moved first, then applied to a game where Black still moves
  first. Black gets both the first move and the komi.
- Pool after filtering: **337,716 of 604,636** normal games.

### Handicap games: compensated only

Handicap games split into two populations, and the opening winrate separates them cleanly:

| | komi at HA 2 / 4 / 6 | White's opening winrate | share |
|---|---|---|---|
| compensated | ~20 / ~47 / ~75 | ~0.50 | 55% |
| uncompensated | roughly −10 to +10 | **0.00** | 37% |
| partial | in between | 0.01–0.30 | ~4% |

KataGo compensates with `adjustKomiToEven` (a 20-visit search) with probability
`handicapCompensateKomiProb` (0.5–0.6 in its configs). Compensation works out to about 13
points per stone — the real value of a handicap stone.

Only compensated games are used (**18,647** games). Measured against `normal` games that
also start balanced, they are statistically ordinary:

| | White visits | moves losing ≥2 pts | White wins | White's moves while below 5% winrate |
|---|---|---|---|---|
| normal, balanced | 350 | 0.27% | 45.8% | 18.6% |
| handicap, compensated | 331 | 0.37% | 45.7% | 19.4% |
| handicap, uncompensated | 240 | 0.40% | **0.3%** | **98.6%** |

In uncompensated games KataGo treats the whole game as lost, drops both sides to its
240-visit cheap-search floor, and wins 37 of 12,328 games as White. Those are not games
where White is trying to win.

**Accepted limitation:** in compensated games White plays as if the game is even, because
its komi cushion makes it so. On KGS, White giving handicap is genuinely behind. The
network cannot see komi either way, so it learns sound even-game moves on handicap boards.

### Mix and split

5% of positions from handicap, 95% from normal, matching the corpus's natural rate. Enough
exposure to handicap boards without over-weighting a game type whose White plays with a
hidden cushion.

The train/val/test split is **by game** (0.93/0.05/0.02), stratified so each split holds
the same handicap share, and seeded. No game appears in two splits.

---

## Conversion

`convert_shuffled.py` builds feature planes and nothing else. Its only judgements:

- **Passes are never positions** — the policy head has no pass output.
- **A move the engine rejects is not emitted** (multi-stone suicide under `sui1` rules).
  The replay stops there and the positions before it are kept.
- **`--max-winrate-loss 0.10`** drops a position whose move KataGo's own search says gave
  up ≥10% winrate. Measured cost **0.137%**. The replay continues, so the position *after*
  a blunder is kept with the move that punishes it. Off by default; pass it explicitly.

This is the only position filter, deliberately: it is the only one resting on a direct
observation rather than an inference. Note it is blind once the winrate saturates (0.015%
blunder rate in fully-decided games vs 0.238% in barely-decided ones), which is a known
limit, not a reason to add a proxy filter on top.

### The two-pass shuffle

The shards of a split, read in file order, are **one uniformly random permutation of every
position in that split**.

```
pass 1  convert every game; send each position to one of K bucket files chosen at random.
        Buckets live on disk and grow until the last game is done, so each ends up with a
        random ~1/K of positions from ALL games.
pass 2  load each bucket, shuffle it in memory, write it as a shard, delete the bucket.
        Buckets are independent, so several run in parallel.
```

Random bucket assignment, a uniform shuffle inside each bucket, and concatenation give a
uniform permutation: given the bucket sizes, every assignment and every within-bucket order
is equally likely, and each pair maps to exactly one final order.

Buckets are raw fixed-size records, not HDF5. Every feature plane is 0/1, so workers
bit-pack each position (17,328 bytes → 2,166 at 48 planes) and the main process only
appends bytes. Compressing into HDF5 there was the pass-1 bottleneck, because HDF5 calls
cannot run in parallel from Python.

**Bucket size and shard size are separate knobs.** A bucket must fit in a pass-2 worker's
memory to be shuffled (`--positions-per-bucket`, default 100k ≈ 220MB packed); a shard is
only how the result is packaged on disk (`--positions-per-file`, default 1M ≈ 2.5GB), and
several whole buckets go into one. Appending bucket by bucket keeps peak memory at one
bucket however large the shards are, and the position order is identical either way — only
where the file boundaries fall changes. The 40M set below predates this and was written
1:1, giving 373 shards of 252MB.

**Shard contents:** `states` (N,19,19,F) uint8, `actions` (N,2) uint8, `game_id` (N,) int32,
`move` (N,) int16, plus `features` and `conversion_args`. `game_id` indexes
`<split>/games.tsv`, so any position traces back to its SGF and move number.

---

## Training

`supervised_policy_trainer_v4.py` is `v3` with only the data layer replaced. It streams
the shards start to finish, wrapping at the end, with **no shuffle buffer**, no game index
and no split logic. Each position gets a random symmetry, chosen as a function of (seed,
position in the stream), so a resumed run continues exactly where an uninterrupted one
would have been.

Every pass sees the same order. If varying it between passes ever matters, reintroduce a
small shuffle over shard order rather than a position buffer.

`v3`, `shuffle_buffer.py` and `game_converter.py` / `game_converter_parallel.py` are
untouched, and still work for shards in the older per-game format.

### Why the shuffle moved out of the trainer

The 400k-position buffer in `v3` holds about 1% of a 40M-position set, so positions from
one game cluster together. Simulated over a full pass of a realistic set:

| | 400k buffer | uniform shuffle | measured on the real shards |
|---|---|---|---|
| next position from the same game | 1 in 2,570 | 1 in 131,580 | **1 in ~100,000** |
| distinct games per 1,024-batch | 851 | 1,020 | — |
| handicap share over 100-batch windows | 3.7%–6.1% | 4.8%–5.2% | — |

The buffer's batch-level handicap mix was already close to ideal; what it could not fix was
same-game clustering and slow drift in the mix.

---

## The 40M set

```
selection   119,047 normal + 6,317 handicap games        seed 20260922
            train 116,589 / val 6,268 / test 2,507 games, ~5% handicap in each

conversion  train 37,200,866 positions in 373 shards     winrate drops 0.137%
            val   1,994,800 positions in  20 shards
            test    799,552 positions in   8 shards
            total 39,995,218 positions, ~96GB
```

Verified after conversion: every position appears exactly once, positions traced back to
their SGFs re-convert to identical tensors, each ~99.5k-position shard draws from ~67,000
different games, and handicap sits at 4.8–5.0% in every shard.

First run on it (`benchmarks/_restower_b15c192_v4shuf40m_mb1024_lr1p6_seed90001`,
LR 1.6, batch 1024): 95 epochs, best **val_loss 1.4836**, val_accuracy 0.5558. The earlier
run on the old pipeline reached 1.5608 at epoch 74. **The two are not directly comparable**:
the training and validation sets differ, and removing blunders and noisy game types lowers
the loss a run can reach regardless of whether the network plays better.

---

## Verified against KataGo's source

Checked in `KataGoFresh` (`cpp/program/play.cpp`, `playutils.cpp`, `playsettings.cpp`):

- **`gameHash` is random**, not derived from the moves, so a repeated hash means a
  duplicated file rather than a repeated game.
- **Handicap stones are chosen by the policy net at temperature 1.0**, not placed on fixed
  points. That is why they land on the corner hoshi ~40% of the time and on neighbouring
  points otherwise. KGS uses the fixed pattern, so the exact KGS configuration is covered,
  plus a neighbourhood around it.
- **Komi is built up in layers:** a per-rules fair komi from an empty-board search
  (`komiAuto`), Gaussian noise (stdev 1.0), a 5–6% chance of stdev 12, a 0.5% chance of
  stdev 45, integer komi allowed half the time, and the flip described above.
- **Wide komi is deliberate**, to teach KataGo's own komi input: *"vary komi to better
  learn komi and large score differences that would never happen in even games."*
- **`gtype=handicap` excludes asymmetric games** — any game with a playout advantage is
  relabelled `asym` — so `handicap` games always have equal search on both sides.

The distributed run's settings come from the server and are not in the repo. Two signs the
live values differ from the shipped defaults: the flip rate looks like ~0.13–0.15 rather
than the code default of 0.25, and a handful of games exceed the noise bounds the configs
imply.

---

## Settled, with the evidence

| decision | why |
|---|---|
| Handicap stones are recorded as played moves in `turns_since` | They really were placed in sequence immediately before White's first move, and the GTP path builds the identical history at play time (`place_handicaps` → `place_handicap_stone` → `do_move`). Pushing them into the "age ≥ 7" plane would describe a board where stones were played 7+ turns ago followed by 7 turns of nothing — a state that cannot occur. |
| No skipping of early positions | The skip existed for `sgfpos`/`fork` blocks, which are now excluded. It was costing the whole handicap opening: plies 0–6 of every handicap game, the exact position a KGS handicap game starts from. |
| Passes consume an age slot in `turns_since` | "Turns since" means turns elapsed and a pass is a turn. The same `Preprocess` runs at training and inference, so no mismatch is possible. Passes are never labels and never appear in any plane. |
| `last_moves` is not used | Absent captures it is identical to `turns_since` planes 0–4; they diverge on 7.30% of positions overall but only 0.47% before move 50. Adding it changes the plane count and forces a retrain from scratch. |
| No hopeless-position filter | It would drop 23.4% of positions on a proxy we cannot observe (winrate loss collapses to zero in decided games), and what it removes is overwhelmingly endgame — the phase the bot is already weakest at. |
| No visit-count filter | Blunder rate by visits is 0.248% below 200 visits against 0.053% above 1000. Real, but 99.75% of low-visit moves are clean; low-visit moves sit near the raw policy's top choice. |
| `shuffle_buffer.py` has no race | No threads, locks or pools; both yield points copy defensively; `model.fit` consumes one iterator. The documented 2-in-8 collapse rate is not explained by it. |
| `file_offsets` are validated at load | Gaps, overlaps and trailing rows were all silent. Now a startup error naming the shard. |

---

## Open

- **Komi conditioning.** ~13.9% of the corpus is live, well-played, and labelled for a komi
  the network cannot see. A tighter band or a komi plane would address it; both cost either
  throughput or a retrain from scratch. Currently left alone.
- **A fair comparison between networks.** The only sound way is to evaluate both on the
  same held-out positions, e.g. this set's `test/` split, which neither was trained on.
- **Search at play time.** The bot is a raw policy net with no search. Even shallow MCTS is
  usually worth more than any data filter here.
- **Two-pass endings.** `run_gtp_player.py` records 0 of 50 matches reaching a clean
  two-pass end, leaving dead stones uncaptured. A play-path problem, not a data one.
- **Bit-packed planes end to end.** The GPU needs floats only at the first convolution.
  Keeping planes packed to the GPU would cut the validation array from 4.2GB to ~130MB and
  the per-batch transfer from 71MB to 2.2MB. Unlikely to matter for speed; not measured.
- **Turn distribution.** ~42% of positions are past move 200 and ~5% are openings. Bigger
  than any filter debated here, and untouched.
- **7–9 stone handicap.** KataGo never generates it, so KGS games at those handicaps have
  no matching training data.
