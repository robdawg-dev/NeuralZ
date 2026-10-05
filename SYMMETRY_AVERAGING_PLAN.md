# Plan: symmetry averaging in `go_server.py`, and a check of whether it helps

Drafted 2026-10-01. Not yet implemented. Open decisions are marked **Decide**.

## The idea

A Go position means the same thing under all 8 symmetries of the board - the 4 rotations
and 4 reflections training already uses as augmentation (`BATCH_TRANSFORMATIONS` in
`AlphaGo/training/shard_stream.py`: `noop`, `rot90`, `rot180`, `rot270`, `fliplr`,
`flipud`, `diag1`, `diag2`). The network only learns this approximately, so a single
evaluation can have orientation-specific quirks (e.g. reading a shape slightly better in
one corner than its mirror image). Today the bot evaluates each position once, as given.

**Symmetry averaging**, for one position:

1. Apply each of the 8 symmetries to its feature planes.
2. Evaluate all 8 views in one batch.
3. Map each view's 361 probabilities back to the original orientation with the inverse
   of that symmetry - via the point permutation each symmetry defines
   (`shard_stream.symmetry_permutations`): for view k, `original[:, perm_k] = view_k`.
4. Average the 8 maps, then normalize over the legal moves and choose a move as now.

Choices within that:

- **What to average:** probabilities (arithmetic mean; simple, standard - the starting
  point), or log-probabilities (geometric mean, renormalized; a move needs support from
  all views, so one view's overconfident outlier counts for less).
- **How many views:** all 8 (most consistent), one random view per move (KataGo's
  default during search - no extra cost, removes fixed orientation bias but adds
  randomness), or 2-4 in between.

Expected effects:

- Consistency: a position gets the same evaluation however the board is oriented.
- A small accuracy gain is typical for policy networks - from a fraction of a point to
  about a point of top-1 - but **not measured on this network yet** (see the check).
- Symmetric positions give exact ties (e.g. all four 4-4 points on the empty board).
  Top-k sampling handles that fine.
- The averaged distribution is slightly flatter than a single view's, so `--temperature`
  and `--top-k` (tuned on single views) behave a little differently; greedy play only
  changes in close calls.
- Cost: 8x the computation per move. Measured single-view on the CPU (b20c256, 2 TF
  threads, while training ran): ~180 ms per move. 8 views in one batch is cheaper than 8
  separate calls - perhaps 0.5-1 s per move. To be timed on the bot server.

## 1. The symmetry check (a one-off script in `workspace/`)

Measure before using it in games.

- **Data:** the 60M data set's **test split** (`workspace/prod_60m/shards/test`, 1.2M
  positions, never used for training or validation). A 20,000-position sample - the first
  20,000, a uniform random sample since the shards are pre-shuffled.
- **Weights:** the final checkpoint of `workspace/runs/newres_b20c256`.
- **Compared:**

  | Method | Network calls per position |
  |---|---|
  | Single view (today) | 1 |
  | 8 views, arithmetic mean of probabilities | 8 |
  | 8 views, geometric mean (mean of log-probabilities, renormalized) | 8 |
  | One random view per position | 1 |

- **Measured for each:** top-1 accuracy, top-5 accuracy, and loss on KataGo's moves - the
  same measures as training's validation, so they compare directly with the run's
  numbers.
- **When:** after training ends, on the GPU - 20,000 x 8 evaluations take minutes and
  disturb nothing. (A smaller CPU sample is possible during training, but slow and it
  competes with training's data loading.)
- **Bar:** at 20,000 positions top-1 noise is about +-0.3 points. A gain clearly above that
  is worth the slower move; otherwise keep 1.
- **Not included unless wanted:** bot-vs-bot games with and without averaging, which would
  show the effect on strength directly but cost far more time.

## 2. The server change

- New option **`go_server.py --symmetries {1,8}`**, default **1**.
  - `--symmetries 1` evaluates only the position as given - exactly today's behavior (a
    test will check the output is identical).
  - `--symmetries 8` evaluates all 8 views and averages (arithmetic mean, unless the
    check favors the geometric mean).
  - 2 and 4 left out to start: which subset to use is a judgment call, and the check
    shows whether 8 is worth it at all.
- **Inside:** in the inference thread, each of the N positions in a batch is expanded into
  its K views with the training symmetry functions, all N x K go through the network in
  one call, each view's probabilities are mapped back through the inverse permutation,
  and the K maps are averaged per position. Batching across bots is unchanged.
  `--max-batch` still counts positions (the model call is N x K inputs).
- `/info` reports the setting, so a running server shows what it is doing.
- **No client change** - every bot gets it.

### Tests

1. `--symmetries 1` returns exactly what the current server returns.
2. `--symmetries 8` is orientation-independent: a rotated or reflected input position gives
   the same probabilities, correspondingly rotated or reflected.
3. The averaging matches a by-hand computation (8 separate evaluations, mapped back,
   averaged).
4. On the empty board, symmetric points get identical probabilities.

## Decide

1. Run the check after training finishes, on the GPU? (Recommended.)
2. Write the server option before the check (defaulting to 1, so nothing changes), or only
   if the check shows it is worth having?
3. Arithmetic or geometric mean - let the check decide?

## Results of the check (2026-10-01)

`workspace/symmetry_check.py`: 20,000 positions from `workspace/prod_60m/shards/test`, final
b20c256 weights (`weights.00067`), GPU (3 minutes for 160,000 evaluations). Differences
are paired against the single view (same positions), +-1 standard error.

| Method | Top-1 | Top-5 | Loss | Top-1 vs today |
|---|---|---|---|---|
| Single view (today) | 56.92% | 88.87% | 1.3941 | - |
| One random view | 57.15% | 88.80% | 1.3940 | +0.23 +- 0.21 pt (noise) |
| **8 views, arithmetic mean** | **57.95%** | **89.53%** | 1.3497 | **+1.04 +- 0.18 pt** |
| **8 views, geometric mean** | 57.89% | **89.53%** | **1.3475** | **+0.98 +- 0.18 pt** |

- **8-view averaging clearly helps**: about +1.0 pt top-1 (over 5x the noise), +0.67 pt top-5,
  -0.045 loss - more than the last three LR cuts of the training run added together.
- Arithmetic and geometric means are tied within noise; arithmetic chosen (simpler).
- One random view gives no measurable gain.
- Mapping-back check: each view's top move matches the unrotated view's ~82% of the time
  (a wrong mapping would match a few percent). So the single-view top choice depends on
  board orientation in about 1 position in 5 - each view alone scores 56.9-57.5% top-1 -
  which is what averaging smooths out.

**Decision:** implement `--symmetries 8` (arithmetic mean) in `go_server.py`, default 1;
time it on the bot server's CPU before switching the bots over.

## Follow-up: 2- and 4-view averages, and CPU timings (2026-10-01)

Same 20,000 test positions and weights (`workspace/symmetry_subsets.py` scores subsets of
the saved views):

| Views averaged | Top-1 | Top-5 | Loss | Top-1 vs 1 view |
|---|---|---|---|---|
| 1 (today) | 56.92% | 88.87% | 1.3941 | - |
| 2: noop + transpose | 57.39% | 89.28% | 1.3702 | +0.48 +- 0.16 pt |
| 2: noop + rot180 | 57.34% | 89.35% | 1.3663 | +0.43 +- 0.16 pt |
| **4: the rotations** | 57.73% | 89.50% | 1.3534 | **+0.81 +- 0.18 pt** |
| 4: noop, both mirrors, rot180 | 57.85% | 89.44% | 1.3568 | +0.94 +- 0.18 pt |
| 4: noop, both diagonals, rot180 | 57.68% | 89.54% | 1.3556 | +0.77 +- 0.18 pt |
| 8: all | 57.96% | 89.53% | 1.3497 | +1.04 +- 0.18 pt |

Averages over every subset: 2 views 57.56% top-1 (28 subsets), 4 views 57.83% (70), 6 views
57.93% (28). The gain flattens: 1->2 about +0.45 pt, 2->4 about +0.4, 4->8 about +0.15.
Which 4 views makes no difference beyond noise; the 4 rotations were chosen as the
simplest.

CPU time per move (b20c256, one position, median of 15, 16-thread desktop):

| `--symmetries` | All 16 threads | 4 threads |
|---|---|---|
| 1 | 162 ms | 149 ms |
| 4 | 337 ms (2.1x) | 354 ms (2.4x) |
| 8 | 490 ms (3.0x) | 575 ms (3.9x) |

Several views cost far less than proportionally because they run as one batch.

**Implemented:** `go_server.py --symmetries {1,4,8}` (1 = unchanged default, 4 = the
rotations, 8 = all), arithmetic mean, with tests in `tests/test_go_server_client.py`.
**Next:** time it on the bot server's CPU; use 8 if moves stay around a second or less
with several bots active, otherwise 4.
