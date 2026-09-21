# Audit notes — deferred items

Working notes from the data-generation / training audit. **Scope of this file:** the
items we deliberately parked to revisit later, plus the reasoning behind everything
already fixed. Regression tests live in `tests/test_pipeline_audit.py` and
`tests/test_gamestate.py`; the suite currently runs **162 passed, 1 xfailed**, and that
single xfail is the one open data defect (`turns_since`, section 1).

Everything below was measured, not inferred from reading code. Reproduction commands are
at the bottom.

---

## 1. Setup stones (`AB`/`AW`) and the move-recency features

### What `AB` / `AW` actually are

SGF separates two concepts:

- **Moves** — `B[db]`, `W[dd]`. A player took a turn. Implies alternation, capture
  logic, ko, and a position in the move sequence.
- **Setup** — `AB[..]` (*Add Black*), `AW[..]` (*Add White*), `AE[..]` (*Add Empty*).
  "These stones are simply on the board now." **No turn was taken, no order is implied.**

Classic uses are handicap stones and tsumego diagrams. KataGo uses them for something
else entirely.

### What KataGo is doing with them in our corpus

Sample from `AlphaGo/tmp_data` (kata1-b28c512nbt selfplay, 60,136 files, all `SZ[19]`):

```
(;FF[4]GM[1]SZ[19] ... HA[0] KM[3.5] RU[koSITUATIONALscoreTERRITORYtaxSEKIsui0] RE[B+5.5]
  AB[pa][ob][qb][dc][hc][kc][pc][qc][rc][nd][qd][cp][kp][eq][nq]     <- 15 black
  AW[nb][cc][nc][oc][od][pd][de][oe][qe][ck][cn][qn][pp]             <- 13 white
  C[startTurnIdx=5,initTurnNum=30,gameHash=...,gtype=sgfpos]
  ;B[db];W[dd];B[dj] ...
)
```

That comment is the key: `initTurnNum=30`, `gtype=sgfpos`. KataGo seeded this selfplay
game from a position **30 moves into another game**, dumped the resulting stone layout in
as setup stones, and played forward. It does this to diversify training positions rather
than always starting from an empty board.

**These are not handicap stones.** Measured over 2,000 files:

| metric | value |
|---|---|
| games with `HA[0]` | 1884 / 2000 |
| games with `AB` | 869 (43%) |
| games with `AW` | 751 (37.5%) |
| median `AB` stones / `AW` stones | 25 / 30 (max 158) |
| `AB` sets that are star-points-only | 25 |
| `AB` sets scattered off star points | 844 |
| games where `HA == len(AB)` | 2 / 869 |

White setup stones alone rule out a handicap interpretation. **The original move order was
discarded** — those ~55 stones were played over ~30 turns by both players and only the
final layout survives. There is no recoverable order to encode.

### The mechanism

`AlphaGo/util.py:_sgf_init_gamestate` places setup stones via
`GameState.place_handicap_stone()`, which calls `do_move()` — so setup stones are pushed
onto `moves_history` and increment `num_handicap`
([game_state.pyx:723-731](AlphaGo/go/game_state.pyx#L723)). Both move-recency features
read `moves_history` with no notion of where real play began, so they report **SGF listing
order as recency**.

Stone *placement itself is correct* — verified on a real game with 15 `AB` + 13 `AW`:
0 misplaced, every colour right. Only the derived recency features are affected.

### `turns_since` — measured impact is small and self-healing

`get_turns_since` ([preprocessing.pyx:70-101](AlphaGo/preprocessing/preprocessing.pyx#L70))
has 8 planes: 0–6 are "played N turns ago", plane 7 is "age >= 7". Measured on a real
game with 28 setup stones:

| move # | setup stones wrongly marked "recent" |
|---|---|
| 0 | 7 of 28 |
| 3 | 4 of 28 |
| 6 | 1 of 28 |
| **7+** | **0** |

Only 7 "recent" planes exist, so once 7 real moves are played they fully occupy planes
0–6 and every setup stone correctly falls into plane 7. The error is bounded at 7 stones,
lasts 7 positions per game, and is fully self-correcting.

Net: **~1% of all positions**, each with ~25% of its setup stones carrying a wrong age
bucket. Corpus-wide check: 41.0% of first-8 positions affected (matching the `AB`/`AW`
rate), which is 41% of ~2.8% of positions.

**The argument for leaving it alone** (raised by Rob, and it largely holds): setup stones
can reasonably be read as "moves that happened first"; the `board` planes already carry
all positional information; the error decays to zero by move 7; and — see below — the
convention is identical at training and inference time, so it is an arbitrary-but-
consistent encoding rather than a train/test mismatch.

**Counter-argument:** these aren't handicap stones but a discarded-order snapshot, so the
invented ordering is pure noise on 7 stones. If changed, the honest convention is "setup
stones are old" (plane 7) — which is what they become by move 7 anyway.

**Current verdict: minor. Left as-is pending a decision.**

### `last_moves` — the one that actually looks wrong

`get_last_moves` ([preprocessing.pyx:103-124](AlphaGo/preprocessing/preprocessing.pyx#L103))
is a newer feature (5 planes, one per last 5 moves) added to mimic KataGo. It is **not in
the default feature list** and is not in the shards currently used for training. Measured:

```
at move 0 (no real move played),  last_moves planes 0..4 mark: [1, 1, 1, 1, 1]
after 3 real moves,               last_moves planes 0..4 mark: [1, 1, 1, 1, 1]
```

At move 0 it presents **5 setup stones as the last 5 moves played**. After 3 real moves,
planes 3 and 4 still point at setup stones. It only clears once 5 real moves exist.

Why this is worse than `turns_since`:

1. **Sparse, high-confidence signal.** Each plane marks exactly one intersection and
   asserts "move N−k was played here." A wrong mark is a much stronger false claim than
   shifting a stone between age buckets.
2. **Diverges from KataGo precisely where it matters.** KataGo knows its random-init
   stones aren't moves; its own last-move planes are blank at such a position. If the
   point of the feature is to mimic KataGo, this is the wrong half to copy.
3. **Locality.** Policy networks lean hard on "respond near the opponent's last move."
   Pointing that at an arbitrary setup stone is actively misleading.

`get_last_moves` reads `moves_history[n_moves - 1 - i]` starting from index 0. The fix is
to floor the index at `num_handicap` so setup stones can never occupy a last-move plane.

**Recommendation: fix before generating shards that include this feature.** Doing it
afterwards means re-running the whole conversion.

### Train/inference consistency — verified, no mismatch

The open worry was whether the deployed GTP bot builds handicap positions the same way
the converter does. It does:

`cmd_set_free_handicap` → `GTPGameConnector.place_handicaps` → `GameState.place_handicaps`
→ `place_handicap_stone` → `do_move`.

Built a 4-stone handicap position both ways and compared:

```
GTP  moves_history: [(3,3), (15,15), (3,15), (15,3)]   num_handicap: 4   to play: WHITE
SGF  moves_history: [(3,3), (15,15), (3,15), (15,3)]   num_handicap: 4   to play: WHITE
TENSORS IDENTICAL:  True
```

Byte-identical, including the invented `turns_since` ages (planes 3, 2, 1, 0 in placement
order). **This removes the train/inference-mismatch concern entirely** and is the main
reason finding 1 was downgraded.

Also verified while in there: **GTP coordinates are correct, no mirroring.** `A1`→(0,0),
`D4`→(3,3), `Q16`→(15,15), `A19`→(0,18), `T1`→(18,0). The asymmetric corners rule out a
transpose. Worth knowing a flip here would have been invisible to every offline test.

One non-bug nuance: training setup stones are ~55 scattered random-init stones; a KGS
handicap game is 2–9 star points. Same convention, different distribution. The `board`
planes carry the real information so this should be harmless, but it's untested.

### FIXED - `AW` was counted as black handicap

`AW` stones go through `place_handicap_stone()`, which increments `num_handicap` — a
counter documented as "handicap stones placed by BLACK". An `HA[0]` game with 15 `AB` +
13 `AW` reports **28 handicaps**.

- **No training impact** — the converter uses `enforce_superko=False`.
- **Did affect** `get_handicaps()` and `save_gamestate_to_sgf()`.

**Fix:** the counter is split in two, because it was serving two incompatible purposes:

- `num_handicap` — the whole setup block, **both colours**. This is the boundary marking
  where alternating play begins, and is what the superko pre-filter (§2) keys off. It was
  already correct for that purpose.
- `num_black_handicap` — only the genuine black handicap stones, used by
  `get_handicaps()`.

This assumes the black setup stones come first in the block, which is how
`_sgf_init_gamestate` (all `AB`, then all `AW`) and `place_handicaps()` place them.

Covered by `test_only_black_setup_stones_count_as_handicap`, which asserts the two black
stones are returned, neither white stone is, and the setup block is still 4 long so the
superko boundary is unchanged.

---

## 2. Superko fix and its residual caveat

### What was fixed

[game_state.pyx:254](AlphaGo/go/game_state.pyx#L254) computed which slice of
`moves_history` holds the current player's own moves:

```python
first = self.num_handicap + (1 if self.current_player == stone_t.WHITE else 0)
```

After N black handicap stones **White moves first**, so White's moves sit at indices
N, N+2, … and Black's at N+1, N+3, …. The formula had it backwards — confirmed
empirically:

```
0 handicap, BLACK to play: moves_history[0::2] -> ['BLACK']   correct
4 handicap, WHITE to play: moves_history[5::2] -> ['BLACK']   WRONG
```

This is the Part 1 fast-path filter ("has the current player ever played here? if not,
superko is impossible"). Scanning the opponent's list means a real hit is missed,
`played` stays `False`, and the function returns "not superko" **without ever running the
Part 2 hash check** — letting genuine superko violations through. It failed in the
permissive direction.

Scope: `enforce_superko=True` is set only in `GTPGameConnector`, so this **never affected
training data** — only handicap games played by the bot on KGS.

**Fix** — derive from parity instead of assuming Black moves first:

```python
first = self.num_handicap
if (first % 2) != (self.moves_history.size() % 2):
    first += 1
```

**Regression test:** `TestKo.test_positional_superko_with_handicap` in
`tests/test_gamestate.py` — the same 9×9 superko position reached through a 2-stone
handicap game. Verified it **fails on the pre-fix build** and **passes on the fixed
build**; the test has teeth.

### The caveat to revisit

The parity derivation assumes **strict alternation after the setup block**. That covers
every real game, but not:

- GTP permits non-alternating sequences (`play B D4` twice in a row).
- `set_current_player()` can change the player without a move, breaking parity after SGF
  setup stones (`PL[..]`).
- If `AW` stones keep incrementing `num_handicap` (§1), the companion check
  `location in moves_history[:num_handicap]` — which assumes that block is all Black —
  is also wrong. **These two issues interact and should be fixed together.**

The old code was *also* wrong in these cases; the fix is strictly better, not complete.

**The robust alternative:** drop the colour distinction entirely — "was this point ever
played by *anyone*." Always correct, no parity reasoning, no interaction with
`num_handicap`. Measured cost over 1,500 real positions:

```
Part 2 (expensive state copy + zobrist hash) reached on:
  own-colour filter:  1.035% of legal moves
  any-colour filter:  3.008% of legal moves   -> 2.91x more
```

≈6 state copies per position instead of ≈2 — negligible next to neural-net inference.
Note the pre-filter cannot simply be removed: `is_positional_superko` is called from
`is_legal_move`, which `update_legal_moves` runs for all ~361 locations after every move,
so unconditional Part 2 would mean ~361 full state copies per move.

**Open decision: keep the parity fix, or switch to the simpler any-colour filter?**
Leaning toward any-colour for robustness, given the measured cost is trivial.

---

## 3. Suicide truncation — investigated, recommendation is "leave it"

KataGo rulesets ending in `sui1` permit suicide. `is_legal_move`
([game_state.pyx:508](AlphaGo/go/game_state.pyx#L508)) rejects it, so the converter raises
`IllegalMove` and **drops the remainder of the game**.

Measured over 3,000 files:

| metric | value |
|---|---|
| games whose ruleset allows suicide (`sui1`) | 1479 (49.3%) |
| games containing an *actual* suicide move | 13 (0.43%) |
| moves lost to truncation | 381 / 892,934 (**0.0427%**) |
| fraction of game completed before failure | median 0.92, max 0.98, **min 0.33** |
| moves lost per affected game | median 27, max 60 |

The suicide point always has 1, 2 or 4 same-colour neighbours — never a lone stone. It is
always a multi-stone group being filled in and dying, i.e. late cleanup under territory
scoring. Extrapolated to the full 60k corpus: ~7,600 moves out of ~18M.

### Skipping the move is NOT a fix — it silently corrupts data

The obvious "just skip the bad move and continue" desyncs the board, because under `sui1`
the suicide **removes the player's own group**; skipping leaves that group in place.
Demonstrated by classifying every rejection that follows a skip:

```
reason for FIRST rejection:        suicide   13
reasons AFTER skipping:            occupied  11,  suicide  3
games hitting further rejections:  7 / 13
```

The 11 `occupied` rejections are direct proof of desync — the real game played on an empty
point that our board still holds a stone on. Worse: **6 of 13 games hit no further
rejections at all**, meaning the wrong board silently accepted every remaining move. Those
would emit corrupt training positions with no error signal whatsoever.

*(A first attempt at this test measured `declared - replayed - skipped`, which is
tautologically 0 — every move is either replayed or skipped. Don't repeat that mistake;
the meaningful signal is the rejection **reason**.)*

### FIXED - the converter used to emit the suicide itself as a training label

Found while writing the truncation tests. `sgf_iter_states` yields `(position, move)`
**before** applying the move, so a move the engine will reject still reaches the consumer
once; only the *following* iteration raises. `convert_game` therefore emitted that move as
a label, then stopped.

Confirmed on real data before the fix:

```
last move yielded by iterator  : move=(9,5)  is_legal=False  player_matches_current=True
last label EMITTED by converter: (9,5)
```

One bad label per affected game - 0.43% of games, roughly 0.004% of all labels. Negligible
in volume, but it is specifically teaching the network a multi-stone suicide: the exact
move type that is illegal under both rulesets KGS offers and that the bot must never play.

**Fix:** `convert_game` skips the yield when the move is not legal from the current
player's perspective. `continue` rather than `return`, so the truncation signal survives -
the next iteration resumes the generator, whose own `do_move` raises, and the caller still
records the partial game. The legality check is only trusted when
`state.get_current_player() == player`, since `do_move` swaps colours for a
non-alternating record which would make `is_legal()`'s answer meaningless.

Verified over 3,000 real files: 13 games truncated, **0** still emitting the illegal move.
Covered by `test_converter_does_not_emit_an_illegal_move_as_a_label`, plus
`test_iterator_yields_the_offending_move_before_raising` which pins the yield-before-apply
behaviour that made this possible.

### Why not implement suicide properly

Making suicide legal puts suicide points into `state.legal_moves`, which
`get_capture_size`, `get_self_atari_size` and `get_liberties_after` all iterate.
`get_liberties_after` computes `min(groups_after[location, 1] - 1, 7)` — for a suicide
that is `min(-1, 7) = -1`, a **negative plane index** into a tensor compiled with
`wraparound=False, boundscheck=False`. That is memory corruption, not an exception.

The conceptually-right variant (execute the suicide so replay stays faithful, but never
emit it as a training label) carries the same engine risk.

**Recommendation: keep the current drop-the-remainder behaviour.** 0.043% of moves is not
worth a risky change to the core engine's legality and capture logic. There is also a
positive argument for it: suicide is illegal under the rules the bot actually plays on
KGS, so the policy net should never see it as a label. The prefix already written is
valid, correctly-labelled data — only the tail is lost.

---

## 4. Latent defects — ALL FIXED

These were real but not biting the current corpus. All five have since been fixed and
their tests flipped from `xfail(strict=True)` to passing. Kept here for the reasoning and
the measurements, not as outstanding work.

### FIXED - `Preprocess.zeros()` had no return statement

[preprocessing.pyx:325](AlphaGo/preprocessing/preprocessing.pyx#L325):

```cython
# Nothing to do; all features begin with zeros.         return offset + 1
```

The `return offset + 1` was absorbed into the trailing comment, so the `cdef int` function
returns **0**. Harmless *only* because `"zeros"` is last in the default feature list — any
feature ordered after it would be written starting at offset 0, overwriting the board
planes. `color()` delegates to `zeros()` and returns its value, so it is **already broken
for the value-net path**. Present since at least commit `3b420fa8`.

**Fix:** the return is now on its own line. `test_zeros_feature_does_not_reset_the_plane_offset`
puts `zeros` FIRST in the feature list so a regression corrupts the board planes, and passes.

### FIXED - `sgf_iter_states` replayed the previous move on a moveless node

[util.py](AlphaGo/util.py) — if a node in `game.rest` carries neither `W` nor `B`,
`move`/`player` retain their previous values and the prior move is replayed:

| SGF shape | result |
|---|---|
| `;B[pd];W[dp];C[comment];B[pp]` | `IllegalMove` — rest of game dropped |
| `;C[hello];B[pd];W[dp]` (leading) | `UnboundLocalError` → caught as "other" → **whole file dropped** |
| `;B[pd];W[dp];AE[pd];B[pp]` | `IllegalMove` — rest of game dropped |
| `B[pd]C[ok]` (comment *attached* to move) | fine |

**Dormant on KataGo selfplay TRAINING games: 0 moveless nodes in all 60,136 files.**
KataGo attaches comments to move nodes there.

**But NOT dormant in general — confirmed live on real data.** KataGo *rating* (gatekeeper)
games end with a standalone `;C[... result=W+R]` node carrying neither `B` nor `W`.
Replayed through the engine:

```
rating games replayed: 90
  clean        : 11
  IllegalMove  : 79  (87.8%)
  last move is a DUPLICATE of the previous: 89 (98.9%)
```

So the bug produced one duplicated training row per game plus a spurious "dropping the
remainder" warning on ~88% of files.

**Fix:** nodes carrying neither `W` nor `B` are now handled by what they actually contain,
because two very different cases hide there:

- **Pure annotation** (`C`, `N`, markup `CR`/`LB`/`MA`/`SL`/`SQ`/`TR`/`AR`/`LN`/`DD`,
  judgements `DM`/`GB`/`GW`/`UC`/`V`, timing `BL`/`WL`/`OB`/`OW`, `FG`/`PM`/`VW`) - no
  board effect, **skipped silently**. This is 100% of the moveless nodes in every corpus
  measured (81/81 in KataGo rating games are a terminal `C[...result=...]`; 0 in 20,000
  training files; 0 in the repo's own test SGFs).
- **Board-altering** (`AB`, `AW`, `AE`, `PL` outside the root) - these change the position
  or whose turn it is without being a move. **Raises `go.IllegalMove`.** Skipping them
  would leave the board silently out of step with the record and every later move would be
  replayed against a position that never occurred - the same desync proven corrupting for
  "skip the suicide and carry on". `go.IllegalMove` deliberately, so every existing caller
  already responds correctly: keep the prefix, drop the remainder.

*(An intermediate version of this fix skipped **all** moveless nodes unconditionally. That
fixed the comment case but converted a board-altering node from a loud truncation into
silent divergence - a regression. Do not simplify it back.)*

Verified on real data after the fix:

```
rating    n=90     clean=90     IllegalMove=0     (was 79, 87.8%)
training  n=2,000  clean=1,990  IllegalMove=10    (the genuine suicide truncations)
```

Covered by `test_moveless_nodes_are_skipped_not_replayed`.

### FIXED - mixed board sizes caused a SIGSEGV

Every `GameState` stores a **raw pointer** into a single set of global lookup tables
(`neighbor`, `neighbor3x3`, `neighbor12d`, `zobrist_lookup`) that are reassigned in place
whenever a state of a different size is constructed
([game_state.pyx:209-225](AlphaGo/go/game_state.pyx#L209)). With `boundscheck=False` a
stale pointer reads out of bounds silently.

Confirmed: build a 19×19 state, construct a 9×9 state, then call `state_to_tensor` on the
19×19 state → **exit code 139 (SIGSEGV)**. `get_legal_moves()` survives; the feature
processors are what crash.

**Fix:** the lookup tables are now keyed by board size in a `std::map`
(`neighbor_by_size`, `neighbor3x3_by_size`, `neighbor12d_by_size`, `zobrist_by_size`),
which was the existing `# TODO - global map from size to lookup table`. `std::map` is used
deliberately: it guarantees references to existing elements stay valid across later
insertions, so `&table[size]` is safe to hold for a GameState's lifetime. A `vector` would
not be, because it reallocates.

Covered by `test_mixed_board_sizes_do_not_corrupt_live_gamestates`, still subprocess-
isolated because a regression is a segfault that would take the whole runner down. It now
asserts the process survives AND that the 19x19 tensor is byte-identical before and after a
9x9 state is constructed - surviving without crashing is not sufficient.

### FIXED (documentation) - `ensure_xla_conv_nhwc()` is load-order dependent

`AlphaGo/training/xla_workarounds.py` claims in its docstring that setting `XLA_FLAGS`
after TensorFlow is imported "still takes effect, since XLA reads XLA_FLAGS lazily at
first compile". Reproduced a hard crash where it did not:

```
Autotuner could not find any supported configs for HLO:
  %cudnn-conv-bias-activation.17 = ... dim_labels=bf01_oi01->bf01
```

`bf01` is NCHW — the NHWC force never applied. Trigger: a script that imported TF and ran
an `evaluate()` *before* importing the trainer module. Production ordering
(`supervised_policy_trainer_v3` imported first) is fine, and `docker-compose.yml` also
sets the env var. So this was a **docstring overstating a guarantee**, not a live bug — but
it would bite anyone writing a harness that touches Keras before the trainer.

**Fix:** the docstring now states plainly that import order matters and records the
observed failure. `ensure_xla_conv_nhwc()` additionally emits a `RuntimeWarning` if
`tensorflow` is already in `sys.modules` when it runs.

**And that warning immediately caught a second instance.**
`supervised_policy_trainer_v3.py` itself imported `tensorflow` on line 6 and only called
`ensure_xla_conv_nhwc()` on line 11 - the exact ordering its own comment warns against. It
worked only because XLA had not parsed flags at that point; it was one import away from the
hard crash above, and the new warning fired on every normal run. The other four callers
(`lr_testing_trainer`, `_debug`, `_v2`, `benchmarks/_plot_sgf_heatmaps`) already had the
right order - v3 was the outlier. Now reordered so the call precedes `import tensorflow`,
which also means the warning only fires when something genuinely is out of order.

---

## 5. Conversion has no data-loss visibility

`sgfs_to_hdf5_parallel` calls `warnings.warn` per failed file. Across 60k files nobody
reads that, and there is **no aggregate count**. A conversion run that silently dropped
10% of the corpus would look identical to a clean one.

Two consequences:

1. The 0.043% suicide loss above is only known because it was measured by hand.
2. Right now *any* `IllegalMove` drops the remainder. If a genuine engine bug started
   rejecting legal moves, it would be indistinguishable from the suicide case.

**Proposed (not implemented):** accumulate per-error-type counters (`illegal_move`,
`parse_exception`, `size_mismatch`, `other`) in `sgfs_to_hdf5_parallel` and print a summary
at the end of the run with loss as a percentage of declared moves. Low risk, touches no
engine code, and converts "0.04%, fine" from an assumption into something verified on every
run.

---

## 6. Verified clean — do NOT re-audit these

All measured on real KataGo data unless noted. This is the expensive part of the audit;
don't redo it.

| area | result |
|---|---|
| **Coordinate frame** | state `[x][y][plane]` and label `[x][y]` agree. 3 independent confirmations (see "Context worth not re-deriving") |
| **Symmetries** | **0 mismatches** across all 8 transforms, 4,000 real positions |
| **HDF5 round-trip** | 300 games → 84,300 positions; `file_offsets` tile `[0,n)` exactly — 0 gaps, 0 overlaps, `sum(lengths) == len(states)`, feature string round-trips |
| **State/action pairing** | 4,000 positions: every stored move on an `EMPTY`, `sensible` point; board planes valid 3-way one-hot; `ones`/`zeros` planes correct |
| **Shuffle buffer coverage** | every position exactly once at buffer sizes 1, 3, 25, 1×, 3× dataset |
| **Train/val/test split** | position-disjoint (0 overlap on all three pairs), covers every position |
| **Seed reproducibility** | same seed → identical stream; different seed → different stream |
| **Batch identity** | the `.copy()` in `shuffle_buffer_batch_generator` verified through Keras's real `GeneratorDataAdapter` — distinct objects, distinct contents, 24 unique positions over 6 batches |
| **Initial loss** | 5.8886 vs `log(361)` = 5.8889 — net starts exactly at the uniform-guess baseline |
| **Overfit capability** | 973 positions from 3 games → **train acc 0.9906**, loss 0.0283 over 60 epochs (val correctly diverges) |
| **Train ↔ inference** | live `state_to_tensor` byte-identical to stored tensors 25/25; `eval_state` top move = played move 25/25; argmax matches `flatten_idx` on **99.4%** of memorised positions vs **7.8%** for a transposed control |
| **LR schedule** | cosine reaches peak exactly at `warmup_steps`, floor at `total_steps` (confirms the `decay_steps = total - warmup` reasoning) |
| **Resume arithmetic** | `_ResumedLRSchedule(s) == base(s + offset)` exactly over 600 steps; resume continues at 0.041250 instead of restarting warmup at 0.0001 |
| **Plateau replay** | `_replay_plateau_state` matches real `ReduceLROnPlateau` on best/wait/cooldown, including `min_delta=0.005` |
| **GTP coordinates** | `A1`→(0,0), `D4`→(3,3), `Q16`→(15,15), `A19`→(0,18), `T1`→(18,0) — asymmetric corners rule out a transpose |

### Known-broken, unrelated

`tests/test_supervised_policy_trainer.py` fails collection — imports `FILE_TEST`, which no
longer exists in `AlphaGo/training/supervised_policy_trainer.py`. Pre-existing, tests the
**v1** trainer, outside the v3 path. Either delete or repair. All test runs in this file
use `--ignore=tests/test_supervised_policy_trainer.py`.

---

## Open decisions to pick up

1. **`last_moves`** — apply the `num_handicap` floor? (Recommended, and before generating
   shards with the feature.)
2. **`turns_since`** — leave as-is, or move setup stones to the age≥7 plane? (Leaning
   leave-as-is; impact is ~1% of positions and self-healing.)
3. **Superko pre-filter** — keep parity, or switch to any-colour? (Leaning any-colour.)
4. **`AW` / `num_handicap`** — stop counting white setup stones as black handicap.
   Interacts with (3).
5. **Conversion loss reporting** — add per-error-type counters and an end-of-run summary,
   classifying `IllegalMove` by reason so genuine suicide truncations stay distinguishable
   from a real engine bug. Low risk, high value. (§5)
6. **Suicide truncation** — leave as-is. (§3)

Either of (1) or (2) requires **re-running the full conversion** — the features are baked
into the shards. Nothing else on this list does.

### Pipeline tooling, as built

```
sgf_analyze                   survey a corpus; decides nothing
sgf_preparation scan/select   file-level policy -> keep-list
game_converter_katago_data    position-level policy -> shard_NNNNN.h5
supervised_policy_trainer_v3  unchanged - the converter's output is a
                              backward-compatible superset
```

See SGF_FILTER_POLICY.md for every filter decision and its measurement, including the two
that were argued for and then **reversed** (`sui1` and `gtype=asym`).

One finding from building the converter that belongs here: `sgf_iter_states` yields
`(position, move)` **before** applying the move, so a move the engine will reject reaches
the consumer once. Both converters now guard against emitting it as a label - see §3.

### Already fixed (no action needed)

Superko handicap parity (§2), `Preprocess.zeros()` missing return, moveless SGF nodes,
mixed board sizes, `AW` handicap count, `xla_workarounds` docstring (all §4). Suite:
**162 passed, 1 xfailed** — the single remaining xfail is (2), `turns_since`.

### Suggested grouping if picking this up cold

- **Cheap + independent, do together:** (1), (5)
- **Interact, do together:** (3) and (4)
- **Judgement calls, no action needed:** (2), (6)

---

## Reproducing this

Everything ran inside the project container with the Cython extensions built:

```bash
# build extensions (needed after any .pyx change)
docker run --rm -v "C:\Users\winst\Documents\NeuralZ:/workspace" -w /workspace \
  neuralzalphago-gpu:latest python setup_cython.py build_ext --inplace

# test suite  (expect: 162 passed, 1 xfailed)
docker run --rm -v "C:\Users\winst\Documents\NeuralZ:/workspace" -w /workspace \
  -e PYTHONPATH=/workspace -e CUDA_VISIBLE_DEVICES="" \
  neuralzalphago-gpu:latest python -m pytest tests/ -q \
  --ignore=tests/test_supervised_policy_trainer.py
```

Note `tests/test_supervised_policy_trainer.py` fails collection (imports `FILE_TEST`,
removed from the v1 trainer). Pre-existing, unrelated to the v3 path.

Corpus used: `AlphaGo/tmp_data` — 60,136 KataGo selfplay SGFs, all `SZ[19]`
(gitignored).

### Key code locations

| what | where |
|---|---|
| setup stones → `moves_history` | `AlphaGo/util.py:_sgf_init_gamestate` |
| handicap placement | [game_state.pyx:723](AlphaGo/go/game_state.pyx#L723) |
| `turns_since` | [preprocessing.pyx:70](AlphaGo/preprocessing/preprocessing.pyx#L70) |
| `last_moves` | [preprocessing.pyx:103](AlphaGo/preprocessing/preprocessing.pyx#L103) |
| superko pre-filter (fixed) | [game_state.pyx:254](AlphaGo/go/game_state.pyx#L254) |
| GTP handicap entry point | `interface/gtp_wrapper.py:cmd_set_free_handicap` |
| audit regression tests | `tests/test_pipeline_audit.py` |
| superko handicap test | `tests/test_gamestate.py:TestKo` |
| suicide rejection | [game_state.pyx:508](AlphaGo/go/game_state.pyx#L508) |
| `zeros()` missing return | [preprocessing.pyx:325](AlphaGo/preprocessing/preprocessing.pyx#L325) |
| moveless-node bug | `AlphaGo/util.py:sgf_iter_states` |
| global lookup tables (SIGSEGV) | [game_state.pyx:209](AlphaGo/go/game_state.pyx#L209) |
| conversion error handling | `AlphaGo/preprocessing/game_converter_parallel.py` (~line 230) |
| XLA workaround | `AlphaGo/training/xla_workarounds.py` |

### Context worth not re-deriving

The **coordinate frame is correct and verified three independent ways** — don't re-audit
it. `calculate_board_location(x, y, size)` is defined as `x + y*size` but *every caller
passes the arguments swapped*, so the real flat index is `x*size + y`, matching
`util.flatten_idx`. State tensors index `[x][y][plane]`, action one-hots index `[x][y]`.
Evidence: 0 symmetry mismatches across all 8 transforms on 4,000 real positions; a 99.1%
overfit; and 99.4% argmax agreement at inference vs 7.8% for a transposed control.
