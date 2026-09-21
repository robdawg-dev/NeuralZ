# NEXT_STEPS

Record of the discussion held after handover, covering the filtering options at each stage
of the new preprocessing pipeline and the decisions still outstanding.

Companion documents: `AUDIT_NOTES.md` (bugs found and fixed, deferred items) and
`SGF_FILTER_POLICY.md` (every filter decision with its supporting measurement, including
the ones argued for and then reversed). This file is the *forward-looking* one.

---

## Context at handover

The pipeline is built and tested. Three tools:

```
sgf_preparation scan   <dir> manifest.jsonl    records facts, decides almost nothing
sgf_preparation select manifest.jsonl keep.txt FILE-level policy
game_converter_katago_data keep.txt out_dir/   POSITION-level policy
```

A full scan has already been run on the corpus: **60,136 files -> 60,136 rows**, 46MB,
31 seconds, at `workspace/generation_testing/manifest.jsonl`. A 5,000-game test conversion
produced 1,467,678 positions / 3.8GB in 282 seconds (17.7 games/sec). All measurements
below come from those two runs unless noted.

**Flagged immediately:** `last_moves` needs a code fix before it can be used at all. It
marks setup stones as "the last 5 moves played" and does not clear until 5 real moves
exist. It is not in the default feature list, so nothing has trained on it. `turns_since`
is the milder cousin - see Decision 3a.

---

## Stage 1 - `scan`

**Nothing to decide here, by design.** The scan records rejection reasons as *advisory* and
acts on none of them, so policy stays re-tunable without re-reading the corpus.

| option | effect | recommendation |
|---|---|---|
| `--board-size` (default 19) | the only short-circuit: a wrong-size file is still recorded, but remaining regexes and move-stats are skipped | leave at 19 |
| `--no-move-stats` | drops per-move aggregates (blunder counts, decided positions, visits) | **do not use** |
| `--resume` / `--sample` / `--workers` | operational | as needed |

`--no-move-stats` saves only **~3.5%** (1,925 vs 1,992 files/sec) and costs the ability to
size any move-level threshold from the manifest. Not worth it.

*(An earlier measurement put its cost at ~50%. That was wrong - it was a double-file-read
bug, since fixed. Recorded so the number is not reused.)*

**Status: settled.** Scan already run.

---

## Stage 2 - `select`

Five criteria always apply; six are opt-in.

**Always on:** unreadable, wrong board size, zero moves, unparseable komi, komi out of range.

Note `no_komi` is **always on** - a file with no parseable `KM` is dropped even though it
was never requested. Defensible (komi is invisible to the network, so unknown is worse than
extreme) but it is not opt-in. Raise if that should change.

### Decision 2a - komi band  *(OPEN)*

Komi is a **hidden variable**: KataGo feeds it to its network
(`nninputs.cpp:1215`, `rowGlobal[5] = selfKomi/15.0`); our 48 planes do not contain it. So a
position labelled under komi 32 and an identical one under komi 6.5 have different correct
moves and no feature to distinguish them.

| band | games kept | moves |
|---|---|---|
| `[-10, 30]` (default, generous) | 52,820 (87.8%) | 16.07M |
| `[-2, 16]` - \|k-7\|<=9 | 44,187 (73.5%) | 13.48M |
| `[1, 13]` - \|k-7\|<=6 | 41,011 (68.2%) | 12.59M |

Supporting measurement - extreme komi drives games into decided territory:

```
|komi-7|  0-3 :  39.9% of positions already decided
|komi-7| 12-15:  84.4%  (median winrate 0.01)
```

**Recommendation: `[-2, 16]`.** The first tightening costs 14 points and removes the
pathological end; the second costs 5 more on a much weaker argument.

**Uncertainty: there is no A/B evidence that any komi band helps.** The existing 4d model
trained on all komi including negative. See Decision 3f for an alternative that avoids the
tradeoff entirely.

### Decision 2b - gtypes  *(recommended: leave defaults)*

Defaults exclude `hintpos`, `hintfork`, `cleanuptraining` (5.1% total).

- **`hintpos` / `hintfork`** - source-verified: *"If a full search gives a different move
  than a quick search and judges the move to be way better... record as a hintpos"*
  (`startposes.cpp:1146`). Games **selected for being surprising**. High confidence.
- **`cleanuptraining`** - synthetic drilling, 0 init moves, p90 first-winrate 0.97. High
  confidence.
- **`asym` - KEPT** (reversed after measurement). The weakened side is real (229 vs 448
  visits, 1.36% vs 0.69% blunders), but in handicap games White *always* receives the
  playout advantage (`play.cpp:629`), so these are a strong player facing handicap stones -
  the KGS deployment case. Also **84% of `asym` games are handicap games** (mode assignment
  overwrites, `play.cpp:1633-1653`), so excluding it would silently remove half the handicap
  data.
- **`sui1` - KEPT** (also reversed). Permitting suicide measurably does not change the move
  stream (`own_eye` 8.24% vs 8.24%; game length +0.16%), and only 1.13% of permitting games
  contain one.

### Decision 2c - setup stones  *(recommended: keep them)*

`--exclude-setup-stones` drops 43.7% of files. Recommended **against** - the problem is
confined to the first 7 positions of those games and costs ~1% to fix at the converter
instead (Decision 3a).

```
komi [-2,16] + exclude-setup-stones : 27,725 games (46.1%)
komi [-2,16] alone                  : 44,187 games (73.5%)
```

Moves/game *rises* to 329 when excluding them, because `sgfpos` games start mid-game
(median `initTurnNum` 54) and are therefore shorter.

### Decision 2d - target size  *(OPEN)*

| target positions | source games | disk | convert |
|---|---|---|---|
| 40M | ~136,000 | 104 GB | ~2.1 h |
| 70M (previous production scale) | ~238,000 | 182 GB | ~3.7 h |

**Open question: what size, and is an A/B in scope?** That determines whether the contested
filter (3c) is settled now or two sets are generated.

---

## Stage 3 - converter

### Decision 3a - `turns_since`: skip positions, or fix the feature?  *(OPEN - recommendation revised)*

Setup stones land in `moves_history`, so `turns_since` reports a pre-placed position as
though it had just been played move by move. Measured decay on a real 28-setup-stone game:

| move # | setup stones wrongly marked "recent" |
|---|---|
| 0 | 7 of 28 |
| 3 | 4 of 28 |
| 6 | 1 of 28 |
| **7+** | **0** |

Only 7 "recent" age planes exist, so it self-corrects once 7 real moves have been played.
Net **~1% of all tensors**, one feature group.

**Option A - `--skip-setup-positions 7`.** Drop those positions. Cost **0.91%**. No code
change.

**Option B - fix `get_turns_since`.** Exclude the setup block when assigning ages so setup
stones stay in the age>=7 plane. Cost **0%**.

**Recommendation revised to B**, for three reasons:

1. **B is more correct, not just cheaper.** The setup stones genuinely *are* old - the
   original move order was discarded, and "age >= 7" is the honest answer. A discards valid
   positions to avoid a wrong feature; B makes the feature right.
2. **B fixes `last_moves` with the same change.** That fix is needed anyway to use the
   feature at all, and `last_moves` is the worse offender.
3. Contained change. Setup stones are `moves_history[0:num_handicap]` and the age loop runs
   backwards, so it is a bounded index rather than a full reverse iteration.
   `preprocessing.pyx` already reaches `num_handicap` through the cimported `GameState`.

**Cost of B:** a Cython edit, a rebuild, and re-conversion - but re-conversion is happening
regardless.

**Uncertainty on both:** it is not known that the `turns_since` error hurts at all. ~1% of
tensors with one feature group partially wrong, and the 4d model trained with it present.

**Looked at and deliberately left alone:** `turns_since` counts a pass as consuming an age
slot (median 8 passes/game). Arguably correct - a pass *is* a turn, and AlphaGo's feature is
"turns since a move was played". Not a bug.

### Decision 3b - `--max-winrate-loss`  *(recommended: 0.10)*

Drops a position whose label gave up more than X winrate. KataGo records
`win/loss/noResult/score` from a constant White perspective, so the delta across a move is
what the mover surrendered.

| threshold | cost |
|---|---|
| 0.05 | 0.50% |
| 0.10 | **0.12%** |
| 0.20 | 0.02% |

**Caveats, all real:**

- **Structurally blind in decided positions.** Above 0.95 or below 0.05 winrate, mean
  \|delta\| is ~8x smaller - a terrible move at 0.99 might register 0.01. This filter cannot
  see the positions where move selection is least meaningful.
- **Coarse resolution.** Winrate is written with `%.2f`; good for catching real blunders, not
  for ranking near-equal moves.
- **Biased mean.** A mover's own search is slightly optimistic about the move it just chose,
  so this must threshold on the loss *tail*, never an absolute level. The thresholds above do.

**Recommendation: 0.10.** Unambiguous blunders at 0.12% cost. 0.05 is defensible but starts
catching moves that are merely inferior rather than wrong.

**Uncertainty:** no evidence it helps; downside bounded at 0.12%.

### Decision 3c - `--drop-hopeless-mover`  *(OPEN - the contested one)*

Drops positions where the player to move is already at or below X winrate. **20.9% at
0.05** - the only expensive filter, and the only one with a genuine argument on both sides.

**For:**
- In a decided position the value signal is flat, so move selection carries less information.
- KataGo's losing-side behaviour is calibrated against an opponent that never errs, so it
  does score-minimisation. Against fallible humans one would want complications. Arguably not
  just low-signal but the **wrong behaviour to imitate**.
- The blunder filter (3b) is blind exactly here, so these positions otherwise receive no
  quality filtering at all.

**Against:**
- The moves are not random - the policy prior plus `dynamicScoreUtilityFactor = 0.40` keep
  them sensible. "Slack" is not "bad".
- **Phase-correlated**: removes 45.9% of endgame vs 8.0% of opening. KataGo's
  play-to-completion endgame coverage is a real advantage over human corpora (which end in
  resignation), and this partially gives it back.

The asymmetric form matters:

| variant | corpus kept | endgame positions/game |
|---|---|---|
| none | 100% | 69.2 |
| symmetric (drop both sides) | 51.9% | **7.0** (median 0) |
| asymmetric (implemented) | 79.1% | 38.2 |

The symmetric version guts the endgame. The asymmetric one keeps the **winner's** moves -
converting a won game, which human corpora teach badly - and drops only the hopeless side's.
Positions at 0.05-0.30 (behind but not hopeless) are kept either way, so the network still
learns to play from disadvantage.

**Recommendation: OFF for the first set.** If one A/B is run, make it this. Two conversions
at ~136k and ~172k games is roughly 2.1 and 2.7 hours.

### Decision 3d - features  *(OPEN)*

`--features all` gives the 11 features / 48 planes currently trained on. **`last_moves` is
not in `all`** - it needs an explicit list, and the fix from 3a first.

**Open question:** keeping the 48-plane set, or is changing the feature set on the table?
It is baked into the h5, so it must be settled before generating.

### Decision 3e - split and shuffle at conversion  *(agreed in principle, sub-decisions open)*

Moving train/val/test splitting and position shuffling into the converter so the trainer
reads sequentially removes ~250 of `shuffle_buffer.py`'s 335 lines, frees the 7GB runtime
buffer, and eliminates the `build_validation_arrays` OOM.

Open sub-decisions:

1. **Split ratios** - currently `0.93 / 0.05 / 0.02`. Keep?
2. **Split must stay at the game level** (positions within a game are correlated), then
   shuffle positions within each split. Considered non-negotiable.
3. **Small runtime shuffle buffer in the trainer, or fully serial?** Pre-shuffling gives one
   fixed order forever. Random symmetry per epoch already presents each position as one of 8
   transforms, so this is probably not serious - but a ~100k ring buffer restores order
   variation for ~20 lines.
4. **Schema break.** Shuffled positions make `file_offsets` meaningless, so these shards will
   not load in the current `v3` trainer. That is a new reader. Acceptable?

### Decision 3f - komi as an input plane instead of a filter  *(OPEN - raised late)*

The alternative most worth considering. Komi is filtered (2a) because it is a hidden
variable. But it could be *added* rather than filtered: a constant-valued komi plane, the
same mechanism as the existing `ones` plane, would

- allow keeping **all** komi values - no band, no 12-30% data loss
- let the bot condition on komi at play time, which it knows on KGS
- cost one plane out of ~49

The same applies to the rules axes measured as behaviourally negligible, so those are
probably not worth it - but komi genuinely changes play.

**Cost:** a new feature in `preprocessing.pyx`, a changed input dimension, retraining from
scratch. Not small. But it addresses the root cause rather than working around it, and it is
strictly more data, not less.

---

## Broader concerns raised

### None of the filtering work is validated

The audit found the pipeline **essentially correct** - coordinate frame, symmetries, HDF5
round-trip, 99.1% overfit, train/inference consistency all verified on real data. The only
measured *correctness* defects in the data were the suicide label (~0.004% of labels, fixed)
and `turns_since` (~1% of tensors).

Everything else - komi bands, gtype exclusions, blunder thresholds, hopeless-mover - is a
**hypothesis about improvement**, not a defect being repaired. The 4d model trained on
completely unfiltered data with `turns_since` broken. Worth weighing when deciding how much
effort the remaining decisions deserve.

### An A/B trap that constrains data generation

Two confounds will bite any filter comparison:

**The validation set changes with the filter.** If config A drops 20% of positions and B
does not, their val sets contain different positions, so `val_loss` is not comparable. A
**single fixed evaluation set** is needed - generated once with no filtering, held out from
every config's training games, used identically by all runs. The current design splits
train/val/test per conversion, which produces exactly the non-comparable case.

**The LR schedule shifts with dataset size.** `steps_per_epoch = epoch_length // minibatch`
and cosine decay is shaped by `--epochs`, so two configs with different data volumes get
different LR trajectories. Either fix `--epoch-length` across runs or compare at matched step
counts.

There is precedent: a comment in `shuffle_buffer.py` records two runs that looked LR-related
but actually differed because their train sets overlapped only ~93%.

**This must be decided before generating data**, since it constrains how splits are produced.

### Search is likely a bigger lever than any of this

Confirmed by reading the code: the deployed bot uses `ProbabilisticPolicyPlayer` - a single
forward pass with top-k sampling. **No search at all.** `MCTSPlayer` exists in `ai.py` but
`run_gtp_player.py` does not use it, and it pins `CUDA_VISIBLE_DEVICES=-1` for CPU-only
deployment.

A 4d rating from a raw policy network is a strong result. Even modest search typically adds
several stones - plausibly far more than a 1% feature fix or a 20% data filter.

The CPU-only constraint is real; a b15c192 forward pass on CPU is slow enough that deep
search is not free. Questions worth answering:

- Has shallow MCTS (20-50 playouts) been tried and measured?
- Is `MCTSPlayer` functional, or does it need the value network that was removed?
- The original AlphaGo design used a small, fast **rollout** policy precisely for this;
  `AlphaGo/models/rollout.py` is present in the tree.

If the goal is "stronger bot on KGS" rather than "better supervised policy", the expected
gain from search is worth knowing before investing many hours in data filtering.

### Housekeeping

- **`tests/test_supervised_policy_trainer.py` fails collection** (imports `FILE_TEST`,
  removed from the v1 trainer). Pre-existing; excluded from every test run. Delete or repair.
- **The counterfactual numbers come from one 5,000-game sample.** They match independent
  measurements closely, but precise threshold tuning should use 30-50k games (~30 min).
- **`sgf_ok` in the manifest means *readable*, not *valid SGF*** - `True` for an empty file.
  Harmless, but the name overstates it.

---

## Decision checklist

Ordered by what blocks what. Nothing below stage A should be generated against until
stage A is settled.

### A. Blocks everything - decide first

- [ ] **A1. A/B measurement protocol.** Will filter configs be compared? If yes, a
      **single fixed evaluation set** must be generated once with no filtering, held out
      from every config's training games, and used identically by all runs. Otherwise
      `val_loss` is not comparable across configs, because each config's val set contains
      different positions. Also fix `--epoch-length` across runs, or compare at matched
      step counts, so the cosine LR schedule is not confounded by dataset size.
      *Constrains how splits are produced, so it cannot be decided later.*

- [ ] **A2. Feature set.** Keeping the 48 planes, or changing? Baked into the h5.
      Sub-questions: add `last_moves` (needs the B1 fix first)? Add a **komi plane** (see
      A3)?

- [ ] **A3. Komi: filter or feature?** Either restrict the komi band (B-stage flag) or add
      a constant-valued komi plane so the network can condition on it. The plane costs one
      of ~49 and keeps **all** komi values; the filter costs 12-30% of the corpus. If the
      plane is chosen, **A4 becomes moot**. Requires a new feature in `preprocessing.pyx`,
      a changed input dimension, and training from scratch.

- [ ] **A4. Komi band** *(only if A3 = filter)*. `[-10,30]` (87.8% kept) / `[-2,16]`
      (73.5%) / `[1,13]` (68.2%). **Recommendation: `[-2,16]`.**

- [ ] **A5. Target size and set count.** 40M positions ~= 136k games / 104GB / ~2.1h;
      70M ~= 238k / 182GB / ~3.7h. One set, or two for an A/B?

### B. Code changes - needed before they can be exercised

- [x] **B1. `turns_since` / `last_moves`: fix the feature, or skip positions?**
      **RESOLVED 2026-09-21: skip positions.** `--skip-setup-positions` now defaults to 7
      in `game_converter_katago_data.py`. Measured cost 0.85% on 1,200 real games.

      This went *against* the original recommendation above ("fix the feature, cost 0%"),
      because that option turned out not to exist in any meaningful form. The AB/AW block
      carries no temporal information at all: it is a row-major board scan (100% of 400
      sampled `sgfpos` files), `util.py` applies all AB then all AW so the "7 most recent
      moves" read off it are 7 same-coloured stones - a shape alternating play cannot
      produce - and captured stones are absent entirely (38.5% of files have
      nAB+nAW < init_turn_num, p90 = 7 missing, max 73). There is no true age to recover,
      so any "fix" would have had to blank or fabricate the ages regardless.

      The skip also preserves plane semantics exactly, so data generated now remains
      directly comparable with already-trained networks. A feature edit would have broken
      that. `last_moves` is not used; a warning block in `get_last_moves` records its
      separate defect and that its skip would need to be 5, not 7.

- [x] **B2. Pass handling in `turns_since`** - passes consume an age slot (median 8/game,
      clustered at game end).
      **RESOLVED 2026-09-21: no change. Not a bug.**

      "Turns since" means turns elapsed and a pass is a turn, so the ages stay accurate.
      The only real effect is that the oldest move drops out of the 7-slot window into the
      age>=7 bucket - an honest loss of depth rather than a misstatement. The alternative
      convention (passes do not consume a slot) would keep 7 stones visible but report
      them at ages they were not played at.

      Crucially the same `Preprocess` object runs at training and at inference, so neither
      convention can produce a train/serve mismatch. Verified identical to what
      already-trained networks saw: the PASS guards in `get_turns_since` have never been
      edited in the file's entire git history, and pass labels were already excluded by
      the old path (`game_converter.py:38`, which `game_converter_parallel.py:53` calls).

      Passes are never training labels - the policy head is a 361-way softmax with no pass
      slot - and are never written into any feature plane. Only 3.05% of emitted positions
      carry a pass within the last 7 plies, and 98% of those sit past move 200, where
      `ai.py:101` makes the bot pass back without consulting the network at all.

- [ ] **B3. `asym` mitigation - currently a gap.** `asym` was KEPT on the stated basis that
      the weakened side would be identified at conversion from visit counts. **That was
      never built.** Today its moves train unmitigated (229 visits, 1.36% blunder rate vs
      0.69%). Choose one:
      (a) build it - per game, median visits by colour, drop the low side above a ~1.5x
      ratio (~15 lines);
      (b) decide 229 visits is acceptable, consistent with keeping the ~240-visit
      cheap-search moves, and correct the doc;
      (c) re-open excluding `asym` - but that also removes ~half the handicap data.
      **Recommendation: (a).**

- [ ] **B4. Split + shuffle move into the converter** *(agreed in principle)*:
      - [ ] B4a. Split ratios - keep `0.93 / 0.05 / 0.02`?
      - [ ] B4b. Split at the **game** level, then shuffle positions within each split.
            *(Considered non-negotiable - positions within a game are correlated.)*
      - [ ] B4c. Small runtime shuffle buffer in the trainer (~100k), or fully serial?
      - [ ] B4d. **Accept the schema break?** Shuffled positions make `file_offsets`
            meaningless, so these shards will not load in the current `v3` trainer. Needs a
            new reader.

- [ ] **B5. Make counterfactual thresholds configurable.** `_CF_LOSS`, `_CF_HOPELESS`,
      `_CF_SKIP` are hard-coded, so sizing `--drop-hopeless-mover` at anything other than
      0.05 needs a code edit - which defeats the purpose of the report. Trivial fix.

- [ ] **B6. `no_komi` opt-out.** It is the only rejection criterion with no flag.

### C. Filter values - set at generation time, no code

- [ ] **C1. gtype exclusions.** **Recommendation: leave defaults**
      (`hintpos`, `hintfork`, `cleanuptraining`; 5.1%).
- [x] **C2. Setup stones.** **RESOLVED 2026-09-21: keep setup-bearing games.**
      `--exclude-setup-stones` stays off; B1's skip handles the defect at 0.85% rather
      than 43.7%. Note that ~87% of setup-bearing files are whole-board mid-game
      resumptions, not handicap - and that a blanket setup-stone filter would have removed
      100% of `handicap` and 84% of `asym`, i.e. exactly the handicap data wanted for
      play against handicap opponents.
- [ ] **C3. `--max-winrate-loss`.** 0.05 (0.50%) / **0.10 (0.12%, recommended)** / 0.20
      (0.02%).
- [ ] **C4. `--drop-hopeless-mover`.** 20.9% at 0.05 - the only expensive filter and the
      only genuinely contested one. **Recommendation: OFF for the first set; this is the
      A/B candidate if A1 says yes.**
- [ ] **C5. Pre-`startTurnIdx` moves** - 7.5% of moves, played **without search**, which
      KataGo excludes from its own training. Currently recorded as KEPT, but the evidence
      is thinner than that status implies (20% of `normal` games are already lopsided by
      the time search begins). **Belongs beside C4 as an open question, not under
      DECIDED.** No flag exists yet - would need one.

### D. Does not block generation

- [ ] **D1. Search at play time.** The bot is a raw policy net with **zero search**
      (`ProbabilisticPolicyPlayer`, CPU-only). Even shallow MCTS is usually worth more than
      any filter here. Is `MCTSPlayer` functional, or does it need the removed value net?
      Worth measuring the expected gain before investing further in data filtering.
- [ ] **D2. Delete `tests/test_supervised_policy_trainer.py`** or repair it (imports
      `FILE_TEST`, removed from the v1 trainer; excluded from every test run).
- [ ] **D3. Rename `sgf_ok`** - it means *readable*, not *valid SGF*.

### Dependency notes

```
A3 = "add komi plane"   ->  A4 moot, A2 changes, retrain from scratch
A1 = "yes, A/B"         ->  C4 stays open, A5 needs two sets
B1                      ->  decide B2 in the same edit
B4d                     ->  a new reader is required before any training
```

### Settled - not worth reopening

Checked and left alone: keeping `sui1` (`own_eye` 8.24% vs 8.24%, game length +0.16%);
not filtering scoring/ko/tax (<=2pp, confined to the last 20% of moves); not using KataGo's
`weight` (its high end is anti-correlated with representative play); drop-the-remainder on
suicide truncation; the counterfactual measurements themselves (reproduced across three
independent samples).

---

## Open decisions, collected

| # | decision | status | recommendation |
|---|---|---|---|
| 2a | komi band | OPEN | `[-2, 16]`, or see 3f |
| 2b | gtype exclusions | open | leave defaults |
| 2c | setup stones | open | keep; fix at converter |
| 2d | target size / A/B scope | **OPEN** | - |
| 3a | `turns_since`: skip vs fix feature | **OPEN** | fix the feature (B) |
| 3b | `--max-winrate-loss` | open | 0.10 |
| 3c | `--drop-hopeless-mover` | **OPEN** | off for first set; the A/B candidate |
| 3d | feature set | **OPEN** | - |
| 3e | split + shuffle at conversion | agreed; 4 sub-decisions open | - |
| 3f | komi as input plane | **OPEN** | worth serious consideration |
| - | A/B measurement protocol | **OPEN** | decide before generating |
| - | search at play time | **OPEN** | measure expected gain first |

The two that change code are **3a** (fix the feature vs skip positions) and **3e.4** (accept
the schema break). The rest are flag values settable at generation time.
