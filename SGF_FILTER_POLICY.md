# SGF filter policy

The decision list for which KataGo selfplay games and positions are healthy to train on,
and the spec the three preprocessing tools implement:

| tool | scope | what it decides |
|---|---|---|
| `sgf_analyze.py` | read-only survey | nothing - reports distributions so criteria can be sized |
| `sgf_preparation.py` | whole **files** | `scan` records every file; `select` emits a keep-list |
| `game_converter_katago_data.py` | individual **positions** | which positions reach the h5 |

All three are built and tested (`tests/test_sgf_preparation.py`,
`tests/test_game_converter_katago.py`).

**Goal:** a valid board position paired with the move a strong player actually chose from
it. Anything that undermines that pairing — or that teaches behaviour which never occurs
in the bot's deployment (19x19 on KGS) — is a filter candidate.

**Guiding principle:** *filter at the level the property lives at.* Board size, ruleset,
komi and `gtype` are properties of a whole **game** → reject the file. Blunders are
properties of a single **move** → suppress that label, keep the game.

**Cost is not a constraint.** The KataGo selfplay supply is effectively unlimited, so any
filter that improves data quality is affordable. When in doubt, discard.

**Real scale, for calibration.** The previous production set was **120GB across 5 shards**
from roughly 240k games - on the order of **50-70M positions**, not the 500M an earlier
draft of this document assumed. At the measured converter rate (~330k games / ~5 hours) a
full production conversion is about **4 hours**, not the 60+ that same draft extrapolated.
Both figures mattered: the "re-converting is prohibitively expensive" argument that
originally pushed decisions toward the trainer does not hold at this scale, which is why
every position-level decision now lives in the converter instead.

All percentages are measured over the 60,136-file sample in `AlphaGo/tmp_data`
(kata1-b28c512nbt, 17,915,007 moves) unless noted.

---

## DECIDED — discard the whole file

These are the defaults `sgf_preparation select` applies. Everything else is an opt-in flag.

| # | Rule | Cost | Why |
|---|---|---|---|
| D1 | `SZ` != 19 | 0.00% here | Only 19x19 will ever be trained. The one criterion that **short-circuits** the scan, since it is never revisited. |
| D2 | `gtype=hintpos` | 2.20% | See below |
| D3 | `gtype=hintfork` | 0.99% | Same as D2 |
| D4 | `gtype=cleanuptraining` | 1.89% | Synthetic endgame/cleanup drilling. 0 initialisation moves, p90 first-winrate 0.97 — starts from already-decided positions. Not representative of play. |
| D5 | komi outside `[-10, 30]`, **non-handicap games only** | ~4.4% | Komi is a **hidden variable**: KataGo feeds it to its network (`nninputs.cpp:1215`, `rowGlobal[5] = selfKomi/15.0`), this project's 48 planes do not contain it. A deliberately **generous outer bound** only — the real band is an open question, so this removes only values no plausible band would keep. |
| D6 | unreadable / zero moves | ~0% | Unusable |
| D5b | handicap games with komi outside `[-120, 120]` | 0.005% (2 files) | **Handicap komi is not a free parameter.** KataGo expresses handicap as compensation at ~13 pts/stone — the real value of a stone on 19x19. Measured medians: HA2 15.5, HA3 27.5, HA4 39.0, HA5 53.5, HA6 65.0, HA9 115.5. Applying the even-board band to these discarded **49.9% of handicap games**, the data most wanted for play against handicap opponents. They now get a loose sanity bound only, which exists because the raw corpus contains komi of -303 and +359 on a 361-point board. Restore the old behaviour with `--handicap-uses-komi-band`. |

Measured together: **91.14% of files kept, 305 moves per surviving game** (60,136 scanned -> 54,806 kept). Before the handicap-komi fix this was 87.83%; the difference is 1,986 handicap games, of which 957 are `gtype=handicap` and 905 `asym`.

### REVERSED — two rules this document previously specified and no longer does

Both were argued for at length earlier and then overturned by measurement. Recorded so
they are not reinstated from memory.

**`sui1` (suicide-permitting rulesets, 49.73% of files) — NOT filtered.**
The original argument was that suicide is illegal on KGS, so it should never be a label,
and that a header regex is cheaper than a per-game replay. Both true, but the premise was
wrong: **merely permitting suicide does not change the move stream.** Measured on clean
`gtype=normal` games:

```
rule   games   real mv med   passes med   blunder%   games w/ an actual suicide
sui0   1,581       319            8        0.482          0.00%
sui1   1,589       317            8        0.477          1.13%

last 20% of moves:  own_eye  8.24% vs 8.24%   own_area 8.55% vs 8.27%
mean real moves:    318.7 vs 319.2   (+0.16%)
```

`own_eye` matches to two decimal places. Only **1.13%** of permitting games contain one,
so filtering costs **49.73% of the corpus to address ~1% of those files** — the same
game-level-versus-move-level overpay rejected for blunders. The converter already replays
every game and truncates on the suicide, at a cost of ~0.09% of moves, and it now
*classifies* the truncation reason so a genuine engine bug stays distinguishable from the
expected background. That recovers the one thing the filter was buying.

Why suicide exists at all, since it comes up: it is the **primitive** case in KataGo's rule
formalisation (`docs/rules.html` — resolving own-colour captures last is what a minimal
definition does; *forbidding* suicide is the added clause), it is real in Tromp-Taylor and
New Zealand rules, and the 50/50 split is simply a uniform draw from
`multiStoneSuicideLegals = false,true`. Only *multi-stone* suicide is ever legal — matching
the observation that the suicide point always had 1, 2 or 4 same-colour neighbours.

**`gtype=asym` (3.48%) — NOT filtered.**
The original argument was that one side searches with fewer visits and is deliberately
weaker. That is true, and a clean controlled comparison shows it:

```
group              B visits  W visits  W/B    B blunder%  W blunder%
handicap (HA>0)       382       381    1.00      0.33%       0.35%
asym     (HA>0)       229       448    1.96      1.36%       0.69%
```

Same positions, only the search budget differs. But discarding the **file** throws away the
wrong half. In a handicap game White *always* receives the playout advantage
(`play.cpp:629`), so an asym handicap game is **a strong player facing handicap stones
against a weakened opponent** — which is this bot's KGS deployment almost exactly. White
sits at 448 visits and a 0.69% blunder rate, comparable to normal games' 0.48%.

And 229 visits is not disqualifying by this project's own standard: the ~240-visit
cheap-search moves are kept everywhere else. The weakened side is identifiable at
conversion time from the per-move visit counts if ever wanted.

Related: **84% of `asym` games are handicap games.** Mode assignment is a sequence of
overrides (`play.cpp:1633-1653`, later wins), so a handicap game that also drew asymmetric
playouts is re-labelled `asym`. Filtering `asym` would therefore have removed half of all
handicap games without that being visible from the rule.

### D2/D3 — why hint positions go

From `cpp/command/startposes.cpp:1146`:

> *"If a full search gives a different move than a quick search and judges the move to be
> way better than the quick search's move, then record as a hintpos."*

These games are **selected precisely for being surprising**. That is the same
active-learning bias rejected for KataGo's per-move `weight` (see "weight" below), applied
here as a whole-game selection criterion. Training on a corpus enriched for tactical
surprises distorts what the network believes normal play looks like.

---

## DECIDED — keep

| Rule | Share | Why kept |
|---|---|---|
| `button1` games | 16.48% | **The button is taken by playing a pass** (`cpp/game/boardhistory.cpp:975-982`: on a pass with `hasButton`, `whiteBonusScore += ±0.5` and the button clears). `convert_game` already skips all passes, so the button action **never becomes a training label**. It is worth half a point in a razor-close endgame and nothing else. Measured: moves/game median 316 vs 312, integer-komi rate 49.4% vs 50.5% — statistically indistinguishable. Discarding would cost 16.5% for no benefit. |
| Moves before `startTurnIdx` | 7.5% of moves | These are played **without search** (`policyInitAreaProp`) and carry no `v=`/`weight=` annotation. KataGo excludes them from its own recording; the leading un-annotated run equals `startTurnIdx` in **100%** of games. Kept because they are raw-policy samples from a superhuman net (legitimate distillation), they carry the temperature-driven move variety that is the selfplay analogue of human diversity, and they are **the only opening coverage available** — 94.5% of games open with at least one, mean 30.4 for `gtype=normal`. Dropping them would nearly erase the empty-board first move from training. **Held loosely — a genuine A/B candidate** (see Open Questions). |
| KataGo's per-move `weight` | — | **Not used as a filter or a sample weight.** See below. |
| All scoring rules (`TERRITORY` 50.09% / `AREA` 49.91%) | — | **No filter.** See below. |
| All ko rules (`POSITIONAL` / `SIMPLE` / `SITUATIONAL`, ~1/3 each) | — | Affects only genuine superko situations, which are rare; simple ko is handled identically regardless. Filtering would cost ~67% of data for a sub-1% behavioural difference. |
| All `tax` variants (`SEKI` / `NONE` / `ALL`) | — | Affects seki *scoring*, not move choice, and seki is uncommon. |

### Why scoring rule is NOT filtered

The theoretical divergence is real but narrow: under area scoring, playing inside your own
territory is free; under territory scoring it costs a point. That only bites when a game
**stops early** and relies on agreement about dead stones and unfilled dame.

It does not bite here, because KataGo's selfplay — like this project's GTP bot — plays
every game out to a fully resolved position. Measured on clean `gtype=normal` games only,
so nothing else confounds it:

```
scoring      games   real moves med   passes med   trailing-pass   resign%
TERRITORY    1,526        317             10           2.00          0.0%
AREA         1,644        318              6           2.00          0.0%

real moves:  TERRITORY  p10 285  p25 299  med 317  p75 335  p90 354  mean 318.3
             AREA       p10 286  p25 301  med 318  p75 338  p90 354  mean 319.6

difference in mean real moves: +1.3  (+0.42%)
```

The scoring rule changes game length by **1.3 moves out of ~319**, and the distributions
match at every percentile. Resign rate is 0.0% for both — these games are always played
out.

Note where the difference *does* appear: **passes** (median 10 territory vs 6 area —
KataGo's territory implementation uses cleanup/encore phases to resolve life-and-death).
`convert_game` discards every pass, so the largest observable ruleset difference lands
entirely in moves that never become training data.

**Direct test of the mechanism** (not a length proxy). Under territory scoring a move
inside your own territory costs a point; under area scoring it is free, so AREA games
should fill their own territory measurably more. Replayed through the engine, last 20% of
each game, `gtype=normal` + `sui0` only:

```
scoring        moves      own_eye   own_area   contact    other
TERRITORY     28,952       8.46%      8.66%    59.05%    23.83%
AREA          32,859       8.05%      8.34%    57.85%    25.76%

"inside own territory" (own_eye + own_area):
   TERRITORY 17.12%   AREA 16.38%   difference -0.74 pp
```

The predicted effect does not appear — AREA comes back marginally *lower*. Likely because
KataGo's territory implementation resolves scoring differences in its encore/cleanup
phases (the source of the extra passes), leaving the main move stream unchanged.

**Caveat on that test:** the classifier is a 4-neighbour check, not a territory flood-fill,
so it undercounts moves filling the interior of a large open territory. Those land in the
catch-all `other` bucket, where AREA is +1.93 pp higher — weak evidence in the opposite
direction.

**Why the decision is robust regardless of sign.** All three measurements are small and
point in mixed directions (+0.42% game length, −0.74 pp own-territory fill, +1.93 pp
`other`). Taking the largest at face value — ~2 pp, confined to the last 20% of moves —
filtering to `AREA` still costs **50% of the corpus**. The trade fails whichever way the
residual effect runs. That, rather than any single measurement, is what settles it.

**Retracted:** an earlier draft argued that AREA-only training would make the bot *more*
inclined to fill its own territory, hurting it against Japanese-rules opponents on KGS.
The data does not support that (AREA fills own territory marginally less). It was
speculation and is not a reason for this decision.

(Precision note: area and territory scores do not literally match on a played-out board —
they differ by the stone-count parity, typically 0–1 point depending on who moved last,
which is exactly why Chinese komi is 7.5 and Japanese 6.5. That is a scoring offset, not a
difference in optimal play.)

### Why `weight` is not used

`weight` conflates two mechanisms that transfer very differently:

- **`policySurpriseDataWeight = 0.5`** — half of KataGo's total training weight is
  deliberately concentrated on positions where its own net disagreed with its search.
  That is residual-driven active learning: correct for KataGo, close to the opposite of
  what a supervised imitation policy wants. Confirmed in the data: `weight > 0` moves show
  a **3x larger** mean winrate gain for the mover (0.0129 vs 0.0045) — the signature of
  "search found something the net missed."
- **`cheapSearchTargetWeight = 0.0`** (70.5% of moves) — KataGo trains on the **full MCTS
  visit distribution** as a soft policy target, and a 100–240 visit distribution is noisy.
  We train on a **single hard one-hot label: the move actually played**. A b28c512 net at
  240 visits still picks an excellent move. Blunder rates confirm the choice is fine:
  0.62% (weight==0) vs 0.32% (weight>0) — both tiny.

Filtering on `weight > 0` would discard **70% of moves, systematically biased toward
tactically surprising positions**. That is exactly the distortion to avoid.

---

## DECIDED — rating games are not used

KataGo's distributed run publishes both **training** (selfplay) and **rating** (gatekeeper)
games. Rating games were evaluated as a possibly-cleaner corpus and **rejected**. Measured
on a 90-game sample against the 60,136-file training corpus:

| | training | rating |
|---|---|---|
| `gtype` | 52% normal, rest diversified | **100% normal** |
| setup stones | 43.7% of files | **0%** |
| unsearched pre-`startTurnIdx` moves | 7.5% of moves | **0%** |
| visits/move | 240 median (cheap) / 1026 (full) | **933 median, min 600, all full** |
| `weight=` | present | **absent** |
| komi | 538 distinct, 14.6% negative | **10 distinct, 91% in {6, 6.5, 7}** |
| resignation | 0.0% | **90.0%** |
| moves/game | 319 median | **203** |
| decided positions | 48.1% | **4.4%** |
| `moveless_node` bug | 0 / 60,136 | **89 / 90** |
| both sides same network | yes | **no — median 2,413M step gap** |

Rating games use `PlaySettings::loadForGatekeeper` (`cpp/program/playsettings.cpp`), which
sets only `allowResignation`, `resignThreshold`, `resignConsecTurns` and
`compensateKomiVisits` — every diversification mechanism stays at its zero default. That
is why they are so clean.

**Rejected for two disqualifying reasons:**

1. **The blunder filter cannot work on them.** Measured 28.16% of moves losing >5% winrate
   versus 0.53% in training games — a 50x difference that is an artifact, not a signal. The
   two sides are **different networks**, so consecutive winrate evaluations come from
   different evaluators and the delta measures *network disagreement*, not move quality.
2. **No endgame.** 90% resignation, median 203 moves vs 319. Resignation removes precisely
   the endgame coverage that is KataGo data's advantage over human corpora — making this
   corpus shallower than KGS/GoGoD in the one dimension where selfplay was winning.

Also: one side is always a weaker network (median 2,413M training steps apart, max 4,689M).
Mitigable by parsing the step counts out of `PB`/`PW`, but it is the same weak-player
concern as `asym` and arguably worse.

Worth keeping in mind if the situation changes: rating games would make D3–D6, the
setup-stone question, the pre-`startTurnIdx` debate and the `weight` debate all moot. They
are a genuinely cleaner *opening/middlegame* corpus. They are just not a complete one.

---

## DECIDED — position-level decisions live in the converter

`game_converter_katago_data.py` applies every position-inclusion decision, and the h5 then
contains only positions intended for training. Nothing is filtered at training time. Three
reasons, in order of weight:

1. **Per-epoch I/O.** The shuffle buffer re-reads positions from disk every epoch, so a
   position discarded in the trainer is decompressed and thrown away once per pass for the
   whole run. At 500M positions with 24% discarded that is ~315GB of wasted reads *per
   epoch*. Disk read is precisely the bottleneck the shuffle buffer exists to work around.
2. **Trainer invariants.** "Every position exactly once per pass" stays true. Filtering
   there would make `n_train_data` - and therefore `steps_per_epoch` - depend on a mask
   rather than on `file_offsets` lengths: a silently-wrong-epoch-length bug waiting to
   happen, in code this audit just certified as correct.
3. **Re-conversion is ~4 hours**, so baking a decision in is cheap to undo.

Note the disk-space argument does NOT apply and was wrong in an earlier draft: the target
is a position count, not a game count, so filtering changes how many source games are
needed, not the output size.

### Counterfactual reporting

The converter reports what each threshold *would* cost even when the filter is off, since
it has already parsed the winrates. Measured on 400 real games:

```
positions written : 117,703
  hopeless_05      25,500  (21.66%)
  loss_gt05           555  ( 0.47%)
  loss_gt10           134  ( 0.11%)
  loss_gt20            19  ( 0.02%)
  skip_setup_7      1,028  ( 0.87%)
```

Those track the standalone measurements elsewhere in this document (24%, 0.53%, 0.14%,
~1%). **One short run sizes every open threshold against the real corpus**, which removes
the need to convert once per variant - and was most of what an earlier "store metadata in
the h5" proposal was trying to buy, without putting anything in the h5.

### Truncation is classified, not merely counted

Every game the engine cannot fully replay keeps its prefix and is recorded by reason:
`illegal_suicide`, `illegal_occupied`, `illegal_ko`, `setup_node`, `parse_error`, `other`.
The summary marks suicide as **expected** and everything else as **UNEXPECTED:
investigate**. This is what makes keeping `sui1` games safe (see REVERSED above): a genuine engine
bug stays visible against the background of expected suicide truncations, instead of being
indistinguishable from it.

---

## DECIDED — move-level (game retained)

| Rule | Cost | Why |
|---|---|---|
| Suppress the label on moves losing > N% winrate | ~0.14% at N=10, ~0.55% at N=5 | KataGo records `win loss noResult score` at every move from a **constant White perspective** (`cpp/dataio/trainingwrite.h:24`), so the winrate delta across a move directly measures *was this move good* — the actual objective, which `weight` does not encode. **Threshold still open.** |

**Why move-level, not game-level.** Discarding whole games on any blunder costs ~112x more
than it buys:

| threshold | games killed | positions lost | vs. dropping just the move |
|---|---|---|---|
| >5% | 59.6% | **61.7%** | 0.55% |
| >10% | 27.6% | **28.5%** | 0.14% |
| >20% | 5.9% | 5.9% | 0.02% |

Only 40.4% of games are blunder-free (mean ~1.5 blunders per ~270-move game), so
game-level rejection deletes most of the corpus.

Two further reasons to keep the game:

- **The position *after* a blunder is high-value data.** The bot plays humans who blunder
  constantly; "opponent erred, here is how a strong player punishes it" is directly useful.
- **Suppressing a label is safe; skipping a move is not.** The converter still calls
  `do_move`, it just does not `yield` that position — no board desync. (Contrast the
  suicide case, where skipping the *move* provably desyncs: 11 `occupied` rejections
  followed a skip, and 6 of 13 games silently accepted every later move on a wrong board.)

Phase skew is negligible at this rate: dropped fraction is 0.39% (opening) / 0.93%
(middlegame) / 0.06% (endgame) at the 5% threshold.

---

## DECIDED — structural rejects

| Rule | Measured | Why |
|---|---|---|
| Unparseable SGF | 0 in 3,000 | Cannot be used. |
| Zero moves | — | No training data. |
| Node carrying neither `B` nor `W` | **0 in 60,136** training, **89/90** rating games | **NOT a file reject.** Since the `sgf_iter_states` fix, annotation-only nodes are skipped cleanly and only board-altering ones (`AB`/`AW`/`AE`/`PL` outside the root) raise — which the converter handles by truncating with the prefix kept. Rejecting the file would discard every KataGo rating game for no benefit. Counted in the manifest as information only. |

---

## OPEN — still to decide

**All of these are now sizable from a single short converter run** (see Counterfactual
reporting above), so none needs a dedicated experiment to cost out - only to evaluate.


1. **Komi band beyond "not negative".** 538 distinct values; only 38.42% fall in
   [5.5, 7.5]. Tightening reduces hidden-variable label noise but costs data. Note this is
   a different kind of argument from the scoring one — komi is genuinely invisible to the
   network *and* genuinely changes correct play, whereas the scoring rule changes almost
   nothing about the move stream.
2. **Setup-stone gtypes** (`sgfpos` 30.61%, `fork` 5.40%, `handicap` 2.94%). Discard
   outright, or keep and skip the first 8 positions at conversion time? Keeping preserves
   KataGo's deliberate position diversity at ~97% of their positions; discarding is
   simpler. See AUDIT_NOTES §1.
3. **Blunder threshold** — 5%, 10% or 20%.
4. **Passes and `turns_since`.** Median 8 passes per game (597,962 total). Passes are
   skipped as labels but still enter `moves_history`, and `get_turns_since` increments its
   age counter on them, so the most recent real move is reported as older than it is.
   Converter-level issue, not a file filter. Note territory-scored games carry noticeably
   more passes (median 10 vs 6), so this interacts with the scoring mix.

---

## Yield with the DECIDED file-level rules

Measured over all 60,136 files (not assumed independent):

```
rejected (any rule) : 36,415  (60.55%)
KEPT                : 23,721  (39.45%)
moves kept          : 7,204,169  (40.21% of all moves)
                    = ~304 moves per surviving game
```

Source games needed, and what each stage costs:

| target positions | source games | scan | convert |
|---|---|---|---|
| 40M (small test set) | ~150,000 | ~2 min | ~2 hours |
| 70M (previous production scale) | ~260,000 | ~4 min | ~4 hours |

Scan rate measured at **1,242-1,992 files/sec** on Windows/Docker - note the ~55% spread on
identical work, which is cold-vs-warm filesystem cache, not parsing. Budget off the low
end. Parsing cost is in the noise: the per-move aggregates add only ~3.5%.

Billions of files are unnecessary; a few hundred thousand suffice.

Adding the open filters (scoring, tighter komi) would roughly halve to quarter these
survival rates and correspondingly increase the games needed.

---

## The pipeline, as built

```
sgf_analyze    <dir>                          survey; decides nothing
sgf_preparation scan   <dir> manifest.jsonl   one row per file, drops NOTHING
sgf_preparation select manifest.jsonl keep.txt   policy -> keep-list, never reads an SGF
game_converter_katago_data keep.txt out_dir/  position decisions -> shard_NNNNN.h5
supervised_policy_trainer_v3 model.json out_dir/ run/   unchanged
```

Design points worth not re-deriving:

- **`scan` records rejections as advisory and acts on none of them.** Policy lives in
  `select`, so re-tuning criteria costs seconds instead of a re-scan.
- **Keep-list, not deletion.** Nothing is ever unrecoverable without a re-scan.
- **Board size short-circuits** in `scan` - it is the one criterion never revisited, so a
  wrong-size file is recorded and the remaining regexes and move-stat pass are skipped.
  The `gtype` rejects deliberately do NOT short-circuit: they are only ~5% of files, and
  we changed our minds about gtypes twice while writing this document.
- **Every manifest row carries every key**, null where absent. A manifest exists to be
  queried, and a consumer should never have to distinguish "field absent" from "null".
- **`game_hash` is captured** because cross-download duplicate detection is free here and
  expensive to add later.
- **Converter output is a backward-compatible superset** - the same datasets
  `build_game_index` already reads, plus a `conversion_args` scalar. Shards train with the
  existing `v3` trainer with no trainer changes. The provenance string matters because a
  final shard set outlives the session that produced it, and "which filters made this?" is
  not answerable from a directory listing.
- **Sharding is by target bytes** (default 20GB), matching previous practice and the `v3`
  trainer's note that a single file degrades badly on spinning disk past ~100-150GB.

### Traps found while building these

- Reading each file twice (once for the header scan, once for move stats) **halved**
  throughput - 912 vs 1,716 files/sec. The open() dominates, not the regexes.
- A shard-size check keyed to a game count never fires on a short run and writes one
  oversized shard. It must be driven by measured bytes-per-position.
- A roll landing on the last write leaves an **empty trailing shard**, which would hand
  `build_game_index` a shard with no `file_offsets`.
- `_RE_ANNOT` requiring `weight=` made the analyzer report "0% annotated" on any corpus
  without training weights - which is how rating games initially looked unannotated.
