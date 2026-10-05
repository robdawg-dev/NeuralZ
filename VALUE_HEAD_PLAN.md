# Value head plan

Add a value head to b20c256 so the bot can tell which of its own top moves is best.

**Why.** In 7 KGS games (2026-10-02, `workspace/sample_ratio/`), the bot's 90 biggest
mistakes cost 679 points. In 50 of them (418 points), KataGo's move was already among
the bot's top 5 policy moves; in 66 (522 points), among its top 10. The bot often ignored
the same urgent point for many moves (21 moves in one game). A policy-only bot plays its
top choice and cannot tell that its 2nd or 4th choice saves a group.

## Decisions

| Question | Decision |
|---|---|
| Value target | KataGo's search win rate for the position (SGF comment), not the game result |
| Score | Also predict KataGo's expected score lead |
| Training | Frozen trunk first: b20c256's trunk and policy head fixed, train only the new head. Joint fine-tune only if needed |
| Komi | Input to the value head only (not an input plane), so the trunk stays frozen |

## What the data holds (verified)

Every move node in a KataGo self-play SGF from turn `startTurnIdx` on (usually 2) carries
`C[win loss noResult score v=N ... weight=W]`: KataGo's search result, **from White's
perspective**, for the position **before** that node's move. This is the same search that
chose the move. KataGoFresh `cpp/program/play.cpp` pushes `whiteValueTargetsByTurn`
during the turn's search, and `cpp/dataio/sgf.cpp` writes entry `i` on move node `i`
before applying the move. So a shard record (position before move *i*, move *i*) gets its
value label from move node *i*'s comment. The last node also has `result=...`, and the
root has `KM[]`.

Each shard record stores `game_id` (row of `<split>/games.tsv`, which gives the SGF path)
and `move` (move number in that game). So targets can be looked up per record without
re-converting the 60M-position shards.

## Steps

### 1. Value targets (sidecar files)

`AlphaGo/preprocessing/add_value_targets.py` (done). For each split, it parses every game
in `games.tsv` once (move-node annotations, `KM`), then for each `shard_NNNNN.h5` writes
`value_NNNNN.h5` with one row per record, in the same order. (The name is not
`shard_NNNNN.value.h5`, which `find_split_shards`' `shard_*.h5` glob would pick up as a
shard.) Datasets:

- `value`: win / (win + loss) **for the player to move** (float16)
- `score`: expected score lead for the player to move (float16)
- `komi`: komi from the player to move's side (+komi for White, -komi for Black)
- `has_target`: 0 where the node has no annotation (the first moves); masked out of the loss

The golden shards are not modified. A shard's `move` is the move-node index
`convert_shuffled` took from `sgf_iter_states` (passes and skipped positions still count),
and the annotations are parsed with `convert_shuffled`'s own regex, which indexes nodes the
same way. **Every record is checked:** the move parsed from its node must equal the record's
stored action, and any mismatch stops the run. On the val split (3.0M records): 0
mismatches, and 93.2% of records have a target. The rest are the first two moves and
games that KataGo started from a later position (`startTurnIdx` > 2, e.g. 152), whose
earlier positions were never searched in that game. Sanity checks: the mean value for the
side to move is 0.497, the win-rate and score signs agree 97% of the time, and komi varies
(6, 6.5, 7 are all common).

### 2. Model

`PolicyValueNet` in `AlphaGo/models/value.py` (done). A value head on b20c256's trunk output (after the final BN + ReLU), in KataGo's shape
(`ValueHead` in `python/katago/train/model_pytorch.py`): 1x1 conv to 32 channels → BN +
ReLU → global mean + max pooling → concatenate komi (scaled) → dense 64 + ReLU → two
outputs:

- `value`: 1 unit, sigmoid, binary cross-entropy against the soft target
- `score`: 1 unit, linear, Huber loss on score / 20 (so a few points matter but 80-point
  blowouts don't dominate)

Inputs: planes plus komi. Outputs: policy (unchanged), value and score. b20c256's weights
load into the shared layers, which are frozen (`trainable=False`). The trunk output is
found as the input of the policy head's first 1x1 conv (every trunk conv is 3x3). On the
real b20c256 the policy output is bit-identical, and 12,610 of 23.3M weights train. The
value and score outputs stay float32 under mixed precision; `NeuralNetBase.load_model` now
keeps the dtype of every non-ReLU activation, not only the softmax.
`deploy/build_deploy.py` ships `value.py`, since the loader imports it.

### 3. Training

`AlphaGo/training/value_head_trainer.py` (done), separate so the policy trainer is untouched.
Adam with cosine decay; win-rate binary cross-entropy plus a Huber loss on score / 20,
both weighted by `has_target`. Each epoch saves the whole policy+value network
(`model.json` + `weights.NNNNN.weights.h5`) and appends its metrics to `metadata.json`.
It reads shards plus sidecars through `shard_stream` (planes decoded on device as now;
board symmetries change the planes but not the value, score or komi), and validates on
the val split.

Metrics:
- value log-loss
- mean absolute error vs. KataGo's win rate
- share of positions where the head picks the side KataGo favours
- score mean absolute error in points

With a frozen trunk only the head is trained, but every position still runs the full
trunk forward pass. A subset of the 57M training positions may be enough for a head this
small; the learning curve will show it.

### 4. Offline evaluation

- Val-split metrics above.
- **The 90 mistake positions:** for each, evaluate the bot's top 5 / top 10 moves with
  one-ply lookahead and check whether the chosen move is one KataGo rates better than
  what the bot played. This uses the KataGo tooling in `workspace/sample_ratio/`.

### 5. Play

- `go_server.py` returns value and score with the policy.
- `ai.py`: one-ply lookahead. Evaluate the positions after the top k policy moves (one
  batched server call) and play the move with the best value for the bot, possibly
  blended with the policy probability. The sampling window works as now, on top of the
  lookahead's candidates.
- Matches against the policy-only bot, then KGS games and the same KataGo point-loss
  analysis.

A small MCTS is a later option if one-ply lookahead isn't enough.

## Known limits

- **Rules:** no rules input. KataGo's data mixes area/territory scoring and tax rules;
  this shifts values by about a point.
- **Komi range:** komi is mostly 7.5 in the training data. Handicap games (part of the
  selection) give some range, but 0.5-komi positions will be rarer than in the bot's KGS
  games.
- **No ownership target:** KataGo also trains on an ownership map, which helps its value
  head learn; our SGFs don't record one.
