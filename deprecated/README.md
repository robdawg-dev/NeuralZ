# deprecated

Superseded code, kept for reference. Not maintained, and not imported by anything in the
project.

## Pure-Python originals

Superseded by the Cython implementations under `AlphaGo/`.

| file | superseded by |
|---|---|
| `go_python.py` | `AlphaGo/go/` (`game_state.pyx`, `group_logic.pyx`, `ladders.pyx`, ...) |
| `preprocessing_python.py` | `AlphaGo/preprocessing/preprocessing.pyx` |
| `game_converter_python.py` | `AlphaGo/preprocessing/convert_shuffled.py` |

They were moved here from `AlphaGo/` and `AlphaGo/preprocessing/`. Their imports of each
other were changed from `AlphaGo.go_python` / `AlphaGo.preprocessing.preprocessing_python`
to plain module names, so they resolve when run from inside this directory. Nothing else
was changed.

Two cautions if you ever read them as a reference:

- **They are not a behavioural oracle for the Cython engine.** The two diverged; where they
  disagree, the Cython version is what trains the models and what plays.
- `preprocessing_python.py` still imports `keras.backend`, which the rest of the project no
  longer uses this way.

## Per-game shard pipeline

The pipeline before `convert_shuffled.py`. Superseded by the one described in
`DATA_PIPELINE.md`.

| file | superseded by |
|---|---|
| `game_converter.py` | `AlphaGo/preprocessing/convert_shuffled.py` |
| `game_converter_parallel.py` | `AlphaGo/preprocessing/convert_shuffled.py` |
| `shuffle_buffer.py` | `AlphaGo/training/shard_stream.py` |
| `supervised_policy_trainer_v3.py` | `AlphaGo/training/supervised_policy_trainer_v4.py` |

The converters wrote each game as a contiguous block of rows, with a `file_offsets` group
mapping every SGF to its `(start, length)`. `shuffle_buffer.py` used those offsets to split
train/val/test by game and to shuffle positions through a 400k-position in-memory buffer;
v3 trained from it. The formats are not interchangeable: v3 cannot read
`convert_shuffled.py` shards (no `file_offsets`), and v4 cannot read these (no `train/` and
`val/` split directories).

They were moved here from `AlphaGo/preprocessing/` and `AlphaGo/training/`. Their imports
of each other were changed to plain module names (`shuffle_buffer`, `game_converter`);
imports of live `AlphaGo` modules were left as they were, so run them from inside this
directory with the repository root on `PYTHONPATH`. Nothing else was changed.

v3's training callbacks were carried into v4 unchanged.
