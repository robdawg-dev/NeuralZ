# deprecated

Pure-Python originals, superseded by the Cython implementations under `AlphaGo/`. Kept for
reference, not maintained, and not imported by anything in the project.

| file | superseded by |
|---|---|
| `go_python.py` | `AlphaGo/go/` (`game_state.pyx`, `group_logic.pyx`, `ladders.pyx`, ...) |
| `preprocessing_python.py` | `AlphaGo/preprocessing/preprocessing.pyx` |
| `game_converter_python.py` | `AlphaGo/preprocessing/convert_shuffled.py` (current), `game_converter.py` / `game_converter_parallel.py` (older shard format) |

They were moved here from `AlphaGo/` and `AlphaGo/preprocessing/`. Their imports of each
other were changed from `AlphaGo.go_python` / `AlphaGo.preprocessing.preprocessing_python`
to plain module names, so they resolve when run from inside this directory. Nothing else
was changed.

Two cautions if you ever read them as a reference:

- **They are not a behavioural oracle for the Cython engine.** The two diverged; where they
  disagree, the Cython version is what trains the models and what plays.
- `preprocessing_python.py` still imports `keras.backend`, which the rest of the project no
  longer uses this way.
