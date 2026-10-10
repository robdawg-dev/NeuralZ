# Benchmarks

Speed measurements for playing (not training). Each script appends its runs to
`results/*.jsonl` (with the time, machine and summary), so results from different code
versions and machines can be compared. Conclusions go in [MEASUREMENTS.md](../MEASUREMENTS.md).

| Script | Measures |
|---|---|
| `move_time.py` | the bot's own work per move, by game stage: input planes (all, and each feature), ladder guard, legal moves; `--server` adds the round trip |
| `server_round_trip.py` | a running `go_server.py`, split by its `X-NeuralZ-Timing` header into decode / batch wait / model call / HTTP; `--bots N` sends from N threads at once, like N bots on one server |

`sample_positions.json` holds 169 positions from KGS bot games (several per game, moves
30-280); both scripts use them by default. Rebuild them with
`move_time.py --build <game folders>`.

## Typical runs

```
uv run python benchmarks/move_time.py                     # client work, no server needed
uv run python go_server.py <model.json> <weights.h5> &     # then:
uv run python benchmarks/server_round_trip.py --bots 8     # server under 8 bots' load
uv run python benchmarks/server_round_trip.py --game-log bot.log --game 1   # one real game
```

On a deployment machine, run `server_round_trip.py --bots <number of bots>` against
servers started with different `--threads`, `--max-batch` and `--batch-wait-ms` before
changing `start_server.sh`. Timings from the dev PC don't transfer to it.

## Profiling

The profilers are an optional extra: `uv sync --extra profile`.

- **Where the client's time goes:** `move_time.py --profile out.prof`, then
  `uv run snakeviz out.prof`. By default each Cython function is one opaque call. To see
  inside them (each feature plane, the ladder reader), build the extensions with profiling
  hooks first:

  ```
  NEURALZ_CYTHON_PROFILE=1 uv run python setup_cython.py build_ext --inplace
  ```

  Those builds are several times slower. Rebuild normally before timing anything or
  deploying (`uv run python setup_cython.py build_ext --inplace`; switching modes
  regenerates the C++ by itself). `move_time.py` warns if it is timing a profiling build.
- **A running process** (go_server, or a bot during a game): `uv run py-spy top --pid N`,
  or `uv run py-spy record --pid N -o profile.svg` for a flame graph. Add `--native` to
  see inside the compiled code without a profiling build.
- **Line by line:** `line_profiler` (`kernprof`) on Python code, and on the `def`/`cpdef`
  functions of a profiling build.
- **A quick call tree:** `uv run pyinstrument script.py`.

Example: `move_time.py` showed the ladder planes growing with the game (up to ~100 ms a
position); the cause was `update_legal_moves`, which every ply of a ladder read called
twice (fixed 2026-10-09, see MEASUREMENTS.md).
