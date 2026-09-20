#!/usr/bin/env python
"""Time SGF-to-feature-tensor conversion over a slice of real game files.

Reports one-time startup cost (import + GameConverter construction, which
includes compiling the Preprocess feature pipeline) separately from the
steady-state per-game conversion rate, then extrapolates to a full corpus
run. This replaces running the converter by hand at two different game
counts and estimating the slope, which was the previous ad hoc methodology.

Pass --processes N (N > 1) to additionally convert the same games with a
process pool and compare wall-clock throughput against the sequential
baseline.

Note on why this is --processes and not --threads: GameState.do_move and
Preprocess.state_to_tensor are Cython/C++ with no 'nogil' sections, and the
SGF parser is pure Python, so the whole per-game cost holds the GIL -
measured threaded runs came out at 0.94-0.97x sequential, i.e. no gain.
Worse, AlphaGo/go/game_state.pyx keeps its neighbor/zobrist lookup tables as
globals that are reinitialized in place whenever a GameState of a different
board size is constructed; since the real corpus mixes board sizes,
concurrent threads hitting different sizes raced on that reinit and
segfaulted. Separate processes don't share that global state (each has its
own copy), and each runs on its own core, so this is the correct way to
parallelize this workload.

Usage:
    python benchmarks/preprocessing_benchmark.py --num-games 30
    python benchmarks/preprocessing_benchmark.py --num-games 200 --processes 8
"""
import argparse
import concurrent.futures
import os
import statistics
import sys
import time
import warnings

import sgf as sgf_module

p = os.path
parentdir = p.abspath(p.join(p.dirname(__file__), ".."))
sys.path.append(parentdir)

start = time.perf_counter()
from AlphaGo import go  # noqa: E402
from AlphaGo.preprocessing.game_converter import GameConverter, SizeMismatchError  # noqa: E402

ALL_FEATURES = [
    "board",
    "ones",
    "turns_since",
    "liberties",
    "capture_size",
    "self_atari_size",
    "liberties_after",
    "ladder_capture",
    "ladder_escape",
    "sensibleness",
    "zeros"]

# Mirrors GameConverter.sgfs_to_hdf5's error handling (the real production path): the
# real corpus has a handful of malformed/non-square-board SGFs that shouldn't derail a
# benchmark run any more than they'd derail a full conversion.
SKIPPABLE_ERRORS = (SizeMismatchError, go.IllegalMove, sgf_module.ParseException, ValueError)

# Set by _init_worker in each worker process; a GameConverter isn't picklable (it wraps a
# Cython Preprocess object), so each process builds its own once instead of one being
# passed in from the parent.
_worker_converter = None


def _init_worker(features):
    global _worker_converter
    _worker_converter = GameConverter(features)


def _convert_one(converter, sgf_file, board_size):
    """Run one game through the converter, returning None for a game that was skipped."""
    try:
        for _ in converter.convert_game(sgf_file, board_size):
            pass
    except SKIPPABLE_ERRORS as e:
        warnings.warn("skipping {}: {}".format(sgf_file, e))
        return None
    return True


def _convert_one_in_worker(sgf_file, board_size):
    return _convert_one(_worker_converter, sgf_file, board_size)


def _find_sgfs(root, limit):
    """Yield up to `limit` SGF file paths under root, walked in sorted order
    for reproducibility across runs."""
    found = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.endswith(".sgf"):
                yield os.path.join(dirpath, filename)
                found += 1
                if found >= limit:
                    return


def time_sequential(converter, sgf_files, board_size):
    per_game_times = []
    skipped = 0
    for sgf_file in sgf_files:
        game_start = time.perf_counter()
        result = _convert_one(converter, sgf_file, board_size)
        if result is None:
            skipped += 1
            continue
        per_game_times.append(time.perf_counter() - game_start)
    return per_game_times, skipped


def time_multiprocess(sgf_files, board_size, features, num_processes):
    """Convert the same games concurrently via a process pool, timing overall wall clock
    (including pool startup: each worker process re-imports AlphaGo and builds its own
    GameConverter, which is a real cost worth including for small batches)."""
    completed = 0
    skipped = 0
    wall_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes, initializer=_init_worker, initargs=(features,)) as pool:
        futures = [pool.submit(_convert_one_in_worker, f, board_size) for f in sgf_files]
        for future in concurrent.futures.as_completed(futures):
            if future.result() is None:
                skipped += 1
            else:
                completed += 1
    wall_elapsed = time.perf_counter() - wall_start
    return completed, skipped, wall_elapsed


def run_benchmark(sgf_dir, num_games, features, board_size, extrapolate_to, num_processes):
    startup_elapsed = time.perf_counter() - start

    converter_start = time.perf_counter()
    converter = GameConverter(features)
    converter_elapsed = time.perf_counter() - converter_start

    sgf_files = list(_find_sgfs(sgf_dir, num_games))
    if not sgf_files:
        raise ValueError("No .sgf files found under {}".format(sgf_dir))

    per_game_times, skipped = time_sequential(converter, sgf_files, board_size)
    if not per_game_times:
        raise ValueError("All {} game(s) were skipped (size mismatch)".format(len(sgf_files)))

    total_game_time = sum(per_game_times)
    mean_per_game = statistics.mean(per_game_times)
    median_per_game = statistics.median(per_game_times)
    sequential_rate = 1.0 / mean_per_game

    print("features:              {}".format(", ".join(features)))
    print("games processed:       {} ({} skipped for size mismatch)".format(
        len(per_game_times), skipped))
    print("import overhead:       {:.3f}s".format(startup_elapsed))
    print("GameConverter() build:  {:.3f}s".format(converter_elapsed))
    print("[sequential] total conversion time: {:.3f}s".format(total_game_time))
    print("[sequential] mean time/game:        {:.4f}s ({:.1f} games/sec)".format(
        mean_per_game, sequential_rate))
    print("[sequential] median time/game:      {:.4f}s".format(median_per_game))

    if extrapolate_to:
        one_time = startup_elapsed + converter_elapsed
        estimate = one_time + extrapolate_to * mean_per_game
        print("[sequential] estimated time for {} games: {:.1f}s ({:.1f} min)"
              .format(extrapolate_to, estimate, estimate / 60.0))

    if num_processes and num_processes > 1:
        completed, proc_skipped, wall_elapsed = time_multiprocess(
            sgf_files, board_size, features, num_processes)
        mp_rate = completed / wall_elapsed
        speedup = mp_rate / sequential_rate
        print()
        print("[processes x{}] games processed: {} ({} skipped)".format(
            num_processes, completed, proc_skipped))
        print("[processes x{}] wall-clock time: {:.3f}s ({:.1f} games/sec)".format(
            num_processes, wall_elapsed, mp_rate))
        print("[processes x{}] speedup vs sequential: {:.2f}x".format(num_processes, speedup))
        if extrapolate_to:
            estimate = extrapolate_to / mp_rate
            print("[processes x{}] estimated time for {} games: {:.1f}s ({:.1f} min)"
                  .format(num_processes, extrapolate_to, estimate, estimate / 60.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sgf-dir", default="SampleGames",
                        help="Directory to search (recursively) for .sgf files (default: SampleGames)")  # noqa: E501
    parser.add_argument("--num-games", type=int, default=30,
                        help="Number of games to time (default: 30)")
    parser.add_argument("--features", default="all",
                        help="Comma-separated feature list, or 'all' (default: all)")
    parser.add_argument("--size", type=int, default=19,
                        help="Board size; games of a different size are skipped (default: 19)")
    parser.add_argument("--extrapolate-to", type=int, default=200000,
                        help="Report an estimated total runtime for this many games (default: 200000; pass 0 to disable)")  # noqa: E501
    parser.add_argument("--processes", type=int, default=1,
                        help="If > 1, also convert the same games with this many worker processes and report the speedup (default: 1, i.e. sequential only)")  # noqa: E501
    args = parser.parse_args()

    features = ALL_FEATURES if args.features.lower() == 'all' else args.features.split(",")
    run_benchmark(args.sgf_dir, args.num_games, features, args.size, args.extrapolate_to,
                 args.processes)
