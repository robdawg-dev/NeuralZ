#!/usr/bin/env python
"""Multiprocessing counterpart to GameConverter.sgfs_to_hdf5 / game_converter.py's CLI.

Kept as a separate module rather than folded into game_converter.py so the simple
sequential path stays simple. Worker processes each build their own GameConverter (a
Cython Preprocess object isn't picklable, so it can't just be handed to workers) and
convert one SGF file at a time fully in memory; only the main process ever touches the
output HDF5 file, so the file format and file_offsets bookkeeping are identical to the
sequential version - workers are used purely as a compute pool, not for the writes.

This is processes, not threads: benchmarks/preprocessing_benchmark.py measured that
threading this workload gains nothing (0.94-0.97x sequential - the SGF parser is pure
Python and GameState.do_move/Preprocess.state_to_tensor are Cython/C++ with no 'nogil'
sections, so the whole per-game cost holds the GIL), and worse, AlphaGo/go/game_state.pyx
keeps its neighbor/zobrist lookup tables as globals that are reinitialized in place
whenever a GameState of a different board size is constructed - since the real corpus
mixes board sizes, concurrent threads hitting different sizes raced on that reinit and
segfaulted. Separate processes each get their own copy of that global state, so there's
nothing to race on. Measured against real SampleGames data: ~1.9x at 2 workers, ~5.4x at
8, ~9x at 16 (sub-linear past 8, likely a mix of hyperthreading and per-task overhead).
"""
import argparse
import concurrent.futures
import os
import sys
import warnings

import h5py as h5
import numpy as np
import sgf

import AlphaGo.go as go
from AlphaGo.preprocessing.game_converter import GameConverter, SizeMismatchError

# Set by _init_worker in each worker process.
_worker_converter = None


def _init_worker(features):
    global _worker_converter
    _worker_converter = GameConverter(features)


def _convert_one_file(file_name, bd_size):
    """Run in a worker process: convert one SGF file fully in-memory, mirroring
    GameConverter.sgfs_to_hdf5's per-file error handling. Returns a plain (picklable)
    dict describing the outcome rather than writing to HDF5 directly.
    """
    pairs = []
    error_type = None
    error_message = None
    try:
        for state, move in _worker_converter.convert_game(file_name, bd_size):
            pairs.append((state[0], move))
    except go.IllegalMove:
        error_type = "illegal_move"
    except sgf.ParseException:
        error_type = "parse_exception"
    except SizeMismatchError:
        error_type = "size_mismatch"
    except Exception as e:
        error_type = "other"
        error_message = str(e)
    return {
        "file_name": file_name,
        "pairs": pairs,
        "error_type": error_type,
        "error_message": error_message,
    }


def sgfs_to_hdf5_parallel(features, sgf_files, hdf5_file, bd_size=19, num_workers=None,
                          ignore_errors=True, verbose=False, resume=False, flush_every=500):
    """Convert all files in sgf_files into an hdf5 group, using a process pool.

    Arguments mirror GameConverter.sgfs_to_hdf5, plus:
    - num_workers : number of worker processes (default: os.cpu_count())
    - resume : if True and a previous incomplete run's tmp file exists, continue from
      it instead of starting fresh - files already recorded in that tmp file's
      file_offsets are skipped, and writing continues from where it left off. Intended
      for long unattended runs (e.g. the full ~330k game / ~5 hour corpus conversion)
      that may not survive to completion in one sitting. If no tmp file exists, this
      behaves like a normal fresh start regardless of this flag.
    - flush_every : force an HDF5 flush (persist metadata to disk, not just data) every
      this many files. Without this, an abrupt kill (SIGKILL, OOM, host reboot - anything
      that doesn't let this process run its own cleanup) can leave the entire file
      unreadable even though the actual bytes are on disk, because HDF5's superblock
      metadata was never persisted. Lower values bound how much work a real crash can
      lose at the cost of more frequent (moderately expensive) flushes.

    Produces the exact same HDF5 layout as GameConverter.sgfs_to_hdf5 (see that
    docstring for the states/actions/file_offsets schema).
    """
    sgf_files = list(sgf_files)
    n_features = GameConverter(features).n_features
    # ProcessPoolExecutor accepts num_workers=None directly (meaning "use os.cpu_count()"),
    # but max_in_flight below needs an actual int to multiply.
    num_workers = num_workers or os.cpu_count()

    if os.path.exists(hdf5_file):
        raise ValueError(
            "{} already exists (a previous run already completed). Move/remove it first "
            "if you want to redo the conversion.".format(hdf5_file))

    # make a hidden temporary file in case of a crash.
    # on success, this is renamed to hdf5_file
    tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
    tmp_exists = os.path.exists(tmp_file)

    if tmp_exists and not resume:
        raise ValueError(
            "{} already exists from a previous incomplete run. Pass resume=True to "
            "continue it, or remove the file first to start over.".format(tmp_file))

    resuming = tmp_exists and resume
    h5f = h5.File(tmp_file, 'a' if resuming else 'w')

    try:
        states = h5f.require_dataset(
            'states',
            dtype=np.uint8,
            shape=(1, bd_size, bd_size, n_features),
            maxshape=(None, bd_size, bd_size, n_features),
            exact=False,
            # Deliberately small (64 rows, ~1.1MB/chunk): training reads positions in a
            # fully random global shuffle, and HDF5 must decompress an entire chunk to
            # read any part of it. A larger chunk cuts write-time chunk-index overhead
            # but multiplies training-time read amplification by the same factor - e.g.
            # at 2048 rows/chunk, one random 17KB position read pulls ~5.3MB compressed
            # (~300x overhead), repeated ~2-3 billion times across a full training run
            # (every position, every epoch). That's certain and severe; the write-side
            # benefit of larger chunks was never actually confirmed (see the sharding
            # approach in run_game_converter_parallel for how write-time scaling is
            # actually being addressed instead: many smaller files rather than fewer,
            # bigger chunks).
            chunks=(64, bd_size, bd_size, n_features),
            compression="lzf")
        actions = h5f.require_dataset(
            'actions',
            dtype=np.uint8,
            shape=(1, 2),
            maxshape=(None, 2),
            exact=False,
            chunks=(64, 2),  # matches states' row-chunking; see the comment there
            compression="lzf")
        file_offsets = h5f.require_group('file_offsets')
        features_str = ','.join(features)
        if 'features' not in h5f:
            h5f['features'] = np.bytes_(features_str)
        elif resuming:
            # Without this check, resuming with a different --features (one that happens
            # to produce the same n_features plane count, so require_dataset's shape check
            # above doesn't catch it) would silently convert the remaining games under
            # different feature semantics while this stored string stays stale - and
            # nothing downstream (build_game_index, the trainer's feature-match guard)
            # would ever know, since they all trust this string.
            stored = h5f['features'][()]
            if isinstance(stored, bytes):
                stored = stored.decode('ascii')
            if stored != features_str:
                raise ValueError(
                    "{} was started with features\n\t{}\nbut this resume is using\n\t{}\n"
                    "All files in one output must use the same feature list - remove the "
                    "file and start over, or match the original --features."
                    .format(tmp_file, stored, features_str))

        if resuming:
            # next_idx comes from the recorded (start, length) pairs, not len(states):
            # if the previous run died mid-write (after resize()/assignment but before
            # its file_offsets entry was recorded), len(states) could be larger than
            # what's actually confirmed-complete. Trim any such orphaned rows so the
            # dataset stays exactly as long as what file_offsets accounts for, then
            # skip every file that's already recorded.
            next_idx = 0
            already_done = set()
            for key in file_offsets:
                start, length = file_offsets[key][()]
                next_idx = max(next_idx, start + length)
                already_done.add(key)
            if len(states) > next_idx:
                states.resize((next_idx, bd_size, bd_size, n_features))
                actions.resize((next_idx, 2))
            before = len(sgf_files)
            sgf_files = [f for f in sgf_files if f.replace('/', ':') not in already_done]
            if verbose:
                print("resuming: {} files already done, {} remaining".format(
                    before - len(sgf_files), len(sgf_files)))
        else:
            next_idx = 0

        if verbose:
            print("{} HDF5 dataset in {}".format("resumed" if resuming else "created", tmp_file))

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, initializer=_init_worker,
                initargs=(features,)) as pool:
            # Bounded in-flight submission (backpressure): submitting every file's future
            # up front lets all num_workers processes race ahead of this single-threaded
            # HDF5-writing loop, and completed-but-not-yet-written results pile up in
            # memory (each holding a full game's worth of uint8 tensors) faster than they
            # can be drained - confirmed via docker stats showing unbounded, steady memory
            # growth toward an OOM kill on a real run. Capping how many files are
            # in flight at once throttles producers to consumer speed instead.
            max_in_flight = num_workers * 4
            sgf_iter = iter(sgf_files)
            pending = set()
            files_since_flush = 0

            def _submit_next():
                try:
                    f = next(sgf_iter)
                except StopIteration:
                    return
                pending.add(pool.submit(_convert_one_file, f, bd_size))

            for _ in range(max_in_flight):
                _submit_next()

            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    file_name = result["file_name"]
                    pairs = result["pairs"]
                    error_type = result["error_type"]

                    if verbose:
                        print(file_name)

                    if error_type == "parse_exception":
                        warnings.warn("Could not parse %s\n\tdropping game" % file_name)
                    elif error_type == "size_mismatch":
                        warnings.warn("Skipping %s; wrong board size" % file_name)
                    elif error_type == "illegal_move":
                        warnings.warn("Illegal Move encountered in %s\n"
                                      "\tdropping the remainder of the game" % file_name)
                    elif error_type == "other":
                        if ignore_errors:
                            warnings.warn("Unkown exception with file %s\n\t%s" %
                                          (file_name, result["error_message"]), stacklevel=2)
                        else:
                            pool.shutdown(cancel_futures=True)
                            raise RuntimeError("{}: {}".format(file_name,
                                                               result["error_message"]))

                    n_pairs = len(pairs)
                    if n_pairs > 0:
                        file_start_idx = next_idx
                        end_idx = next_idx + n_pairs
                        if end_idx > len(states):
                            states.resize((end_idx, bd_size, bd_size, n_features))
                            actions.resize((end_idx, 2))
                        # One batched write per file instead of one HDF5 write per
                        # position - cuts resize()/assignment calls by ~2 orders of
                        # magnitude and was part of why the single-threaded writer here
                        # couldn't keep up with 16 parallel producers.
                        states[next_idx:end_idx] = np.stack([p[0] for p in pairs])
                        actions[next_idx:end_idx] = np.array([p[1] for p in pairs])
                        next_idx = end_idx
                        # '/' has special meaning in HDF5 key names, so they are replaced
                        # with ':'
                        file_name_key = file_name.replace('/', ':')
                        file_offsets[file_name_key] = [file_start_idx, n_pairs]
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")

                    # h5py/HDF5 doesn't persist its metadata (including the superblock's
                    # own record of how much of the file is valid) to disk as it goes -
                    # only h5f.close() or an explicit flush() does. A process that never
                    # gets to run our own cleanup code (a real SIGKILL, a host reboot, an
                    # OOM kill - as opposed to a Python exception, which does hit the
                    # except block below) leaves the file looking like an empty shell
                    # even though the actual data bytes are sitting on disk - confirmed
                    # the hard way after an 11-hour run: the whole file came back
                    # unreadable, not just recoverable-via-resume. Flushing periodically
                    # bounds the loss from a real kill to at most flush_every files
                    # instead of the entire run.
                    files_since_flush += 1
                    if files_since_flush >= flush_every:
                        h5f.flush()
                        files_since_flush = 0

                    _submit_next()
    except Exception as e:
        # Deliberately NOT removing tmp_file here (unlike GameConverter.sgfs_to_hdf5,
        # which has no resume support and just cleans up): keeping it is what makes
        # resume=True able to continue a run that died partway through - e.g. an
        # unattended multi-hour full-corpus conversion.
        print("sgfs_to_hdf5_parallel failed; {} preserved for resume=True".format(tmp_file))
        h5f.close()
        raise e

    if verbose:
        print("finished. renaming %s to %s" % (tmp_file, hdf5_file))

    # processing complete; rename tmp_file to hdf5_file
    h5f.close()
    os.rename(tmp_file, hdf5_file)


def run_game_converter_parallel(cmd_line_args=None):
    """Run conversions using a process pool. command-line args may be passed in as a list

    Same CLI as game_converter.py's run_game_converter, plus --workers.
    """
    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the neural network model '
                    '(multiprocessing version - see game_converter_parallel.py module '
                    'docstring for why this exists as processes and not threads).',
        epilog="Available features are: board, ones, turns_since, liberties, capture_size, "
        "self_atari_size, liberties_after, ladder_capture, ladder_escape, sensibleness, "
        "and zeros.")
    parser.add_argument("--features", "-f", help="Comma-separated list of features to compute and store or 'all'", default='all')  # noqa: E501
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)  # noqa: E501
    parser.add_argument("--recurse", "-R", help="Set to recurse through directories searching for SGF files", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--directory", "-d", help="Directory containing SGF files to process. if not present, expects files from stdin", default=None)  # noqa: E501
    parser.add_argument("--size", "-s", help="Size of the game board. SGFs not matching this are discarded with a warning", type=int, default=19)  # noqa: E501
    parser.add_argument("--workers", "-w", help="Number of worker processes (default: os.cpu_count())", type=int, default=None)  # noqa: E501
    parser.add_argument("--resume", help="Continue a previous incomplete run's output file instead of starting fresh", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--flush-every", help="Force an HDF5 metadata flush every N files, bounding how much a real crash (not a caught exception) can lose. Default: 500", type=int, default=500)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if args.features.lower() == 'all':
        feature_list = [
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
    else:
        feature_list = args.features.split(",")

    if args.verbose:
        print("using features", feature_list)
        print("using {} worker processes".format(args.workers or os.cpu_count()))

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _walk_all_sgfs(root):
        """a helper function/generator to get all SGF files in subdirectories of root
        """
        for (dirpath, dirname, files) in os.walk(root):
            for filename in files:
                if _is_sgf(filename):
                    yield os.path.join(dirpath, filename)

    def _list_sgfs(path):
        """helper function to get all SGF files in a directory (does not recurse)
        """
        files = os.listdir(path)
        return (os.path.join(path, f) for f in files if _is_sgf(f))

    # get an iterator of SGF files according to command line args
    if args.directory:
        if args.recurse:
            files = _walk_all_sgfs(args.directory)
        else:
            files = _list_sgfs(args.directory)
    else:
        files = (f.strip() for f in sys.stdin if _is_sgf(f))

    sgfs_to_hdf5_parallel(feature_list, files, args.outfile, bd_size=args.size,
                          num_workers=args.workers, verbose=args.verbose, resume=args.resume,
                          flush_every=args.flush_every)


if __name__ == '__main__':
    run_game_converter_parallel()
