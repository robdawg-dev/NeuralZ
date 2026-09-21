#!/usr/bin/env python
"""Convert KataGo selfplay SGFs into training shards, applying every position-inclusion
decision here rather than deferring any of them to the trainer.

A separate module from game_converter.py / game_converter_parallel.py, which are left
untouched - they work, and the shards they produce trained the current models.

Design
------
**The h5 is training data.** Everything written to it is, by construction, a position we
want. Nothing is filtered at training time, for three reasons:

- the shuffle buffer re-reads positions from disk EVERY epoch, so a position discarded in
  the trainer is decompressed and thrown away once per pass for the whole run. Disk read is
  precisely the bottleneck the shuffle buffer exists to work around.
- it keeps the trainer's "every position exactly once per pass" invariant intact.
  Filtering there would make n_train_data - and therefore steps_per_epoch - depend on a
  mask rather than on file_offsets lengths, which is a silently-wrong-epoch-length bug
  waiting to happen.
- re-converting is ~4 hours at this project's real scale (~330k games / ~5 hours measured),
  so baking a decision in is cheap to undo.

**Output is a backward-compatible superset.** The datasets are exactly what
shuffle_buffer.build_game_index already reads - states, actions, features, file_offsets -
plus one `conversion_args` scalar recording how the shard was made. Shards from this tool
train with the existing supervised_policy_trainer_v3 with no trainer changes. The
provenance string matters because a final shard set outlives the session that produced it,
and "which filters made this?" is not answerable from a directory listing.

**Sharding is by target bytes**, defaulting to 20GB, because that is what this project has
used and because supervised_policy_trainer_v3 notes a single file degrades badly on
spinning disk past ~100-150GB. Output file names match what find_shard_files globs.

Correctness (default ON)
------------------------
--skip-setup-positions N            [default 7]
    Drop the first N positions of games carrying AB/AW setup stones. Those games arrive
    with their real move order discarded, so turns_since reports the pre-placed position as
    though it had just been played move by move. There are only 7 "recent" age planes, so
    the error is bounded at 7 stones and self-corrects once 7 real moves exist - measured
    at ~1% of all positions. N=7 makes turns_since exactly correct for ~1% cost.

    This defaults to 7 because turns_since is in the 48-plane feature set: at 0 the
    converter emits tensors that are known-wrong. Set 0 only to reproduce older output.
    See get_turns_since in preprocessing.pyx. The value is tuned for turns_since -
    last_moves is not in use and would need its own treatment (5, not 7).

Filters (all default OFF)
-------------------------

--max-winrate-loss X
    Drop a position whose label is a move that gave up more than X winrate. KataGo records
    win/loss/noResult/score from a CONSTANT White perspective, so the change across a move
    measures what the mover gave up. Note this is structurally blind once the winrate
    saturates (mean |delta| is ~8x smaller above 0.95 / below 0.05), which is what
    --drop-hopeless-mover covers instead.

--drop-hopeless-mover X
    Drop positions where the player to move is already at or below X winrate. Judged from
    the MOVER's perspective, not symmetrically: in a decided game this keeps the winner's
    moves (converting a won game is a real skill, and one human corpora teach badly because
    those games end in resignation) and drops only the hopeless side's.

Even with the filters off, the end-of-run summary reports what each WOULD have removed, so
a single short test run sizes every threshold against your own data without converting once
per variant.
"""
import argparse
import collections
import json
import os
import re
import sys
import time

import h5py as h5
import numpy as np
import sgf

import AlphaGo.go as go
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import sgf_iter_states

ALL_FEATURES = [
    "board", "ones", "turns_since", "liberties", "capture_size", "self_atari_size",
    "liberties_after", "ladder_capture", "ladder_escape", "sensibleness", "zeros"]

# Same comment format as sgf_preparation parses; duplicated rather than imported so this
# module does not depend on the preparation tool (it accepts a plain directory too).
_RE_MOVE_ANNOT = re.compile(
    r";([BW])\[([^\]]*)\]"
    r"(?:C\[\s*(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) v=(\d+)"
    r"(?: rv=(\d+))?(?: weight=([\d.]+))?[^\]]*\])?")
_RE_SETUP = re.compile(r"\b(?:AB|AW)((?:\[[^\]]*\])+)")
_RE_BRACKET = re.compile(r"\[([^\]]*)\]")

# Thresholds reported as counterfactuals when the corresponding filter is off.
_CF_LOSS = (0.05, 0.10, 0.20)
_CF_HOPELESS = (0.05,)
_CF_SKIP = (7,)


class _Cfg(object):
    """Picklable filter configuration handed to each worker."""

    def __init__(self, features, bd_size, skip_setup_positions, max_winrate_loss,
                 drop_hopeless_mover):
        self.features = features
        self.bd_size = bd_size
        self.skip_setup_positions = skip_setup_positions
        self.max_winrate_loss = max_winrate_loss
        self.drop_hopeless_mover = drop_hopeless_mover


_worker = {}


def _init_worker(cfg):
    # A Cython Preprocess is not picklable, so each worker builds its own - same reason
    # game_converter_parallel uses processes rather than threads.
    _worker["cfg"] = cfg
    _worker["proc"] = Preprocess(cfg.features, size=cfg.bd_size)


def _annotations(text):
    """Per declared move: (mover_winrate, loss_for_mover, visits), or None where KataGo
    recorded no search for that move (the un-searched policy-init opening moves).

    Indexes align with sgf_iter_states, which yields one tuple per declared move.
    """
    raw = []
    for m in _RE_MOVE_ANNOT.finditer(text):
        if m.group(3) is None:
            raw.append(None)
        else:
            raw.append((m.group(1), float(m.group(3)), int(m.group(7))))

    out = []
    for i, entry in enumerate(raw):
        if entry is None:
            out.append(None)
            continue
        color, white_w, visits = entry
        mover_w = white_w if color == "W" else 1.0 - white_w
        loss = None
        # the next ANNOTATED move gives the position's value after this one
        for j in range(i + 1, len(raw)):
            if raw[j] is not None:
                nxt = raw[j][1]
                loss = (white_w - nxt) if color == "W" else (nxt - white_w)
                break
        out.append((mover_w, loss, visits))
    return out


def _n_setup_stones(text):
    total = 0
    for m in _RE_SETUP.finditer(text):
        total += len(_RE_BRACKET.findall(m.group(1)))
    return total


def _classify_truncation(exc, state, move):
    """Why the replay stopped. Keeping these apart is the point: once suicide is an
    expected, counted event, any OTHER reason becomes a real alarm rather than noise."""
    msg = str(exc)
    if "board-altering" in msg:
        return "setup_node"
    if state is None or move is None or move is go.PASS:
        return "illegal_other"
    try:
        board = state.get_board()
        if board[move[0]][move[1]] != go.EMPTY:
            return "illegal_occupied"
        if state.get_ko_location() == move:
            return "illegal_ko"
    except Exception:                                     # noqa: BLE001
        return "illegal_other"
    return "illegal_suicide"


def convert_one(path):
    """Convert one SGF in a worker. Returns a picklable dict; never raises."""
    cfg, proc = _worker["cfg"], _worker["proc"]
    res = {
        "path": path, "pairs": [], "truncation": None, "error": None,
        "n_declared": 0, "n_dropped": collections.Counter(),
        "counterfactual": collections.Counter(),
    }
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
    except OSError as e:
        res["truncation"] = "unreadable"
        res["error"] = str(e)
        return res

    ann = _annotations(text)
    has_setup = _n_setup_stones(text) > 0
    state = move = None

    try:
        for i, (state, move, player) in enumerate(sgf_iter_states(text, include_end=False)):
            res["n_declared"] += 1
            if state.get_size() != cfg.bd_size:
                res["truncation"] = "size_mismatch"
                res["pairs"] = []
                return res
            if move is go.PASS:
                continue
            # sgf_iter_states yields (position, move) BEFORE applying the move, so a move
            # the engine will reject still arrives here once. Emitting it would teach the
            # network a move it must never play - in practice a multi-stone suicide from a
            # sui1 ruleset. Skip the emit; the next iteration's do_move raises and the
            # prefix collected so far is kept.
            if state.get_current_player() == player and not state.is_legal(move):
                continue

            a = ann[i] if i < len(ann) else None
            mover_w = a[0] if a else None
            loss = a[1] if a else None

            # --- counterfactuals: what each threshold WOULD cost, filters off or on ---
            if has_setup:
                for n in _CF_SKIP:
                    if i < n:
                        res["counterfactual"]["skip_setup_{}".format(n)] += 1
            if loss is not None:
                for thr in _CF_LOSS:
                    if loss > thr:
                        res["counterfactual"]["loss_gt{:02d}".format(int(thr * 100))] += 1
            if mover_w is not None:
                for thr in _CF_HOPELESS:
                    if mover_w <= thr:
                        res["counterfactual"]["hopeless_{:02d}".format(int(thr * 100))] += 1

            # --- filters ---
            if has_setup and cfg.skip_setup_positions and i < cfg.skip_setup_positions:
                res["n_dropped"]["skip_setup"] += 1
                continue
            if cfg.max_winrate_loss is not None and loss is not None \
                    and loss > cfg.max_winrate_loss:
                res["n_dropped"]["winrate_loss"] += 1
                continue
            if cfg.drop_hopeless_mover is not None and mover_w is not None \
                    and mover_w <= cfg.drop_hopeless_mover:
                res["n_dropped"]["hopeless_mover"] += 1
                continue

            res["pairs"].append((proc.state_to_tensor(state)[0], move))
    except go.IllegalMove as e:
        res["truncation"] = _classify_truncation(e, state, move)
    except sgf.ParseException:
        res["truncation"] = "parse_error"
    except Exception as e:                                # noqa: BLE001
        res["truncation"] = "other"
        res["error"] = "{}: {}".format(type(e).__name__, e)
    return res


class _ShardWriter(object):
    """Writes games into shard_NNNNN.h5, rolling over once a shard passes the byte target.

    Only the main process ever touches HDF5, so the file layout and file_offsets
    bookkeeping stay identical to the existing converters - workers are a compute pool,
    not writers.
    """

    # Positions to write before the FIRST real size check of a shard. Derived from the
    # byte target rather than fixed, because a fixed 1000 meant a small --shard-bytes
    # never reached its first check and wrote one oversized shard. After the first check
    # the interval adapts to the MEASURED bytes-per-position, so a 20GB shard is not
    # flushed needlessly often.
    MIN_CHECK_INTERVAL = 64
    _ASSUMED_BYTES_PER_POSITION = 4000   # deliberately low, so the first check is early

    def _first_check(self):
        return max(self.MIN_CHECK_INTERVAL,
                   min(1000, int(self.shard_bytes / self._ASSUMED_BYTES_PER_POSITION)))

    def __init__(self, out_dir, features, n_features, bd_size, shard_bytes,
                 conversion_args, first_index=0):
        self.out_dir = out_dir
        self.features = features
        self.n_features = n_features
        self.bd_size = bd_size
        self.shard_bytes = shard_bytes
        self.conversion_args = conversion_args
        self.index = first_index
        self.total_positions = 0
        self.shards = []
        self._open()

    def _open(self):
        path = os.path.join(self.out_dir, "shard_{:05d}.h5".format(self.index))
        self.path = path
        self.h5f = h5.File(path, "w")
        self.states = self.h5f.create_dataset(
            "states", dtype=np.uint8,
            shape=(0, self.bd_size, self.bd_size, self.n_features),
            maxshape=(None, self.bd_size, self.bd_size, self.n_features),
            # 64 rows (~1.1MB/chunk): training reads positions in a fully random global
            # shuffle and HDF5 must decompress a whole chunk to serve any row, so a larger
            # chunk multiplies read amplification by the same factor. See
            # game_converter_parallel for the full rationale.
            chunks=(64, self.bd_size, self.bd_size, self.n_features), compression="lzf")
        self.actions = self.h5f.create_dataset(
            "actions", dtype=np.uint8, shape=(0, 2), maxshape=(None, 2),
            chunks=(64, 2), compression="lzf")
        self.offsets = self.h5f.create_group("file_offsets")
        self.h5f["features"] = np.bytes_(",".join(self.features))
        self.h5f["conversion_args"] = np.bytes_(self.conversion_args)
        self.next_idx = 0
        self.check_at = self._first_check()
        self.shards.append(path)

    def _roll(self):
        self.h5f.close()
        self.index += 1
        self._open()

    def write_game(self, path, pairs):
        if not pairs:
            return
        start = self.next_idx
        end = start + len(pairs)
        self.states.resize((end, self.bd_size, self.bd_size, self.n_features))
        self.actions.resize((end, 2))
        self.states[start:end] = np.stack([p[0] for p in pairs])
        self.actions[start:end] = np.array([p[1] for p in pairs], dtype=np.uint8)
        # '/' has special meaning in HDF5 key names
        self.offsets[path.replace("/", ":").replace("\\", ":")] = [start, len(pairs)]
        self.next_idx = end
        self.total_positions += len(pairs)

        # Compressed size is not predictable from the position count alone, so measure it
        # occasionally and extrapolate. Checking on a fixed game count instead meant a
        # short run (or a small --shard-bytes) never reached the first check and wrote one
        # oversized shard.
        if self.next_idx >= self.check_at:
            self.h5f.flush()
            size = self.h5f.id.get_filesize()
            if size >= self.shard_bytes:
                self._roll()
            else:
                bytes_per_position = size / float(max(1, self.next_idx))
                remaining = (self.shard_bytes - size) / max(1.0, bytes_per_position)
                self.check_at = self.next_idx + max(
                    self.MIN_CHECK_INTERVAL, int(remaining * 0.5))

    def close(self):
        # A roll that fires on the last write leaves an empty trailing shard. Leaving it
        # would make find_shard_files hand build_game_index a shard with no file_offsets.
        empty = self.next_idx == 0
        path = self.path
        self.h5f.close()
        if empty:
            self.shards.remove(path)
            try:
                os.remove(path)
            except OSError:
                pass


def _done_paths(out_dir):
    """Source paths already converted, read back from existing shards' file_offsets."""
    done = set()
    if not os.path.isdir(out_dir):
        return done, 0
    shards = sorted(f for f in os.listdir(out_dir)
                    if f.startswith("shard_") and f.endswith(".h5"))
    for name in shards:
        try:
            with h5.File(os.path.join(out_dir, name), "r") as f:
                done.update(f["file_offsets"].keys())
        except (OSError, KeyError):
            continue
    next_index = 0
    if shards:
        next_index = int(shards[-1][len("shard_"):-len(".h5")]) + 1
    return done, next_index


def _read_inputs(source):
    """A keep-list (one path per line) or a directory to walk."""
    if os.path.isdir(source):
        for dirpath, _dirs, files in os.walk(source):
            for name in files:
                if name.endswith(".sgf"):
                    yield os.path.join(dirpath, name)
    else:
        with open(source) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def _chunks(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def convert(source, out_dir, features, bd_size=19, workers=None, shard_bytes=20 * 10 ** 9,
            skip_setup_positions=7, max_winrate_loss=None, drop_hopeless_mover=None,
            resume=False, limit=None, quiet=False, conversion_args="{}"):
    import concurrent.futures

    workers = workers or os.cpu_count()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    done, first_index = (_done_paths(out_dir) if resume else (set(), 0))
    if resume and done:
        print("resuming: {:,} games already converted, continuing at shard_{:05d}".format(
            len(done), first_index), file=sys.stderr)
    elif not resume and os.listdir(out_dir):
        raise ValueError(
            "{} is not empty. Move it aside, or pass resume=True to continue it."
            .format(out_dir))

    cfg = _Cfg(features, bd_size, skip_setup_positions, max_winrate_loss,
               drop_hopeless_mover)
    n_features = Preprocess(features, size=bd_size).get_output_dimension()

    paths = _read_inputs(source)
    if done:
        paths = (p for p in paths
                 if p.replace("/", ":").replace("\\", ":") not in done)
    if limit:
        # Counts the TOTAL target, so `--limit N --resume` tops the set up to N rather
        # than adding another N on top of what is already there.
        remaining = max(0, limit - len(done))
        paths = (p for i, p in enumerate(paths) if i < remaining)

    writer = _ShardWriter(out_dir, features, n_features, bd_size, shard_bytes,
                          conversion_args, first_index=first_index)
    stats = collections.Counter()
    dropped = collections.Counter()
    counterfactual = collections.Counter()
    errors = []
    start = time.time()
    last = start

    try:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            work = _chunks(paths, 8)
            pending = set()

            def _submit():
                try:
                    pending.add(pool.submit(_convert_chunk, next(work)))
                except StopIteration:
                    pass

            # Bounded in-flight submission: workers must not outrun the single-threaded
            # HDF5 writer, or completed-but-unwritten games pile up toward an OOM.
            for _ in range(workers * 4):
                _submit()

            while pending:
                batch, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in batch:
                    for res in fut.result():
                        stats["games"] += 1
                        stats["declared_moves"] += res["n_declared"]
                        dropped.update(res["n_dropped"])
                        counterfactual.update(res["counterfactual"])
                        if res["truncation"]:
                            stats["truncated"] += 1
                            stats["trunc:" + res["truncation"]] += 1
                            if res["error"] and len(errors) < 5:
                                errors.append((res["path"], res["error"]))
                        writer.write_game(res["path"], res["pairs"])
                    _submit()
                now = time.time()
                if not quiet and now - last >= 5.0:
                    print("  ... {:,} games, {:,} positions ({:,.0f} games/sec)".format(
                        stats["games"], writer.total_positions,
                        stats["games"] / (now - start) if now > start else 0),
                        file=sys.stderr)
                    last = now
    finally:
        writer.close()

    _report(stats, dropped, counterfactual, writer, errors, time.time() - start,
            skip_setup_positions, max_winrate_loss, drop_hopeless_mover)
    return writer


def _convert_chunk(paths):
    return [convert_one(p) for p in paths]


def _pct(n, d):
    return 0.0 if not d else 100.0 * n / d


def _report(stats, dropped, counterfactual, writer, errors, elapsed,
            skip_setup_positions, max_winrate_loss, drop_hopeless_mover):
    written = writer.total_positions
    emitted_plus_dropped = written + sum(dropped.values())
    print("\n" + "=" * 72)
    print("CONVERSION SUMMARY")
    print("=" * 72)
    print("games read        : {:,}".format(stats["games"]))
    print("declared moves    : {:,}".format(stats["declared_moves"]))
    print("positions written : {:,}".format(written))
    print("shards this run   : {} ({})".format(
        len(writer.shards), ", ".join(os.path.basename(s) for s in writer.shards[:4]) +
        (", ..." if len(writer.shards) > 4 else "")))
    print("elapsed           : {:.1f}s  ({:,.1f} games/sec)".format(
        elapsed, stats["games"] / elapsed if elapsed else 0))

    print("\nTRUNCATED GAMES (prefix kept, remainder dropped)")
    if stats["truncated"]:
        print("  total: {:,}  ({:.2f}% of games)".format(
            stats["truncated"], _pct(stats["truncated"], stats["games"])))
        for key in sorted(k for k in stats if k.startswith("trunc:")):
            reason = key.split(":", 1)[1]
            note = ""
            if reason == "illegal_suicide":
                note = "   <- expected: KataGo sui1 rulesets permit multi-stone suicide"
            elif reason in ("illegal_occupied", "illegal_ko", "illegal_other", "other"):
                note = "   <- UNEXPECTED: investigate"
            print("    {:22s} {:>8,}{}".format(reason, stats[key], note))
    else:
        print("  none")
    for path, err in errors:
        print("    e.g. {}: {}".format(os.path.basename(path), err))

    if dropped:
        print("\nPOSITIONS DROPPED BY FILTERS")
        for k, v in dropped.most_common():
            print("    {:22s} {:>10,}  ({:5.2f}% of eligible)".format(
                k, v, _pct(v, emitted_plus_dropped)))

    print("\nWHAT EACH THRESHOLD WOULD COST (on the positions actually written)")
    if not counterfactual:
        print("    (no KataGo annotations found - are these selfplay SGFs?)")
    else:
        active = []
        if skip_setup_positions:
            active.append("skip_setup_{}".format(skip_setup_positions))
        for key, count in sorted(counterfactual.items()):
            flag = "  [ACTIVE]" if key in active else ""
            print("    {:22s} {:>10,}  ({:5.2f}%){}".format(
                key, count, _pct(count, emitted_plus_dropped), flag))
        print("    (skip_setup_N counts positions in setup-stone games only;"
              " loss_/hopeless_ need annotations)")


def main(cmd_line_args=None):
    parser = argparse.ArgumentParser(
        description="Convert KataGo selfplay SGFs into training shards. All decisions "
                    "about which positions to include are made here - the resulting h5 "
                    "contains only positions intended for training.")
    parser.add_argument("source", help="Keep-list file (one SGF path per line, from sgf_preparation select) or a directory to walk")  # noqa: E501
    parser.add_argument("out_directory", help="Directory to write shard_NNNNN.h5 into")
    parser.add_argument("--features", "-f", default="all", help="Comma-separated feature list, or 'all'")  # noqa: E501
    parser.add_argument("--size", "-s", type=int, default=19, help="Board size. Default: 19")  # noqa: E501
    parser.add_argument("--workers", "-w", type=int, default=None, help="Worker processes (default: os.cpu_count())")  # noqa: E501
    parser.add_argument("--shard-bytes", type=float, default=20e9, help="Roll to a new shard once one passes this size. Default: 20e9 (20GB) - a single file degrades badly on spinning disk past ~100-150GB")  # noqa: E501
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many source games - for quick test sets")  # noqa: E501
    parser.add_argument("--resume", action="store_true", help="Continue into an existing output directory, skipping games already present in its shards")  # noqa: E501
    parser.add_argument("--quiet", action="store_true", help="Suppress the periodic progress line")  # noqa: E501

    g = parser.add_argument_group("position filters (all default OFF; the summary reports what each would cost regardless)")  # noqa: E501
    g.add_argument("--skip-setup-positions", type=int, default=7, help="Drop the first N positions of games carrying AB/AW setup stones. Default 7, which makes turns_since exactly correct for ~1%% of positions; 0 emits known-wrong tensors.")  # noqa: E501
    g.add_argument("--max-winrate-loss", type=float, default=None, help="Drop a position whose label gave up more than this much winrate (e.g. 0.10). Blind once the winrate saturates - see --drop-hopeless-mover.")  # noqa: E501
    g.add_argument("--drop-hopeless-mover", type=float, default=None, help="Drop positions where the player to move is at or below this winrate (e.g. 0.05). Asymmetric: keeps the winning side's moves.")  # noqa: E501

    args = parser.parse_args(cmd_line_args)
    features = ALL_FEATURES if args.features.lower() == "all" else args.features.split(",")

    convert(
        args.source, args.out_directory, features, bd_size=args.size,
        workers=args.workers, shard_bytes=int(args.shard_bytes),
        skip_setup_positions=args.skip_setup_positions,
        max_winrate_loss=args.max_winrate_loss,
        drop_hopeless_mover=args.drop_hopeless_mover,
        resume=args.resume, limit=args.limit, quiet=args.quiet,
        conversion_args=json.dumps(vars(args), sort_keys=True))


if __name__ == "__main__":
    main()
