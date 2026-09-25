#!/usr/bin/env python
"""Scan a tree of SGF files once, recording one JSONL row per file.

    python -m AlphaGo.preprocessing.sgf_preparation scan <root> <manifest.jsonl>

Nothing is filtered, modified or deleted here. The manifest is a record of what the
corpus contains, and every later question - which games to train on, how big a set to
build, what the komi or handicap distribution looks like - is answered by reading it
rather than by touching the SGFs again.

Conversion is the expensive step in this pipeline (hours per 100k games), while the scan
runs at ~1,400 files/sec. So the rule is: anything cheap to record should be recorded,
not decided. A criterion baked in at scan time costs a full re-scan to change.

That is why a row is written for EVERY file, including ones no sensible policy would
keep. Rejection reasons are recorded as advisory strings, never acted on.

Downstream:
    select_games.py   reads this manifest and writes the train/val/test keep-lists
    convert_shuffled.py  turns those keep-lists into shuffled training shards
    sgf_cull.py       separately, deletes files that are not 19x19 normal/handicap games

Manifest schema (one JSON object per line)
------------------------------------------
Identity and provenance:
    path, bytes, sgf_ok, size, gtype, start_turn_idx, init_turn_num, game_hash
Rules and setup:
    ko, score, tax, sui, rules_extra, komi, handicap, result, n_ab, n_aw
Move counts:
    n_moves, n_passes, n_moveless_nodes, n_annotated, n_weighted
Per-move aggregates (so move-level questions can be answered without per-move storage):
    visits_p10, visits_median, first_searched_winrate,
    n_blunder_gt05, n_blunder_gt10, n_blunder_gt20,
    n_decided, n_mover_hopeless
Bookkeeping:
    reasons  (list of strings; advisory only - nothing here acts on them)

`game_hash` is KataGo's own per-game identifier. It is captured because cross-download
duplicate detection is free here and expensive to add later, once you are pulling from an
effectively unlimited supply over time.
"""
import argparse
import collections
import gzip
import io
import json
import os
import re
import sys
import time

from AlphaGo.preprocessing.sgf_analyze import (
    SUSPECT_GTYPES, iter_sgf_files, scan_file_cheap, _chunks)


# A move together with its KataGo annotation, if present. Needed as one pattern (rather
# than sgf_analyze's separate move/annotation regexes) because the blunder and
# decided-position aggregates require the winrate to be paired with the colour that moved.
#
# The trailing [^\]]* absorbs anything after weight= - notably the terminal
# "result=W+R" that KataGo appends to the final comment.
_RE_MOVE_ANNOT = re.compile(
    r";([BW])\[([^\]]*)\]"
    r"(?:C\[\s*(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) v=(\d+)"
    r"(?: rv=(\d+))?(?: weight=([\d.]+))?[^\]]*\])?")

def _percentile(sorted_values, q):
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def move_stats(text, blunder_thresholds=(0.05, 0.10, 0.20), decided=0.95):
    """Per-file aggregates derived from KataGo's per-move annotations.

    win/loss/noResult/score are recorded from a CONSTANT White perspective
    (cpp/dataio/trainingwrite.h: "As usual, these are from the perspective of white"), so
    the change in white-winrate across a move measures how much the player who moved gave
    up. That is a direct measure of move quality, which KataGo's own training `weight`
    does not encode.

    Two caveats, both recorded in DATA_PIPELINE.md:
    - winrate is written with %.2f, so resolution is coarse. Useful for catching real
      blunders, not for ranking near-equal moves.
    - the mean is biased (a mover's own search is slightly optimistic about the move it
      just chose), so thresholds belong on the loss tail, never on the absolute level.
    """
    out = {
        "visits_p10": None, "visits_median": None, "first_searched_winrate": None,
        "n_decided": 0, "n_mover_hopeless": 0,
    }
    for thr in blunder_thresholds:
        out["n_blunder_gt{:02d}".format(int(round(thr * 100)))] = 0

    seq = []
    for m in _RE_MOVE_ANNOT.finditer(text):
        if m.group(3) is None:
            continue
        seq.append((m.group(1), float(m.group(3)), int(m.group(7))))
    if not seq:
        return out

    visits = sorted(v for _c, _w, v in seq)
    out["visits_p10"] = _percentile(visits, 0.10)
    out["visits_median"] = _percentile(visits, 0.50)
    out["first_searched_winrate"] = seq[0][1]

    for i, (color, white_w, _v) in enumerate(seq):
        mover_w = white_w if color == "W" else 1.0 - white_w
        if white_w >= decided or white_w <= (1.0 - decided):
            out["n_decided"] += 1
        if mover_w < (1.0 - decided):
            out["n_mover_hopeless"] += 1
        if i + 1 < len(seq):
            nxt = seq[i + 1][1]
            loss = (white_w - nxt) if color == "W" else (nxt - white_w)
            for thr in blunder_thresholds:
                if loss > thr:
                    out["n_blunder_gt{:02d}".format(int(round(thr * 100)))] += 1
    return out


# Every row carries every key, with None where the file did not supply a value. A manifest
# exists to be queried - by select_games.py, and by hand when sizing a criterion - and a consumer
# should never have to distinguish "field absent" from "field null". Files lacking a root
# comment (anything not written by KataGo selfplay) would otherwise silently omit
# gtype/game_hash/start_turn_idx and make every query guard for it.
_HEADER_FIELDS = (
    "path", "bytes", "sgf_ok", "size", "gtype", "start_turn_idx", "init_turn_num",
    "game_hash", "rules", "ko", "score", "tax", "sui", "rules_extra", "komi", "handicap",
    "result", "n_ab", "n_aw", "n_moves", "n_passes", "n_moveless_nodes", "n_annotated",
    "n_reanalyzed", "n_weighted", "n_weight_zero", "n_weight_pos", "reasons",
)
_MOVE_STAT_FIELDS = (
    "visits_p10", "visits_median", "first_searched_winrate", "n_decided",
    "n_mover_hopeless", "n_blunder_gt05", "n_blunder_gt10", "n_blunder_gt20",
)


def scan_one(path, with_move_stats=True, board_size=19):
    """One manifest row. Never raises: an unreadable or unparseable file is itself a
    finding and is recorded as such."""
    # Read ONCE and hand the text to both passes. Reading the file a second time for the
    # move stats measured roughly half the throughput (912 vs 1,700 files/sec) - at corpus
    # scale the open() dominates, not the regexes.
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
    except OSError as e:
        row = {k: None for k in _HEADER_FIELDS}
        if with_move_stats:
            row.update({k: None for k in _MOVE_STAT_FIELDS})
        row.update(path=path, sgf_ok=False, reasons=["unreadable:" + str(e)[:80]])
        return row

    # Board size is the one criterion that is never revisited, so a non-19x19 file stops
    # here rather than paying for the remaining regexes and the move-stat pass.
    rec = scan_file_cheap(path, text=text, board_size=str(board_size), short_circuit=True)
    # sgf_analyze keeps Counters for its own aggregation; they are not per-file facts and
    # would bloat every row.
    rec.pop("visit_buckets", None)
    rec.pop("weight_buckets", None)
    rec["sgf_ok"] = "unreadable" not in rec["reasons"]
    wrong_size = any(r.startswith("not_") for r in rec["reasons"])

    if with_move_stats and rec["sgf_ok"] and not wrong_size:
        try:
            rec.update(move_stats(text))
        except Exception as e:                            # noqa: BLE001
            # A corrupt annotation must not cost the file its row, let alone the run.
            # The regexes are permissive by design ([\d.]+ matches "1.2.3"), and at
            # corpus scale a handful of malformed files is a certainty - one of them
            # previously killed an entire multi-hour scan and left an EMPTY manifest,
            # because the exception escaped the worker and tore down the pool.
            rec["reasons"] = list(rec.get("reasons", ())) + [
                "move_stats_error:" + type(e).__name__]

    fields = _HEADER_FIELDS + (_MOVE_STAT_FIELDS if with_move_stats else ())
    return {k: rec.get(k) for k in fields}


def _scan_chunk(args):
    """Belt and braces: a single unexpected failure must cost one row, never the chunk
    (and therefore never the run). scan_one already guards the paths that are known to
    fail; this catches anything not yet anticipated."""
    paths, with_move_stats, board_size = args
    out = []
    for p in paths:
        try:
            out.append(scan_one(p, with_move_stats, board_size))
        except Exception as e:                            # noqa: BLE001
            row = {k: None for k in _HEADER_FIELDS}
            if with_move_stats:
                row.update({k: None for k in _MOVE_STAT_FIELDS})
            row.update(path=p, sgf_ok=False,
                       reasons=["scan_error:" + type(e).__name__])
            out.append(row)
    return out


def _open_out(path, mode="w"):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return io.open(path, mode, encoding="utf-8", newline="\n")


def _open_in(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, "r", encoding="utf-8")


def _already_scanned(manifest_path):
    """Path hashes of rows already in the manifest, for --resume.

    Hashes rather than paths deliberately: at several million files a set of full path
    strings runs to hundreds of MB, while 64-bit hashes cost ~8 bytes each. Collision
    risk at this scale is negligible (~4e-7 for 4M entries), and the cost of a collision
    is one file silently not re-scanned.
    """
    seen = set()
    if not os.path.exists(manifest_path):
        return seen
    with _open_in(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(hash(json.loads(line)["path"]))
            except (ValueError, KeyError):
                continue
    return seen


def cmd_scan(args):
    import concurrent.futures

    workers = args.workers or os.cpu_count()
    resume_skip = _already_scanned(args.manifest) if args.resume else set()
    if resume_skip:
        print("resuming: {:,} files already in {}".format(len(resume_skip), args.manifest),
              file=sys.stderr)

    paths = iter_sgf_files(args.directory, limit=args.sample)
    if resume_skip:
        paths = (p for p in paths if hash(p) not in resume_skip)

    work = ((chunk, not args.no_move_stats, args.board_size)
            for chunk in _chunks(paths, args.chunk_size))

    n = 0
    start = time.time()
    last = start
    reasons = collections.Counter()

    out = _open_out(args.manifest, "a" if args.resume else "w")
    try:
        if workers == 1:
            results_iter = (_scan_chunk(w) for w in work)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
            results_iter = _bounded_map(pool, _scan_chunk, work, workers * 4)

        for batch in results_iter:
            for rec in batch:
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                for r in rec.get("reasons", ()):
                    reasons[r] += 1
                n += 1
            now = time.time()
            if not args.quiet and now - last >= 2.0:
                print("  ... {:,} files  ({:,.0f}/sec)".format(
                    n, n / (now - start) if now > start else 0), file=sys.stderr)
                last = now
        if workers != 1:
            pool.shutdown()
    finally:
        out.close()

    elapsed = time.time() - start
    print("\nscanned {:,} files in {:.1f}s  ({:,.0f} files/sec)".format(
        n, elapsed, n / elapsed if elapsed else 0))
    print("manifest: {}".format(args.manifest))
    if reasons:
        print("\nadvisory flags recorded (NOT applied - see `select`):")
        for r, c in reasons.most_common():
            print("   {:28s} {:>10,}  ({:5.2f}%)".format(r, c, 100.0 * c / max(1, n)))


def _bounded_map(pool, fn, work, max_in_flight):
    """Submit at most max_in_flight tasks at a time. Submitting everything up front would
    materialise the whole file list and let workers outrun the single-threaded writer -
    the same backpressure pattern game_converter_parallel uses, for the same reason."""
    import concurrent.futures
    pending = set()

    def _submit():
        try:
            pending.add(pool.submit(fn, next(work)))
        except StopIteration:
            pass

    for _ in range(max_in_flight):
        _submit()
    while pending:
        done, pending = concurrent.futures.wait(
            pending, return_when=concurrent.futures.FIRST_COMPLETED)
        for fut in done:
            yield fut.result()
            _submit()


def main(cmd_line_args=None):
    parser = argparse.ArgumentParser(
        description="Scan an SGF corpus into a JSONL manifest. Never modifies or "
                    "deletes any SGF; selection lives in select_games.py.")
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p = sub.add_parser("scan", help="Walk a tree and write one JSONL row per SGF file")
    p.add_argument("directory", help="Root directory (recurses into subdirectories)")
    p.add_argument("manifest", help="Output JSONL path; .gz is compressed transparently")
    p.add_argument("--workers", "-w", type=int, default=None, help="Worker processes (default: os.cpu_count())")  # noqa: E501
    p.add_argument("--sample", "-n", type=int, default=None, help="Stop after this many files")  # noqa: E501
    p.add_argument("--chunk-size", type=int, default=200, help="Files per worker task. Default: 200")  # noqa: E501
    p.add_argument("--board-size", type=int, default=19, help="Board size to keep. A file of any other size SHORT-CIRCUITS: it is recorded but the remaining regexes and the move-stat pass are skipped, since this is the one criterion never revisited. Default: 19")  # noqa: E501
    p.add_argument("--no-move-stats", action="store_true", help="Skip the per-move aggregates (blunders, decided positions, visits). Roughly 2x faster, header-only.")  # noqa: E501
    p.add_argument("--resume", action="store_true", help="Append to an existing manifest, skipping files already recorded in it")  # noqa: E501
    p.add_argument("--quiet", action="store_true", help="Suppress the periodic throughput line")  # noqa: E501
    p.set_defaults(func=cmd_scan)

    args = parser.parse_args(cmd_line_args)
    args.func(args)


if __name__ == "__main__":
    main()
