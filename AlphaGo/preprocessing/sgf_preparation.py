#!/usr/bin/env python
"""Two-phase corpus preparation: scan a tree of SGF files once, then select from the
result as often as you like.

    scan    walk a directory tree, record one JSONL row per file, write nothing else
    select  read that manifest, apply criteria, emit a keep-list for the converter

Why two phases
--------------
Conversion is the expensive step in this pipeline - roughly 5 hours per 330k games, so a
multi-million game corpus is a 60+ hour run. The scan is cheap by comparison (~1,800
files/sec measured). So the rule throughout is: **anything cheap to record should be
recorded, not decided.** A criterion baked in at scan time costs a full re-scan to change;
a criterion applied at select time costs seconds.

That is why `scan` writes a row for EVERY file, including ones no sensible policy would
keep. Rejection reasons are recorded, not acted on. `select` is where policy lives, and
re-running it never re-reads the corpus.

Nothing here deletes or modifies an SGF. `select` emits a keep-list (a plain text file of
paths, one per line) which the converter consumes. Deletion, if ever wanted, is a separate
step that reads the same manifest.

What is deliberately NOT decided here
-------------------------------------
File-level properties only. Anything per-POSITION - blunder suppression, decided-position
filtering, skipping the first N positions of a setup-stone game - belongs in
game_converter, because those decisions need to travel with the position they describe.
See SGF_FILTER_POLICY.md for the full split and the reasoning behind each criterion.

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
    reasons  (list of strings; advisory, applied by `select` not by `scan`)

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

# Defaults for `select`. These are the criteria SGF_FILTER_POLICY.md records as "confident
# there is no value" - everything else is left to explicit flags.
# An ALLOWLIST, not a denylist. Two of KataGo's eight game types are kept:
#
#   normal    52% of files, zero AB/AW setup stones, the baseline for every quality metric
#   handicap   3% of files, 1-5 setup stones that are GENUINELY placed moves - in sequence,
#              immediately before White's first move, exactly as the GTP path places them
#              at play time. Nothing about them needs special handling.
#
# The six excluded types, and why:
#   sgfpos, fork         AB block is a serialized mid-game BOARD (median 69 and 50 stones)
#                        written in row-major raster order, all-black-then-all-white, with
#                        captured stones absent. It carries no move order, so no faithful
#                        turns_since representation exists. This is what --skip-setup-
#                        positions existed to paper over.
#   asym                 asymmetric playouts: the handicap-RECEIVING side runs ~142 visits
#                        against ~373 and blunders 3.6x more. Worst blunder rate of any
#                        gtype (0.27% vs normal's 0.12%) and the source of the 100-point
#                        handicap blowouts.
#   hintpos, hintfork    positions carrying a hinted move, deliberately biased toward it
#   cleanuptraining      endgame/scoring drill, starts ~106 stones in
#
# An allowlist also fails safe: a new gtype in future KataGo data is excluded until looked
# at, rather than silently included.
DEFAULT_INCLUDE_GTYPES = ("normal", "handicap")
DEFAULT_KOMI_MIN = -10.0
DEFAULT_KOMI_MAX = 30.0

# The komi band above applies ONLY to games without handicap stones.
#
# KataGo expresses handicap as komi compensation, at roughly 13 points per stone - the
# real value of a handicap stone on 19x19. Measured medians: HA2 15.5, HA3 27.5, HA4 39.0,
# HA5 53.5, HA6 65.0, HA9 115.5. So a flat komi-max of 30 is in effect a handicap filter:
# it discarded 49.9% of handicap games while catching only 1.9% of non-handicap ones.
#
# Handicap games are instead bounded by the sanity limits below, which exist only to strip
# values no board can support (the raw corpus contains komi of -303 and +359 on a
# 361-point board). Only 2 handicap games of 3,735 exceeded 120.
DEFAULT_HANDICAP_KOMI_MIN = -120.0
DEFAULT_HANDICAP_KOMI_MAX = 120.0


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

    Two caveats, both recorded in SGF_FILTER_POLICY.md:
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
# exists to be queried - by `select`, and by hand when sizing a criterion - and a consumer
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


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_reasons(rec, args):
    """Why this file would be excluded. Empty list means keep."""
    out = []
    if not rec.get("sgf_ok", False):
        out.append("unreadable")
    if rec.get("size") != str(args.board_size):
        out.append("not_{}x{}".format(args.board_size, args.board_size))
    if not rec.get("n_moves"):
        out.append("no_moves")

    gtype = rec.get("gtype")
    if gtype not in set(args.include_gtype):
        out.append("gtype_{}".format(gtype or "none"))

    komi = _float(rec.get("komi"))
    if komi is None:
        if not args.allow_no_komi:
            out.append("no_komi")
    else:
        # Handicap games get the loose sanity bound, not the training band - their komi is
        # principled compensation (~13 pts/stone) and the board itself shows the imbalance,
        # so it is not the hidden variable it is on an even board. See the note by
        # DEFAULT_HANDICAP_KOMI_MIN.
        try:
            handicap = int(rec.get("handicap") or 0)
        except (TypeError, ValueError):
            handicap = 0
        if handicap > 0:
            if not (args.handicap_komi_min <= komi <= args.handicap_komi_max):
                out.append("handicap_komi_out_of_range")
        elif not (args.komi_min <= komi <= args.komi_max):
            out.append("komi_out_of_range")

    return out


def cmd_select(args):
    kept = 0
    total = 0
    rejected = collections.Counter()
    kept_moves = 0
    out = _open_out(args.keeplist)
    try:
        with _open_in(args.manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except ValueError:
                    rejected["unparseable_manifest_row"] += 1
                    continue
                reasons = select_reasons(rec, args)
                if reasons:
                    for r in reasons:
                        rejected[r] += 1
                    rejected["__any__"] += 1
                    continue
                out.write(rec["path"] + "\n")
                kept += 1
                kept_moves += rec.get("n_moves") or 0
                if args.limit and kept >= args.limit:
                    break
    finally:
        out.close()

    print("manifest rows read : {:,}".format(total))
    print("rejected           : {:,}  ({:.2f}%)".format(
        rejected["__any__"], 100.0 * rejected["__any__"] / max(1, total)))
    for r, c in rejected.most_common():
        if r == "__any__":
            continue
        print("   {:28s} {:>10,}  ({:5.2f}%)".format(r, c, 100.0 * c / max(1, total)))
    print("KEPT               : {:,}  ({:.2f}%)".format(kept, 100.0 * kept / max(1, total)))
    print("moves in kept games: {:,}  ({:,.0f} per game)".format(
        kept_moves, kept_moves / max(1, kept)))
    print("keep-list          : {}".format(args.keeplist))
    if args.limit and kept >= args.limit:
        print("\n(stopped at --limit {:,}; the manifest was not read to the end)".format(
            args.limit))


def main(cmd_line_args=None):
    parser = argparse.ArgumentParser(
        description="Prepare an SGF corpus for conversion: scan once into a manifest, "
                    "then select from it as often as needed. Never modifies or deletes "
                    "any SGF.")
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

    p = sub.add_parser("select", help="Filter a manifest into a keep-list of paths")
    p.add_argument("manifest", help="JSONL manifest produced by `scan`")
    p.add_argument("keeplist", help="Output path; one SGF path per line")
    p.add_argument("--board-size", type=int, default=19, help="Required SZ. Default: 19")
    p.add_argument("--include-gtype", action="append", default=None,
                   help="KataGo gtype to KEEP; repeatable. Everything else is dropped. "
                        "Default: {} - see the note by DEFAULT_INCLUDE_GTYPES for why "
                        "the other six are excluded.".format(
                            ",".join(DEFAULT_INCLUDE_GTYPES)))
    p.add_argument("--komi-min", type=float, default=DEFAULT_KOMI_MIN, help="Default: {}".format(DEFAULT_KOMI_MIN))  # noqa: E501
    p.add_argument("--komi-max", type=float, default=DEFAULT_KOMI_MAX, help="Default: {}. A deliberately GENEROUS outer bound - a tighter band is a training-time knob, not a file filter. Applies to non-handicap games only.".format(DEFAULT_KOMI_MAX))  # noqa: E501
    p.add_argument("--handicap-komi-min", type=float, default=DEFAULT_HANDICAP_KOMI_MIN, help="Sanity bound for games WITH handicap stones, whose komi is compensation at ~13pts/stone rather than a free parameter. Default: {}".format(DEFAULT_HANDICAP_KOMI_MIN))  # noqa: E501
    p.add_argument("--handicap-komi-max", type=float, default=DEFAULT_HANDICAP_KOMI_MAX, help="Default: {}".format(DEFAULT_HANDICAP_KOMI_MAX))  # noqa: E501
    p.add_argument("--allow-no-komi", action="store_true", help="Keep games with no parseable KM. Off by default: komi is not a feature plane, so an unknown komi is an unknown offset.")  # noqa: E501
    p.add_argument("--limit", type=int, default=None, help="Stop once this many files have been kept - use to build a fixed-size training set")  # noqa: E501
    p.set_defaults(func=cmd_select)

    args = parser.parse_args(cmd_line_args)
    if getattr(args, "include_gtype", None) is None:
        args.include_gtype = list(DEFAULT_INCLUDE_GTYPES)
    args.func(args)


if __name__ == "__main__":
    main()
