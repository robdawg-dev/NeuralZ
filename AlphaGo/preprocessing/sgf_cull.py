#!/usr/bin/env python
"""Find, and then delete, KataGo selfplay SGFs we have decided never to use.

Two steps, deliberately separate:

    scan    <root> <out_dir>   walk <root>, write <out_dir>/to_delete.txt. Deletes nothing.
    delete  <out_dir>          delete exactly the files listed in to_delete.txt.

Review the scan summary before running delete.

A file is listed for deletion for the FIRST of these that applies:

    empty        zero bytes, or only whitespace
    unreadable   cannot be opened, or the root node cannot be parsed as SGF
    no_moves     no B[] or W[] move after the root node
    not_19x19    SZ is anything other than 19 (a missing SZ means 19, per the SGF spec)
    gtype_<x>    the root comment's gtype is anything other than normal or handicap,
                 including a file with no gtype at all (gtype_none)
    duplicate    same KataGo gameHash as a file already kept, AND byte-identical to it

Duplicates: files are visited in sorted path order, so the first copy seen (the earliest
date folder) is the one kept. A later file with the same gameHash is compared byte for
byte against the kept copy. Identical -> listed for deletion. Different -> NOT deleted;
recorded in hash_conflicts.txt instead, since that means gameHash is not the per-game
identifier we assume it is and the two files need a human look.

Outputs in <out_dir>:
    to_delete.txt        path<TAB>reason[<TAB>kept copy, for duplicates]
    hash_conflicts.txt   gameHash<TAB>kept path<TAB>other path   (same hash, different bytes)
    scan_summary.txt     counts and bytes per reason

Only the root node is parsed, from the first few KB of each file, so a scan reads a
fraction of the corpus.
"""
import argparse
import concurrent.futures
import filecmp
import os
import re
import sys
import time

KEEP_GTYPES = ("normal", "handicap")
HEAD_BYTES = 8192
BATCH = 2000

_RE_GTYPE = re.compile(r"gtype=([^,\]\s]+)")
_RE_GAMEHASH = re.compile(r"gameHash=([0-9A-Fa-f]+)")
_RE_MOVE = re.compile(r"[;(]\s*[BW]\s*\[")

REASON_ORDER = ("empty", "unreadable", "no_moves", "not_19x19")


# ------------------------------------------------------------------------ parsing

def _root_node(text):
    """Properties of the root node, and the index just past it.

    Returns (props, end) or None if the text does not hold a complete root node - which
    may just mean the read was too short, so the caller retries on the whole file.
    Values are unescaped; property identifiers are kept exactly as written.
    """
    start = text.find("(")
    if start < 0:
        return None
    pos = text.find(";", start)
    if pos < 0 or text[start + 1:pos].strip():
        return None
    pos += 1
    n = len(text)
    props = {}
    while pos < n:
        c = text[pos]
        if c.isspace():
            pos += 1
            continue
        if c in ";()":
            return props, pos
        ident_start = pos
        while pos < n and text[pos].isalpha():
            pos += 1
        ident = text[ident_start:pos]
        if not ident:
            return None
        values = []
        while True:
            while pos < n and text[pos].isspace():
                pos += 1
            if pos >= n or text[pos] != "[":
                break
            pos += 1
            buf = []
            while pos < n and text[pos] != "]":
                if text[pos] == "\\":
                    pos += 1
                    if pos >= n:
                        break
                buf.append(text[pos])
                pos += 1
            if pos >= n:
                return None
            pos += 1
            values.append("".join(buf))
        if not values:
            return None
        props[ident] = values
    return None


def _read(path, limit=None):
    with open(path, "rb") as f:
        data = f.read() if limit is None else f.read(limit)
    return data.decode("latin-1")


def check_file(path):
    """(path, reason or None, gameHash or None, size_bytes) for one file."""
    try:
        size = os.path.getsize(path)
        if size == 0:
            return path, "empty", None, 0
        text = _read(path, HEAD_BYTES)
    except OSError:
        return path, "unreadable", None, 0

    if not text.strip():
        return path, "empty", None, size

    whole = size <= HEAD_BYTES
    parsed = _root_node(text)
    if parsed is None and not whole:
        try:
            text, whole = _read(path), True
        except OSError:
            return path, "unreadable", None, size
        parsed = _root_node(text)
    if parsed is None:
        return path, "unreadable", None, size
    props, end = parsed

    if not _RE_MOVE.search(text, end):
        if not whole:
            try:
                text, whole = _read(path), True
            except OSError:
                return path, "unreadable", None, size
        if not _RE_MOVE.search(text, end):
            return path, "no_moves", None, size

    board = props.get("SZ", ["19"])[0].strip()
    if board not in ("19", "19:19"):
        return path, "not_19x19", None, size

    comment = " ".join(props.get("C", []))
    m = _RE_GTYPE.search(comment)
    gtype = m.group(1) if m else "none"
    if gtype not in KEEP_GTYPES:
        return path, "gtype_" + gtype, None, size

    h = _RE_GAMEHASH.search(comment)
    return path, None, (h.group(1).upper() if h else None), size


# ------------------------------------------------------------------------ scan

def _walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.lower().endswith(".sgf"):
                yield os.path.join(dirpath, name)


def _batches(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def cmd_scan(args):
    if not os.path.isdir(args.root):
        sys.exit("not a directory: {}".format(args.root))
    os.makedirs(args.out_dir, exist_ok=True)
    delete_path = os.path.join(args.out_dir, "to_delete.txt")
    conflict_path = os.path.join(args.out_dir, "hash_conflicts.txt")
    summary_path = os.path.join(args.out_dir, "scan_summary.txt")
    if os.path.exists(delete_path) and not args.force:
        sys.exit("{} already exists - pass --force to overwrite it".format(delete_path))

    counts = {}
    freed = {}
    kept = kept_bytes = total = total_bytes = no_hash = conflicts = 0
    first_by_hash = {}
    start = time.time()

    with open(delete_path, "w", encoding="utf-8", newline="\n") as out, \
            open(conflict_path, "w", encoding="utf-8", newline="\n") as conf, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch in _batches(_walk(args.root), BATCH):
            for path, reason, game_hash, size in pool.map(check_file, batch):
                total += 1
                total_bytes += size
                extra = ""
                if reason is None:
                    if game_hash is None:
                        no_hash += 1
                    elif game_hash in first_by_hash:
                        original = first_by_hash[game_hash]
                        if filecmp.cmp(original, path, shallow=False):
                            reason, extra = "duplicate", "\t" + original
                        else:
                            conflicts += 1
                            conf.write("{}\t{}\t{}\n".format(game_hash, original, path))
                    else:
                        first_by_hash[game_hash] = path
                if reason is None:
                    kept += 1
                    kept_bytes += size
                    continue
                counts[reason] = counts.get(reason, 0) + 1
                freed[reason] = freed.get(reason, 0) + size
                out.write("{}\t{}{}\n".format(path, reason, extra))
            if not args.quiet:
                rate = total / max(time.time() - start, 1e-9)
                print("  ... {:,} files  ({:,.0f}/sec)".format(total, rate), flush=True)

    elapsed = time.time() - start
    lines = _summary_lines(args.root, total, total_bytes, kept, kept_bytes, counts, freed,
                           no_hash, conflicts, elapsed)
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote {}".format(delete_path))
    if conflicts:
        print("wrote {}  <- REVIEW: same gameHash, different bytes".format(conflict_path))
    print("nothing has been deleted; run `delete {}` after reviewing".format(args.out_dir))


def _gb(n):
    return n / 1e9


def _summary_lines(root, total, total_bytes, kept, kept_bytes, counts, freed, no_hash,
                   conflicts, elapsed):
    to_delete = total - kept
    lines = [
        "root            : {}".format(root),
        "files scanned   : {:,}  ({:.2f} GB)".format(total, _gb(total_bytes)),
        "kept            : {:,}  ({:.2f} GB)".format(kept, _gb(kept_bytes)),
        "to delete       : {:,}  ({:.2f} GB, {:.1f}% of files)".format(
            to_delete, _gb(total_bytes - kept_bytes), 100.0 * to_delete / max(total, 1)),
        "elapsed         : {:.0f}s  ({:,.0f} files/sec)".format(
            elapsed, total / max(elapsed, 1e-9)),
        "",
        "to delete, by reason:",
    ]
    ordered = [r for r in REASON_ORDER if r in counts]
    ordered += sorted(r for r in counts if r.startswith("gtype_"))
    ordered += [r for r in ("duplicate",) if r in counts]
    for r in ordered:
        lines.append("  {:<22} {:>10,}  ({:.2f} GB)".format(r, counts[r], _gb(freed[r])))
    lines += [
        "",
        "kept files with no gameHash (not deduplicated): {:,}".format(no_hash),
        "hash conflicts (same gameHash, different bytes; NOT deleted): {:,}".format(conflicts),
    ]
    return lines


# ------------------------------------------------------------------------ delete

def cmd_delete(args):
    delete_path = os.path.join(args.out_dir, "to_delete.txt")
    if not os.path.exists(delete_path):
        sys.exit("no {} - run scan first".format(delete_path))
    log_path = os.path.join(args.out_dir, "delete_log.txt")

    deleted = missing = failed = skipped = 0
    freed = 0
    with open(delete_path, encoding="utf-8") as f, \
            open(log_path, "a", encoding="utf-8", newline="\n") as log:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            path, reason = parts[0], (parts[1] if len(parts) > 1 else "")
            if not os.path.exists(path):
                missing += 1
                continue
            # A duplicate is only safe to remove while its kept copy still exists and still
            # matches - otherwise deleting it could remove the last copy of the game.
            if reason == "duplicate":
                original = parts[2] if len(parts) > 2 else ""
                if not (original and os.path.exists(original)
                        and filecmp.cmp(original, path, shallow=False)):
                    skipped += 1
                    log.write("SKIPPED\t{}\tkept copy missing or changed: {}\n".format(
                        path, original))
                    continue
            size = os.path.getsize(path)
            if args.dry_run:
                deleted += 1
                freed += size
                continue
            try:
                os.remove(path)
            except OSError as e:
                failed += 1
                log.write("FAILED\t{}\t{}\n".format(path, e))
                continue
            deleted += 1
            freed += size

    verb = "would delete" if args.dry_run else "deleted"
    print("{:<14}: {:,}  ({:.2f} GB)".format(verb, deleted, _gb(freed)))
    print("{:<14}: {:,}".format("already gone", missing))
    print("{:<14}: {:,}".format("skipped dup", skipped))
    print("{:<14}: {:,}".format("failed", failed))
    if skipped or failed:
        print("details in {}".format(log_path))


# ------------------------------------------------------------------------ cli

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Find, then delete, SGFs that are not 19x19 normal/handicap KataGo games.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="Write to_delete.txt. Deletes nothing.")
    p.add_argument("root", help="Directory to walk (recursively) for .sgf files")
    p.add_argument("out_dir", help="Where to write to_delete.txt and the summary")
    p.add_argument("--workers", type=int, default=8, help="Reader threads. Default: 8")
    p.add_argument("--force", action="store_true", help="Overwrite an existing to_delete.txt")
    p.add_argument("--quiet", action="store_true", help="No progress lines")
    p.set_defaults(func=cmd_scan)

    d = sub.add_parser("delete", help="Delete the files listed in <out_dir>/to_delete.txt")
    d.add_argument("out_dir")
    d.add_argument("--dry-run", action="store_true", help="Report what would be deleted")
    d.set_defaults(func=cmd_delete)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
