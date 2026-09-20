#!/usr/bin/env python
"""Survey a directory tree of SGF files and report the distributions that decide which
games are healthy to train on. READ-ONLY - this module never writes to or deletes the
files it scans.

Why this exists
---------------
This project was originally trained on KGS/GoGoD human games. It is now trained on
KataGo selfplay, which is a materially different kind of data: it contains games seeded
from arbitrary mid-game positions, games where one side was deliberately weakened,
deliberate low-confidence exploration moves, and a wide spread of rulesets and komi. None
of that exists in human game records, so none of the original pipeline's assumptions were
ever tested against it.

The goal of training data here is narrow: a valid board position, and the move a strong
player actually chose from it. This tool measures how often each candidate problem occurs
so that filtering policy can be chosen from numbers rather than guesses. Deciding and
acting on that policy is sgf_preparation.py's job; this module only reports.

Two-tier scanning
-----------------
Almost everything worth knowing (board size, provenance, ruleset, komi, per-move search
metadata) lives in the SGF header or in KataGo's own move comments, and can be read with
compiled regexes at roughly filesystem speed. Only illegal-move detection needs a real
replay through the Cython engine, which is orders of magnitude slower. So the deep pass
is opt-in via --replay, and is normally run on a sample rather than the whole corpus.

The corpus this is aimed at is potentially billions of files, so all aggregation is into
bounded-cardinality Counters (never lists of per-move values) and the file list is
streamed rather than materialised where possible. --sample caps the work for quick
iteration; --progress-every reports throughput so a full run can be sized honestly before
being started.

KataGo move comment format
--------------------------
Verified against KataGo's own source (cpp/dataio/sgf.cpp, WriteSgf::writeSgf):

    C[<win> <loss> <noResult> <score> v=<visits> [rv=<reanalysisVisits>] weight=<w>]

- v= is the visit count of the search that chose the move (unreducedNumVisits).
- rv= appears only when the game was re-analysed after the fact; v= is then the original
  cheap search and rv= the reanalysis.
- weight= is targetWeightByTurnUnrounded, KataGo's own training weight for the position:
    * exactly 0.00  -> the move came from a "cheap search" (reduced visits). KataGo's
      shipped selfplay configs set cheapSearchTargetWeight = 0.0
      (cpp/configs/training/selfplay1.cfg), meaning KataGo EXCLUDES these positions from
      its own training data.
    * 0 < w < 1     -> reduced-visit taper toward game end (reducedVisitsWeight = 0.1).
    * w > 1         -> upweighted by policySurpriseDataWeight, i.e. positions where the
      search disagreed most with the raw network - the most informative ones.
  Visit count alone is NOT a usable proxy for weight (measured: the weight>0 and
  weight==0 visit distributions overlap heavily), so read weight directly.

KataGo game provenance (gtype, in the root node comment)
--------------------------------------------------------
    normal            ordinary selfplay from an empty board
    sgfpos            seeded from a position lifted out of an existing SGF
    fork / hintfork   branched from another game, optionally at a hinted move
    hintpos           seeded from a position with a hint move
    handicap          genuine handicap game
    asym              asymmetric playouts - one side searches with fewer visits and is
                      therefore deliberately weaker
    cleanuptraining   synthetic endgame/cleanup drilling

Positions seeded from a prior position (sgfpos/fork/hintpos/...) arrive as AB/AW setup
stones with their real move order discarded, which is what makes the move-recency
features (turns_since, last_moves) unreliable for the first few plies of such games - see
AUDIT_NOTES.md.
"""
import argparse
import collections
import json
import os
import re
import sys
import time


# --- header fields -----------------------------------------------------------------
_RE_SZ = re.compile(r"SZ\[([^\]]*)\]")
_RE_KM = re.compile(r"KM\[([^\]]*)\]")
_RE_RU = re.compile(r"RU\[([^\]]*)\]")
_RE_HA = re.compile(r"HA\[([^\]]*)\]")
_RE_RE = re.compile(r"RE\[([^\]]*)\]")
_RE_AB = re.compile(r"AB((?:\[[^\]]*\])+)")
_RE_AW = re.compile(r"AW((?:\[[^\]]*\])+)")
_RE_BRACKETS = re.compile(r"\[([^\]]*)\]")

# --- root provenance comment -------------------------------------------------------
_RE_ROOT_C = re.compile(r"C\[([^\]]*gtype[^\]]*)\]")

# --- moves and per-move annotations ------------------------------------------------
_RE_MOVE = re.compile(r";([BW])\[([^\]]*)\]")
_RE_ANNOT = re.compile(
    r"C\[\s*(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) v=(\d+)(?: rv=(\d+))? weight=([\d.]+)")
# A node that opens with ';' but carries neither a B nor a W property. sgf_iter_states
# replays the PREVIOUS move on such a node (see AUDIT_NOTES.md), truncating or dropping
# the game, so it is worth counting even though KataGo does not currently emit any.
_RE_NODE = re.compile(r";([A-Z]{1,2})\[")

# Ruleset strings look like "koSITUATIONALscoreTERRITORYtaxSEKIsui1button1".
_RE_RULES = re.compile(
    r"ko(?P<ko>[A-Z]+)score(?P<score>[A-Z]+)tax(?P<tax>[A-Z]+)sui(?P<sui>[01])(?P<rest>.*)")

SETUP_GTYPES = {"sgfpos", "fork", "hintpos", "hintfork"}
# gtypes whose moves are not a strong player's genuine best attempt
SUSPECT_GTYPES = {"asym", "cleanuptraining"}


def _bucket(value, edges):
    """Map a number into a bounded set of labelled bins, so aggregation stays O(1) in
    memory no matter how many files are scanned."""
    prev = None
    for e in edges:
        if value < e:
            return "<{}".format(e) if prev is None else "{}-{}".format(prev, e)
        prev = e
    return ">={}".format(edges[-1])


VISIT_EDGES = [100, 200, 300, 500, 1000, 2000, 4000]
WEIGHT_EDGES = [0.001, 0.1, 0.5, 1.0, 2.0]
MOVES_EDGES = [50, 100, 200, 300, 400, 600]


def scan_file_cheap(path):
    """Text-only scan of one SGF. Returns a flat dict of findings; never raises for
    malformed content (a file that cannot be read or has no moves is itself a finding)."""
    rec = {"path": path, "reasons": []}
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
    except OSError as e:
        rec["reasons"].append("unreadable")
        rec["error"] = str(e)
        return rec

    rec["bytes"] = len(text)

    # board size
    m = _RE_SZ.search(text)
    size = m.group(1) if m else None
    rec["size"] = size
    if size != "19":
        rec["reasons"].append("not_19x19")

    # provenance
    gtype = None
    m = _RE_ROOT_C.search(text)
    if m:
        kv = dict(p.split("=", 1) for p in m.group(1).split(",") if "=" in p)
        gtype = kv.get("gtype")
        rec["start_turn_idx"] = kv.get("startTurnIdx")
        rec["init_turn_num"] = kv.get("initTurnNum")
        rec["game_hash"] = kv.get("gameHash")
    rec["gtype"] = gtype
    if gtype in SUSPECT_GTYPES:
        rec["reasons"].append("suspect_gtype:{}".format(gtype))

    # setup stones
    n_ab = n_aw = 0
    m = _RE_AB.search(text)
    if m:
        n_ab = len(_RE_BRACKETS.findall(m.group(1)))
    m = _RE_AW.search(text)
    if m:
        n_aw = len(_RE_BRACKETS.findall(m.group(1)))
    rec["n_ab"], rec["n_aw"] = n_ab, n_aw
    if n_ab or n_aw:
        rec["reasons"].append("has_setup_stones")

    m = _RE_HA.search(text)
    rec["handicap"] = m.group(1) if m else None

    # rules / komi
    m = _RE_RU.search(text)
    ru = m.group(1) if m else None
    rec["rules"] = ru
    if ru:
        rm = _RE_RULES.match(ru)
        if rm:
            rec["ko"] = rm.group("ko")
            rec["score"] = rm.group("score")
            rec["tax"] = rm.group("tax")
            rec["sui"] = rm.group("sui")
            rec["rules_extra"] = rm.group("rest") or None
            if rm.group("sui") == "1":
                rec["reasons"].append("suicide_allowed")
        else:
            rec["reasons"].append("unparsed_ruleset")
    m = _RE_KM.search(text)
    rec["komi"] = m.group(1) if m else None

    m = _RE_RE.search(text)
    rec["result"] = m.group(1) if m else None

    # moves / passes
    moves = _RE_MOVE.findall(text)
    rec["n_moves"] = len(moves)
    rec["n_passes"] = sum(1 for _c, v in moves if v == "" or v == "tt")
    if rec["n_moves"] == 0:
        rec["reasons"].append("no_moves")

    # nodes carrying neither B nor W (would desync sgf_iter_states)
    node_props = _RE_NODE.findall(text)
    # first node is the root; every later node should open with B or W
    later = node_props[1:] if node_props else []
    rec["n_moveless_nodes"] = sum(1 for p in later if p not in ("B", "W"))
    if rec["n_moveless_nodes"]:
        rec["reasons"].append("moveless_node")

    # per-move search metadata
    annots = _RE_ANNOT.findall(text)
    rec["n_annotated"] = len(annots)
    rec["n_reanalyzed"] = sum(1 for a in annots if a[5])
    if annots:
        weights = [float(a[6]) for a in annots]
        visits = [int(a[4]) for a in annots]
        rec["n_weight_zero"] = sum(1 for w in weights if w == 0.0)
        rec["n_weight_pos"] = len(weights) - rec["n_weight_zero"]
        rec["visit_buckets"] = collections.Counter(_bucket(v, VISIT_EDGES) for v in visits)
        rec["weight_buckets"] = collections.Counter(_bucket(w, WEIGHT_EDGES) for w in weights)
    else:
        rec["n_weight_zero"] = rec["n_weight_pos"] = 0
        rec["visit_buckets"] = collections.Counter()
        rec["weight_buckets"] = collections.Counter()
        if rec["n_moves"]:
            rec["reasons"].append("no_move_annotations")

    return rec


def scan_file_deep(path):
    """Replay the game through the real engine to find moves it rejects. Much slower than
    the cheap scan (full Cython GameState replay per game), so this is opt-in.

    Imports are function-local so that a cheap-only run never pays for loading the Cython
    engine, and so this module stays importable if the extensions aren't built.
    """
    import sgf as sgflib
    import AlphaGo.go as go
    from AlphaGo.util import _sgf_init_gamestate, _parse_sgf_move

    out = {"replay": "ok", "replay_moves_ok": 0, "replay_moves_declared": 0}
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
        game = sgflib.parse(text)[0]
        gs = _sgf_init_gamestate(game.root)
    except Exception as e:                                    # noqa: BLE001 - any parse issue
        out["replay"] = "parse_error"
        out["replay_error"] = "{}: {}".format(type(e).__name__, e)
        return out

    if game.rest is None:
        out["replay"] = "no_moves"
        return out

    declared = [n for n in game.rest if "B" in n.properties or "W" in n.properties]
    out["replay_moves_declared"] = len(declared)
    for node in declared:
        props = node.properties
        if "B" in props:
            move, color = _parse_sgf_move(props["B"][0]), go.BLACK
        else:
            move, color = _parse_sgf_move(props["W"][0]), go.WHITE
        try:
            gs.do_move(move, color)
            out["replay_moves_ok"] += 1
        except go.IllegalMove:
            # classify so a genuine engine disagreement can be told apart from a rules
            # difference we already understand (KataGo's sui1 permits suicide)
            board = gs.get_board()
            if move is not None and board[move[0]][move[1]] != go.EMPTY:
                out["replay"] = "illegal_occupied"
            elif gs.get_ko_location() == move:
                out["replay"] = "illegal_ko"
            else:
                out["replay"] = "illegal_suicide"
            out["replay_fail_move"] = out["replay_moves_ok"]
            return out
        except Exception as e:                                # noqa: BLE001
            out["replay"] = "error"
            out["replay_error"] = "{}: {}".format(type(e).__name__, e)
            return out
    return out


def _scan_chunk(args):
    """Worker entry point. Takes a list of paths rather than one path so per-task
    overhead stays negligible at corpus scale."""
    paths, deep = args
    results = []
    for p in paths:
        rec = scan_file_cheap(p)
        if deep and "unreadable" not in rec["reasons"]:
            rec.update(scan_file_deep(p))
        results.append(rec)
    return results


def iter_sgf_files(root, limit=None):
    """Stream SGF paths from a (possibly deeply nested) tree without materialising the
    whole list - at corpus scale the file list itself is large."""
    n = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".sgf"):
                yield os.path.join(dirpath, name)
                n += 1
                if limit is not None and n >= limit:
                    return


def _chunks(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class Aggregator:
    """Accumulates findings into bounded-cardinality counters. Deliberately stores no
    per-file rows except a capped sample, so memory stays flat across a billion files."""

    def __init__(self, keep_examples=5):
        self.n_files = 0
        self.counters = collections.defaultdict(collections.Counter)
        self.reasons = collections.Counter()
        self.totals = collections.Counter()
        self.examples = collections.defaultdict(list)
        self.keep_examples = keep_examples

    def add(self, rec):
        self.n_files += 1
        c = self.counters
        c["size"][rec.get("size")] += 1
        c["gtype"][rec.get("gtype")] += 1
        c["komi"][rec.get("komi")] += 1
        c["ko"][rec.get("ko")] += 1
        c["score"][rec.get("score")] += 1
        c["tax"][rec.get("tax")] += 1
        c["sui"][rec.get("sui")] += 1
        c["rules_extra"][rec.get("rules_extra")] += 1
        c["handicap"][rec.get("handicap")] += 1
        c["start_turn_idx"][rec.get("start_turn_idx")] += 1
        c["n_moves"][_bucket(rec.get("n_moves", 0), MOVES_EDGES)] += 1
        c["n_passes"][min(rec.get("n_passes", 0), 20)] += 1
        c["n_ab"][min(rec.get("n_ab", 0), 40)] += 1
        c["n_aw"][min(rec.get("n_aw", 0), 40)] += 1
        c["visits"].update(rec.get("visit_buckets", {}))
        c["weights"].update(rec.get("weight_buckets", {}))
        if "replay" in rec:
            c["replay"][rec["replay"]] += 1

        self.totals["moves"] += rec.get("n_moves", 0)
        self.totals["passes"] += rec.get("n_passes", 0)
        self.totals["annotated"] += rec.get("n_annotated", 0)
        self.totals["reanalyzed"] += rec.get("n_reanalyzed", 0)
        self.totals["weight_zero"] += rec.get("n_weight_zero", 0)
        self.totals["weight_pos"] += rec.get("n_weight_pos", 0)
        self.totals["moveless_nodes"] += rec.get("n_moveless_nodes", 0)
        self.totals["setup_stones"] += rec.get("n_ab", 0) + rec.get("n_aw", 0)
        if "replay_moves_declared" in rec:
            self.totals["replay_declared"] += rec["replay_moves_declared"]
            self.totals["replay_ok"] += rec["replay_moves_ok"]

        for reason in rec["reasons"]:
            self.reasons[reason] += 1
            if len(self.examples[reason]) < self.keep_examples:
                self.examples[reason].append(rec["path"])
        if rec.get("replay", "ok") not in ("ok", None):
            key = "replay:" + rec["replay"]
            self.reasons[key] += 1
            if len(self.examples[key]) < self.keep_examples:
                self.examples[key].append(rec["path"])


def _pct(n, d):
    return 0.0 if not d else 100.0 * n / d


def _print_counter(title, counter, limit=12, total=None):
    total = total or sum(counter.values())
    print("\n{}".format(title))
    for key, n in counter.most_common(limit):
        print("    {:<46} {:>10,}  ({:5.2f}%)".format(str(key), n, _pct(n, total)))
    if len(counter) > limit:
        print("    ... {} more distinct values".format(len(counter) - limit))


def report(agg, elapsed, deep):
    n = agg.n_files
    t = agg.totals
    print("=" * 78)
    print("SGF CORPUS ANALYSIS")
    print("=" * 78)
    print("files scanned : {:,}".format(n))
    print("elapsed       : {:.1f}s   ({:,.0f} files/sec)".format(
        elapsed, n / elapsed if elapsed else 0))
    print("declared moves: {:,}".format(t["moves"]))
    print("deep replay   : {}".format("on" if deep else "off (pass --replay to enable)"))

    _print_counter("BOARD SIZE", agg.counters["size"], total=n)
    _print_counter("GAME PROVENANCE (gtype)", agg.counters["gtype"], total=n)
    print("\n  setup-stone gtypes (sgfpos/fork/hintpos/hintfork): {:,} ({:.2f}%)".format(
        sum(agg.counters["gtype"][g] for g in SETUP_GTYPES),
        _pct(sum(agg.counters["gtype"][g] for g in SETUP_GTYPES), n)))
    print("  suspect gtypes (asym/cleanuptraining):            {:,} ({:.2f}%)".format(
        sum(agg.counters["gtype"][g] for g in SUSPECT_GTYPES),
        _pct(sum(agg.counters["gtype"][g] for g in SUSPECT_GTYPES), n)))

    _print_counter("KO RULE", agg.counters["ko"], total=n)
    _print_counter("SCORING", agg.counters["score"], total=n)
    _print_counter("TAX", agg.counters["tax"], total=n)
    _print_counter("SUICIDE ALLOWED (sui)", agg.counters["sui"], total=n)
    _print_counter("EXTRA RULE FLAGS", agg.counters["rules_extra"], total=n)
    _print_counter("KOMI", agg.counters["komi"], limit=15, total=n)
    _print_counter("HANDICAP (HA)", agg.counters["handicap"], total=n)

    _print_counter("MOVES PER GAME", agg.counters["n_moves"], total=n)
    _print_counter("PASSES PER GAME (capped at 20)", agg.counters["n_passes"], total=n)
    print("\n  total passes: {:,} ({:.2f} per game, {:.2f}% of all moves)".format(
        t["passes"], t["passes"] / n if n else 0, _pct(t["passes"], t["moves"])))

    _print_counter("AB SETUP STONES PER GAME (capped at 40)", agg.counters["n_ab"], total=n)
    _print_counter("AW SETUP STONES PER GAME (capped at 40)", agg.counters["n_aw"], total=n)

    print("\nPER-MOVE SEARCH METADATA")
    print("    moves with v=/weight= annotation : {:,} ({:.2f}% of moves)".format(
        t["annotated"], _pct(t["annotated"], t["moves"])))
    print("    post-game reanalysed (rv=)       : {:,} ({:.2f}% of annotated)".format(
        t["reanalyzed"], _pct(t["reanalyzed"], t["annotated"])))
    print("    weight == 0 (KataGo discards)    : {:,} ({:.2f}% of annotated)".format(
        t["weight_zero"], _pct(t["weight_zero"], t["annotated"])))
    print("    weight >  0 (KataGo trains on)   : {:,} ({:.2f}% of annotated)".format(
        t["weight_pos"], _pct(t["weight_pos"], t["annotated"])))
    _print_counter("  visit-count distribution", agg.counters["visits"], total=t["annotated"])
    _print_counter("  weight distribution", agg.counters["weights"], total=t["annotated"])

    if deep:
        _print_counter("ENGINE REPLAY OUTCOME", agg.counters["replay"], total=n)
        print("\n    moves replayed ok : {:,} / {:,} ({:.4f}% lost)".format(
            t["replay_ok"], t["replay_declared"],
            _pct(t["replay_declared"] - t["replay_ok"], t["replay_declared"])))

    print("\n" + "=" * 78)
    print("FLAGGED FILES (a file may carry several flags)")
    print("=" * 78)
    for reason, count in agg.reasons.most_common():
        print("  {:<34} {:>10,}  ({:5.2f}% of files)".format(reason, count, _pct(count, n)))
        for ex in agg.examples[reason][:2]:
            print("        e.g. {}".format(ex))
    if not agg.reasons:
        print("  (none)")


def analyze(directory, workers=None, limit=None, deep=False, chunk_size=200,
            progress_every=0, json_out=None):
    import concurrent.futures

    workers = workers or os.cpu_count()
    agg = Aggregator()
    start = time.time()
    last_report = start

    paths = iter_sgf_files(directory, limit=limit)
    work = ((chunk, deep) for chunk in _chunks(paths, chunk_size))

    if workers == 1:
        results_iter = (_scan_chunk(w) for w in work)
        for results in results_iter:
            for rec in results:
                agg.add(rec)
            if progress_every and agg.n_files - (agg.n_files % progress_every) != 0:
                pass
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            # Bounded in-flight submission: the file list can be enormous, and submitting
            # everything up front would both materialise it and let workers outrun the
            # aggregation loop. Same backpressure pattern as game_converter_parallel.
            max_in_flight = workers * 4
            pending = set()

            def _submit_next():
                try:
                    pending.add(pool.submit(_scan_chunk, next(work)))
                except StopIteration:
                    pass

            for _ in range(max_in_flight):
                _submit_next()

            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    for rec in fut.result():
                        agg.add(rec)
                    _submit_next()
                if progress_every:
                    now = time.time()
                    if now - last_report >= 2.0:
                        rate = agg.n_files / (now - start) if now > start else 0
                        print("  ... {:,} files  ({:,.0f}/sec)".format(agg.n_files, rate),
                              file=sys.stderr)
                        last_report = now

    elapsed = time.time() - start
    report(agg, elapsed, deep)

    if json_out:
        payload = {
            "n_files": agg.n_files,
            "elapsed_sec": elapsed,
            "files_per_sec": agg.n_files / elapsed if elapsed else 0,
            "deep": deep,
            "totals": dict(agg.totals),
            "reasons": dict(agg.reasons),
            "counters": {k: {str(kk): vv for kk, vv in v.items()}
                         for k, v in agg.counters.items()},
            "examples": {k: v for k, v in agg.examples.items()},
        }
        with open(json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print("\nmachine-readable summary written to {}".format(json_out))
    return agg


def main(cmd_line_args=None):
    parser = argparse.ArgumentParser(
        description="Survey a directory tree of SGF files for issues that affect training "
                    "data quality. Read-only: never modifies or deletes anything.")
    parser.add_argument("directory", help="Root directory to scan (recurses into subdirectories)")  # noqa: E501
    parser.add_argument("--workers", "-w", type=int, default=None, help="Worker processes (default: os.cpu_count())")  # noqa: E501
    parser.add_argument("--sample", "-n", type=int, default=None, help="Stop after this many files - use for quick iteration on a huge corpus")  # noqa: E501
    parser.add_argument("--replay", action="store_true", help="Also replay every game through the Cython engine to find moves it rejects (suicide, ko, occupied). Much slower - normally run on a --sample.")  # noqa: E501
    parser.add_argument("--chunk-size", type=int, default=200, help="Files per worker task. Default: 200")  # noqa: E501
    parser.add_argument("--json", dest="json_out", default=None, help="Also write the full summary as JSON to this path")  # noqa: E501
    parser.add_argument("--quiet", action="store_true", help="Suppress the periodic throughput line on stderr")  # noqa: E501
    args = parser.parse_args(cmd_line_args)

    analyze(args.directory, workers=args.workers, limit=args.sample, deep=args.replay,
            chunk_size=args.chunk_size, progress_every=0 if args.quiet else 1,
            json_out=args.json_out)


if __name__ == "__main__":
    main()
