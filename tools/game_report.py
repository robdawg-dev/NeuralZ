"""Routine check of the bot's own games (KGS or OGS SGFs): results, repeat opponents,
scoring and passing problems, dead ladders, and where its losses were lost. Run it after a
deploy, or every week or so, on the games downloaded since:

    uv run python tools/game_report.py <folders, .sgf or .zip files> --since 2026-10-06 \\
        --katago <exe> --katago-model <net.bin.gz> --katago-config <analysis.cfg> \\
        [--archive-pages <dir>] [--quick] [--out report.txt]

KataGo can also come from the environment: KATAGO_EXE, KATAGO_MODEL, KATAGO_CONFIG. Without
it, only the sections that need no judging are written.

Sections:
  results       record by handicap, by game type (ranked/free, from saved KGS archive pages,
                --archive-pages) and by how games ended
  opponents     anyone with 3+ wins against the bot, and opening lines an opponent repeated
  scoring       recorded results KataGo disagrees with (winner, or by > 10 points), and the
                bot passing back on a board KataGo still sees as open (> 10 contested points)
  ladders       bot moves extending a group in atari into a ladder the engine reads as dead
                (what the ladder guard prevents)
  losses        per lost game, KataGo's lead at the bot's first move and at the end, the
                bot's moves that lost >= 10 points, and "ignored fights": the bot throwing
                away 30+ points and the opponent handing them back, twice or more
                (skipped with --quick, the slow part: every position of every loss)

KGS records an undo as a side branch; the line actually played is the first child at each
fork, and that is the line analyzed. The bot is the player whose name starts with
--bot-prefix (default NeuralZ).
"""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLS = "abcdefghijklmnopqrs"
GTP_COLS = "ABCDEFGHJKLMNOPQRST"
CONTESTED_MAX = 10  # as SETTLED_MAX_CONTESTED in interface/gtp_wrapper.py
BIG_DROP = 10.0
FIGHT_SWING = 30.0


# --- reading games --------------------------------------------------------------------------

def parse_tree(text):
    """SGF text -> root node {"props": str, "children": [...]}, variations kept."""
    pos = 0

    def node_seq():
        nonlocal pos
        first = cur = None
        while pos < len(text):
            ch = text[pos]
            if ch == ";":
                pos += 1
                start, in_value = pos, False
                while pos < len(text) and (in_value or text[pos] not in ";()"):
                    if text[pos] == "[" and not in_value:
                        in_value = True
                    elif text[pos] == "]" and in_value and text[pos - 1] != "\\":
                        in_value = False
                    pos += 1
                node = {"props": text[start:pos], "children": []}
                if cur is None:
                    first = node
                else:
                    cur["children"].append(node)
                cur = node
            elif ch == "(":
                pos += 1
                sub = node_seq()
                if sub is not None and cur is not None:
                    cur["children"].append(sub)
            elif ch == ")":
                pos += 1
                return first
            else:
                pos += 1
        return first

    start = text.find("(")
    if start < 0:
        return None
    pos = start + 1
    return node_seq()


def played_line(root):
    """[(color, sgf point or "" for a pass), ...] along the first child at every fork."""
    line, node = [], root
    while node and node["children"]:
        node = node["children"][0]
        m = re.match(r"\s*([BW])\[([a-s]{2}|tt|)\]", node["props"])
        if m:
            line.append((m.group(1), "" if m.group(2) == "tt" else m.group(2)))
    return line


def prop(text, key):
    m = re.search(r"(?<![A-Z])" + key + r"\[((?:[^\]\\]|\\.)*)\]", text)
    return m.group(1).strip() if m else ""


def read_game(text, name, bot_prefix):
    """One SGF's text -> a game dict, or None if the bot isn't one of the players."""
    pb, pw = prop(text, "PB"), prop(text, "PW")
    if pb.startswith(bot_prefix):
        bot = "B"
    elif pw.startswith(bot_prefix):
        bot = "W"
    else:
        return None
    root = parse_tree(text)
    line = played_line(root) if root else []
    ab = re.search(r"AB((?:\s*\[[a-s]{2}\])+)", root["props"] if root else "")
    result = prop(text, "RE")
    m = re.match(r"([BW])\+(.*)$", result)
    winner = m.group(1) if m else None
    how = m.group(2).strip() if m else ""
    score = None
    if m and re.match(r"^[\d.]+$", how):
        score = float(how) * (1 if winner == bot else -1)  # bot's view
        how = "score"
    elif how:
        how = {"r": "resign", "t": "time", "f": "forfeit"}.get(how[0].lower(), how)
    return {
        "name": name, "date": prop(text, "DT")[:10], "bot": bot,
        "account": pb if bot == "B" else pw, "opponent": pw if bot == "B" else pb,
        "opp_rank": prop(text, "WR" if bot == "B" else "BR"),
        "ha": int(prop(text, "HA") or 0), "komi": float(prop(text, "KM") or 0),
        "rules": prop(text, "RU").lower() or "chinese", "result": result,
        "won": None if winner is None else winner == bot, "how": how, "score": score,
        "setup": re.findall(r"\[([a-s]{2})\]", ab.group(1)) if ab else [], "line": line,
        # free-placement handicap: Black's stones are recorded as its first moves
        "handicap_moves": handicap_moves(line, int(prop(text, "HA") or 0), bool(ab))}


def handicap_moves(line, ha, has_setup):
    """How many of the line's first moves are handicap stones (KGS free placement records
    them as Black moves in a row; with AB setup stones there are none)."""
    if ha < 2 or has_setup:
        return 0
    n = 0
    while n < min(ha, len(line)) and line[n][0] == "B":
        n += 1
    return n if n == ha else 0


def iter_sgfs(paths):
    """(name, text) for every .sgf under the given folders, files and .zip archives. name is
    the path from the year folder on (e.g. 2026/10/6/NeuralZ01-foo.sgf) when there is one."""
    def short(path):
        parts = re.split(r"[\\/]", path)
        for i, part in enumerate(parts):
            if re.match(r"^(19|20)\d\d$", part) and len(parts) - i == 4:
                return "/".join([parts[i], str(int(parts[i + 1])), str(int(parts[i + 2])),
                                 parts[i + 3]])
        return parts[-1]

    def from_file(path):
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                for member in z.namelist():
                    if member.lower().endswith(".sgf"):
                        yield short(member), z.read(member)
        elif path.lower().endswith(".sgf"):
            with open(path, "rb") as f:
                yield short(path), f.read()

    for path in paths:
        files = [path] if os.path.isfile(path) else sorted(
            glob.glob(os.path.join(path, "**", "*.sgf"), recursive=True) +
            glob.glob(os.path.join(path, "**", "*.zip"), recursive=True))
        for f in files:
            for name, data in from_file(f):
                yield name, data


def load_games(paths, bot_prefix, since=None):
    games, seen = [], set()
    for name, data in iter_sgfs(paths):
        digest = hashlib.sha1(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        game = read_game(data.decode("utf-8", "replace"), name, bot_prefix)
        if game and (since is None or game["date"] >= since):
            games.append(game)
    games.sort(key=lambda g: (g["date"], g["name"]))
    for i, g in enumerate(games):
        g["id"] = str(i)  # file names can repeat across folders and servers
    return games


def archive_types(folder):
    """Game type per file from saved KGS archive pages (gameArchives.jsp, saved as .html or
    .xhtml): {"2026/10/6/NeuralZ01-foo.sgf": "Ranked" | "Free" | ...}."""
    row = re.compile(r'files\.gokgs\.com/games/(\d+/\d+/\d+/[^"]+\.sgf)".*?</td>'
                     r'(?:<td>.*?</td>){3}<td>[^<]*</td><td>([^<]*)</td>')
    types = {}
    for page in glob.glob(os.path.join(folder, "*.*html")):
        with open(page, encoding="utf-8", errors="replace") as f:
            for url, kind in row.findall(f.read()):
                types[url] = kind
    return types


# --- KataGo ---------------------------------------------------------------------------------

def gtp(point, size=19):
    if not point:
        return "pass"
    return GTP_COLS[COLS.index(point[0])] + str(size - COLS.index(point[1]))


def katago_query(game, turns, visits, ownership=False):
    moves = [[c, gtp(p)] for c, p in game["line"]]
    rules = game["rules"] if game["rules"] in ("japanese", "korean", "aga") else "chinese"
    q = {"initialStones": [["B", gtp(p)] for p in game["setup"]], "moves": moves,
         "rules": rules, "komi": game["komi"], "boardXSize": 19, "boardYSize": 19,
         "maxVisits": visits, "analyzeTurns": sorted(set(turns))}
    if ownership:
        q["includeOwnership"] = True
    return q


def run_katago(command, queries, label=""):
    """{(query id, turn): KataGo's reply} for every analyzed turn. Black's view throughout."""
    # KataGo's example configs log to ./analysis_logs: keep that out of the working tree
    log_dir = os.path.join(tempfile.gettempdir(), "game_report_katago_logs")
    proc = subprocess.Popen(command + ["-override-config",
                                       "reportAnalysisWinratesAs=BLACK,logDir=" + log_dir],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)

    def feed():  # from a thread: writing everything first deadlocks on a full stdout pipe
        for q in queries:
            proc.stdin.write(json.dumps(q) + "\n")
        proc.stdin.close()
    threading.Thread(target=feed, daemon=True).start()
    total = sum(len(q["analyzeTurns"]) for q in queries)
    out = {}
    for line in proc.stdout:
        r = json.loads(line)
        if "error" in r or "warning" in r:
            continue
        out[(r["id"], r["turnNumber"])] = r
        if label and len(out) % 2000 == 0:
            sys.stderr.write("  {} {}/{}\n".format(label, len(out), total))
    proc.wait()
    return out


def contested(reply):
    return sum(1 for o in reply.get("ownership", []) if 0.3 <= abs(o) < 0.9)


# --- checks ---------------------------------------------------------------------------------

def finished(g):
    return g["won"] is not None and len(g["line"]) >= 50


def bot_pass_backs(g):
    """Turns (index after the bot's pass) where the bot passed right after an opponent's pass
    past move 100."""
    out = []
    for i in range(1, len(g["line"])):
        (c0, p0), (c1, p1) = g["line"][i - 1], g["line"][i]
        if i > 100 and c0 != g["bot"] and p0 == "" and c1 == g["bot"] and p1 == "":
            out.append(i + 1)
    return out


def last_board_turn(g):
    """Turn number of the final position, closing passes excluded."""
    n = len(g["line"])
    while n and g["line"][n - 1][1] == "":
        n -= 1
    return n


def bot_leads(g, replies, gid):
    """The bot's lead (KataGo) at every turn 0..last, None where missing."""
    sign = 1 if g["bot"] == "B" else -1
    return [sign * replies[(gid, t)]["rootInfo"]["scoreLead"] if (gid, t) in replies else None
            for t in range(last_board_turn(g) + 1)]


def drops(g, leads, threshold):
    """(move number, vertex, points lost) for the bot's moves that lost >= threshold."""
    out = []
    for i, (c, p) in enumerate(g["line"][:len(leads) - 1]):
        if c == g["bot"] and leads[i] is not None and leads[i + 1] is not None:
            lost = leads[i] - leads[i + 1]
            if lost >= threshold:
                out.append((i + 1, gtp(p), lost))
    return out


def ignored_fights(g, leads, swing=FIGHT_SWING):
    """Move numbers of the bot's moves that threw away >= swing points which the opponent's
    reply then gave back - the signature of both sides ignoring the same big fight. leads:
    the bot's lead per turn (bot_leads)."""
    out = []
    for i in range(1, len(leads) - 1):  # move i takes leads[i - 1] to leads[i]
        a, b, c = leads[i - 1], leads[i], leads[i + 1]
        if (g["line"][i - 1][0] == g["bot"] and None not in (a, b, c)
                and a - b >= swing and c - b >= swing):
            out.append(i)
    return out


def dead_ladder_moves(g):
    """Bot moves that extended a group in atari into a ladder the engine reads as dead."""
    from AlphaGo import go
    from AlphaGo.ai import ProbabilisticPolicyPlayer
    state = go.GameState(enforce_superko=False)
    if g["setup"]:
        state.place_handicaps([(COLS.index(p[0]), COLS.index(p[1])) for p in g["setup"]])
    bot = go.BLACK if g["bot"] == "B" else go.WHITE
    out = []
    for i, (c, p) in enumerate(g["line"]):
        color = go.BLACK if c == "B" else go.WHITE
        move = (COLS.index(p[0]), COLS.index(p[1])) if p else None
        if color == bot and move is not None:
            state.set_current_player(color)
            if move in ProbabilisticPolicyPlayer._failed_ladder_extensions(state):
                out.append((i + 1, gtp(p)))
        try:
            state.do_move(move, color)
        except go.IllegalMove:
            break
    return out


# --- report ---------------------------------------------------------------------------------

def pct(a, b):
    return "{:.0%}".format(a / b) if b else "-"


def report(games, types=None, katago=None, quick=False, visits=30, write=print):
    fin = [g for g in games if finished(g)]
    write("# Game report: {} games ({} to {}), {} finished".format(
        len(games), games[0]["date"] if games else "-", games[-1]["date"] if games else "-",
        len(fin)))
    accounts = sorted({g["account"] for g in games})
    write("accounts: {}".format(", ".join(accounts)))

    # results
    write("\n## Results (bot wins-losses)")
    won = sum(g["won"] for g in fin)
    write("overall {}-{} ({})".format(won, len(fin) - won, pct(won, len(fin))))

    def table(title, key):
        rows = collections.defaultdict(lambda: [0, 0])
        for g in fin:
            rows[key(g)][0 if g["won"] else 1] += 1
        write("{}: {}".format(title, ", ".join(
            "{} {}-{}".format(k, w, l_) for k, (w, l_) in sorted(rows.items()))))
    table("by handicap", lambda g: "HA{}".format(g["ha"]))
    table("by how it ended (bot won-lost)", lambda g: g["how"] or "?")
    if types:
        table("by game type", lambda g: types.get(g["name"], "not in pages"))

    # opponents
    write("\n## Opponents with 3+ wins against the bot")
    rec = collections.defaultdict(lambda: [0, 0, set(), set()])
    for g in fin:
        r = rec[g["opponent"]]
        r[1 if g["won"] else 0] += 1
        r[2].add(g["opp_rank"] or "-")
        r[3].add(g["ha"])
    for opp, (w, l_, ranks, has) in sorted(rec.items(), key=lambda kv: -kv[1][0]):
        if w >= 3:
            write("- {}: {}-{} against the bot | rank {} | handicap {}".format(
                opp, w, l_, "/".join(sorted(ranks)), "/".join(map(str, sorted(has)))))
    write("\nRepeated openings (40+ moves identical to an earlier game by the same opponent):")
    by_opp, any_repeat = collections.defaultdict(list), False
    for g in games:
        best = 0
        for h in by_opp[g["opponent"]]:
            n = 0
            for a, b in zip(g["line"], h["line"]):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        if best >= 40:
            any_repeat = True
            write("- {} {}: {} moves, {}".format(g["opponent"], g["name"], best, g["result"]))
        by_opp[g["opponent"]].append(g)
    if not any_repeat:
        write("- none")

    # ladders
    write("\n## Dead-ladder moves by the bot")
    ladder_hits = [(g, m) for g in games for m in dead_ladder_moves(g)]
    for g, (n, v) in ladder_hits:
        write("- {} move {} {} ({})".format(g["name"], n, v, g["result"]))
    if not ladder_hits:
        write("- none")

    if katago is None:
        write("\n(no KataGo: scoring and loss sections skipped)")
        return

    # scoring and passing
    scored = [g for g in fin if g["score"] is not None]
    passes = {g["id"]: bot_pass_backs(g) for g in fin}
    queries = []
    for g in fin:
        if g["score"] is not None:  # the final board, searched for the count
            q = katago_query(g, [last_board_turn(g)], 100)
            q["id"] = g["id"]
            queries.append(q)
        if passes[g["id"]]:
            # each pass-back's board as the opponent passed on it (before their pass: the
            # same stones), at 1 visit: the bot's gate was calibrated on the network's own
            # ownership, and a search spreads ownership out, inflating the contested count
            q = katago_query(g, [t - 2 for t in passes[g["id"]]], 1, ownership=True)
            q["id"] = g["id"] + "|pass"
            queries.append(q)
    replies = run_katago(katago, queries, "final positions")
    write("\n## Scoring: recorded result vs KataGo (bot's view)")
    flagged = 0
    for g in scored:
        r = replies.get((g["id"], last_board_turn(g)))
        if r is None:
            continue
        kata = r["rootInfo"]["scoreLead"] * (1 if g["bot"] == "B" else -1)
        if (kata > 0) != (g["score"] > 0) or abs(kata - g["score"]) > 10:
            flagged += 1
            write("- {}{}: recorded {:+.1f}, KataGo {:+.1f}{}".format(
                g["name"], " ({})".format(types.get(g["name"])) if types else "", g["score"],
                kata, " <- bot was ahead" if kata > 0 > g["score"] else ""))
    write("{} of {} scored games flagged".format(flagged, len(scored)))
    write("\nBot passing back on an open board (> {} contested points, the network at 1 visit as "
          "the bot judges it):".format(CONTESTED_MAX))
    open_passes = 0
    for g in fin:
        for t in passes[g["id"]]:
            r = replies.get((g["id"] + "|pass", t - 2))
            if r is not None and contested(r) > CONTESTED_MAX:
                open_passes += 1
                write("- {} move {}: {} contested points".format(g["name"], t, contested(r)))
    write("{} of {} pass-backs".format(open_passes, sum(len(v) for v in passes.values())))

    if quick:
        write("\n(--quick: loss analysis skipped)")
        return

    # losses
    losses = [g for g in fin if not g["won"]]
    qs = []
    for g in losses:
        q = katago_query(g, range(last_board_turn(g) + 1), visits)
        q["id"] = g["id"]
        qs.append(q)
    lead_replies = run_katago(katago, qs, "loss positions")
    write("\n## Losses ({}), KataGo {} visits per position".format(len(losses), visits))
    rows = []
    for g in losses:
        leads = bot_leads(g, lead_replies, g["id"])
        # the expectation the game started from: the bot's first real move, after the
        # handicap stones (which the bot itself places when it is Black)
        first = next((i for i, (c, _p) in enumerate(g["line"])
                      if c == g["bot"] and i >= g["handicap_moves"]), None)
        if first is None or first >= len(leads) or leads[first] is None or leads[-1] is None:
            continue
        rows.append((g, leads[first], leads[-1], drops(g, leads, BIG_DROP),
                     ignored_fights(g, leads)))
    by_ha = collections.defaultdict(list)
    for row in rows:
        by_ha[row[0]["ha"]].append(row)
    for ha in sorted(by_ha):
        rs = by_ha[ha]
        write("HA{}: {} losses | start median {:+.1f} | final median {:+.1f} | "
              "drops >= {:.0f} per game {:.1f}".format(
                  ha, len(rs), statistics.median(r[1] for r in rs),
                  statistics.median(r[2] for r in rs), BIG_DROP,
                  statistics.mean(len(r[3]) for r in rs)))
    write("\nLargest falls from the starting expectation:")
    falls = sorted((r for r in rows if r[2] < r[1]), key=lambda r: r[2] - r[1])
    for g, start, final, ds, fights in falls[:15]:
        top = sorted(ds, key=lambda d: -d[2])[:3]
        write("- {} vs {} ({}) HA{}: start {:+.1f}, final {:+.1f} | {}".format(
            g["name"], g["opponent"], g["opp_rank"] or "-", g["ha"], start, final,
            ", ".join("move {} {} -{:.0f}".format(*d) for d in top) or "no single big drop"))
    write("\nIgnored fights (bot gave away {:.0f}+ points and got them back, twice or "
          "more):".format(FIGHT_SWING))
    hits = [r for r in rows if len(r[4]) >= 2]
    for g, _s, _f, _d, fights in sorted(hits, key=lambda r: -len(r[4])):
        write("- {} vs {}: {} times, first at move {}".format(
            g["name"], g["opponent"], len(fights), fights[0]))
    if not hits:
        write("- none")
    gradual = sum(1 for r in rows if not r[3])
    write("\nLost gradually (no single drop >= {:.0f}): {} of {}".format(
        BIG_DROP, gradual, len(rows)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="folders, .sgf or .zip files")
    parser.add_argument("--since", help="only games on or after this date (YYYY-MM-DD)")
    parser.add_argument("--bot-prefix", default="NeuralZ")
    parser.add_argument("--archive-pages", help="folder of saved KGS archive pages (game types)")
    parser.add_argument("--katago", default=os.environ.get("KATAGO_EXE"))
    parser.add_argument("--katago-model", default=os.environ.get("KATAGO_MODEL"))
    parser.add_argument("--katago-config", default=os.environ.get("KATAGO_CONFIG"))
    parser.add_argument("--visits", type=int, default=30, help="per loss position. Default: 30")
    parser.add_argument("--quick", action="store_true", help="skip the loss analysis")
    parser.add_argument("--out", help="also write the report to this file")
    args = parser.parse_args(argv)
    if args.since:
        datetime.date.fromisoformat(args.since)  # a typo fails here, not as an empty report
    games = load_games(args.paths, args.bot_prefix, args.since)
    if not games:
        sys.exit("game_report: no games by {}* found".format(args.bot_prefix))
    katago = None
    if args.katago and args.katago_model and args.katago_config:
        katago = [args.katago, "analysis", "-config", args.katago_config,
                  "-model", args.katago_model]
    types = archive_types(args.archive_pages) if args.archive_pages else None
    lines = []

    def write(text):
        print(text, flush=True)
        lines.append(text)
    report(games, types, katago, args.quick, args.visits, write)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
