"""Profile one opponent from the bot's games against them: is a run of wins skill, too
large a handicap, a program, or an exploit?

    uv run python tools/opponent_report.py <opponent> <game folders, .sgf or .zip> \\
        [--since 2026-10-06] [--visits 30]

Without KataGo: the record (by handicap), margins, how games ended (who passed when), the
opponent's time per move (an engine's is fast and steady - a median of 1-2 s with little
spread), and how often the opponent answered the same position the same way across
games (an engine playing its top choice always does). With KataGo, also each side's points
lost per move and the lead at the start and end - how big a head start the handicap gave,
and how much of it the bot won back. KataGo: --katago/--katago-model/--katago-config or
KATAGO_EXE/KATAGO_MODEL/KATAGO_CONFIG.
"""
import argparse
import collections
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import game_report as gr  # noqa: E402
from katago_util import add_katago_args, katago_command, run  # noqa: E402


def move_seconds(text, color):
    """Seconds the player spent per move, from the SGF's BL/WL clock values."""
    left = [float(x) for x in re.findall(r";{0}\[[a-s]*\]{0}L\[([\d.]+)\]".format(color), text)]
    return [a - b for a, b in zip(left, left[1:]) if a - b >= 0]


def repeated_replies(games, opp_color_of):
    """Over positions reached in two or more games with the opponent to move: (positions,
    how many of them the opponent always answered the same way)."""
    seen = collections.defaultdict(set)
    for g in games:
        opp = opp_color_of(g)
        for i, (c, p) in enumerate(g["line"][:60]):
            if c == opp:
                seen[tuple(g["line"][:i])].add(p)
    repeated = {k: v for k, v in seen.items()
                if sum(1 for g in games if tuple(g["line"][:len(k)]) == k
                       and len(g["line"]) > len(k)) >= 2}
    return len(repeated), sum(1 for v in repeated.values() if len(v) == 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("opponent")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--since")
    parser.add_argument("--bot-prefix", default="NeuralZ")
    parser.add_argument("--visits", type=int, default=30)
    add_katago_args(parser)
    args = parser.parse_args(argv)

    texts = {name: data.decode("utf-8", "replace") for name, data in gr.iter_sgfs(args.paths)}
    games = [g for g in gr.load_games(args.paths, args.bot_prefix, args.since)
             if g["opponent"].lower() == args.opponent.lower()]
    if not games:
        sys.exit("opponent_report: no games against {}".format(args.opponent))
    fin = [g for g in games if gr.finished(g)]
    opp_of = {g["id"]: "W" if g["bot"] == "B" else "B" for g in games}
    print("# {} - {} games ({} finished), {} to {}".format(
        args.opponent, len(games), len(fin), games[0]["date"], games[-1]["date"]))
    print("ranks: {} | accounts played: {}".format(
        ", ".join(sorted({g["opp_rank"] or "-" for g in games})),
        ", ".join(sorted({g["account"] for g in games}))))

    print("\n## Record ({}'s wins-losses against the bot)".format(args.opponent))
    by = collections.defaultdict(lambda: [0, 0])
    for g in fin:
        by[(g["ha"], g["komi"])][1 if g["won"] else 0] += 1
    for (ha, komi), (w, l_) in sorted(by.items()):
        print("  HA{} komi {}: {}-{}".format(ha, komi, w, l_))
    margins = collections.defaultdict(list)
    for g in fin:
        if g["score"] is not None:
            margins["opponent won" if not g["won"] else "bot won"].append(abs(g["score"]))
    for k, ms in margins.items():
        print("  margins when the {}: {}".format(k, " ".join("{:g}".format(m) for m in sorted(ms))))
    how = collections.Counter("{} {}".format("bot won" if g["won"] else "opponent won",
                                             g["how"] or "?") for g in fin)
    print("  how games ended: " + ", ".join("{} {}".format(n, k) for k, n in how.most_common()))
    opp_passes = [i + 1 for g in fin for i, (c, p) in enumerate(g["line"])
                  if c == opp_of[g["id"]] and p == "" and i + 1 < len(g["line"])]
    if opp_passes:
        print("  opponent passed at moves: median {}, earliest {}".format(
            int(statistics.median(opp_passes)), min(opp_passes)))

    print("\n## Play style")
    secs = [s for g in games for s in move_seconds(texts.get(g["name"], ""), opp_of[g["id"]])]
    if secs:
        q = statistics.quantiles(secs, n=4) if len(secs) >= 4 else [min(secs), 0, max(secs)]
        print("  time per move: median {:.1f} s, middle half {:.1f}-{:.1f} s, max {:.0f} s "
              "({} moves)".format(statistics.median(secs), q[0], q[2], max(secs), len(secs)))
    positions, same = repeated_replies(games, lambda g: opp_of[g["id"]])
    if positions:
        print("  positions reached in 2+ games (first 60 moves): {}; answered the same way "
              "every time: {} ({:.0%})".format(positions, same, same / positions))

    command = katago_command(args)
    if not command:
        print("\n(no KataGo: move quality skipped)")
        return
    qs = []
    for g in fin:
        q = gr.katago_query(g, range(gr.last_board_turn(g) + 1), args.visits)
        q["id"] = g["id"]
        qs.append(q)
    replies = run(command, qs, "positions")
    loss = {"bot": [], "opponent": []}
    starts, finals = [], []
    for g in fin:
        leads = gr.bot_leads(g, replies, g["id"])
        ha = g["handicap_moves"]
        if ha < len(leads) and leads[ha] is not None and leads[-1] is not None:
            starts.append(-leads[ha])
            finals.append(-leads[-1])
        for i, (c, _p) in enumerate(g["line"][:len(leads) - 1]):
            if i < ha or leads[i] is None or leads[i + 1] is None:
                continue
            d = leads[i] - leads[i + 1]  # the bot's lead lost by this move
            if c == g["bot"]:
                loss["bot"].append(max(d, 0))
            else:
                loss["opponent"].append(max(-d, 0))
    print("\n## Move quality (KataGo, {} visits)".format(args.visits))
    for who, v in loss.items():
        if v:
            print("  {:<9} points lost per move {:.2f} (median {:.2f}), moves losing 5+: "
                  "{} of {}".format(who, statistics.mean(v), statistics.median(v),
                                    sum(x >= 5 for x in v), len(v)))
    if starts:
        print("  {}'s expected lead: at the start (after any handicap) median {:+.1f}, at the "
              "end median {:+.1f}".format(args.opponent, statistics.median(starts),
                                          statistics.median(finals)))


if __name__ == "__main__":
    main()
