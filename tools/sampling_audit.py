"""What does the opening sampling cost? For the bot's first --sample-moves own moves in a
set of games, the close-call candidates the player would sample among (at least
--sample-ratio as likely as the top move), each weighted by its chance of being picked
(with --temperature) and judged by KataGo against the top move. The answer is the expected
points the sampling gives away per sampled position and per game - for any network and
any settings, independent of what was actually played.

    uv run python tools/sampling_audit.py <game folders> [--model b20c256] \\
        [--sample-ratio 0.5] [--sample-moves 20] [--temperature 1.0] [--max-games 200]

Positions come from the games' actual lines (the bot's side, after any handicap stones;
none with a bot stone in atari - the player is greedy there). Measured for b20c256 at
0.5 / 20: +0.10 points per sampled move, about +0.4 per game (MEASUREMENTS.md). KataGo:
--katago/--katago-model/--katago-config or KATAGO_EXE/KATAGO_MODEL/KATAGO_CONFIG.
"""
import argparse
import collections
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import game_report as gr  # noqa: E402
from katago_util import COLS, GTP_COLS, add_katago_args, katago_command, run  # noqa: E402


def candidates(player, state, probs):
    """[(move, pick probability)] the player would sample among - [] when it plays greedy."""
    move_probs = player._close_candidates(probs)
    if len(move_probs) < 2 or player._has_stone_in_atari(state):
        return []
    picks = player.apply_temperature([p for _m, p in move_probs])
    return list(zip([m for m, _p in move_probs], [float(x) for x in picks]))


def positions(games, policy, ratio, temperature, window):
    """For each bot move in the window: the game, turn, top move and sampling candidates."""
    from AlphaGo import go
    from AlphaGo.ai import ProbabilisticPolicyPlayer as P
    player = P(policy, temperature=temperature, sample_ratio=ratio)
    out = []
    for g in games:
        state = go.GameState(enforce_superko=False)
        if g["setup"]:
            state.place_handicaps([(COLS.index(p[0]), COLS.index(p[1])) for p in g["setup"]])
        bot = go.BLACK if g["bot"] == "B" else go.WHITE
        own = 0
        for i, (c, p) in enumerate(g["line"]):
            color = go.BLACK if c == "B" else go.WHITE
            if color == bot and i >= g["handicap_moves"] and own < window:
                own += 1
                state.set_current_player(color)
                probs = policy.eval_state(state, state.get_legal_moves(include_eyes=False))
                dead = P._failed_ladder_extensions(state)
                probs = [(m, q) for m, q in probs if m not in dead] or probs
                if probs:
                    top = max(probs, key=lambda t: t[1])[0]
                    cands = candidates(player, state, probs)
                    out.append({"game": g, "turn": i, "top": top, "candidates": cands})
            try:
                state.do_move((COLS.index(p[0]), COLS.index(p[1])) if p else None, color)
            except go.IllegalMove:
                break
    return out


def vertex(move):
    return "{}{}".format(GTP_COLS[move[0]], 19 - move[1])


def judge(rows, command, visits):
    """Adds each row's expected excess loss (points, the bot's view) over its top move."""
    queries, keys = [], {}
    for k, r in enumerate(rows):
        g = r["game"]
        base = gr.katago_query(g, [r["turn"]], visits)["moves"][:r["turn"]]
        for move in {r["top"]} | {m for m, _p in r["candidates"]}:
            q = gr.katago_query(g, [r["turn"] + 1], visits)
            q["moves"] = base + [[g["bot"], vertex(move)]]
            q["id"] = "{}|{}".format(k, vertex(move))
            keys[(k, move)] = q["id"]
            queries.append(q)
    replies = run(command, queries, "positions")
    for k, r in enumerate(rows):
        sign = 1 if r["game"]["bot"] == "B" else -1

        def lead(move):
            rep = replies.get((keys[(k, move)], r["turn"] + 1))
            return None if rep is None else sign * rep["rootInfo"]["scoreLead"]
        top = lead(r["top"])
        if not r["candidates"] or top is None:
            r["excess"] = 0.0 if r["candidates"] == [] else None
            r["worst"] = None
            continue
        losses = [(top - lead(m), m, p) for m, p in r["candidates"] if lead(m) is not None]
        r["excess"] = sum(loss * p for loss, _m, p in losses)
        r["worst"] = max(losses) if losses else None
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--model", default="b20c256")
    parser.add_argument("--sample-ratio", type=float, default=0.5)
    parser.add_argument("--sample-moves", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--visits", type=int, default=50)
    parser.add_argument("--bot-prefix", default="NeuralZ")
    add_katago_args(parser, required=True)
    args = parser.parse_args(argv)
    command = katago_command(args, parser)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from play_tests.policy_loading import load_policy
    games = gr.load_games(args.paths, args.bot_prefix)[:args.max_games]
    rows = judge(positions(games, load_policy(args.model), args.sample_ratio,
                           args.temperature, args.sample_moves), command, args.visits)
    rows = [r for r in rows if r["excess"] is not None]
    sampled = [r for r in rows if r["candidates"]]
    print("{} games, {} window positions; sampling possible at {} ({:.0%}), {:.1f} candidates "
          "on average".format(len(games), len(rows), len(sampled),
                              len(sampled) / max(len(rows), 1),
                              statistics.mean(len(r["candidates"]) for r in sampled)
                              if sampled else 0))
    if not sampled:
        return
    ex = [r["excess"] for r in sampled]
    print("expected cost vs always playing the top move: {:+.2f} points per sampled position, "
          "{:+.2f} per game".format(statistics.mean(ex), sum(ex) / len(games)))
    by = collections.defaultdict(list)
    for r in sampled:
        own = sum(1 for c, _p in r["game"]["line"][r["game"]["handicap_moves"]:r["turn"]]
                  if c == r["game"]["bot"]) + 1
        by[(own - 1) // 5].append(r["excess"])
    print("by own move: " + ", ".join("{}-{}: {:+.2f}".format(k * 5 + 1, k * 5 + 5,
                                                              statistics.mean(v))
                                      for k, v in sorted(by.items())))
    worst = sorted((r for r in sampled if r["worst"]), key=lambda r: -r["worst"][0])[:8]
    print("riskiest candidates (points worse than the top move, chance of being picked):")
    for r in worst:
        loss, m, p = r["worst"]
        print("  {} move {}: {} instead of {}: -{:.1f} ({:.0%})".format(
            r["game"]["name"], r["turn"] + 1, vertex(m), vertex(r["top"]), loss, p))


if __name__ == "__main__":
    main()
