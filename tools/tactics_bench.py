"""A benchmark of positions where the bot went badly wrong in real games, each with the
moves that would have been right - for measuring a new network or a new guard (like the
ladder guard) on exactly the mistakes that cost games.

    # build the set from the bot's games (KataGo; slow: every bot move is evaluated)
    uv run python tools/tactics_bench.py build <game folders> [--out tools/data/tactics.json]
    # score a network, as the bot plays (greedy, ladder guard on unless --no-ladder-guard)
    uv run python tools/tactics_bench.py run [--model b20c256] [--no-ladder-guard]

build: in the bot's losses (--all-games: all games), KataGo (--fast-visits) finds the
bot's moves that lost --min-loss points or more; each such position is analyzed again
(--visits) and every move within --tolerance points of KataGo's best is accepted. A
position is kept if the bot's move isn't accepted and at most --max-accepted moves are (a
clear answer), at most --per-game per game. Each is labeled:
"ladder" (the bot extended into a dead ladder), "fight" (the opponent's reply handed the
points back - both sides ignoring the same fight), or "blunder".

run: plays each position with ProbabilisticPolicyPlayer exactly as go_client.py does past
the sampling window (greedy), and counts a move right if it is accepted. Also reports how
often the network's own top move is accepted, and the policy rank of KataGo's best move.
KataGo (build only): --katago/--katago-model/--katago-config or KATAGO_* variables.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import game_report as gr  # noqa: E402
from katago_util import COLS, GTP_COLS, add_katago_args, gtp, katago_command, run  # noqa: E402

DEFAULT_SET = os.path.join(HERE, "data", "tactics.json")


def engine_state(setup, moves, to_move):
    """A GameState after the setup stones and moves (GTP vertices), to_move to play."""
    from AlphaGo import go
    state = go.GameState(enforce_superko=False)
    if setup:
        state.place_handicaps([(GTP_COLS.index(v[0]), 19 - int(v[1:])) for _c, v in setup])
    for c, v in moves:
        move = None if v == "pass" else (GTP_COLS.index(v[0]), 19 - int(v[1:]))
        state.do_move(move, go.BLACK if c == "B" else go.WHITE)
    state.set_current_player(go.BLACK if to_move == "B" else go.WHITE)
    return state


def vertex(move):
    return "pass" if move is None else "{}{}".format(GTP_COLS[move[0]], 19 - move[1])


def label(g, i, leads):
    """'ladder', 'fight' or 'blunder' for the bot's move i (0-based) of game g."""
    from AlphaGo.ai import ProbabilisticPolicyPlayer
    c, p = g["line"][i]
    setup = [["B", gtp(q)] for q in g["setup"]]
    state = engine_state(setup, [[cc, gtp(pp)] for cc, pp in g["line"][:i]], c)
    if p and (COLS.index(p[0]), COLS.index(p[1])) in \
            ProbabilisticPolicyPlayer._failed_ladder_extensions(state):
        return "ladder"
    if i + 2 < len(leads) and None not in (leads[i], leads[i + 1], leads[i + 2]) and \
            leads[i + 2] - leads[i + 1] >= 0.7 * (leads[i] - leads[i + 1]):
        return "fight"
    return "blunder"


def build(paths, command, min_loss=20.0, fast_visits=30, visits=400, tolerance=3.0,
          max_accepted=5, per_game=2, bot_prefix="NeuralZ", all_games=False):
    games = [g for g in gr.load_games(paths, bot_prefix) if len(g["line"]) >= 30
             and (all_games or g["won"] is False)]
    qs = []
    for g in games:
        q = gr.katago_query(g, range(gr.last_board_turn(g) + 1), fast_visits)
        q["id"] = g["id"]
        qs.append(q)
    fast = run(command, qs, "scan")
    candidates = []
    for g in games:
        leads = gr.bot_leads(g, fast, g["id"])
        drops = sorted((d for d in gr.drops(g, leads, min_loss)
                        if d[0] - 1 >= g["handicap_moves"]), key=lambda d: -d[2])
        for move_no, _v, lost in drops[:per_game * 2]:
            candidates.append((g, move_no - 1, lost, leads))
    deep = []
    for k, (g, i, _lost, _leads) in enumerate(candidates):
        q = gr.katago_query(g, [i], visits)
        q["id"] = "c{}".format(k)
        deep.append(q)
    replies = run(command, deep, "candidates")
    out, kept = [], collections.Counter()
    for k, (g, i, lost, leads) in enumerate(candidates):
        r = replies.get(("c{}".format(k), i))
        if r is None or kept[g["id"]] >= per_game:
            continue
        sign = 1 if g["bot"] == "B" else -1
        infos = [m for m in r.get("moveInfos", []) if m["visits"] >= 5 and m["move"] != "pass"]
        if not infos:
            continue
        best = max(sign * m["scoreLead"] for m in infos)
        accepted = sorted(m["move"] for m in infos if sign * m["scoreLead"] >= best - tolerance)
        played = gtp(g["line"][i][1])
        if played in accepted or len(accepted) > max_accepted:
            continue
        kept[g["id"]] += 1
        out.append({"id": "{}#{}".format(g["name"], i + 1), "source": g["name"],
                    "move_number": i + 1, "to_move": g["bot"], "komi": g["komi"],
                    "rules": g["rules"], "setup": [["B", gtp(q)] for q in g["setup"]],
                    "moves": [[c, gtp(p)] for c, p in g["line"][:i]],
                    "played": played, "played_lost": round(lost, 1), "accepted": accepted,
                    "kind": label(g, i, leads)})
    return out


def score(positions, policy, ladder_guard=True):
    """Per position: the player's move, whether it is accepted, the network's top move and
    the policy rank of the best accepted move."""
    from AlphaGo.ai import ProbabilisticPolicyPlayer
    player = ProbabilisticPolicyPlayer(policy, sample_moves=0, ladder_guard=ladder_guard)
    rows = []
    for p in positions:
        state = engine_state(p["setup"], p["moves"], p["to_move"])
        move = vertex(player.get_move(state, own_moves=10 ** 6))
        probs = sorted(policy.eval_state(state, state.get_legal_moves(include_eyes=False)),
                       key=lambda t: -t[1])
        order = [vertex(m) for m, _q in probs]
        best_rank = min((order.index(a) + 1 for a in p["accepted"] if a in order), default=None)
        rows.append({"id": p["id"], "kind": p["kind"], "move": move,
                     "right": move in p["accepted"], "top_right": bool(order) and
                     order[0] in p["accepted"], "best_rank": best_rank})
    return rows


def report(rows):
    lines = []
    by = collections.defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for kind in ["all"] + sorted(by):
        rs = rows if kind == "all" else by[kind]
        ranks = sorted(r["best_rank"] for r in rs if r["best_rank"])
        lines.append("{:<8} {:>4} positions | right {:>3} ({:.0%}) | network's top move right "
                     "{:.0%} | right move's policy rank: median {}".format(
                         kind, len(rs), sum(r["right"] for r in rs),
                         sum(r["right"] for r in rs) / len(rs),
                         sum(r["top_right"] for r in rs) / len(rs),
                         ranks[len(ranks) // 2] if ranks else "-"))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("paths", nargs="+")
    b.add_argument("--out", default=DEFAULT_SET)
    b.add_argument("--min-loss", type=float, default=20.0)
    b.add_argument("--fast-visits", type=int, default=30)
    b.add_argument("--visits", type=int, default=400)
    b.add_argument("--tolerance", type=float, default=3.0)
    b.add_argument("--max-accepted", type=int, default=5)
    b.add_argument("--per-game", type=int, default=2)
    b.add_argument("--bot-prefix", default="NeuralZ")
    b.add_argument("--all-games", action="store_true",
                   help="scan wins too (default: losses only - about 5x faster)")
    add_katago_args(b, required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", default=DEFAULT_SET)
    r.add_argument("--model", default="b20c256")
    r.add_argument("--json", help="model JSON instead of --model (needs --weights)")
    r.add_argument("--weights")
    r.add_argument("--no-ladder-guard", dest="ladder_guard", action="store_false")
    args = parser.parse_args(argv)

    if args.command == "build":
        positions = build(args.paths, katago_command(args, b), args.min_loss, args.fast_visits,
                          args.visits, args.tolerance, args.max_accepted, args.per_game,
                          args.bot_prefix, args.all_games)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"positions": positions}, f, indent=1)
        print("{} positions ({}) -> {}".format(len(positions), dict(collections.Counter(
            p["kind"] for p in positions)), args.out))
        return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    with open(args.set) as f:
        positions = json.load(f)["positions"]
    if args.json:
        from AlphaGo.models.nn_util import NeuralNetBase
        policy = NeuralNetBase.load_model(args.json)
        policy.model.load_weights(args.weights)
    else:
        from play_tests.policy_loading import load_policy
        policy = load_policy(args.model)
    print("\n".join(report(score(positions, policy, args.ladder_guard))))


if __name__ == "__main__":
    main()
