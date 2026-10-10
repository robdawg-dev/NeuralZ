"""Review one game: writes a copy of the SGF with a comment on every move - KataGo's score
lead before and after, the points the move lost, KataGo's preferred move, and (unless
--no-policy) where the played and the preferred moves rank in a NeuralZ network's policy.
Moves that lost --mark points or more get KataGo's move labeled "A" on the board, and are
listed at the end. Open the result in Sabaki (or any SGF viewer) and step through.

    uv run python tools/game_review.py game.sgf [--out game_review.sgf] [--visits 100] \\
        [--model b20c256 | --no-policy] [--mark 5]

The played line is the first branch at each fork (KGS records undos as later branches).
Komi, rules, AB setup stones and free-placement handicap stones come from the record.
KataGo: --katago/--katago-model/--katago-config or KATAGO_EXE/KATAGO_MODEL/KATAGO_CONFIG.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from game_report import handicap_moves, parse_tree, played_line, prop  # noqa: E402
from katago_util import COLS, GTP_COLS, add_katago_args, gtp, katago_command, run  # noqa: E402

HEADER = ("CA", "KM", "RU", "HA", "PB", "PW", "BR", "WR", "DT", "EV", "RO",
          "PC", "RE", "TM", "OT")


def sgf_point(vertex, size=19):
    """A GTP vertex as an SGF point: 'D16' -> 'dd', 'pass' -> ''."""
    if vertex.lower() == "pass":
        return ""
    return COLS[GTP_COLS.index(vertex[0].upper())] + COLS[size - int(vertex[1:])]


def escape(text):
    return text.replace("\\", "\\\\").replace("]", "\\]")


def policy_ranks(line, setup, model):
    """For each position before move i: {vertex: (rank, probability)} from the network,
    over every legal move; None entries where the line couldn't be replayed."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from AlphaGo import go
    from play_tests.policy_loading import load_policy
    policy = load_policy(model)
    state = go.GameState(enforce_superko=False)
    if setup:
        state.place_handicaps([(COLS.index(p[0]), COLS.index(p[1])) for c, p in setup if c == "B"])
    out = []
    for c, p in line:
        color = go.BLACK if c == "B" else go.WHITE
        state.set_current_player(color)
        probs = sorted(policy.eval_state(state, state.get_legal_moves(include_eyes=True)),
                       key=lambda t: -t[1])
        out.append({"{}{}".format(GTP_COLS[m[0]], 19 - m[1]): (k + 1, float(q))
                    for k, (m, q) in enumerate(probs)})
        try:
            state.do_move((COLS.index(p[0]), COLS.index(p[1])) if p else None, color)
        except go.IllegalMove:
            out += [None] * (len(line) - len(out))
            break
    return out


def review(text, command, visits=100, model="b20c256", use_policy=True, mark=5.0):
    """-> (annotated SGF text, [(loss, move number, color, played, best)] for marked moves)."""
    root = parse_tree(text)
    line = played_line(root)
    setup = []
    for key, color in (("AB", "B"), ("AW", "W")):
        m = re.search(key + r"((?:\s*\[[a-s]{2}\])+)", root["props"])
        setup += [(color, p) for p in re.findall(r"\[([a-s]{2})\]", m.group(1))] if m else []
    ru = prop(text, "RU").lower()
    query = {"id": "g", "initialStones": [[c, gtp(p)] for c, p in setup],
             "moves": [[c, gtp(p)] for c, p in line],
             "rules": "japanese" if ru.startswith("jap") else
             "korean" if ru.startswith("kor") else "chinese",
             "komi": float(prop(text, "KM") or 0), "boardXSize": 19, "boardYSize": 19,
             "maxVisits": visits, "analyzeTurns": list(range(len(line) + 1))}
    replies = {t: r for (_id, t), r in run(command, [query], "positions").items()}
    ranks = policy_ranks(line, setup, model) if use_policy else [None] * len(line)
    skip = handicap_moves(line, int(prop(text, "HA") or 0), bool(setup))

    head = "GM[1]FF[4]SZ[19]" + "".join("{}[{}]".format(k, escape(prop(text, k)))
                                        for k in HEADER if prop(text, k))
    if setup:
        for color in "BW":
            pts = [p for c, p in setup if c == color]
            if pts:
                head += "A{}{}".format(color, "".join("[{}]".format(p) for p in pts))
    nodes, marked = [], []
    for i, (c, p) in enumerate(line):
        node = ";{}[{}]".format(c, p)
        before, after = replies.get(i), replies.get(i + 1)
        if i < skip or before is None or after is None:
            nodes.append(node)
            continue
        sign = 1 if c == "B" else -1
        lead_b, lead_a = before["rootInfo"]["scoreLead"], after["rootInfo"]["scoreLead"]
        lost = sign * (lead_b - lead_a)
        infos = sorted(before.get("moveInfos", []), key=lambda m: m.get("order", 0))
        best = infos[0]["move"] if infos else "?"
        played = gtp(p)
        parts = ["Move {} {} {}: Black's lead {:+.1f} -> {:+.1f}, lost {:.1f} for {}".format(
            i + 1, c, played, lead_b, lead_a, max(lost, 0), c)]
        if best != "?" and best.upper() != played.upper():
            parts.append("KataGo preferred {}".format(best))
        r = ranks[i]
        if r:
            pr = r.get(played.upper())
            top = min(r.items(), key=lambda kv: kv[1][0]) if r else None
            txt = "NeuralZ: played move #{} ({:.1%})".format(*pr) if pr else "NeuralZ: -"
            if best != "?" and best.upper() != played.upper() and best.upper() in r:
                txt += ", KataGo's move #{} ({:.1%})".format(*r[best.upper()])
            if top:
                txt += ", its top {} ({:.1%})".format(top[0], top[1][1])
            parts.append(txt)
        node += "C[{}]".format(escape("\n".join(parts)))
        if lost >= mark and best != "?" and best.lower() != "pass":
            node += "LB[{}:A]".format(sgf_point(best))
            marked.append((lost, i + 1, c, played, best))
        nodes.append(node)
    return "(;" + head + "".join("\n" + n for n in nodes) + ")\n", marked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sgf")
    parser.add_argument("--out", help="default: <sgf name>_review.sgf beside it")
    parser.add_argument("--visits", type=int, default=100, help="per position. Default: 100")
    parser.add_argument("--model", default="b20c256",
                        help="network for the policy ranks (play_tests/policy_loading.py)")
    parser.add_argument("--no-policy", action="store_true", help="skip the policy ranks")
    parser.add_argument("--mark", type=float, default=5.0,
                        help="label KataGo's move where a move lost this much. Default: 5")
    add_katago_args(parser, required=True)
    args = parser.parse_args(argv)
    with open(args.sgf, encoding="utf-8", errors="replace") as f:
        text = f.read()
    out_text, marked = review(text, katago_command(args, parser), args.visits, args.model,
                              not args.no_policy, args.mark)
    out = args.out or os.path.splitext(args.sgf)[0] + "_review.sgf"
    with open(out, "w", encoding="utf-8") as f:
        f.write(out_text)
    print("wrote {}".format(out))
    for color in "BW":
        worst = sorted((m for m in marked if m[2] == color), reverse=True)[:8]
        if worst:
            print("{} ({}): ".format(prop(text, "P" + color) or color, color) + ", ".join(
                "move {} {} -{:.0f} (KataGo {})".format(n, pl, loss, best)
                for loss, n, _c, pl, best in worst))


if __name__ == "__main__":
    main()
