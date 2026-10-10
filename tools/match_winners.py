"""Decide a play_tests/match_networks.py match's winners with KataGo. Each game's final
position (before the closing passes) is searched under Tromp-Taylor rules with the match's
komi, and KataGo's score lead decides. The built-in count can't: dead stones the bots never
captured count as area there (their last liberties look like the capturer's own eyes,
which the bots' sensible-move filter won't fill).

Writes katago_results.json next to the match's results.json and prints the tally.

    uv run python tools/match_winners.py play_tests/sgf/<match dir> [--visits 400]

KataGo: --katago/--katago-model/--katago-config or KATAGO_EXE/KATAGO_MODEL/KATAGO_CONFIG
(see tools/katago_util.py).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from katago_util import add_katago_args, gtp, katago_command, run  # noqa: E402


def final_moves(sgf_text):
    """The game's moves as [[color, vertex], ...], closing passes removed."""
    raw = re.findall(r";([BW])\[([a-t]{0,2})\]", sgf_text)
    while raw and raw[-1][1] in ("", "tt"):
        raw.pop()
    return [[c, gtp(p)] for c, p in raw]


def decide(match_dir, command, visits=400):
    """KataGo's winner for every game of the match; writes and returns katago_results."""
    with open(os.path.join(match_dir, "results.json")) as f:
        results = json.load(f)
    komi = results.get("komi", 7.5)
    queries, n_moves = [], {}
    for g in results["games"]:
        with open(os.path.join(match_dir, g["sgf_file"]), encoding="utf-8") as f:
            moves = final_moves(f.read())
        n_moves[g["sgf_file"]] = len(moves)
        queries.append({"id": g["sgf_file"], "moves": moves, "rules": "tromp-taylor",
                        "komi": komi, "boardXSize": 19, "boardYSize": 19,
                        "maxVisits": int(visits), "analyzeTurns": [len(moves)]})
    replies = run(command, queries)
    wins, games, flipped = {}, [], 0
    for g in results["games"]:
        reply = replies.get((g["sgf_file"], n_moves[g["sgf_file"]]))
        if reply is None:
            raise RuntimeError("KataGo gave no answer for {}".format(g["sgf_file"]))
        black = reply["rootInfo"]["scoreLead"]
        winner = g["black_model"] if black > 0 else g["white_model"]
        wins[winner] = wins.get(winner, 0) + 1
        flipped += winner != g["winner_model"]
        games.append({"game_index": g["game_index"], "sgf_file": g["sgf_file"],
                      "black_model": g["black_model"], "white_model": g["white_model"],
                      "katago_black_lead": round(black, 1), "katago_winner_model": winner,
                      "recorded_score": g["score"], "recorded_winner_model": g["winner_model"]})
    for m in (results["model_a"], results["model_b"]):
        wins.setdefault(m, 0)
    out = {"visits": int(visits), "rules": "tromp-taylor", "komi": komi, "wins": wins,
           "flipped_vs_recorded": flipped, "games": games}
    with open(os.path.join(match_dir, "katago_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match_dir", help="a match_networks.py output directory")
    parser.add_argument("--visits", type=int, default=400, help="per game. Default: 400")
    add_katago_args(parser, required=True)
    args = parser.parse_args(argv)
    out = decide(args.match_dir, katago_command(args, parser), args.visits)
    for g in out["games"]:
        print("game {:3d}  B={:<14} W={:<14} KataGo {}{:.1f}  winner {:<14} (recorded {}{})".format(
            g["game_index"], g["black_model"], g["white_model"],
            "B+" if g["katago_black_lead"] > 0 else "W+", abs(g["katago_black_lead"]),
            g["katago_winner_model"], "B+" if g["recorded_score"] > 0 else "W+",
            abs(g["recorded_score"])))
    print("KataGo wins: {}  ({} of {} games differ from the recorded winner)".format(
        out["wins"], out["flipped_vs_recorded"], len(out["games"])))


if __name__ == "__main__":
    main()
