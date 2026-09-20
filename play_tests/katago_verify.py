"""Cross-check a match_networks.py run's naive winner calls against KataGo, CPU-only.

GameState.get_score()/get_winner_color() does pure Tromp-Taylor-style area
scoring with no life-and-death resolution - every game in a match run hits
the move cap rather than ending via two passes, so dead groups are never
actually removed before scoring. KataGo's own search-based scoring (run per
game via `katago evalsgf -print-score-now`, which reports a real `RE[B+X.X]`/
`RE[W+X.X]` result after searching the position) accounts for life and death
properly, and is the trustworthy reference this project doesn't have to
implement itself. This script re-scores every game in a match_networks.py
output directory with KataGo, compares the winner to what results.json
recorded, and writes a katago_verification.json alongside it.

Requires a local KataGo install (binary + a .bin.gz model file + a gtp-style
config - see --katago-exe/--model/--config defaults, which point at what's
on this machine; override them if KataGo lives somewhere else). Uses the
OpenCL (GPU) backend by default - unlike the models under test, KataGo isn't
being evaluated for commodity-CPU feasibility here, it's just the reference
oracle, so there's no reason to hold it to that constraint.

Run natively via uv, from the repo root:
    uv run python play_tests/katago_verify.py play_tests/sgf/match_simplecnn_vs_resnet_20260908_193218
"""
import argparse
import json
import os
import re
import subprocess
import time

DEFAULT_KATAGO_EXE = r"C:\Users\winst\.katrain\katago-v1.11.0-opencl-windows-x64.exe"
DEFAULT_CONFIG = r"C:\Users\winst\Documents\KataGo\cpp\configs\gtp_example.cfg"
DEFAULT_MODEL = r"C:\Users\winst\.katrain\kata1-b40c256-s11101799168-d2715431527.bin.gz"

FINAL_RE_PATTERN = re.compile(r"Final:\s*RE\[([BW])\+([\d.]+)\]")


def score_one_sgf(katago_exe, config, model, sgf_path, move_num, visits, timeout):
    cmd = [
        katago_exe, "evalsgf",
        "-config", config,
        "-model", model,
        "-m", str(move_num),
        "-v", str(visits),
        "-print-score-now",
        sgf_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    output = result.stdout + result.stderr
    match = FINAL_RE_PATTERN.search(output)
    if match is None:
        return None, output
    winner_color = "BLACK" if match.group(1) == "B" else "WHITE"
    margin = float(match.group(2))
    return {"winner_color": winner_color, "margin": margin}, output


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="A match_networks.py output directory "
                                        "(containing results.json and the game SGFs)")
    parser.add_argument("--katago-exe", default=DEFAULT_KATAGO_EXE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--visits", type=int, default=300,
                        help="KataGo search visits per position. Default: 300")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-game subprocess timeout in seconds. Default: 300")
    args = parser.parse_args()

    results_path = os.path.join(args.run_dir, "results.json")
    with open(results_path) as f:
        results = json.load(f)

    print("Verifying {} games from {} against KataGo ({} visits)...".format(
        len(results["games"]), args.run_dir, args.visits))

    verified_games = []
    agree_count = 0
    for game in results["games"]:
        sgf_path = os.path.join(args.run_dir, game["sgf_file"])
        print("  {} (naive winner: {})...".format(game["sgf_file"], game["winner_model"]), end=" ")
        start = time.perf_counter()
        katago_result, raw_output = score_one_sgf(
            args.katago_exe, args.config, args.model, sgf_path,
            game["moves_played"], args.visits, args.timeout)
        elapsed = time.perf_counter() - start

        entry = {
            "game_index": game["game_index"],
            "sgf_file": game["sgf_file"],
            "naive_winner_color": game["winner_color"],
            "naive_winner_model": game["winner_model"],
            "naive_score": game["score"],
        }
        if katago_result is None:
            entry["katago_result"] = None
            entry["agrees"] = None
            print("FAILED TO PARSE (see raw output below)")
            print(raw_output[-2000:])
        else:
            black_model, white_model = game["black_model"], game["white_model"]
            katago_winner_model = black_model if katago_result["winner_color"] == "BLACK" \
                else white_model
            agrees = katago_winner_model == game["winner_model"]
            agree_count += int(agrees)
            entry["katago_result"] = {
                "winner_color": katago_result["winner_color"],
                "winner_model": katago_winner_model,
                "margin": katago_result["margin"],
            }
            entry["agrees"] = agrees
            print("KataGo: {}+{:.1f} ({}) [{}] ({:.1f}s)".format(
                katago_result["winner_color"][0], katago_result["margin"], katago_winner_model,
                "AGREE" if agrees else "DISAGREE", elapsed))
        verified_games.append(entry)

    scored = [g for g in verified_games if g["agrees"] is not None]
    verification = {
        "run_dir": args.run_dir,
        "katago_exe": args.katago_exe,
        "model": args.model,
        "visits": args.visits,
        "games_scored": len(scored),
        "games_failed": len(verified_games) - len(scored),
        "agree_count": agree_count,
        "disagree_count": len(scored) - agree_count,
        "games": verified_games,
    }
    out_path = os.path.join(args.run_dir, "katago_verification.json")
    with open(out_path, "w") as f:
        json.dump(verification, f, indent=2)

    print("\n{}/{} scored games agree with the naive winner ({} failed to parse).".format(
        agree_count, len(scored), len(verified_games) - len(scored)))
    print("Saved to {}".format(out_path))


if __name__ == "__main__":
    main()
