# tools

Command-line tools for running and studying the bot. Each script's `--help` (its
docstring) has the details. Run them from the repository root, e.g.
`uv run python tools/game_report.py --help`.

Most use KataGo. It isn't part of this project: pass `--katago`, `--katago-model` and
`--katago-config` (an analysis-engine config, such as KataGo's `analysis_example.cfg`), or
set `KATAGO_EXE`, `KATAGO_MODEL` and `KATAGO_CONFIG` once per shell.

## The bot's games

| Tool | What it answers |
|---|---|
| `game_report.py` | The routine check after a deploy: results by handicap and game type, repeat opponents, scoring and pass-back problems, dead ladders, where losses were lost |
| `game_review.py` | One game, move by move: an annotated SGF (KataGo's lead and preferred move, NeuralZ's policy ranks) to step through in Sabaki |
| `opponent_report.py` | One opponent: record, margins, how games ended, time per move, repeated replies, move quality - skill, too much handicap, a program, or an exploit? |
| `gtp_log.py` | A bot's `--gtp-log`: the games in it, answer times, a summary across all bots' logs, and a KataGo check of a game's end and cleanup |

## Networks

| Tool | What it answers |
|---|---|
| `playoff.py` | Round robin among networks: plays every pair (`play_tests/match_networks.py`), KataGo judges, crosstable and Elo |
| `match_winners.py` | One match's winners, decided by KataGo instead of the built-in count |
| `tactics_bench.py` | How a network (or a guard) does on positions where the bot went badly wrong in real games; the set is `data/tactics.json` |
| `pro_agreement.py` | How often each network's first choice is the move strong players played (a quick comparison, not a strength measure) |
| `sampling_audit.py` | What the opening sampling (`--sample-ratio`, `--sample-moves`) costs a network, in points per game |
| `plot_heatmaps.py` | A network's move probabilities drawn on the board, for chosen positions or a whole game |
| `policy_reference.py` | Whether a model's outputs stay the same across an environment upgrade |

## Records and releases

| Tool | What it answers |
|---|---|
| `sgf_check.py` | Is a game record legal and plausible (illegal moves, cut-off records, move quality)? |
| `package_models.py` | Zips models, checksums and release notes for a GitHub release (`MODEL_RELEASE_PLAN.md`) |
| `fetch_model.py` | Downloads and installs a released model (standard library only) |

`katago_util.py` holds the KataGo plumbing the tools share.
