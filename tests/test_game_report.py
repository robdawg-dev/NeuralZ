"""tools/game_report.py: reading the bot's SGFs (undo branches, zips, duplicates), the
pass-back and ignored-fight detectors, and the report with and without a fake KataGo."""
import os
import sys
import textwrap
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))

import game_report as gr  # noqa: E402


def sgf(moves, pb="NeuralZ01", pw="rival", result="W+12.5", ha=0, extra="", date="2026-10-06"):
    body = "".join(";{}[{}]".format(c, p) for c, p in moves)
    return "(;GM[1]SZ[19]KM[6.50]RU[Japanese]HA[{}]PB[{}]PW[{}]WR[2k]DT[{}]RE[{}]{}{})".format(
        ha, pb, pw, date, result, extra, body)


def long_line(n, start=0):
    """n alternating moves on distinct points (no captures), Black first."""
    pts = [a + b for a in "abcdefghijklmnopqrs" for b in "abcdefghijklmnopqrs"]
    return [("B" if i % 2 == 0 else "W", pts[start + i]) for i in range(n)]


def test_the_played_line_is_the_first_branch_at_an_undo():
    # KGS keeps the line actually played first; the undone moves are a later variation
    text = "(;GM[1]PB[NeuralZ01]PW[x];B[dd];W[pp](;B[dp];W[pd])(;B[qq]))"
    assert gr.played_line(gr.parse_tree(text)) == [("B", "dd"), ("W", "pp"), ("B", "dp"),
                                                   ("W", "pd")]


def test_read_game_takes_the_bots_side():
    g = gr.read_game(sgf([("B", "dd"), ("W", "")], result="W+12.5"), "a.sgf", "NeuralZ")
    assert (g["bot"], g["opponent"], g["opp_rank"], g["won"], g["how"], g["score"]) == \
        ("B", "rival", "2k", False, "score", -12.5)
    assert gr.read_game(sgf([], pb="someone"), "b.sgf", "NeuralZ") is None
    assert gr.read_game(sgf([], result="B+Resign"), "c.sgf", "NeuralZ")["how"] == "resign"


def test_free_placement_handicap_stones_are_not_the_bots_first_moves():
    stones = [("B", "dd"), ("B", "pp"), ("B", "dp")]
    assert gr.handicap_moves(stones + [("W", "qq"), ("B", "cc")], 3, False) == 3
    assert gr.handicap_moves(stones + [("W", "qq")], 3, True) == 0  # AB setup instead
    assert gr.handicap_moves([("B", "dd"), ("W", "pp")], 0, False) == 0
    g = gr.read_game(sgf(stones + [("W", "qq")], ha=3), "a.sgf", "NeuralZ")
    assert g["handicap_moves"] == 3


def test_games_load_from_folders_and_zips_once_each(tmp_path):
    text = sgf(long_line(60))
    day = tmp_path / "2026" / "10" / "06"
    day.mkdir(parents=True)
    (day / "NeuralZ01-rival.sgf").write_text(text)
    with zipfile.ZipFile(tmp_path / "neuralz01-2026-10.zip", "w") as z:
        z.writestr("2026/10/6/NeuralZ01-rival.sgf", text)  # the same game again
        z.writestr("2026/10/6/NeuralZ01-other.sgf",
                   sgf(long_line(60), pw="other", date="2026-10-01"))
    games = gr.load_games([str(tmp_path)], "NeuralZ")
    assert [g["name"] for g in games] == ["2026/10/6/NeuralZ01-other.sgf",
                                          "2026/10/6/NeuralZ01-rival.sgf"]
    assert [g["opponent"] for g in gr.load_games([str(tmp_path)], "NeuralZ", "2026-10-05")] \
        == ["rival"]


def test_a_pass_back_is_the_bots_pass_right_after_the_opponents_past_move_100():
    line = long_line(101) + [("W", ""), ("B", "")]  # W passes at 102, the bot (B) at 103
    g = gr.read_game(sgf(line), "a.sgf", "NeuralZ")
    assert gr.bot_pass_backs(g) == [103]
    early = gr.read_game(sgf(long_line(41) + [("W", ""), ("B", "")]), "b.sgf", "NeuralZ")
    assert gr.bot_pass_backs(early) == []


def test_an_ignored_fight_is_a_bot_blunder_the_opponent_hands_back():
    g = gr.read_game(sgf(long_line(6)), "a.sgf", "NeuralZ")  # bot is Black: moves 1, 3, 5
    leads = [0, 0, -40, 0, -35, 0, 0]  # moves 2 and 4 (the opponent's) would not count
    assert gr.ignored_fights(g, leads) == []
    leads = [0, -40, 0, -35, 0, 0, 0]
    assert gr.ignored_fights(g, leads) == [1, 3]


def test_report_without_katago_lists_results_opponents_and_repeats(tmp_path):
    for i in range(3):
        # the same moves, different files (identical files would count once)
        (tmp_path / "g{}.sgf".format(i)).write_text(sgf(long_line(60), result="W+{}.5".format(i)))
    (tmp_path / "won.sgf").write_text(sgf(long_line(60, 100), pw="weak", result="B+Resign"))
    lines = []
    gr.report(gr.load_games([str(tmp_path)], "NeuralZ"), write=lines.append)
    text = "\n".join(lines)
    assert "overall 1-3 (25%)" in text
    assert "- rival: 3-0 against the bot" in text
    assert "60 moves" in text  # rival repeated the whole game
    assert "no KataGo" in text


FAKE_KATAGO = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        q = json.loads(line)
        for t in q["analyzeTurns"]:
            # Black ahead by 20 everywhere; ownership half-settled (0.5) at turn 101
            own = [0.5 if t == 101 else 1.0] * 361
            print(json.dumps({"id": q["id"], "turnNumber": t, "ownership": own,
                              "rootInfo": {"scoreLead": 20.0 - t * 0.01}}), flush=True)
''')


def test_report_with_katago_flags_misscored_games_and_open_pass_backs(tmp_path):
    fake = tmp_path / "fake_katago.py"
    fake.write_text(FAKE_KATAGO)
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # the bot (Black) passed back at move 103 on a board KataGo still sees as open, and the
    # game was scored as a loss although KataGo has Black 19 ahead
    (games_dir / "a.sgf").write_text(sgf(long_line(101) + [("W", ""), ("B", "")],
                                         result="W+3.5"))
    lines = []
    gr.report(gr.load_games([str(games_dir)], "NeuralZ"),
              katago=[sys.executable, str(fake)], quick=True, write=lines.append)
    text = "\n".join(lines)
    assert "recorded -3.5, KataGo +19.0 <- bot was ahead" in text
    assert "move 103: 361 contested points" in text
    assert "--quick" in text
