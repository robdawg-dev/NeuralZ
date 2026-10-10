"""tools/: match_winners (KataGo decides a match), playoff (crosstable and Elo), gtp_log
(reading go_client --gtp-log files) and sgf_check (record legality), with a fake KataGo
where one is needed."""
import json
import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))

import gtp_log  # noqa: E402
import katago_util  # noqa: E402
import match_winners  # noqa: E402
import playoff  # noqa: E402
import sgf_check  # noqa: E402

# Black leads by +5 when the query's last move is Black's, -5 when it's White's
FAKE_KATAGO = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        q = json.loads(line)
        for t in q["analyzeTurns"]:
            last = q["moves"][t - 1][0] if t else "W"
            print(json.dumps({"id": q["id"], "turnNumber": t,
                              "rootInfo": {"scoreLead": 5.0 if last == "B" else -5.0}}),
                  flush=True)
''')


@pytest.fixture
def fake_katago(tmp_path):
    path = tmp_path / "fake_katago.py"
    path.write_text(FAKE_KATAGO)
    return [sys.executable, str(path)]


def test_gtp_vertices_from_sgf_points():
    assert katago_util.gtp("dd") == "D16"
    assert katago_util.gtp("jj") == "K10"  # GTP skips I
    assert katago_util.gtp("") == katago_util.gtp("tt") == "pass"


# --- match_winners --------------------------------------------------------------------------

def _match(tmp_path, games):
    """A match directory: games = [(black, white, sgf moves, recorded winner)]."""
    d = tmp_path / "match"
    d.mkdir()
    rows = []
    for i, (b, w, moves, recorded) in enumerate(games, 1):
        name = "game{:02d}.sgf".format(i)
        body = "".join(";{}[{}]".format(c, p) for c, p in moves)
        (d / name).write_text("(;GM[1]SZ[19]" + body + ")")
        rows.append({"game_index": i, "sgf_file": name, "black_model": b, "white_model": w,
                     "score": 1.0 if recorded == b else -1.0, "winner_model": recorded})
    (d / "results.json").write_text(json.dumps({"model_a": "A", "model_b": "B", "komi": 7.5,
                                                "games": rows}))
    return str(d)


def test_closing_passes_are_not_part_of_the_judged_position():
    assert match_winners.final_moves("(;B[dd];W[pp];B[];W[])") == [["B", "D16"], ["W", "Q4"]]


def test_katago_decides_and_flags_disagreements(tmp_path, fake_katago):
    # game 1 ends on Black's move (Black +5): A wins, as recorded; game 2 ends on White's
    # move (Black -5): A as White wins, though B was recorded
    d = _match(tmp_path, [("A", "B", [("B", "dd"), ("W", "pp"), ("B", "dp"), ("W", "")], "A"),
                          ("B", "A", [("B", "dd"), ("W", "pp")], "B")])
    out = match_winners.decide(d, fake_katago, visits=10)
    assert out["wins"] == {"A": 2, "B": 0}
    assert out["flipped_vs_recorded"] == 1
    assert json.load(open(os.path.join(d, "katago_results.json")))["wins"] == out["wins"]


# --- playoff --------------------------------------------------------------------------------

def _judged(pairs):
    """katago_results-like dicts: pairs = [(winner, loser, games)]."""
    games = []
    for w, loser, n in pairs:
        games += [{"black_model": w, "white_model": loser, "katago_winner_model": w}] * n
    return [{"games": games}]


def test_crosstable_and_elo_order():
    results = _judged([("A", "B", 25), ("B", "A", 5), ("A", "C", 20), ("C", "A", 10),
                       ("B", "C", 18), ("C", "B", 12)])
    lines = playoff.table(results)
    assert lines[2].split()[:4] == ["A", "-", "25-5", "20-10"]  # sorted by Elo, A first
    ratings = playoff.elo(playoff.tally(results), ["A", "B", "C"])
    assert ratings["A"] > ratings["B"] > ratings["C"] == 0


def test_a_sweep_still_gets_a_finite_rating():
    ratings = playoff.elo(playoff.tally(_judged([("A", "B", 30)])), ["A", "B"])
    assert 0 < ratings["A"] < 1000 and ratings["B"] == 0


# --- gtp_log --------------------------------------------------------------------------------

LOG = [
    "2026-10-08T20:30:00.000 pid=7 > 'boardsize 19'", "2026-10-08T20:30:00.001 pid=7 < '='",
    "2026-10-08T20:30:00.002 pid=7 > 'kgs-rules japanese'", "2026-10-08T20:30:00.003 pid=7 < '='",
    "2026-10-08T20:30:00.004 pid=7 > 'clear_board'", "2026-10-08T20:30:00.005 pid=7 < '='",
    "2026-10-08T20:30:00.006 pid=7 > 'komi 0.5'", "2026-10-08T20:30:00.007 pid=7 < '='",
    "2026-10-08T20:30:01.000 pid=7 > 'set_free_handicap d4 q16'",
    "2026-10-08T20:30:01.001 pid=7 < '='",
    "2026-10-08T20:30:02.000 pid=7 > 'genmove w'", "2026-10-08T20:30:02.250 pid=7 < '= Q4'",
    "2026-10-08T20:30:03.000 pid=7 > 'play b c3'", "2026-10-08T20:30:03.001 pid=7 < '='",
    "2026-10-08T20:30:04.000 pid=7 > 'undo'", "2026-10-08T20:30:04.001 pid=7 < '='",
    "2026-10-08T20:30:05.000 pid=7 > 'play b d16'", "2026-10-08T20:30:05.001 pid=7 < '='",
    "2026-10-08T20:30:06.000 pid=7 > 'final_status_list dead'",
    "2026-10-08T20:30:06.100 pid=7 < '= Q4 D16'",
    "2026-10-08T20:30:07.000 pid=7 > 'kgs-game_over'", "2026-10-08T20:30:07.001 pid=7 < '='",
    "2026-10-08T20:31:00.000 pid=7 > 'clear_board'", "2026-10-08T20:31:00.001 pid=7 < '='",
    "2026-10-08T20:31:01.000 pid=7 > 'genmove b'",
]


def test_games_are_rebuilt_from_the_log():
    gs = gtp_log.games(gtp_log.exchanges(LOG))
    assert len(gs) == 2
    g = gs[0]
    assert (g["komi"], g["rules"], g["over"]) == (0.5, "japanese", True)
    # handicap stones, the bot's move, and the undone C3 replaced by D16
    assert g["moves"] == [["B", "D4"], ["B", "Q16"], ["W", "Q4"], ["B", "D16"]]
    assert g["final_status"] == [{"after_move": 4, "dead": ["Q4", "D16"]}]
    assert g["bot"] == [{"command": "genmove", "move_number": 3, "vertex": "Q4", "ms": 250.0}]
    assert gs[1]["over"] is False and gs[1]["bot"] == []  # the log ends mid-command


def test_board_applies_captures():
    stones = gtp_log.board([["B", "A2"], ["W", "A1"], ["B", "B1"]])  # White's A1 captured
    assert stones == {(0, 17): "B", (1, 18): "B"}


# --- sgf_check ------------------------------------------------------------------------------

def test_legal_record_passes():
    r = sgf_check.check("(;GM[1]SZ[19]KM[6.5];B[dd];W[pp];B[dp];W[];B[pd])")
    assert r["problems"] == [] and r["out_of_turn"] == 0 and not r["cut_off"]
    assert len(r["legal"]) == 5


def test_occupied_point_and_where_legal_play_ends():
    r = sgf_check.check("(;GM[1];B[dd];W[pp];B[dd];W[qq])")
    assert r["problems"] == ["move 3 BD16 occupied point"]
    assert r["legal"] == [("B", "dd"), ("W", "pp")]


def test_suicide_is_illegal():
    # Black on B19 and A18: White at A19 would have no liberty and capture nothing
    r = sgf_check.check("(;GM[1];B[ba];W[pp];B[ab];W[aa])")
    assert r["problems"] == ["move 4 WA19 suicide"]


# A ko in the top-left: Black B19 A18 B17, White C19 D18 C17; White B18 has one liberty
# (C18), Black takes it at C18, and White may not retake at B18 at once.
KO = "(;GM[1];B[ba];W[ca];B[ab];W[db];B[bc];W[cc];B[pp];W[bb];B[cb]"


def test_retaking_a_ko_at_once_is_illegal():
    r = sgf_check.check(KO + ";W[bb])")
    assert r["problems"] == ["move 10 WB18 ko violation"]


def test_retaking_a_ko_after_a_move_elsewhere_is_legal():
    r = sgf_check.check(KO + ";W[pq];B[qq];W[bb])")
    assert r["problems"] == []


def test_cut_off_record_is_noticed():
    assert sgf_check.check("(;GM[1];B[dd];W[pp];B[d")["cut_off"]


# --- plot_heatmaps --------------------------------------------------------------------------

def test_heatmap_labels_stay_short():
    import plot_heatmaps
    assert [plot_heatmaps.label(x) for x in (40.44, 0.553, 0.0003)] == ["40.4", "0.55", "3e-4"]
    assert plot_heatmaps.vertex((14, 9)) == "P10" and plot_heatmaps.vertex(None) == "pass"


@pytest.mark.parametrize("all_points", [False, True])
def test_heatmap_is_drawn(tmp_path, all_points):
    pytest.importorskip("matplotlib")
    import plot_heatmaps
    from AlphaGo import go
    state = go.GameState()
    state.do_move((3, 3))
    probs = [((15, 15), 0.6), ((15, 3), 0.39), ((9, 9), 0.0099), ((0, 0), 1e-7)]
    path = tmp_path / "h.png"
    plot_heatmaps.draw(state, probs, str(path), cmap="plasma", all_points=all_points)
    assert path.stat().st_size > 10000


# --- game_review ----------------------------------------------------------------------------

REVIEW_KATAGO = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        q = json.loads(line)
        for t in q["analyzeTurns"]:
            # Black's lead falls from 0 to -8 with move 3 (Black's second move)
            lead = 0.0 if t < 3 else -8.0
            print(json.dumps({"id": q["id"], "turnNumber": t, "rootInfo": {"scoreLead": lead},
                              "moveInfos": [{"move": "K10", "order": 0}]}), flush=True)
''')


def test_review_annotates_moves_and_marks_big_losses(tmp_path):
    import game_review
    fake = tmp_path / "fake.py"
    fake.write_text(REVIEW_KATAGO)
    text = "(;GM[1]SZ[19]KM[6.5]PB[NeuralZ01]PW[x];B[dd];W[pp];B[dp];W[pd])"
    out, marked = game_review.review(text, [sys.executable, str(fake)], visits=5,
                                     use_policy=False, mark=5.0)
    assert "Move 3 B D4: Black's lead +0.0 -> -8.0, lost 8.0 for B" in out
    assert "KataGo preferred K10" in out and "LB[jj:A]" in out
    assert marked == [(8.0, 3, "B", "D4", "K10")]
    assert sgf_check.check(out)["problems"] == []  # the review is itself a valid record


# --- opponent_report ------------------------------------------------------------------------

def test_time_per_move_from_the_clock():
    import opponent_report
    text = "(;B[dd]BL[600];W[pp]WL[600];B[dp]BL[598.5];W[pd]WL[590];B[pq]BL[597])"
    assert opponent_report.move_seconds(text, "B") == [1.5, 1.5]
    assert opponent_report.move_seconds(text, "W") == [10.0]


def test_repeated_replies_count_identical_positions():
    import opponent_report
    g1 = {"line": [("B", "dd"), ("W", "pp"), ("B", "dp"), ("W", "pd")]}
    g2 = {"line": [("B", "dd"), ("W", "pp"), ("B", "dp"), ("W", "cc")]}
    positions, same = opponent_report.repeated_replies([g1, g2], lambda g: "W")
    assert (positions, same) == (2, 1)  # same reply after B dd; different after B dp


# --- gtp_log summary ------------------------------------------------------------------------

def test_log_summary_counts_games_errors_and_speed(tmp_path):
    log = tmp_path / "bot.log"
    log.write_text("\n".join(LOG + ["2026-10-08T20:31:01.100 pid=7 < '= D4'",
                                    "2026-10-08T20:31:02.000 pid=7 > 'play b Z9'",
                                    "2026-10-08T20:31:02.001 pid=7 < '? illegal move'"]) + "\n")
    lines = gtp_log.summary([str(log)])
    assert lines[1].split()[:5] == ["bot.log", "2", "1", "0", "1"]
    assert any("GTP errors from {'play': 1}" in line for line in lines)


# --- sampling_audit -------------------------------------------------------------------------

def test_sampling_candidates_follow_the_player():
    import sampling_audit
    from AlphaGo import go
    from AlphaGo.ai import ProbabilisticPolicyPlayer
    player = ProbabilisticPolicyPlayer(None, sample_ratio=0.5)
    probs = [((3, 3), 0.4), ((15, 15), 0.3), ((9, 9), 0.1)]
    cands = sampling_audit.candidates(player, go.GameState(), probs)
    assert [m for m, _p in cands] == [(3, 3), (15, 15)]
    assert abs(sum(p for _m, p in cands) - 1) < 1e-9
    assert sampling_audit.candidates(player, go.GameState(), [((3, 3), 0.9), ((4, 4), 0.1)]) == []


# --- tactics_bench --------------------------------------------------------------------------

def test_tactics_scoring_counts_accepted_moves():
    import tactics_bench

    class Policy(object):  # always prefers Q16, then D4
        def eval_state(self, state, moves=None):
            return [((15, 3), 0.7), ((3, 15), 0.2), ((9, 9), 0.1)]
    positions = [{"id": "a", "kind": "blunder", "setup": [], "moves": [["B", "D16"]],
                  "to_move": "W", "accepted": ["Q16"]},
                 {"id": "b", "kind": "fight", "setup": [], "moves": [["B", "D16"]],
                  "to_move": "W", "accepted": ["D4", "K10"]}]
    rows = tactics_bench.score(positions, Policy(), ladder_guard=False)
    assert [(r["move"], r["right"], r["best_rank"]) for r in rows] == \
        [("Q16", True, 1), ("Q16", False, 2)]
    assert tactics_bench.report(rows)[0].split()[:5] == ["all", "2", "positions", "|", "right"]


# --- package_models / fetch_model -----------------------------------------------------------

def test_models_package_and_fetch_round_trip(tmp_path):
    import fetch_model
    import package_models
    models = tmp_path / "models" / "2016net"
    models.mkdir(parents=True)
    (models / "model.json").write_text("{}")
    (models / "model.weights.h5").write_bytes(b"\0" * 1000)
    release = tmp_path / "release"
    package_models.package(["2016net"], str(release), models_dir=str(tmp_path / "models"))
    root = tmp_path / "checkout"
    folder = fetch_model.fetch("2016net", release.resolve().as_uri(), str(root))
    assert sorted(os.listdir(folder)) == ["model.json", "model.weights.h5"]
    (release / "2016net.zip").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch_model.fetch("2016net", release.resolve().as_uri(), str(tmp_path / "other"))


def test_a_katago_that_dies_is_reported_with_its_output(tmp_path):
    crash = tmp_path / "crash.py"
    crash.write_text("import sys\nsys.stderr.write('no CUDA device found\n')\nsys.exit(2)\n")
    with pytest.raises(RuntimeError, match="no CUDA device found"):
        katago_util.run([sys.executable, str(crash)],
                        [{"id": "a", "moves": [], "analyzeTurns": [0]}])


def test_katago_sees_the_gpu_the_tools_hide_from_tensorflow(tmp_path, monkeypatch):
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent('''
        import json, os, sys
        for line in sys.stdin:
            q = json.loads(line)
            print(json.dumps({"id": q["id"], "turnNumber": 0,
                              "cuda": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}),
                  flush=True)
    '''))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    out = katago_util.run([sys.executable, str(probe)],
                          [{"id": "a", "moves": [], "analyzeTurns": [0]}])
    assert out[("a", 0)]["cuda"] == "unset"
