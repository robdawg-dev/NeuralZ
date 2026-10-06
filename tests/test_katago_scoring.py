"""End-of-game judging: interface/katago_scorer.py against a fake KataGo analysis engine,
go_server's /final_status and /cleanup_move, and the GTP commands that use them
(final_status_list, final_score, kgs-rules, kgs-genmove_cleanup) with their GNU Go fallback."""
import sys
import textwrap
import threading
from http.server import ThreadingHTTPServer

import pytest

import go_client
import go_server
from interface.gtp_wrapper import ExtendedGtpEngine, GTPGameConnector
from interface.katago_scorer import KataGoScorer, vertex_index
from tests.test_go_server_client import model_files  # noqa: F401 - pytest fixture

# A stand-in for `katago analysis`: Black owns every point (so every White stone is dead),
# komi 99 -> an error reply, komi 77 -> no reply at all; with avoidMoves, two move infos
# with the pass first (which must be skipped). Ownership is +1 everywhere (a settled board),
# or 0.5 everywhere at komi 55 (every point contested).
FAKE_KATAGO = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        q = json.loads(line)
        if q["komi"] == 77:
            continue
        if q["komi"] == 99:
            print(json.dumps({"id": q["id"], "error": "boom"}), flush=True)
            continue
        n = q["boardXSize"] * q["boardYSize"]
        reply = {"id": q["id"], "ownership": [0.5 if q["komi"] == 55 else 1.0] * n,
                 "rootInfo": {"scoreLead": 12.5}, "query": q}
        if "avoidMoves" in q:
            reply["moveInfos"] = [{"move": "pass", "order": 0}, {"move": "A1", "order": 1}]
        print(json.dumps(reply), flush=True)
''')


@pytest.fixture(scope="module")
def fake_katago(tmp_path_factory):
    path = tmp_path_factory.mktemp("katago") / "fake_katago.py"
    path.write_text(FAKE_KATAGO)
    return [sys.executable, str(path)]


@pytest.fixture
def scorer(fake_katago):
    s = KataGoScorer(command=fake_katago, timeout=2.0)
    yield s
    s.close()


STONES = [["B", "D4"], ["W", "Q16"], ["W", "Q17"], ["B", "C3"]]


# --- the scorer ----------------------------------------------------------------------------

def test_vertex_index_reads_ownership_from_the_top_left():
    assert vertex_index("A19", 19) == 0
    assert vertex_index("T19", 19) == 18      # GTP skips I
    assert vertex_index("A1", 19) == 18 * 19
    assert vertex_index("j10", 19) == 9 * 19 + 8


def test_dead_stones_are_those_owned_by_the_other_color(scorer):
    verdict = scorer.final_status(STONES, "B", 0.5, "chinese")
    assert sorted(verdict["dead"]) == ["Q16", "Q17"]
    assert verdict["score_lead"] == 12.5


def test_contested_counts_points_whose_owner_is_open(scorer):
    assert scorer.final_status(STONES, "B", 0.5, "chinese")["contested"] == 0
    assert scorer.final_status(STONES, "B", 55, "chinese")["contested"] == 361


def test_cleanup_captures_before_passing(scorer):
    # Black to move, White's stones dead: KataGo's best move other than a pass
    assert scorer.cleanup_move(STONES, "B", 0.5, "chinese") == "A1"
    # White to move: none of Black's stones are dead, so White may pass
    assert scorer.cleanup_move(STONES, "W", 0.5, "chinese") == "pass"


def test_katago_errors_and_silence_become_exceptions(scorer):
    with pytest.raises(ValueError, match="boom"):
        scorer.final_status(STONES, "B", 99, "chinese")
    with pytest.raises(RuntimeError, match="did not answer"):
        scorer.final_status(STONES, "B", 77, "chinese")
    # and the engine still answers afterwards
    assert scorer.final_status(STONES, "B", 0.5, "chinese")["dead"]


def test_concurrent_queries_each_get_their_own_answer(scorer):
    out = {}

    def ask(i):
        out[i] = scorer.final_status([["W", "A{}".format(i + 1)]], "B", 0.5, "chinese")

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(out[i]["dead"] == ["A{}".format(i + 1)] for i in range(8))


# --- go_server endpoints ------------------------------------------------------------------

@pytest.fixture(scope="module")
def judging_server(model_files, fake_katago):  # noqa: F811
    policy = go_server.BatchingPolicy(*model_files)
    katago = KataGoScorer(command=fake_katago, timeout=2.0)
    servers = {}
    for name, sc in (("with", katago), ("without", None)):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), go_server.make_handler(policy, sc))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers[name] = (httpd, "http://127.0.0.1:{}".format(httpd.server_address[1]))
    yield {name: url for name, (_h, url) in servers.items()}
    for httpd, _url in servers.values():
        httpd.shutdown()
        httpd.server_close()
    katago.close()


def test_client_judges_through_the_server(judging_server):
    remote = go_client.RemotePolicy(judging_server["with"])
    assert remote.judges_games
    assert sorted(remote.final_status(STONES, "B", 0.5, "chinese")["dead"]) == ["Q16", "Q17"]
    assert remote.cleanup_move(STONES, "B", 0.5, "chinese") == "A1"


def test_server_without_katago_says_so(judging_server):
    remote = go_client.RemotePolicy(judging_server["without"])
    assert not remote.judges_games
    with pytest.raises(RuntimeError, match="503"):
        remote.final_status(STONES, "B", 0.5, "chinese")


# --- GTP commands -------------------------------------------------------------------------

class FixedPlayer(object):
    """Plays (1, 1) in engine coordinates (GTP B18); records pass_when_offered per call."""

    def __init__(self, move=(1, 1)):
        self.pass_when_offered = True
        self.seen = []
        self.move = move

    def get_move(self, state, own_moves=None):
        self.seen.append(self.pass_when_offered)
        return self.move


class FakeScorer(object):
    def __init__(self, dead=("Q16",), cleanup="D4", fail=False, contested=0):
        self.dead, self.cleanup, self.fail = list(dead), cleanup, fail
        self.contested = contested
        self.asked = []

    def final_status(self, stones, to_move, komi, rules):
        self.asked.append((stones, to_move, komi, rules))
        if self.fail:
            raise RuntimeError("katago down")
        return {"dead": self.dead, "score_lead": -6.5, "contested": self.contested}

    def cleanup_move(self, stones, to_move, komi, rules):
        if self.fail:
            raise RuntimeError("katago down")
        return self.cleanup


def engine(scorer=None, cleanup=False, player=None):
    game = GTPGameConnector(player or FixedPlayer())
    e = ExtendedGtpEngine(game, scorer=scorer, cleanup=cleanup)
    for cmd in ("komi 0.5", "play b D4", "play w Q16", "play b C3"):
        assert e.send(cmd).startswith("=")
    return e


def test_cleanup_is_only_advertised_when_enabled_with_a_scorer():
    assert "kgs-genmove_cleanup" not in engine(FakeScorer()).send("list_commands")
    assert "kgs-genmove_cleanup" not in engine(None, cleanup=True).send("list_commands")
    on = engine(FakeScorer(), cleanup=True)
    assert "kgs-genmove_cleanup" in on.send("list_commands")
    assert on.send("known_command kgs-genmove_cleanup").startswith("= true")
    assert "kgs-rules" in on.send("list_commands")


def test_final_status_comes_from_the_scorer_with_the_game_position():
    sc = FakeScorer(dead=["Q16"])
    e = engine(sc)
    assert e.send("kgs-rules japanese").startswith("=")
    assert e.send("final_status_list dead").strip() == "= Q16"
    alive = e.send("final_status_list alive").strip()[2:].split()
    assert sorted(alive) == ["C3", "D4"]
    stones, to_move, komi, rules = sc.asked[0]
    assert sorted(v for _c, v in stones) == ["C3", "D4", "Q16"] and ["W", "Q16"] in stones
    assert (to_move, komi, rules) == ("W", 0.5, "japanese")
    assert e.send("final_score").strip() == "= W+6.5"


def test_gnu_go_is_the_fallback_and_silence_is_an_error(monkeypatch):
    e = engine(FakeScorer(fail=True))
    monkeypatch.setattr(ExtendedGtpEngine, "call_gnugo", lambda self, f, c: "Q16 Q17")
    assert e.send("final_status_list dead").strip() == "= Q16 Q17"
    monkeypatch.setattr(ExtendedGtpEngine, "call_gnugo", lambda self, f, c: None)
    assert e.send("final_status_list dead").startswith("? cannot judge")
    assert engine(None).send("final_score").startswith("? cannot score")


def test_gnu_go_no_dead_stones_is_an_empty_answer(monkeypatch):
    monkeypatch.setattr(ExtendedGtpEngine, "call_gnugo", lambda self, f, c: "")
    assert engine(None).send("final_status_list dead").strip() == "="


def test_cleanup_plays_katagos_move():
    e = engine(FakeScorer(cleanup="E5"), cleanup=True)
    assert e.send("kgs-genmove_cleanup w").strip() == "= E5"
    assert ["W", "E5"] in e._game.position()["stones"]


def test_cleanup_falls_back_to_the_bots_move_without_passing_on_offer():
    player = FixedPlayer()
    # KataGo proposes an occupied point: unusable, so the bot's own move - asked with
    # pass_when_offered off, and the player's setting restored afterwards
    e = engine(FakeScorer(cleanup="D4"), cleanup=True, player=player)
    assert e.send("kgs-genmove_cleanup w").strip() == "= B18"
    assert player.seen == [False] and player.pass_when_offered is True
    e2 = engine(FakeScorer(fail=True), cleanup=True, player=FixedPlayer())
    e2.send("play b A1")
    assert e2.send("kgs-genmove_cleanup w").strip() == "= B18"


# --- passing back only on a finished board ---------------------------------------------

COLUMNS = "ABCDEFGHJKLMNOPQRST"


def _long_game(e, opponent_passes=True):
    """101+ moves without contact (Black on rows 19-17, White on rows 9-7), ending with
    Black's pass - where the player's own rule would pass back."""
    for i in range(50):
        col = COLUMNS[i % 19]
        assert e.send("play b {}{}".format(col, 19 - i // 19)).startswith("=")
        assert e.send("play w {}{}".format(col, 9 - i // 19)).startswith("=")
    assert e.send("play b {}".format("pass" if opponent_passes else "T1")).startswith("=")


@pytest.mark.parametrize("contested,seen", [(150, [False]), (11, [False]), (10, [True]),
                                            (0, [True])])
def test_bot_passes_back_only_on_a_settled_board(contested, seen):
    player = FixedPlayer(move=(10, 9))  # K10: empty in _long_game
    sc = FakeScorer(contested=contested)
    e = ExtendedGtpEngine(GTPGameConnector(player), scorer=sc)
    _long_game(e)
    assert e.send("genmove w").startswith("=")
    assert player.seen == seen and player.pass_when_offered is True
    assert sc.asked[-1][1] == "W"


def test_pass_check_only_after_an_opponents_pass_and_falls_back_without_katago():
    player = FixedPlayer(move=(10, 9))
    sc = FakeScorer(contested=150)
    e = ExtendedGtpEngine(GTPGameConnector(player), scorer=sc)
    _long_game(e, opponent_passes=False)
    e.send("genmove w")
    assert sc.asked == [] and player.seen == [True]
    for scorer in (None, FakeScorer(fail=True)):
        player = FixedPlayer(move=(10, 9))
        e = ExtendedGtpEngine(GTPGameConnector(player), scorer=scorer)
        _long_game(e)
        e.send("genmove w")
        assert player.seen == [True]   # the player's own rule decides
