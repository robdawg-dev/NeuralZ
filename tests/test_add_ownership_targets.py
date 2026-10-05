"""Tests for add_ownership_targets' pure parts: SGF -> analysis query, KataGo's point order
-> this project's. (Running KataGo itself isn't part of the test suite.)"""
from AlphaGo.preprocessing import add_ownership_targets as aot
from AlphaGo.util import flatten_idx


def test_gtp_vertex_skips_i_and_counts_rows_from_the_bottom():
    assert aot.gtp_vertex("aa", 19) == "A19"
    assert aot.gtp_vertex("ia", 19) == "J19"  # column 8 is J: GTP has no I
    assert aot.gtp_vertex("ss", 19) == "T1"
    assert aot.gtp_vertex("", 19) == aot.gtp_vertex("tt", 19) == "pass"


def test_katago_rules_string_becomes_a_rules_object():
    text = "(;RU[koSITUATIONALscoreTERRITORYtaxSEKIsui1];B[pd])"
    assert aot.katago_rules(text) == {"ko": "SITUATIONAL", "scoring": "TERRITORY",
                                      "tax": "SEKI", "suicide": True}
    assert aot.katago_rules("(;RU[Japanese];B[pd])") == "chinese"


def test_final_position_query_has_setup_moves_and_passes():
    text = ("(;GM[1]SZ[19]KM[0.5]RU[koPOSITIONALscoreAREAtaxNONEsui0]AB[dd][pp]"
            "C[startTurnIdx=2];W[dp]C[0.4 0.6 0.0 1.0 v=1];B[pd];W[];B[])")
    q = aot.final_position_query("7", text)
    assert q["id"] == "7" and q["komi"] == 0.5
    assert q["initialStones"] == [["B", "D16"], ["B", "Q4"]]
    assert q["moves"] == [["W", "D4"], ["B", "Q16"], ["W", "pass"], ["B", "pass"]]
    assert q["analyzeTurns"] == [4] and q["includeOwnership"]
    assert q["rules"]["scoring"] == "AREA"


def test_ownership_is_reordered_to_x_major_like_the_planes():
    size = 19
    katago = [0.0] * (size * size)
    x, y = 3, 15  # SGF 'dp': column d, row p
    katago[y * size + x] = 1.0  # KataGo: row-major from the top-left
    katago[0] = -0.5            # A19, the top-left corner
    out = aot.to_project_order(katago, size)
    assert out[flatten_idx((x, y), size)] == 127
    assert out[flatten_idx((0, 0), size)] == -64
    assert sum(1 for v in out if v) == 2
