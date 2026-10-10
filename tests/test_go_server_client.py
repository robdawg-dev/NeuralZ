"""Tests for go_server.py and go_client.py: a shared inference server and the GTP bots
that use it."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import go_client
import go_server
from AlphaGo import go
from AlphaGo.ai import ProbabilisticPolicyPlayer
from AlphaGo.go import GameState
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.training.shard_stream import BATCH_TRANSFORMATIONS

FEATURES = ["board", "ones", "liberties", "sensibleness"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("go_server")
    policy = CNNPolicy(FEATURES, layers=2, filters_per_layer=8, filter_width_1=3)
    model, weights = str(tmp / "model.json"), str(tmp / "model.weights.h5")
    policy.save_model(model)
    policy.model.save_weights(weights)
    return model, weights


@pytest.fixture(scope="module")
def server(model_files):
    """A live go_server on a free localhost port: (url, BatchingPolicy)."""
    policy = go_server.BatchingPolicy(*model_files, max_batch=16, batch_wait_ms=20)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), go_server.make_handler(policy))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:{}".format(httpd.server_address[1]), policy
    httpd.shutdown()
    httpd.server_close()


def _positions():
    """A few distinct positions: the empty board, then one per move of a short opening."""
    states = []
    moves = [(3, 3), (15, 15), (3, 15), (15, 3), (2, 3), (4, 3), (3, 2), (3, 4), (16, 16)]
    state = GameState()
    states.append(state.copy())
    for move in moves:
        state.do_move(move)
        states.append(state.copy())
    return states


def test_info_describes_the_served_model(server):
    url, policy = server
    remote = go_client.RemotePolicy(url)
    assert remote.info["features"] == FEATURES
    assert remote.info["board_size"] == 19
    assert remote.info["planes"] == policy.planes == remote.preprocessor.get_output_dimension()


def test_remote_moves_match_the_local_network(server, model_files):
    """The point of the split: a bot using the server chooses from exactly the
    distribution a bot loading the network itself would."""
    url, _policy = server
    remote = go_client.RemotePolicy(url)
    local = CNNPolicy.load_model(model_files[0])
    local.model.load_weights(model_files[1])
    for state in _positions():
        got, want = remote.eval_state(state), local.eval_state(state)
        assert [m for m, _p in got] == [m for m, _p in want]
        np.testing.assert_allclose([p for _m, p in got], [p for _m, p in want],
                                   rtol=1e-5, atol=1e-7)


def test_concurrent_requests_are_batched_and_each_gets_its_own_answer(server, monkeypatch):
    url, policy = server
    batch_sizes = []
    run = policy._run

    def counting_run(planes_list):
        batch_sizes.append(len(planes_list))
        return run(planes_list)

    monkeypatch.setattr(policy, "_run", counting_run)
    states = _positions()
    remote = go_client.RemotePolicy(url)
    expected = [remote.move_probabilities(s) for s in states]
    batch_sizes.clear()

    results = [None] * len(states)
    barrier = threading.Barrier(len(states))

    def ask(i):
        barrier.wait()
        results[i] = go_client.RemotePolicy(url).move_probabilities(states[i])

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(len(states))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for got, want in zip(results, expected):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)
    assert sum(batch_sizes) == len(states)
    assert max(batch_sizes) > 1, "simultaneous requests were never batched together"


def test_bad_requests_are_rejected(server):
    url, _policy = server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(urllib.request.Request(url + "/policy", data=b"\x00" * 10))
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url + "/nope")
    assert e.value.code == 404


def test_player_plays_legal_moves_through_the_server(server):
    url, _policy = server
    player = ProbabilisticPolicyPlayer(go_client.RemotePolicy(url), pass_when_offered=True,
                                       sample_ratio=0.5, sample_moves=20)
    state = GameState()
    for _ in range(20):
        move = player.get_move(state)
        assert move is go.PASS or state.is_legal(move)
        state.do_move(move)


def test_unreachable_server_is_a_clear_error():
    with pytest.raises(RuntimeError, match="unreachable"):
        go_client.RemotePolicy("http://127.0.0.1:1", timeout=1, server_wait=0)


def test_client_waits_for_a_restarting_server(server):
    """A server that is down when the client asks, and back a few seconds later (as during
    a restart), costs the client a wait, not its game."""
    import socket
    _url, policy = server
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    def start_later():
        time.sleep(2.5)
        httpd = ThreadingHTTPServer(("127.0.0.1", port), go_server.make_handler(policy))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

    threading.Thread(target=start_later, daemon=True).start()
    t0 = time.time()
    remote = go_client.RemotePolicy("http://127.0.0.1:{}".format(port), timeout=5,
                                    server_wait=20, retry_wait=0.5)
    assert time.time() - t0 >= 2.0
    assert remote.move_probabilities(GameState()).shape == (361,)


def _counting_server(policy, drop_after_reply=False):
    """A go_server that counts the connections it accepts, and with drop_after_reply closes
    each one after its first reply without saying so - as a restarted server does to the
    connection a client was keeping open. (url, connection count list, httpd)."""
    connections = []

    class Handler(go_server.make_handler(policy)):
        def setup(self):
            connections.append(self.client_address)
            super().setup()

        def _reply(self, *args, **kwargs):
            super()._reply(*args, **kwargs)
            if drop_after_reply:
                self.close_connection = True

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return "http://127.0.0.1:{}".format(httpd.server_address[1]), connections, httpd


def test_client_keeps_one_connection_across_moves(server):
    url, connections, httpd = _counting_server(server[1])
    try:
        remote = go_client.RemotePolicy(url)
        for state in _positions():
            assert remote.move_probabilities(state).shape == (361,)
        with pytest.raises(RuntimeError, match="404"):
            remote._request("/nope", b"some body")
        assert remote.move_probabilities(GameState()).shape == (361,)  # still usable
        assert len(connections) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_client_reconnects_at_once_when_the_server_drops_the_connection(server, capsys):
    url, connections, httpd = _counting_server(server[1], drop_after_reply=True)
    try:
        remote = go_client.RemotePolicy(url, retry_wait=5)
        t0 = time.time()
        for state in _positions():
            assert remote.move_probabilities(state).shape == (361,)
        assert time.time() - t0 < 5  # no retry wait
        assert len(connections) == 1 + len(_positions())
        assert "unreachable" not in capsys.readouterr().err
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_client_never_loads_tensorflow():
    """The memory saving depends on it: each bot process must stay TensorFlow-free."""
    script = ("import sys, go_client; "
              "print([m for m in ('tensorflow', 'keras') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]"


# --- symmetry averaging ----------------------------------------------------------------

@pytest.fixture(scope="module")
def symmetric_policy(model_files):
    return go_server.BatchingPolicy(*model_files, symmetries=8)


@pytest.fixture(scope="module")
def rotation_policy(model_files):
    return go_server.BatchingPolicy(*model_files, symmetries=4)


def _planes(policy, state):
    return policy.policy.preprocessor.state_to_tensor(state)[0]


def test_one_symmetry_is_exactly_the_plain_network(server, model_files):
    """--symmetries 1, the default, changes nothing (exactly so with eager calls; the
    compiled calls are compared with them below)."""
    _url, served = server
    assert served.symmetries == ["noop"] and served.info["symmetries"] == 1
    policy = go_server.BatchingPolicy(*model_files, compiled=False)
    local = CNNPolicy.load_model(model_files[0])
    local.model.load_weights(model_files[1])
    for state in _positions()[:4]:
        planes = _planes(policy, state)
        np.testing.assert_array_equal(
            policy._run([planes])[0],
            local.forward(planes[None].astype(np.float32))[0].astype("<f4"))


@pytest.mark.parametrize("which", ["eight", "four"])
def test_symmetry_average_matches_a_by_hand_computation(symmetric_policy, rotation_policy,
                                                        which):
    """Each view evaluated on its own, mapped back to the original orientation by
    inverting its board transform directly (not via the permutation table), and
    averaged."""
    policy = symmetric_policy if which == "eight" else rotation_policy
    names = (list(BATCH_TRANSFORMATIONS) if which == "eight"
             else ["noop", "rot90", "rot180", "rot270"])
    assert policy.symmetries == names
    inverse = {"noop": "noop", "rot90": "rot270", "rot180": "rot180", "rot270": "rot90",
               "fliplr": "fliplr", "flipud": "flipud", "diag1": "diag1", "diag2": "diag2"}
    for state in _positions()[3:6]:
        planes = _planes(policy, state)
        expected = np.zeros((19, 19))
        for name in names:
            view = BATCH_TRANSFORMATIONS[name](planes[None]).astype(np.float32)
            probs = policy.policy.forward(view)[0].reshape(1, 19, 19)
            expected += BATCH_TRANSFORMATIONS[inverse[name]](probs)[0]
        np.testing.assert_allclose(policy._run([planes])[0],
                                   (expected / len(names)).reshape(-1),
                                   rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", list(BATCH_TRANSFORMATIONS))
def test_eight_symmetries_do_not_depend_on_board_orientation(symmetric_policy, name):
    """A rotated or reflected position gets the same answer, rotated or reflected to
    match."""
    policy = symmetric_policy
    planes = _planes(policy, _positions()[5])
    fn = BATCH_TRANSFORMATIONS[name]
    original = policy._run([planes])[0].reshape(1, 19, 19)
    transformed = policy._run([fn(planes[None])[0]])[0].reshape(1, 19, 19)
    np.testing.assert_allclose(transformed, fn(original), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", ["rot90", "rot180", "rot270"])
def test_four_rotations_do_not_depend_on_rotating_the_board(rotation_policy, name):
    policy = rotation_policy
    planes = _planes(policy, _positions()[5])
    fn = BATCH_TRANSFORMATIONS[name]
    original = policy._run([planes])[0].reshape(1, 19, 19)
    transformed = policy._run([fn(planes[None])[0]])[0].reshape(1, 19, 19)
    np.testing.assert_allclose(transformed, fn(original), rtol=1e-5, atol=1e-7)


def test_eight_symmetries_give_symmetric_points_equal_probabilities(symmetric_policy):
    policy = symmetric_policy
    probs = policy._run([_planes(policy, GameState())])[0].reshape(19, 19)
    corners = [probs[3, 3], probs[3, 15], probs[15, 3], probs[15, 15]]
    np.testing.assert_allclose(corners, corners[0], rtol=1e-5)
    np.testing.assert_allclose(probs, probs.T, rtol=1e-5, atol=1e-8)


def test_info_reports_the_symmetry_setting(symmetric_policy, rotation_policy):
    assert symmetric_policy.info["symmetries"] == 8
    assert rotation_policy.info["symmetries"] == 4


def test_unsupported_symmetry_counts_are_rejected(model_files):
    with pytest.raises(ValueError, match="symmetries"):
        go_server.BatchingPolicy(*model_files, symmetries=3)


# --- compiled calls ----------------------------------------------------------------------

@pytest.mark.parametrize("symmetries", [1, 8])
@pytest.mark.parametrize("n_positions", [1, 3, 5])
def test_compiled_calls_match_eager_ones(model_files, symmetries, n_positions):
    """Compiled calls (batches padded up to a compiled size) answer as the plain Keras
    call does, position by position."""
    compiled = go_server.BatchingPolicy(*model_files, max_batch=8, symmetries=symmetries)
    eager = go_server.BatchingPolicy(*model_files, max_batch=8, symmetries=symmetries,
                                     compiled=False)
    planes = [_planes(compiled, s) for s in _positions()[:n_positions]]
    got, want = compiled._run(planes), eager._run(planes)
    assert got.shape == want.shape == (n_positions, 361) and got.dtype == np.dtype("<f4")
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


def test_every_batch_size_up_to_max_batch_has_a_compiled_call(model_files):
    policy = go_server.BatchingPolicy(*model_files, max_batch=12)
    assert sorted(policy._compiled) == [1, 2, 4, 8, 12]
