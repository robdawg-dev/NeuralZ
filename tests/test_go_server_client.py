"""Tests for go_server.py and go_client.py: a shared inference server and the GTP bots
that use it."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import subprocess
import sys
import threading
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
                                       greedy_start=2, top_k=12, top_k_responding=3)
    state = GameState()
    for _ in range(20):
        move = player.get_move(state)
        assert move is go.PASS or state.is_legal(move)
        state.do_move(move)


def test_unreachable_server_is_a_clear_error():
    with pytest.raises(RuntimeError, match="unreachable"):
        go_client.RemotePolicy("http://127.0.0.1:1", timeout=1, retries=0)


def test_client_never_loads_tensorflow():
    """The memory saving depends on it: each bot process must stay TensorFlow-free."""
    script = ("import sys, go_client; "
              "print([m for m in ('tensorflow', 'keras') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]"
