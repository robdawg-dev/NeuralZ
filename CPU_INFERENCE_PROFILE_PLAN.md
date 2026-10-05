# CPU inference profiling plan

Where do the 150-300 ms per bot move go on the CPU server? This plan measures each stage of
a `genmove` separately, on the real server, before anything is optimized. It also sizes
the cost of the planned value-head lookahead, which adds about 5 network evaluations per
move (`VALUE_HEAD_PLAN.md`, step 5).

## A rough estimate first

b20c256's 3x3 convolutions hold about 23M weights, each applied at all 361 points: about
**8.4 G multiply-adds (~17 GFLOP) per position**. A CPU sustaining 100-200 GFLOP/s on this
work needs **~85-170 ms per evaluation**. So the network alone could account for most of
the observed time; the measurements below check that, and find whatever else adds up.

One observation already points at **batch-1 efficiency**: evaluating 8 symmetric views at
once (`--symmetries 8`) roughly doubled the time per move, for 8x the work. So a single
position probably runs at only about a quarter of the CPU's throughput: one 19x19
position gives each conv's matrix multiply too little work to keep many cores busy, and
the eager Keras call pays a fixed cost per layer (~100 layers) whatever the batch size.
That makes the per-call options below (compiled call, thread settings, a CPU inference
runtime) prime suspects, and means the lookahead's batch of ~6 positions may cost only
~1.5-2x a single evaluation, not 6x - which the batch-size measurements will confirm.

## What one `genmove` does (server/client setup)

**Client (`go_client.py`, one process per bot)**

| # | Stage | Code |
|---|---|---|
| C1 | GTP read and dispatch | `interface/gtp_wrapper.py` |
| C2 | Move-limit / pass checks: `get_history()` up to 3 times, each building the whole move list in Python | `AlphaGo/ai.py` `get_move` |
| C3 | Legal moves, excluding own eyes | `GameState.get_legal_moves(include_eyes=False)` |
| C4 | Atari check, in the sampling window: `get_board()` + `get_liberty()`, two 361-point passes | `ai.py` `_has_stone_in_atari` |
| C5 | **Feature planes**: 48 planes, including lookahead features (`liberties_after`, `self_atari_size`, `capture_size`) and ladder searches (`ladder_capture`, `ladder_escape`) | `Preprocess.state_to_tensor` |
| C6 | Pack planes, HTTP POST to the server, wait, read the reply. `urllib` opens a new TCP connection per request | `RemotePolicy._request` |
| C7 | Normalize over the legal moves: Python list comprehension over ~250 moves | `RemotePolicy.eval_state` |
| C8 | Choose: sample or argmax | `ai.py` |
| C9 | Apply the bot's move to its board, with positional superko enforced (the GTP board uses `enforce_superko=True`) | `GTPGameConnector.make_move` |

**Server (`go_server.py`)**

| # | Stage | Code |
|---|---|---|
| S1 | HTTP handling: a new thread per connection (`ThreadingHTTPServer`) | `Handler.do_POST` |
| S2 | Unpack the planes | `BatchingPolicy.decode` |
| S3 | Queue, then the batch wait (`--batch-wait-ms`, default 2 ms) | `_worker` |
| S4 | **Network forward**: an eager Keras call on CPU, batch = positions x `--symmetries` (1, 4 or 8) | `policy.forward` |
| S5 | Map the symmetric views back and average them | `_run` |

## Measurements

### 1. Stage timings: `workspace/profiling/profile_stages.py`

A standalone script, run from the deployed bot folder on the server, with no GTP and no
server process. It replays ~200 positions from real KGS games (moves 20-250, so
opening, middle game and endgame are all covered), and times each stage separately with
`time.perf_counter()`, warm-up excluded. It reports the median, 90th percentile and maximum
per stage.

- **C2, C3, C4, C7, C9** individually. C9 both with superko enforced and without.
- **C5 total, then per feature:** build a `Preprocess` for each feature on its own (`board`,
  `turns_since`, `liberties`, `capture_size`, `self_atari_size`, `liberties_after`,
  `ladder_capture`, `ladder_escape`, `sensibleness`) to see which ones cost. The ladder and
  lookahead features are the main suspects; their cost should grow in tactical positions,
  so report it by game phase too.
- **Pack + unpack** (C6 / S2 without the network).
- **S4 network forward** in-process, the same eager call the server makes:
  - batch sizes 1, 4, 5, 8 (single position, 4 / 8 symmetries, a top-5 lookahead);
  - with TensorFlow's default threads and the server's `--threads` settings;
  - eager call vs a `tf.function`-compiled call vs `model.predict`, since the call style
    alone can change CPU time.

### 2. Transport: `workspace/profiling/profile_transport.py`

The HTTP round trip with the network taken out: a stub server returning a fixed reply of
the right size, timed from the client's side. That gives S1 + C6's overhead (new TCP
connection, server thread, headers). Then the same against the real `go_server.py`, at 1
and 4 symmetries; the difference is S3-S5.

### 3. End to end, and what's unaccounted for

Time real `genmove` round trips: pipe a scripted GTP game (the `check_deploy.py` pattern)
into `go_client.py` against a running `go_server.py`, and log the time between each
`genmove` and its reply. The stage sums from 1 and 2 should add up to this. A large
remainder points at something not modeled: thread handoffs, GIL contention between the
server's HTTP and inference threads, or several bots sharing the server at once.

Then run `go_client.py` under `cProfile` for one scripted game, as a check for Python
hotspots the stage list missed.

### 4. Where to run it

**On the Ubuntu server, CPU only, from the deploy bundle.** The dev machine's CPU, its
Windows/Docker setup and the GPU-container builds all give different numbers. Record the
server's CPU model, core count and load (how many bots were running).

## What the results decide

| If the time is mostly... | Options |
|---|---|
| **Network forward (S4)** | Fewer symmetries (if in use); TF thread settings; a compiled `tf.function` call; ONNX Runtime or OpenVINO for CPU inference, possibly int8 quantization (needs a strength check); a smaller network (b15c192 is ~2.3x fewer FLOPs). For the lookahead: whether batch 6 costs much less than 6 x batch 1 on this CPU. |
| **Feature planes (C5)** | Profile the Cython feature code (ladder searches); compute ladder features only where needed; cache work shared between the current position and its lookahead children. |
| **Transport (C6 / S1)** | Persistent HTTP connections (keep-alive) or a Unix socket instead of a new connection per request. |
| **Game state (C2, C3, C9)** | Avoid rebuilding `get_history()` lists; check whether superko enforcement is the cost. |

The lookahead's cost estimate comes from the same numbers: about 5 more feature-plane
builds on the client (C5) plus one batch-6 forward on the server (S4) per move.

## Notes: lookahead with the score network (added 2026-10-03)

The score network (`SCORE_NET_PLAN.md`) makes a move cost **two network calls**: the
current position (policy, top k) and then the k positions after those moves in one batch
(scores). The second can't start before the first returns, so they can't be merged - but
most of what a call costs besides the arithmetic can be paid once or hidden. Measure these
along with the stages above:

1. **Server-side lookahead: one request per move.** The client sends the position once;
   the server runs the policy, builds the k candidate boards itself, runs the batch, and
   returns the move (or the scores). One HTTP round trip and one batch wait instead of
   two, and the two network calls run back to back. Needs the game engine (Cython
   `GameState` + `Preprocess`) on the server too, to build the candidate boards - today
   only the client has it.
2. **Compiled network calls.** `tf.function` with fixed batch sizes (e.g. 1 and 16,
   padding up to them) instead of the eager call: removes most per-layer Python and
   dispatch overhead, which is paid on every call - twice per move with lookahead. (The
   batch-size measurements under stage S4 above cover eager vs compiled.)
3. **Pondering.** Call 2 also gives the opponent's likely replies (the policy output for
   each candidate board). After our move, evaluate the positions after the opponent's top
   few replies during their thinking time; if they play one of them, call 1 for our next
   move is already done and only the batch of k remains at our turn. KGS time controls
   leave plenty of idle time.

With all three, the cost waited for at our turn could approach one batch-of-k call -
roughly 2x today's time per move by the "batch of 8 costs ~2x one position" observation,
rather than ~3x. Which are worth building depends on what the measurements show.

## Deliverable

A short results table per stage (median / p90, by game phase), saved as
`workspace/profiling/results_<host>.md`, and a recommendation for the first thing to
speed up, if anything.
