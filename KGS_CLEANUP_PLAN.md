# Plan: `kgs-genmove_cleanup`, and honest GNU Go-backed scoring commands

Drafted 2026-10-01. Not yet implemented. Open decisions are marked **Decide**.

## Background

KGS's kgsGtp uses two optional commands at the end of a game:

- **`final_status_list`** - "Optional, required for rated games." In scoring, kgsGtp asks
  for the dead stones and marks them. If the engine returns an error, kgsGtp marks none
  and the opponent has to. In tournament games it is only used if the engine also
  supports `kgs-genmove_cleanup`; otherwise all the engine's stones are assumed alive.
- **`kgs-genmove_cleanup`** - "recommended for playing ranked or tournament games." Used
  when the engine and the opponent disagree about dead stones: like `genmove`, except the
  engine must not pass until all dead stones are removed. In tournament games, an engine
  without it must make plain `genmove` behave that way.

## Where things stand

- `interface/gtp_wrapper.py` answers `final_status_list` and `final_score` by writing the
  game to a temporary SGF and running `gnugo --chinese-rules --mode gtp -l <file>`
  (`run_gnugo`, `call_gnugo`, `_ask_gnugo_about_current_game`), with a 10-second
  timeout. If GNU Go is not found, or does not answer in time, the reply is an **empty
  success** - indistinguishable from "no dead stones".
- A stub `cmd_kgs_genmove_cleanup` is commented out. It would not have worked anyway:
  pygtp finds commands by method name (`cmd_<name>`), so it would register
  `kgs_genmove_cleanup`, while KGS sends **`kgs-genmove_cleanup`**. The method has to be
  attached under the exact hyphenated name with `setattr`.
- The bot's normal pass logic (`AlphaGo/ai.py`): pass once the opponent has just passed
  after move 100, when no "sensible" move is left, or at `--max-moves` (800).

### Measured (GNU Go 3.8, Ubuntu 24.04 package, 2026-10-01)

- GNU Go 3.8 supports `kgs-genmove_cleanup` (in its `list_commands`; answers e.g. `= D6`).
- On finished KataGo games: `final_status_list dead` and `final_score` each answer in
  **0.1-0.2 s** (3 games tested). Well within the 10-second limit.
- On a near-empty board (one stone): **~98 s** - GNU Go plays the whole game out to
  decide status. Past the timeout, so an empty reply. A manual test must therefore use a
  finished position, not a nearly empty board.
- Ubuntu installs GNU Go as `/usr/games/gnugo`. The wrapper finds it via
  `shutil.which('gnugo')`, so `/usr/games` must be on the bot's PATH - which is inherited
  from whatever launches kgsGtp, not necessarily your login shell. Check a running bot
  with `tr '\0' '\n' < /proc/<PID>/environ | grep ^PATH=` (PID from
  `pgrep -af run_gtp_player`). Fix if missing: `sudo ln -s /usr/games/gnugo
  /usr/local/bin/gnugo`. Version: `gnugo --version`.

## The plan

### `kgs-genmove_cleanup <color>`

1. Ask GNU Go `kgs-genmove_cleanup <color>` on the current game, through the same
   SGF-and-subprocess path as `final_status_list`.
2. Parse the reply defensively: a point (e.g. `D6`) -> play it; `PASS` -> pass (GNU Go
   only passes once it sees no dead stones left); anything else (empty/timeout, a GTP
   `?` error, `resign`, unparseable) -> "no answer".
3. Record the move in the bot's own game state exactly as `genmove` does (same
   `make_move` path), so the bot's board and KGS's stay in step.
4. If there is no usable answer, or GNU Go's move is one our engine rejects (possible
   where rules differ, e.g. suicide): see **Decide 2** below.
5. Register it under its exact name (`setattr(ExtendedGtpEngine, "cmd_kgs-genmove_cleanup",
   ...)`), so `list_commands` / `known_command` report it - that is how kgsGtp detects
   support. Remove the dead stub.

Both bots get it: `go_client.py` and `run_gtp_player.py` share `interface/gtp_wrapper.py`.

### Why the bot's own `genmove` is a poor fallback for cleanup

- **Passing too early (the real risk):** normal `genmove` passes as soon as the opponent
  passes after move 100. In cleanup the opponent often passes with dead stones still on
  the board; if the bot passes too, those stones count as alive - which can swing the
  score and lose a won game. It breaks exactly the rule the command exists for.
- **Not removing dead stones:** the network was trained on KataGo games, which pass at the
  end rather than capture dead stones, so it may play ordinary moves instead of attacking
  the dead groups.
- **Pointless or costly moves:** filling neutral points or playing inside its own area -
  mostly harmless under area scoring, possibly costly under territory scoring.
- It cannot desync the board: fallback moves go through the same recording path.

A **cleanup-safe fallback** would remove the first risk: the network's move but *without*
the "pass when the opponent passes" rule - passing only when no sensible move is left, or
at the move limit. The second risk would remain, logged when it happens.

### GNU Go missing / not answering - the direction agreed so far

- **GNU Go not installed / not on PATH:** do not advertise the GNU Go-backed commands
  (`final_status_list`, `kgs-genmove_cleanup`, and `final_score` for consistency), so
  kgsGtp knows the bot cannot do them instead of receiving empty answers. Print a clear
  warning to stderr at startup.
- **GNU Go present but not answering** (timeout, error, unusable reply): return a **GTP
  error** rather than an empty success. For `final_status_list` this is documented as
  safe: kgsGtp marks nothing and the opponent marks the stones.

## Decide

1. **Missing GNU Go: refuse to start, or start with the commands hidden?**
   `final_status_list` is required for rated games, so a bot that hides it presumably
   stops getting rated games - possibly unnoticed for a while (e.g. after an OS update
   or a PATH change). Option: refuse to start by default with a clear message, with a
   flag such as `--allow-no-gnugo` to start anyway with the commands hidden.
2. **`kgs-genmove_cleanup` when GNU Go does not answer: GTP error, or the cleanup-safe
   fallback?** The KGS text does not say what kgsGtp does when a move request returns an
   error - it might retry, pass, resign, or drop the engine; losing or abandoning a game
   over one slow reply would be worse than a network cleanup move. Options: (a) test
   kgsGtp's reaction in a free game first, then decide; (b) use the cleanup-safe fallback
   for timeouts and errors only for genuine failures (e.g. an illegal move) until known.
3. **Timeout:** keep 10 s, or raise to ~20 s to make timeouts rarer on a busy server with
   many bots?

## Tests

With a stand-in for GNU Go (it is not in the Docker image):

- `kgs-genmove_cleanup` appears in `list_commands` and `known_command` when GNU Go is
  available; none of the GNU Go-backed commands appear when it is not.
- A GNU Go answer like `D6` comes back as `= D6` and the stone is on the bot's board.
- `PASS` is returned and recorded as a pass.
- An empty answer, a GTP error, and an illegal move each produce the chosen behavior
  (GTP error or cleanup-safe fallback, per Decide 2).
- `final_status_list` / `final_score` return a GTP error when GNU Go does not answer.
- Optional, run only where GNU Go is installed (e.g. the bot server): play out a finished
  game, check that cleanup moves are legal and that GNU Go passes once the dead stones
  are gone.

## Rollout

1. Implement and test here.
2. Copy the updated `interface/gtp_wrapper.py` into `C:\Users\winst\Desktop\new-bot`.
3. On the bot server: check GNU Go is on the bots' PATH and its version (above), then a
   manual GTP session on a **finished** position - `list_commands`, then
   `kgs-genmove_cleanup black` a few times. A small one-off script that turns an SGF into
   GTP `play` commands would make this easy.
4. Restart the bots.

Side effect: once `kgs-genmove_cleanup` is advertised, kgsGtp also starts using
`final_status_list` in tournament games (previously it assumed all the bot's stones
alive).
