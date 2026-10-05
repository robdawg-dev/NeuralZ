#!/usr/bin/env bash
# Runs LR-study experiments one after another through docker compose - see
# LR_STUDY_PLAN.md.
#
#   benchmarks/lr_study/run_queue.sh benchmarks/lr_study/phase1.queue
#
# Queue file: one experiment per line, "RUN_NAME python-module-and-arguments", where the
# token {out} is replaced by the run's directory, workspace/lr_study/runs/RUN_NAME.
# Blank lines and lines starting with # are ignored.
#
# Each run's output (stdout+stderr) goes to <run dir>.log, followed by its exit code. A
# run whose log already ends in "exit 0" is skipped, so an interrupted queue can simply
# be started again. When a run ends, every checkpoint but the last is deleted (each is
# ~121 MB, and these runs write one per epoch).
set -u
queue="$1"
runs="workspace/lr_study/runs"
mkdir -p "$runs"
export MSYS_NO_PATHCONV=1   # Git Bash on Windows: don't rewrite container paths

grep -v '^\s*#' "$queue" | grep -v '^\s*$' | while read -r name command; do
    out="$runs/$name"
    log="$out.log"
    if [ -f "$log" ] && tail -1 "$log" | grep -qx "exit 0"; then
        echo "$(date '+%F %T') skip  $name (already finished)"
        continue
    fi
    rm -rf "$out"
    echo "$(date '+%F %T') start $name"
    # < /dev/null: docker must not read the rest of the queue from this loop's stdin
    docker compose run --rm gpu python -m ${command//\{out\}/$out} > "$log" 2>&1 < /dev/null
    status=$?
    echo "exit $status" >> "$log"
    ls "$out"/weights.*.weights.h5 2>/dev/null | sort | head -n -1 | xargs -r rm -f
    echo "$(date '+%F %T') done  $name (exit $status)"
done
