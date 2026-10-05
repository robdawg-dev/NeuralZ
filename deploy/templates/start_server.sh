#!/usr/bin/env bash
# Start the policy server (holds the network, CPU only). Extra arguments go to
# go_server.py, e.g.:  ./start_server.sh --port 5005 --threads 4
# See: .venv/bin/python go_server.py --help
#
# If this folder holds KataGo (katago/), the server also runs it to judge finished games
# for the bots (dead stones; cleanup moves for clients started with --cleanup).
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
KATAGO=()
if [ -x katago/katago ] && [ -n "@KATAGO_MODEL@" ]; then
    KATAGO=(--katago katago/katago --katago-model "@KATAGO_MODEL@"
            --katago-config katago_analysis.cfg)
fi
exec .venv/bin/python go_server.py @MODEL_JSON@ @WEIGHTS@ "${KATAGO[@]}" "$@"
