#!/usr/bin/env bash
# Start the policy server (holds the network, CPU only). Extra arguments go to
# go_server.py, e.g.:  ./start_server.sh --port 5005 --symmetries 4
# See: .venv/bin/python go_server.py --help
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
exec .venv/bin/python go_server.py @MODEL_JSON@ @WEIGHTS@ "$@"
