#!/usr/bin/env bash
# The GTP engine command for kgsGtp: talks GTP on stdin/stdout and gets move
# probabilities from start_server.sh. Extra arguments go to go_client.py, e.g.:
#   start_client.sh --server http://127.0.0.1:5005 --sample-moves 20
# See: .venv/bin/python go_client.py --help
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/.venv/bin/python" "$DIR/go_client.py" "$@"
