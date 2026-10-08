#!/usr/bin/env bash
# Install (or update) the bot in this folder: Python + packages into .venv, compile the
# game engine, then check that the server and client work together.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
    echo "install.sh: uv not found - install it with:" >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
if ! command -v g++ >/dev/null; then
    echo "install.sh: g++ not found (needed to compile the game engine) - install it with:" >&2
    echo "    sudo apt install build-essential" >&2
    exit 1
fi

echo "== installing Python and packages into .venv"
uv sync --frozen

echo "== compiling the game engine"
.venv/bin/python setup_cython.py build_ext --inplace > build_engine.log 2>&1 || {
    echo "install.sh: engine build failed - see build_engine.log" >&2
    exit 1
}

if [ -f katago/katago ]; then
    chmod +x katago/katago   # copies (e.g. from Windows) can drop the execute bit
fi

echo "== checking the bot"
.venv/bin/python check_deploy.py
