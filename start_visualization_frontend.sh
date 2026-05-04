#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"

cd "$PROJECT_ROOT"
python3 -m visualization_frontend.server --host "$HOST" --port "$PORT"
