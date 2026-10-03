#!/usr/bin/env bash
# Runs the web app and the daily scheduler together (native setup).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m radar.jobs.scheduler &
worker=$!
trap 'kill $worker 2>/dev/null' EXIT
uv run uvicorn radar.app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
