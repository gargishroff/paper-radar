#!/usr/bin/env bash
# Builds ../paper-radar.zip for sharing, excluding secrets, virtualenvs, data and caches.
set -euo pipefail
cd "$(dirname "$0")/../.."
rm -f paper-radar.zip
zip -rq paper-radar.zip paper-radar \
  -x 'paper-radar/.env' 'paper-radar/.venv/*' 'paper-radar/data/*' 'paper-radar/backups/*' \
     '*/__pycache__/*' 'paper-radar/.pytest_cache/*' 'paper-radar/.ruff_cache/*' '*.DS_Store'
if unzip -l paper-radar.zip | grep -qE '/\.env$|/\.venv/|/data/'; then
  echo "Refusing: secrets or local data ended up in the zip." >&2
  rm -f paper-radar.zip
  exit 1
fi
echo "Wrote $(pwd)/paper-radar.zip ($(du -h paper-radar.zip | cut -f1), $(unzip -l paper-radar.zip | tail -1 | awk '{print $2}') files)"
