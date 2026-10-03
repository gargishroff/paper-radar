#!/usr/bin/env bash
# Native setup (no Docker): checks prerequisites, creates the database, loads recent papers.
set -euo pipefail
cd "$(dirname "$0")/.."
DB="${PAPERRADAR_DB:-paperradar}"

need() { command -v "$1" >/dev/null || { echo "Missing $1. $2" >&2; exit 1; }; }
need uv "Install it: https://docs.astral.sh/uv/getting-started/installation/"
need psql "Install PostgreSQL 16+ (macOS: brew install postgresql@17)."
pg_isready -q || { echo "PostgreSQL is not running (macOS: brew services start postgresql@17)." >&2; exit 1; }

DATABASE_URL="postgresql+psycopg://localhost/${DB}" scripts/init-env.sh

psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB}'" | grep -q 1 || createdb "$DB"
psql -d "$DB" -tAc "SELECT 1 FROM pg_available_extensions WHERE name='vector'" | grep -q 1 || {
  echo "pgvector is not installed (macOS: brew install pgvector; Linux: apt install postgresql-17-pgvector)." >&2
  exit 1
}

uv sync
uv run alembic upgrade head
DAYS="${BOOTSTRAP_DAYS:-3}"
echo "Loading the last ${DAYS} announcement days of arXiv papers (~5 Cohere calls and ~2 minutes each)..."
uv run python -m radar.cli bootstrap --days "$DAYS"
echo
echo "Done. Start PaperRadar with:  make run    then open http://localhost:8000"
