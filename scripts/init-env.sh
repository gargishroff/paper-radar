#!/usr/bin/env bash
# Creates .env from .env.example with generated secrets, asking for the Cohere key.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]] && grep -q '^COHERE_API_KEY=.\+' .env; then
  echo ".env already has a Cohere key; leaving it alone."
  exit 0
fi
key="${COHERE_API_KEY:-}"
if [[ -z "$key" ]]; then
  read -r -p "Cohere API key (https://dashboard.cohere.com/api-keys): " key
fi
[[ -n "$key" ]] || { echo "A Cohere API key is required." >&2; exit 1; }

rand() { python3 -c 'import secrets; print(secrets.token_urlsafe(32))'; }
sed -e "s|^COHERE_API_KEY=.*|COHERE_API_KEY=${key}|" \
    -e "s|^SECRET_KEY=.*|SECRET_KEY=$(rand)|" \
    -e "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(rand)|" \
    .env.example > .env
if [[ -n "${DATABASE_URL:-}" ]]; then
  sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=${DATABASE_URL}|" .env && rm -f .env.bak
fi
echo "Wrote .env"
