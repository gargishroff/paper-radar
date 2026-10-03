.PHONY: help setup run up down logs onboard test lint deploy

help:  ## Show the available commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

setup:  ## Native setup: .env, database, migrations, recent papers
	scripts/setup.sh

run:  ## Native: start the web app and the daily scheduler
	scripts/run.sh

up:  ## Docker: create .env if needed, then build and start everything
	scripts/init-env.sh
	docker compose up -d --build
	@echo "Open http://localhost:8000 (first start loads recent papers; see: make logs)"

down:  ## Docker: stop everything (data is kept)
	docker compose down

logs:  ## Docker: follow the app and worker logs
	docker compose logs -f app worker

test:  ## Run the test suite (needs a paperradar_test database)
	uv run pytest

lint:  ## Lint and format-check
	uv run ruff check . && uv run ruff format --check .

deploy:  ## Docker on a server: HTTPS, basic auth and backups
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
