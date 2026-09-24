.PHONY: install dev-install setup-env migrate run-api run-worker run-ingestion
.PHONY: test test-unit test-integration lint format type-check
.PHONY: docker-up docker-down docker-logs backfill

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

setup-env:
	cp .env.example .env
	@echo "Edit .env with your configuration before running."

# ── Database ──────────────────────────────────────────────────────────────────
migrate:
	alembic upgrade head

migrate-down:
	alembic downgrade -1

# ── Services ──────────────────────────────────────────────────────────────────
run-api:
	uvicorn ai_crypto_trader.api.main:app --host 0.0.0.0 --port 8000 --reload

run-worker:
	celery -A ai_crypto_trader.workers.celery_app worker --loglevel=info

run-ingestion:
	python -m ai_crypto_trader.main.ingestion_service

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=ai_crypto_trader --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check ai_crypto_trader/ tests/

format:
	black ai_crypto_trader/ tests/

type-check:
	mypy ai_crypto_trader/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

# ── Data ──────────────────────────────────────────────────────────────────────
backfill:
	python -m ai_crypto_trader.main.cli backfill --symbol BTCUSDT --timeframe 1h --days 365

# ── Emergency ─────────────────────────────────────────────────────────────────
kill-switch-on:
	python -m ai_crypto_trader.main.cli risk kill-switch on --reason "manual"

kill-switch-off:
	python -m ai_crypto_trader.main.cli risk kill-switch off
