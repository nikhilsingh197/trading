# =============================================================================
# AI Crypto Trader — Multi-stage Dockerfile
# =============================================================================

# Base: Python 3.12 slim
FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Copy source code and assets
COPY ai_crypto_trader/ ./ai_crypto_trader/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY dashboard.html ./
COPY run_paper_trading.py ./

# Create data directories
RUN mkdir -p data/raw data/processed data/live data/models data/reports logs

# =============================================================================
# Worker stage
# =============================================================================
FROM base AS worker
CMD ["celery", "-A", "ai_crypto_trader.workers.celery_app", "worker", "--loglevel=info"]

# =============================================================================
# Ingestion stage
# =============================================================================
FROM base AS ingestion
CMD ["python", "-m", "ai_crypto_trader.main.ingestion_service"]

# =============================================================================
# API stage (Default final stage for Railway / web deployments)
# =============================================================================
FROM base AS api
EXPOSE 8000
CMD ["sh", "-c", "uvicorn ai_crypto_trader.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
