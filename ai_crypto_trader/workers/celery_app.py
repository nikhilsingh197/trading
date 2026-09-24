"""Celery application configuration."""
from __future__ import annotations

from celery import Celery

from ai_crypto_trader.config.settings import get_settings

settings = get_settings()

app = Celery(
    "ai_crypto_trader",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "ai_crypto_trader.workers.data_worker",
        "ai_crypto_trader.workers.research_worker",
        "ai_crypto_trader.workers.monitor_worker",
    ],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,         # Re-queue if worker crashes
    worker_prefetch_multiplier=1,  # One task at a time per worker
    task_soft_time_limit=3600,   # 1 hour soft limit
    task_time_limit=7200,        # 2 hour hard limit
)
