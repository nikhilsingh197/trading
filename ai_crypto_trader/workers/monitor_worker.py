"""Background monitoring tasks."""
from __future__ import annotations

from ai_crypto_trader.workers.celery_app import app
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@app.task(name="monitor.check_drift")
def check_drift_task():
    """Periodic task to check for strategy/data drift."""
    log.info("drift_check_started")
    # Placeholder - full implementation in Milestone 14
    return {"status": "ok"}
