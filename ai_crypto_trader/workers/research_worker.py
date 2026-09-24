"""Background research and training tasks."""
from __future__ import annotations

from ai_crypto_trader.workers.celery_app import app
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@app.task(name="research.run_experiment", bind=True, max_retries=1)
def run_experiment_task(self, experiment_config: dict):
    """Run a strategy experiment in the background."""
    log.info("experiment_task_started", config=experiment_config)
    # Placeholder - full implementation in Milestone 7
    return {"status": "queued", "config": experiment_config}
