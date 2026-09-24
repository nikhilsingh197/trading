"""Background data collection tasks."""
from __future__ import annotations

from ai_crypto_trader.workers.celery_app import app
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@app.task(name="data.backfill", bind=True, max_retries=3)
def backfill_task(self, symbol: str, timeframe: str, days: int = 30):
    """Celery task to backfill historical data."""
    import asyncio
    from datetime import datetime, timedelta, timezone
    import ccxt.async_support as ccxt
    from ai_crypto_trader.config.settings import get_settings
    from ai_crypto_trader.ingestion.rest_backfill import RestBackfill

    log.info("backfill_task_started", symbol=symbol, timeframe=timeframe, days=days)
    settings = get_settings()

    async def _run():
        exchange_cls = getattr(ccxt, settings.exchange_name)
        exchange = exchange_cls({"enableRateLimit": True})
        backfiller = RestBackfill(exchange)
        start = datetime.now(timezone.utc) - timedelta(days=days)
        candles = await backfiller.fetch_candles(symbol, timeframe, start)
        await backfiller.close()
        return len(candles)

    try:
        count = asyncio.run(_run())
        log.info("backfill_task_complete", symbol=symbol, candles=count)
        return {"symbol": symbol, "timeframe": timeframe, "candles": count}
    except Exception as exc:
        log.error("backfill_task_error", error=str(exc))
        raise self.retry(exc=exc, countdown=60)
