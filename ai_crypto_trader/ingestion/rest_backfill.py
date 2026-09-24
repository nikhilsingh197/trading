"""REST API historical data backfill.

Fetches OHLCV candles from the exchange REST API in paginated
batches and stores them in the database.

Design choices:
- Tenacity retry with exponential backoff for API errors.
- Validates every candle before persistence.
- Detects gaps and logs them as warnings.
- Never overwrites existing validated candles.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import ccxt.async_support as ccxt
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai_crypto_trader.core.exceptions import ExchangeConnectionError
from ai_crypto_trader.core.interfaces import Candle
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.ingestion.data_validator import CandleValidator, SeriesValidator

log = get_logger(__name__)

CANDLES_PER_REQUEST = 1000  # Binance maximum


class RestBackfill:
    """Fetches and validates historical OHLCV data."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        candle_validator: CandleValidator | None = None,
        series_validator: SeriesValidator | None = None,
    ) -> None:
        self._exchange = exchange
        self._candle_validator = candle_validator or CandleValidator()
        self._series_validator = series_validator or SeriesValidator()

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None = None,
    ) -> list[Candle]:
        """Fetch all candles for a symbol/timeframe within a time range.

        Args:
            symbol: e.g. 'BTC/USDT'
            timeframe: e.g. '1h'
            start: UTC datetime to start from
            end: UTC datetime to end at (defaults to now)

        Returns:
            List of validated Candle objects (sorted by open_time).
        """
        if end is None:
            end = datetime.now(timezone.utc)

        all_candles: list[Candle] = []
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        log.info(
            "backfill_starting",
            symbol=symbol,
            timeframe=timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        while since_ms < end_ms:
            raw_candles = await self._fetch_batch(symbol, timeframe, since_ms)
            if not raw_candles:
                break

            batch = self._parse_raw_candles(symbol, timeframe, raw_candles)
            # Filter to requested range
            batch = [c for c in batch if c.open_time.timestamp() * 1000 < end_ms]

            if not batch:
                break

            all_candles.extend(batch)
            last_ts = batch[-1].open_time
            since_ms = int(last_ts.timestamp() * 1000) + 1

            log.debug(
                "backfill_batch_fetched",
                symbol=symbol,
                timeframe=timeframe,
                count=len(batch),
                last_ts=last_ts.isoformat(),
            )

            await asyncio.sleep(0.1)  # Gentle rate limit

            if len(raw_candles) < CANDLES_PER_REQUEST:
                break  # No more data

        # Validate the entire series
        validated, warnings = self._series_validator.validate_series(all_candles, timeframe)

        # Validate individual candles (price/volume checks)
        good: list[Candle] = []
        for c in validated:
            try:
                good.append(self._candle_validator.validate_candle(c))
            except Exception as exc:
                log.warning(
                    "candle_validation_failed",
                    symbol=symbol,
                    timeframe=timeframe,
                    open_time=str(c.open_time),
                    error=str(exc),
                )

        log.info(
            "backfill_complete",
            symbol=symbol,
            timeframe=timeframe,
            total=len(good),
            warnings=len(warnings),
        )
        return good

    async def _fetch_batch(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
    ) -> list[Any]:
        """Fetch a single paginated batch with retry."""
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout)),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        ):
            with attempt:
                return await self._exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=CANDLES_PER_REQUEST,
                )
        return []

    def _parse_raw_candles(self, symbol: str, timeframe: str, raw: list[Any]) -> list[Candle]:
        candles = []
        for row in raw:
            # CCXT format: [timestamp_ms, open, high, low, close, volume]
            ts_ms, open_, high, low, close, vol = row[:6]
            open_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            # close_time = open_time + timeframe_delta - 1ms
            tf_seconds = SeriesValidator._timeframe_seconds(timeframe)
            close_time = open_time + timedelta(seconds=tf_seconds - 1)
            candles.append(
                Candle(
                    symbol=symbol.replace("/", ""),
                    timeframe=timeframe,
                    open_time=open_time,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(vol),
                    close_time=close_time,
                )
            )
        return candles

    async def close(self) -> None:
        await self._exchange.close()
