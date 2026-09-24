"""WebSocket real-time data collector.

Maintains persistent WebSocket connections per (symbol, timeframe).
Publishes validated candles to Redis pub/sub for fan-out to consumers.
Auto-reconnects with exponential backoff on disconnection.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import redis.asyncio as aioredis

from ai_crypto_trader.core.constants import CHANNEL_CANDLE
from ai_crypto_trader.core.exceptions import DataValidationError
from ai_crypto_trader.core.interfaces import Candle
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.ingestion.data_validator import CandleValidator

log = get_logger(__name__)


class WebSocketCollector:
    """Streams live OHLCV data via WebSocket and publishes to Redis."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        redis_client: aioredis.Redis,
        symbols: list[str],
        timeframes: list[str],
        validator: CandleValidator | None = None,
    ) -> None:
        self._exchange = exchange
        self._redis = redis_client
        self._symbols = symbols
        self._timeframes = timeframes
        self._validator = validator or CandleValidator()
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start all WebSocket subscription tasks."""
        self._running = True
        for symbol in self._symbols:
            for timeframe in self._timeframes:
                task = asyncio.create_task(
                    self._subscribe_with_retry(symbol, timeframe),
                    name=f"ws:{symbol}:{timeframe}",
                )
                self._tasks.append(task)
        log.info(
            "websocket_collector_started",
            symbols=self._symbols,
            timeframes=self._timeframes,
        )

    async def stop(self) -> None:
        """Stop all WebSocket tasks gracefully."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._exchange.close()
        log.info("websocket_collector_stopped")

    async def _subscribe_with_retry(
        self,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Subscribe to a (symbol, timeframe) stream with exponential backoff."""
        delay = 1.0
        max_delay = 60.0
        while self._running:
            try:
                await self._subscribe(symbol, timeframe)
                delay = 1.0  # Reset on success
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error(
                    "ws_subscription_error",
                    symbol=symbol,
                    timeframe=timeframe,
                    error=str(exc),
                    retry_in=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    async def _subscribe(self, symbol: str, timeframe: str) -> None:
        """Subscribe and process candles for a single stream."""
        log.info("ws_subscribing", symbol=symbol, timeframe=timeframe)
        while self._running:
            ohlcv = await self._exchange.watch_ohlcv(symbol, timeframe)
            if not ohlcv:
                continue
            # CCXT returns list of [ts, o, h, l, c, v]
            for row in ohlcv:
                await self._process_row(symbol, timeframe, row)

    async def _process_row(self, symbol: str, timeframe: str, row: list) -> None:
        ts_ms, open_, high, low, close, vol = row[:6]
        open_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        candle = Candle(
            symbol=symbol.replace("/", ""),
            timeframe=timeframe,
            open_time=open_time,
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(vol),
            close_time=open_time,  # Will be corrected downstream
        )

        try:
            validated = self._validator.validate_candle(candle)
        except DataValidationError as exc:
            log.warning(
                "live_candle_rejected",
                symbol=symbol,
                timeframe=timeframe,
                error=str(exc),
            )
            return

        # Publish to Redis
        payload = {
            "symbol": validated.symbol,
            "timeframe": validated.timeframe,
            "open_time": validated.open_time.isoformat(),
            "open": validated.open,
            "high": validated.high,
            "low": validated.low,
            "close": validated.close,
            "volume": validated.volume,
            "quality_flag": validated.quality_flag,
        }
        channel = f"{CHANNEL_CANDLE}:{symbol.replace('/', '')}:{timeframe}"
        await self._redis.publish(channel, json.dumps(payload))
