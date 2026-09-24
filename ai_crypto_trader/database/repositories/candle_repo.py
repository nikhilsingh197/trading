"""Candle database repository.

Provides async persistence and query operations for OHLCV candles.
Supports PostgreSQL (TimescaleDB) with ON CONFLICT DO UPDATE/NOTHING,
as well as standard SQL dialects for cross-environment testing.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ai_crypto_trader.core.interfaces import Candle as CandleEntity
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.database.models import Candle as CandleModel

log = get_logger(__name__)


class CandleRepository:
    """Async repository for Candle persistence and queries."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_many(self, candles: Sequence[CandleEntity | CandleModel]) -> int:
        """Insert or update a batch of candles, skipping duplicates.

        Args:
            candles: List of Candle dataclass instances or CandleModel objects.

        Returns:
            Number of candles processed.
        """
        if not candles:
            return 0

        # Transform dataclass entities to dicts for bulk insert
        records = []
        for c in candles:
            if isinstance(c, CandleModel):
                records.append({
                    "symbol": c.symbol,
                    "timeframe": c.timeframe,
                    "open_time": c.open_time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "close_time": c.close_time,
                    "quote_volume": c.quote_volume,
                    "trade_count": c.trade_count,
                    "source": getattr(c, "source", "binance"),
                    "quality_flag": c.quality_flag,
                })
            else:
                records.append({
                    "symbol": c.symbol,
                    "timeframe": c.timeframe,
                    "open_time": c.open_time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "close_time": c.close_time,
                    "quote_volume": None,
                    "trade_count": None,
                    "source": "binance",
                    "quality_flag": c.quality_flag,
                })

        bind = self._session.bind
        dialect_name = bind.dialect.name if bind else "postgresql"

        if dialect_name == "postgresql":
            stmt = pg_insert(CandleModel).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timeframe", "open_time"],
                set_={
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "quality_flag": stmt.excluded.quality_flag,
                },
            )
            await self._session.execute(stmt)
        elif dialect_name == "sqlite":
            stmt = sqlite_insert(CandleModel).values(records)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["symbol", "timeframe", "open_time"]
            )
            await self._session.execute(stmt)
        else:
            # Fallback standard merge/insert
            for r in records:
                obj = CandleModel(**r)
                await self._session.merge(obj)

        await self._session.flush()
        log.debug("candles_upserted", count=len(records), dialect=dialect_name)
        return len(records)

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[CandleModel]:
        """Fetch candles ordered chronologically by open_time."""
        stmt = (
            select(CandleModel)
            .where(CandleModel.symbol == symbol, CandleModel.timeframe == timeframe)
            .order_by(CandleModel.open_time.asc())
        )
        if start:
            stmt = stmt.where(CandleModel.open_time >= start)
        if end:
            stmt = stmt.where(CandleModel.open_time <= end)
        if limit:
            stmt = stmt.limit(limit)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_candle(
        self,
        symbol: str,
        timeframe: str,
    ) -> CandleModel | None:
        """Fetch the single most recent candle."""
        stmt = (
            select(CandleModel)
            .where(CandleModel.symbol == symbol, CandleModel.timeframe == timeframe)
            .order_by(CandleModel.open_time.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def get_candle_count(self, symbol: str, timeframe: str) -> int:
        """Count total candles stored for symbol and timeframe."""
        stmt = (
            select(func.count(CandleModel.id))
            .where(CandleModel.symbol == symbol, CandleModel.timeframe == timeframe)
        )
        result = await self._session.execute(stmt)
        return result.scalar() or 0

    async def get_date_range(
        self,
        symbol: str,
        timeframe: str,
    ) -> tuple[datetime | None, datetime | None]:
        """Return (earliest_time, latest_time) for stored candles."""
        stmt = (
            select(
                func.min(CandleModel.open_time),
                func.max(CandleModel.open_time),
            )
            .where(CandleModel.symbol == symbol, CandleModel.timeframe == timeframe)
        )
        result = await self._session.execute(stmt)
        row = result.first()
        if row:
            return row[0], row[1]
        return None, None

    async def delete_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """Delete candles in an interval."""
        stmt = delete(CandleModel).where(
            CandleModel.symbol == symbol,
            CandleModel.timeframe == timeframe,
        )
        if start:
            stmt = stmt.where(CandleModel.open_time >= start)
        if end:
            stmt = stmt.where(CandleModel.open_time <= end)

        res = await self._session.execute(stmt)
        await self._session.flush()
        return res.rowcount  # type: ignore[return-value]
