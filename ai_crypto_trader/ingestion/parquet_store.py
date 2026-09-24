"""High-performance Parquet storage engine for OHLCV candles.

Provides fast, columnar local storage with deduplication, atomic writes,
and efficient slicing by datetime range.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from ai_crypto_trader.config.settings import get_settings
from ai_crypto_trader.core.interfaces import Candle
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


class ParquetStore:
    """Manages local Parquet files for raw and processed market data."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        if base_dir is None:
            settings = get_settings()
            self._base_dir = Path(settings.data_raw_dir)
        else:
            self._base_dir = Path(base_dir)

        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, symbol: str, timeframe: str) -> Path:
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        return self._base_dir / f"{clean_symbol}_{timeframe}.parquet"

    def save_candles(self, candles: Sequence[Candle]) -> int:
        """Save candles to Parquet, appending and deduplicating by open_time.

        Uses atomic write (temp file + rename) to avoid file corruption on interruption.

        Args:
            candles: List of Candle dataclass objects.

        Returns:
            Total number of candles in the dataset after save.
        """
        if not candles:
            return 0

        symbol = candles[0].symbol
        timeframe = candles[0].timeframe
        target_path = self._file_path(symbol, timeframe)

        new_rows = [
            {
                "symbol": c.symbol,
                "timeframe": c.timeframe,
                "open_time": c.open_time if c.open_time.tzinfo else c.open_time.replace(tzinfo=timezone.utc),
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
                "close_time": c.close_time if c.close_time.tzinfo else c.close_time.replace(tzinfo=timezone.utc),
                "quality_flag": int(c.quality_flag),
            }
            for c in candles
        ]
        new_df = pd.DataFrame(new_rows)

        if target_path.exists():
            try:
                existing_df = pd.read_parquet(target_path)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                log.warning("parquet_read_failed_rebuilding", path=str(target_path), error=str(e))
                combined = new_df
        else:
            combined = new_df

        # Deduplicate on open_time, keeping last record
        combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True)
        combined = combined.drop_duplicates(subset=["open_time"], keep="last")
        combined = combined.sort_values(by="open_time").reset_index(drop=True)

        # Atomic write
        temp_path = target_path.with_suffix(".parquet.tmp")
        combined.to_parquet(temp_path, index=False, engine="pyarrow")
        if target_path.exists():
            target_path.unlink()
        temp_path.rename(target_path)

        log.info(
            "parquet_saved",
            symbol=symbol,
            timeframe=timeframe,
            total_bars=len(combined),
            new_added=len(candles),
        )
        return len(combined)

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load candles as a pandas DataFrame indexed by open_time."""
        path = self._file_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(path)
        if df.empty:
            return df

        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

        if start is not None:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            df = df[df["open_time"] >= start]

        if end is not None:
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            df = df[df["open_time"] <= end]

        return df.sort_values("open_time").reset_index(drop=True)

    def load_as_entities(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        """Load candles as a list of Candle dataclass instances."""
        df = self.load_candles(symbol, timeframe, start=start, end=end)
        if df.empty:
            return []

        candles: list[Candle] = []
        for _, row in df.iterrows():
            candles.append(
                Candle(
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    open_time=row["open_time"].to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    close_time=row["close_time"].to_pydatetime(),
                    quality_flag=int(row.get("quality_flag", 0)),
                )
            )
        return candles

    def get_summary(self) -> list[dict]:
        """Scan stored parquet files and return dataset summaries."""
        summaries = []
        if not self._base_dir.exists():
            return summaries

        for file in sorted(self._base_dir.glob("*.parquet")):
            name = file.stem
            if "_" in name:
                parts = name.split("_")
                symbol = parts[0]
                timeframe = "_".join(parts[1:])
            else:
                symbol = name
                timeframe = "unknown"

            try:
                df = pd.read_parquet(file, columns=["open_time"])
                count = len(df)
                if count > 0:
                    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
                    earliest = df["open_time"].min().isoformat()
                    latest = df["open_time"].max().isoformat()
                else:
                    earliest = None
                    latest = None
            except Exception:
                count = 0
                earliest = None
                latest = None

            size_kb = round(file.stat().st_size / 1024, 2)
            summaries.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "count": count,
                "start": earliest,
                "end": latest,
                "size_kb": size_kb,
                "path": str(file),
            })

        return summaries

    def exists(self, symbol: str, timeframe: str) -> bool:
        return self._file_path(symbol, timeframe).exists()

    def delete_dataset(self, symbol: str, timeframe: str) -> bool:
        path = self._file_path(symbol, timeframe)
        if path.exists():
            path.unlink()
            return True
        return False
