"""Data integrity and quality audit engine.

Scans datasets for completeness, gap distribution, anomalous prices,
and produces actionable audit reports and health scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from ai_crypto_trader.core.constants import QUALITY_INTERPOLATED, QUALITY_OK, QUALITY_SUSPECT
from ai_crypto_trader.core.interfaces import Candle
from ai_crypto_trader.ingestion.data_cleaner import DataCleaner, DataGap


@dataclass
class DataQualityReport:
    """Detailed dataset quality report."""
    symbol: str
    timeframe: str
    total_candles: int
    expected_candles: int
    completeness_pct: float
    gap_count: int
    total_missing_candles: int
    largest_gap_candles: int
    ok_count: int
    interpolated_count: int
    suspect_count: int
    invalid_ohlc_count: int
    negative_volume_count: int
    health_score: float  # 0.0 - 100.0
    status: str          # EXCELLENT / GOOD / DEGRADED / CORRUPT
    gaps: list[DataGap] = field(default_factory=list)

    @property
    def is_acceptable_for_backtest(self) -> bool:
        """Threshold: backtests require at least GOOD data quality (>=95% complete, score >= 90)."""
        return self.completeness_pct >= 95.0 and self.health_score >= 90.0 and self.invalid_ohlc_count == 0


class DataIntegrityChecker:
    """Audits market data quality and generates reports."""

    def __init__(self, cleaner: DataCleaner | None = None) -> None:
        self._cleaner = cleaner or DataCleaner()

    def audit(self, candles: Sequence[Candle], timeframe: str) -> DataQualityReport:
        """Run an exhaustive data quality audit on a candle sequence."""
        if not candles:
            return DataQualityReport(
                symbol="UNKNOWN",
                timeframe=timeframe,
                total_candles=0,
                expected_candles=0,
                completeness_pct=0.0,
                gap_count=0,
                total_missing_candles=0,
                largest_gap_candles=0,
                ok_count=0,
                interpolated_count=0,
                suspect_count=0,
                invalid_ohlc_count=0,
                negative_volume_count=0,
                health_score=0.0,
                status="CORRUPT",
            )

        sorted_candles = sorted(candles, key=lambda c: c.open_time)
        symbol = sorted_candles[0].symbol
        total = len(sorted_candles)
        start = sorted_candles[0].open_time
        end = sorted_candles[-1].open_time

        step_sec = self._cleaner.timeframe_to_seconds(timeframe)
        total_seconds = (end - start).total_seconds()
        expected = int(total_seconds // step_sec) + 1 if total_seconds > 0 else 1

        # Gaps
        gaps = self._cleaner.detect_gaps(sorted_candles, timeframe)
        gap_count = len(gaps)
        total_missing = sum(g.missing_count for g in gaps)
        largest_gap = max((g.missing_count for g in gaps), default=0)

        # Quality flag breakdown & structural checks
        ok_count = 0
        interpolated_count = 0
        suspect_count = 0
        invalid_ohlc = 0
        negative_vol = 0

        for c in sorted_candles:
            if c.quality_flag == QUALITY_OK:
                ok_count += 1
            elif c.quality_flag == QUALITY_INTERPOLATED:
                interpolated_count += 1
            elif c.quality_flag == QUALITY_SUSPECT:
                suspect_count += 1

            if c.high < c.low or c.high < c.open or c.high < c.close or c.low > c.open or c.low > c.close:
                invalid_ohlc += 1

            if c.volume < 0:
                negative_vol += 1

        # Completeness calculation
        completeness = min(100.0, (total / expected) * 100.0) if expected > 0 else 0.0

        # Health score (out of 100)
        # Deduct for missing bars, invalid OHLC, suspect bars
        score = completeness
        score -= min(30.0, (total_missing / max(expected, 1)) * 50.0)
        score -= min(50.0, invalid_ohlc * 10.0)
        score -= min(15.0, suspect_count * 0.5)
        score -= min(20.0, negative_vol * 5.0)
        score = max(0.0, min(100.0, score))

        if score >= 98.0 and completeness >= 99.0 and invalid_ohlc == 0:
            status = "EXCELLENT"
        elif score >= 90.0 and completeness >= 95.0 and invalid_ohlc == 0:
            status = "GOOD"
        elif score >= 75.0:
            status = "DEGRADED"
        else:
            status = "CORRUPT"

        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_candles=total,
            expected_candles=expected,
            completeness_pct=round(completeness, 2),
            gap_count=gap_count,
            total_missing_candles=total_missing,
            largest_gap_candles=largest_gap,
            ok_count=ok_count,
            interpolated_count=interpolated_count,
            suspect_count=suspect_count,
            invalid_ohlc_count=invalid_ohlc,
            negative_volume_count=negative_vol,
            health_score=round(score, 1),
            status=status,
            gaps=gaps,
        )
