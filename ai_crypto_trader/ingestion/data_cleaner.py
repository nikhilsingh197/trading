"""Data cleaning, gap detection, and repair module.

Ensures candle series continuity without introducing look-ahead bias or
inventing arbitrary market movements. Small gaps (<= max_gap_bars) can be
linearly interpolated with quality_flag=QUALITY_INTERPOLATED. Large gaps are
reported for targeted exchange re-fetching.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from ai_crypto_trader.core.constants import QUALITY_INTERPOLATED, QUALITY_OK, QUALITY_SUSPECT
from ai_crypto_trader.core.interfaces import Candle
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class DataGap:
    """Represents an identified missing candle interval."""
    symbol: str
    timeframe: str
    start_time: datetime  # Last known candle open_time
    end_time: datetime    # Resuming candle open_time
    missing_count: int    # Number of missing candles
    expected_step_seconds: int


class DataCleaner:
    """Identifies and resolves gaps, duplicates, and anomalies in candle series."""

    @staticmethod
    def timeframe_to_seconds(timeframe: str) -> int:
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "12h": 43200,
            "1d": 86400,
        }
        if timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return mapping[timeframe]

    def detect_gaps(
        self,
        candles: Sequence[Candle],
        timeframe: str,
    ) -> list[DataGap]:
        """Detect missing candle intervals in chronological candle series."""
        if len(candles) < 2:
            return []

        step_sec = self.timeframe_to_seconds(timeframe)
        sorted_candles = sorted(candles, key=lambda c: c.open_time)
        gaps: list[DataGap] = []

        for i in range(len(sorted_candles) - 1):
            curr = sorted_candles[i]
            nxt = sorted_candles[i + 1]

            diff_sec = int((nxt.open_time - curr.open_time).total_seconds())
            if diff_sec > int(step_sec * 1.5):
                missing = int(diff_sec // step_sec) - 1
                if missing > 0:
                    gaps.append(
                        DataGap(
                            symbol=curr.symbol,
                            timeframe=timeframe,
                            start_time=curr.open_time,
                            end_time=nxt.open_time,
                            missing_count=missing,
                            expected_step_seconds=step_sec,
                        )
                    )

        return gaps

    def deduplicate(self, candles: Sequence[Candle]) -> list[Candle]:
        """Deduplicate candles by open_time, preserving the latest record."""
        seen: dict[datetime, Candle] = {}
        for c in candles:
            seen[c.open_time] = c
        return sorted(seen.values(), key=lambda c: c.open_time)

    def interpolate_short_gaps(
        self,
        candles: Sequence[Candle],
        timeframe: str,
        max_gap_bars: int = 5,
    ) -> list[Candle]:
        """Fill short missing intervals (<= max_gap_bars) with linear interpolation.

        Synthesized candles are explicitly tagged with QUALITY_INTERPOLATED so
        strategies and audit mechanisms know they are non-exchange candles.
        Gaps larger than max_gap_bars are left untouched to prevent fabricating data.
        """
        if len(candles) < 2:
            return list(candles)

        step_sec = self.timeframe_to_seconds(timeframe)
        step_delta = timedelta(seconds=step_sec)
        sorted_candles = self.deduplicate(candles)
        filled: list[Candle] = []

        for i in range(len(sorted_candles) - 1):
            curr = sorted_candles[i]
            nxt = sorted_candles[i + 1]
            filled.append(curr)

            diff_sec = int((nxt.open_time - curr.open_time).total_seconds())
            missing = int(diff_sec // step_sec) - 1

            if 0 < missing <= max_gap_bars:
                log.info(
                    "interpolating_gap",
                    symbol=curr.symbol,
                    timeframe=timeframe,
                    missing=missing,
                    from_time=curr.open_time.isoformat(),
                    to_time=nxt.open_time.isoformat(),
                )
                # Linear price step
                price_step = (nxt.open - curr.close) / (missing + 1)
                for step_idx in range(1, missing + 1):
                    synth_open_time = curr.open_time + (step_delta * step_idx)
                    synth_close_time = synth_open_time + step_delta - timedelta(milliseconds=1)
                    synth_price = curr.close + (price_step * step_idx)

                    synth_candle = Candle(
                        symbol=curr.symbol,
                        timeframe=timeframe,
                        open_time=synth_open_time,
                        open=synth_price,
                        high=synth_price,
                        low=synth_price,
                        close=synth_price,
                        volume=0.0,
                        close_time=synth_close_time,
                        quality_flag=QUALITY_INTERPOLATED,
                    )
                    filled.append(synth_candle)

        # Append last candle
        filled.append(sorted_candles[-1])
        return filled

    def flag_flash_spikes(
        self,
        candles: Sequence[Candle],
        spike_threshold_pct: float = 25.0,
    ) -> list[Candle]:
        """Detect and tag extreme transient spikes (jump and immediate return)."""
        if len(candles) < 3:
            return list(candles)

        cleaned: list[Candle] = []
        for i, c in enumerate(candles):
            if 0 < i < len(candles) - 1:
                prev_c = candles[i - 1]
                next_c = candles[i + 1]

                # Check if this candle deviated sharply from both neighbors
                up_prev = (c.high - prev_c.close) / prev_c.close * 100
                up_next = (c.high - next_c.open) / next_c.open * 100
                down_prev = (prev_c.close - c.low) / prev_c.close * 100
                down_next = (next_c.open - c.low) / next_c.open * 100

                if (up_prev > spike_threshold_pct and up_next > spike_threshold_pct) or \
                   (down_prev > spike_threshold_pct and down_next > spike_threshold_pct):
                    c.quality_flag = QUALITY_SUSPECT
                    log.warning(
                        "flash_spike_flagged",
                        symbol=c.symbol,
                        open_time=c.open_time.isoformat(),
                        high=c.high,
                        low=c.low,
                    )
            cleaned.append(c)

        return cleaned
