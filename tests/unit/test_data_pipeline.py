"""Automated tests for Milestone 2: Historical Data Pipeline.

Tests:
- ParquetStore (atomic write, read, slice, deduplicate, summary)
- DataCleaner (gap detection, deduplication, linear interpolation, spike flagging)
- DataIntegrityChecker (audit reports, health scoring, acceptance criteria)
- CandleRepository (async database upsert, query, date range, deduplication)
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ai_crypto_trader.core.constants import QUALITY_INTERPOLATED, QUALITY_OK, QUALITY_SUSPECT
from ai_crypto_trader.core.interfaces import Candle
from ai_crypto_trader.database.models import Base, Candle as CandleModel
from ai_crypto_trader.database.repositories.candle_repo import CandleRepository
from ai_crypto_trader.ingestion.data_cleaner import DataCleaner, DataGap
from ai_crypto_trader.ingestion.data_integrity import DataIntegrityChecker
from ai_crypto_trader.ingestion.parquet_store import ParquetStore


@pytest.fixture
def temp_data_dir() -> Path:
    """Create a temporary directory for Parquet storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_hourly_candles() -> list[Candle]:
    """Generate 10 consecutive hourly candles."""
    base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    candles = []
    price = 60000.0
    for i in range(10):
        t = base_time + timedelta(hours=i)
        candles.append(
            Candle(
                symbol="BTCUSDT",
                timeframe="1h",
                open_time=t,
                open=price + i * 10,
                high=price + i * 10 + 50,
                low=price + i * 10 - 30,
                close=price + i * 10 + 20,
                volume=150.0 + i * 5,
                close_time=t + timedelta(minutes=59, seconds=59),
                quality_flag=QUALITY_OK,
            )
        )
    return candles


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARQUET STORE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestParquetStore:
    def test_save_and_load(self, temp_data_dir, sample_hourly_candles):
        store = ParquetStore(temp_data_dir)
        count = store.save_candles(sample_hourly_candles)
        assert count == 10
        assert store.exists("BTCUSDT", "1h")

        df = store.load_candles("BTCUSDT", "1h")
        assert len(df) == 10
        assert "close" in df.columns
        assert df["close"].iloc[0] == sample_hourly_candles[0].close

    def test_deduplication_on_save(self, temp_data_dir, sample_hourly_candles):
        store = ParquetStore(temp_data_dir)
        store.save_candles(sample_hourly_candles[:6])  # Save first 6

        # Overlapping save of bars 4 to 9 (bars 4 and 5 are duplicates)
        total = store.save_candles(sample_hourly_candles[4:])
        assert total == 10  # Total unique bars must be 10

    def test_slice_by_date_range(self, temp_data_dir, sample_hourly_candles):
        store = ParquetStore(temp_data_dir)
        store.save_candles(sample_hourly_candles)

        start = sample_hourly_candles[2].open_time
        end = sample_hourly_candles[6].open_time

        sliced_df = store.load_candles("BTCUSDT", "1h", start=start, end=end)
        assert len(sliced_df) == 5  # index 2, 3, 4, 5, 6 inclusive

    def test_load_as_entities(self, temp_data_dir, sample_hourly_candles):
        store = ParquetStore(temp_data_dir)
        store.save_candles(sample_hourly_candles)

        entities = store.load_as_entities("BTCUSDT", "1h")
        assert len(entities) == 10
        assert isinstance(entities[0], Candle)
        assert entities[0].symbol == "BTCUSDT"

    def test_summary_report(self, temp_data_dir, sample_hourly_candles):
        store = ParquetStore(temp_data_dir)
        store.save_candles(sample_hourly_candles)

        summary = store.get_summary()
        assert len(summary) == 1
        assert summary[0]["symbol"] == "BTCUSDT"
        assert summary[0]["timeframe"] == "1h"
        assert summary[0]["count"] == 10
        assert summary[0]["size_kb"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANER & GAP REPAIR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDataCleaner:
    def test_detect_no_gaps_in_continuous_series(self, sample_hourly_candles):
        cleaner = DataCleaner()
        gaps = cleaner.detect_gaps(sample_hourly_candles, "1h")
        assert len(gaps) == 0

    def test_detect_gap_in_missing_interval(self, sample_hourly_candles):
        cleaner = DataCleaner()
        # Remove bars at index 4 and 5 (creating a 2-bar gap)
        broken = sample_hourly_candles[:4] + sample_hourly_candles[6:]
        gaps = cleaner.detect_gaps(broken, "1h")

        assert len(gaps) == 1
        assert gaps[0].missing_count == 2
        assert gaps[0].start_time == sample_hourly_candles[3].open_time
        assert gaps[0].end_time == sample_hourly_candles[6].open_time

    def test_interpolate_short_gap(self, sample_hourly_candles):
        cleaner = DataCleaner()
        broken = sample_hourly_candles[:4] + sample_hourly_candles[6:]  # 2 missing
        repaired = cleaner.interpolate_short_gaps(broken, "1h", max_gap_bars=5)

        assert len(repaired) == 10
        # Check synthesized bars
        synth_1 = repaired[4]
        synth_2 = repaired[5]
        assert synth_1.quality_flag == QUALITY_INTERPOLATED
        assert synth_2.quality_flag == QUALITY_INTERPOLATED
        assert synth_1.volume == 0.0

    def test_do_not_interpolate_large_gap(self, sample_hourly_candles):
        cleaner = DataCleaner()
        # Create a 6-bar gap (exceeds max_gap_bars=3)
        broken = sample_hourly_candles[:2] + sample_hourly_candles[8:]
        repaired = cleaner.interpolate_short_gaps(broken, "1h", max_gap_bars=3)

        assert len(repaired) == 4  # Unchanged, did not synthesize fake data

    def test_flag_flash_spikes(self, sample_hourly_candles):
        cleaner = DataCleaner()
        # Inject an extreme flash spike at index 5 (+50% jump and immediate drop)
        spiked_candles = list(sample_hourly_candles)
        spiked_candles[5] = Candle(
            symbol="BTCUSDT",
            timeframe="1h",
            open_time=sample_hourly_candles[5].open_time,
            open=90000.0,
            high=95000.0,
            low=89000.0,
            close=92000.0,
            volume=5000.0,
            close_time=sample_hourly_candles[5].close_time,
            quality_flag=QUALITY_OK,
        )

        cleaned = cleaner.flag_flash_spikes(spiked_candles, spike_threshold_pct=25.0)
        assert cleaned[5].quality_flag == QUALITY_SUSPECT


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA INTEGRITY CHECKER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDataIntegrityChecker:
    def test_audit_perfect_series(self, sample_hourly_candles):
        checker = DataIntegrityChecker()
        report = checker.audit(sample_hourly_candles, "1h")

        assert report.completeness_pct == 100.0
        assert report.gap_count == 0
        assert report.status == "EXCELLENT"
        assert report.is_acceptable_for_backtest is True

    def test_audit_with_gap(self, sample_hourly_candles):
        checker = DataIntegrityChecker()
        broken = sample_hourly_candles[:3] + sample_hourly_candles[7:]  # Missing 4 bars
        report = checker.audit(broken, "1h")

        assert report.completeness_pct < 100.0
        assert report.gap_count == 1
        assert report.total_missing_candles == 4

    def test_audit_invalid_ohlc_fails_acceptance(self, sample_hourly_candles):
        checker = DataIntegrityChecker()
        bad_candle = Candle(
            symbol="BTCUSDT",
            timeframe="1h",
            open_time=sample_hourly_candles[0].open_time,
            open=60000.0,
            high=55000.0,  # High < Open (invalid)
            low=54000.0,
            close=56000.0,
            volume=100.0,
            close_time=sample_hourly_candles[0].close_time,
        )
        report = checker.audit([bad_candle], "1h")
        assert report.invalid_ohlc_count > 0
        assert report.is_acceptable_for_backtest is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATABASE REPOSITORY TESTS (ASYNC SQLITE)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestCandleRepository:
    async def test_upsert_and_query(self, sample_hourly_candles):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(CandleModel.__table__.create)

        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

        async with session_factory() as session:
            repo = CandleRepository(session)

            # Insert batch
            count = await repo.upsert_many(sample_hourly_candles)
            assert count == 10

            # Count check
            total = await repo.get_candle_count("BTCUSDT", "1h")
            assert total == 10

            # Query candles chronologically
            queried = await repo.get_candles("BTCUSDT", "1h")
            assert len(queried) == 10
            assert queried[0].close == sample_hourly_candles[0].close

            # Date range
            min_t, max_t = await repo.get_date_range("BTCUSDT", "1h")
            assert min_t is not None and max_t is not None
            min_utc = min_t if min_t.tzinfo else min_t.replace(tzinfo=timezone.utc)
            max_utc = max_t if max_t.tzinfo else max_t.replace(tzinfo=timezone.utc)
            assert min_utc == sample_hourly_candles[0].open_time
            assert max_utc == sample_hourly_candles[-1].open_time

            # Latest candle
            latest = await repo.get_latest_candle("BTCUSDT", "1h")
            assert latest is not None
            latest_utc = latest.open_time if latest.open_time.tzinfo else latest.open_time.replace(tzinfo=timezone.utc)
            assert latest_utc == sample_hourly_candles[-1].open_time

            # Deduplication: re-insert same batch
            re_count = await repo.upsert_many(sample_hourly_candles)
            assert re_count == 10
            total_after = await repo.get_candle_count("BTCUSDT", "1h")
            assert total_after == 10  # No duplicate rows inserted

        await engine.dispose()
