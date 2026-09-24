"""Shared pytest fixtures."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate a realistic OHLCV DataFrame for testing.

    Returns 500 bars of synthetic BTC-like price data.
    Uses a geometric random walk so price is always positive.
    """
    n = 500
    np.random.seed(42)

    # Geometric random walk
    returns = np.random.normal(0.0001, 0.02, n)  # Small positive drift
    close = 50000.0 * np.cumprod(1 + returns)

    # Realistic OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close.copy()
    open_[1:] = close[:-1]  # Open = previous close
    volume = np.abs(np.random.normal(1000, 200, n))

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = [start + timedelta(hours=i) for i in range(n)]

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    # Ensure high >= low and high >= open/close
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


@pytest.fixture
def sample_candle():
    """Return a single valid Candle for testing."""
    from ai_crypto_trader.core.interfaces import Candle
    now = datetime.now(timezone.utc)
    return Candle(
        symbol="BTCUSDT",
        timeframe="1h",
        open_time=now - timedelta(hours=1),
        open=50000.0,
        high=51000.0,
        low=49500.0,
        close=50800.0,
        volume=1250.5,
        close_time=now,
    )


@pytest.fixture
def feature_df(sample_ohlcv_df) -> pd.DataFrame:
    """OHLCV with features pre-computed."""
    from ai_crypto_trader.features.feature_engine import FeatureEngine
    engine = FeatureEngine()
    return engine.compute(sample_ohlcv_df)
