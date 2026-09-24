"""Unit tests for technical indicators.

Each indicator is tested for:
1. Correct output shape
2. No look-ahead bias (output length matches input)
3. Value range validity (e.g., RSI in [0, 100])
4. NaN handling (leading NaN is expected for rolling windows)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_crypto_trader.indicators import momentum, trend, volatility, volume


@pytest.fixture
def price_series() -> pd.Series:
    np.random.seed(0)
    returns = np.random.normal(0.0001, 0.01, 300)
    prices = 50000 * np.cumprod(1 + returns)
    return pd.Series(prices)


@pytest.fixture
def ohlcv(price_series):
    n = len(price_series)
    close = price_series
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    return {
        "close": close,
        "high": pd.Series(high),
        "low": pd.Series(low),
        "volume": pd.Series(np.abs(np.random.normal(1000, 100, n))),
    }


class TestTrendIndicators:
    def test_ema_length(self, price_series):
        result = trend.ema(price_series, 20)
        assert len(result) == len(price_series)

    def test_ema_not_nan_at_end(self, price_series):
        result = trend.ema(price_series, 20)
        assert not pd.isna(result.iloc[-1])

    def test_sma_leading_nan(self, price_series):
        result = trend.sma(price_series, 20)
        assert pd.isna(result.iloc[0])  # Not enough data for first bar
        assert not pd.isna(result.iloc[-1])

    def test_macd_returns_three_series(self, price_series):
        macd_line, signal, hist = trend.macd(price_series)
        assert len(macd_line) == len(price_series)
        assert len(signal) == len(price_series)
        assert len(hist) == len(price_series)

    def test_donchian_upper_gte_lower(self, ohlcv):
        upper, mid, lower = trend.donchian_channel(ohlcv["high"], ohlcv["low"], 20)
        valid = ~(upper.isna() | lower.isna())
        assert (upper[valid] >= lower[valid]).all()

    def test_adx_positive(self, ohlcv):
        result = trend.adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all()


class TestMomentumIndicators:
    def test_rsi_range(self, price_series):
        result = momentum.rsi(price_series, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_length(self, price_series):
        result = momentum.rsi(price_series, 14)
        assert len(result) == len(price_series)

    def test_stochastic_range(self, ohlcv):
        k, d = momentum.stochastic(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        k_valid = k.dropna()
        assert (k_valid >= 0).all()
        assert (k_valid <= 100).all()

    def test_roc_length(self, price_series):
        result = momentum.roc(price_series, 10)
        assert len(result) == len(price_series)


class TestVolatilityIndicators:
    def test_atr_positive(self, ohlcv):
        result = volatility.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_bollinger_upper_gte_lower(self, price_series):
        upper, mid, lower = volatility.bollinger_bands(price_series, 20, 2.0)
        valid = ~(upper.isna() | lower.isna())
        assert (upper[valid] >= lower[valid]).all()

    def test_historical_volatility_positive(self, price_series):
        result = volatility.historical_volatility(price_series, 20)
        valid = result.dropna()
        assert (valid >= 0).all()


class TestVolumeIndicators:
    def test_obv_length(self, ohlcv):
        result = volume.obv(ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])

    def test_volume_ratio_around_one(self, ohlcv):
        result = volume.volume_ratio(ohlcv["volume"], 20)
        valid = result.dropna()
        # Rolling average ratio should have median near 1.0
        assert 0.5 < valid.median() < 2.0

    def test_mfi_range(self, ohlcv):
        result = volume.mfi(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"], 14
        )
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()
