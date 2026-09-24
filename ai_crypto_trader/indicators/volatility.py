"""Volatility indicators."""
from __future__ import annotations

import pandas as pd
import numpy as np

from ai_crypto_trader.indicators.trend import ema, true_range


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, middle, lower."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def bollinger_bandwidth(upper: pd.Series, lower: pd.Series, middle: pd.Series) -> pd.Series:
    """Bollinger Bandwidth (normalized volatility)."""
    return (upper - lower) / (middle + 1e-9) * 100


def bollinger_pct_b(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    """Bollinger %B: position within the bands."""
    return (close - lower) / (upper - lower + 1e-9)


def historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Annualized historical volatility from log returns."""
    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(period).std()
    return rolling_std * np.sqrt(365)  # Crypto: 365-day annualization


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels: upper, middle, lower."""
    middle = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower
