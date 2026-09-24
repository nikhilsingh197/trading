"""Volume-based indicators."""
from __future__ import annotations

import pandas as pd
import numpy as np


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Rolling VWAP (Volume-Weighted Average Price)."""
    typical_price = (high + low + close) / 3
    return (
        (typical_price * volume).rolling(period).sum() /
        volume.rolling(period).sum()
    )


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Simple Moving Average."""
    return volume.rolling(period).mean()


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume ratio: current / rolling average."""
    avg = volume_sma(volume, period)
    return volume / (avg + 1e-9)


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    direction = typical_price.diff().fillna(0)
    pos_mf = money_flow.where(direction > 0, 0).rolling(period).sum()
    neg_mf = money_flow.where(direction <= 0, 0).rolling(period).sum()
    mfr = pos_mf / (neg_mf + 1e-9)
    return 100 - 100 / (1 + mfr)


def accumulation_distribution(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Accumulation/Distribution Line."""
    clv = ((close - low) - (high - close)) / (high - low + 1e-9)
    return (clv * volume).cumsum()
