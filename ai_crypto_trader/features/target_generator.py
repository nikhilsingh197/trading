"""Target label generation for ML models.

CRITICAL: Targets use future data, so they can ONLY be used for training.
They must NEVER appear in the live feature vector used for inference.

This module is used exclusively during offline training/research.
It is deliberately separated from feature_engine.py to prevent
accidental contamination of live features with future information.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class TargetConfig:
    """Configuration for target generation."""
    horizon: int = 20                    # Forward-looking candles
    tp_pct: float = 2.0                  # Take-profit percentage
    sl_pct: float = 1.0                  # Stop-loss percentage
    min_return_pct: float = 0.5          # Minimum return to be labeled positive


class TargetGenerator:
    """Generates ML training targets from price data.

    Target types:
    1. future_return: Raw % return after N candles
    2. binary_direction: 1 if close[t+N] > close[t] else 0
    3. tp_before_sl: 1 if TP is hit before SL within horizon
    4. regime: Market regime label (from RegimeDetector)
    5. volatility_forecast: Realized volatility of next N candles
    """

    def add_targets(self, df: pd.DataFrame, config: TargetConfig) -> pd.DataFrame:
        """Add target columns to df.

        WARNING: These columns contain FUTURE information.
        Ensure they are DROPPED before inference.
        The columns are prefixed with 'target_' for easy identification.

        Args:
            df: OHLCV DataFrame with features already computed.
            config: Target configuration.

        Returns:
            DataFrame with target columns added (future data embedded).
        """
        result = df.copy()
        h = config.horizon
        close = result["close"]

        # 1. Future return
        result["target_future_return"] = close.shift(-h) / close - 1

        # 2. Binary direction
        result["target_direction"] = (result["target_future_return"] > 0).astype(int)

        # 3. Significant move (return exceeds min threshold)
        result["target_significant_up"] = (
            result["target_future_return"] > config.min_return_pct / 100
        ).astype(int)
        result["target_significant_down"] = (
            result["target_future_return"] < -config.min_return_pct / 100
        ).astype(int)

        # 4. TP before SL within horizon
        result["target_tp_before_sl"] = self._compute_tp_sl_labels(
            result, config.tp_pct / 100, config.sl_pct / 100, h
        )

        # 5. Volatility forecast
        log_ret = np.log(close / close.shift(1))
        result["target_future_volatility"] = (
            log_ret.shift(-h).rolling(h).std() * np.sqrt(365)
        )

        log.debug(
            "targets_generated",
            horizon=h,
            tp_pct=config.tp_pct,
            sl_pct=config.sl_pct,
        )
        return result

    @staticmethod
    def _compute_tp_sl_labels(
        df: pd.DataFrame,
        tp_pct: float,
        sl_pct: float,
        horizon: int,
    ) -> pd.Series:
        """Compute whether TP is hit before SL within horizon.

        Returns 1 if TP is reached before SL, 0 otherwise.
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)
        labels = np.zeros(n, dtype=int)

        for i in range(n - horizon):
            entry = close[i]
            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)

            for j in range(i + 1, min(i + horizon + 1, n)):
                if high[j] >= tp_price:
                    labels[i] = 1
                    break
                if low[j] <= sl_price:
                    labels[i] = 0
                    break

        # Last 'horizon' rows have no forward data — mark as NaN
        return pd.Series(labels, index=df.index).where(
            pd.Series(range(n), index=df.index) < n - horizon,
            other=np.nan,
        )
