"""Comprehensive feature engineering pipeline.

Critical design constraints:
1. NO look-ahead bias: features at time T may only use data up to time T.
2. Features are computed on a copy of the input DataFrame; original is not mutated.
3. All features are named consistently for traceability.
4. NaN rows at the beginning (due to rolling windows) are preserved — callers must handle.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from ai_crypto_trader.core.interfaces import FeatureEngineABC
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.indicators import trend, momentum, volatility, volume

log = get_logger(__name__)


class FeatureEngine(FeatureEngineABC):
    """Computes the full feature set from OHLCV data.

    Each feature group is computed independently for modularity.
    Features are prefixed by category (e.g., trend_, vol_, mom_, vol_).
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features on a copy of the input DataFrame.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume].
                Must be indexed by datetime (UTC) and sorted ascending.

        Returns:
            New DataFrame with all feature columns appended.
            Original df is NOT modified.
        """
        result = df.copy()

        # Validate required columns
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"FeatureEngine: missing columns {missing}")

        result = self._add_trend_features(result)
        result = self._add_momentum_features(result)
        result = self._add_volatility_features(result)
        result = self._add_volume_features(result)
        result = self._add_structure_features(result)
        result = self._add_return_features(result)

        log.debug("features_computed", n_features=len(result.columns) - len(df.columns))
        return result

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # EMAs
        for p in [9, 21, 50, 100, 200]:
            df[f"trend_ema_{p}"] = trend.ema(close, p)

        # SMAs
        for p in [20, 50, 200]:
            df[f"trend_sma_{p}"] = trend.sma(close, p)

        # EMA crossovers (ratio, not raw price)
        df["trend_ema9_ema21_ratio"] = df["trend_ema_9"] / df["trend_ema_21"]
        df["trend_ema21_ema50_ratio"] = df["trend_ema_21"] / df["trend_ema_50"]
        df["trend_ema50_ema200_ratio"] = df["trend_ema_50"] / df["trend_ema_200"]

        # MACD
        macd_line, signal_line, histogram = trend.macd(close)
        df["trend_macd"] = macd_line
        df["trend_macd_signal"] = signal_line
        df["trend_macd_hist"] = histogram

        # ADX (trend strength)
        df["trend_adx"] = trend.adx(high, low, close, 14)

        # Donchian channels
        dc_upper, dc_mid, dc_lower = trend.donchian_channel(high, low, 20)
        df["trend_dc_upper"] = dc_upper
        df["trend_dc_mid"] = dc_mid
        df["trend_dc_lower"] = dc_lower
        df["trend_dc_pct"] = (close - dc_lower) / (dc_upper - dc_lower + 1e-9)

        # Price relative to moving averages (normalized)
        df["trend_close_vs_sma50"] = close / df["trend_sma_50"] - 1
        df["trend_close_vs_sma200"] = close / df["trend_sma_200"] - 1
        df["trend_close_vs_ema21"] = close / df["trend_ema_21"] - 1

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        df["mom_rsi_14"] = momentum.rsi(close, 14)
        df["mom_rsi_7"] = momentum.rsi(close, 7)

        # Stochastic
        k, d = momentum.stochastic(high, low, close, 14, 3)
        df["mom_stoch_k"] = k
        df["mom_stoch_d"] = d
        df["mom_stoch_kd_diff"] = k - d

        # ROC
        for p in [5, 10, 20]:
            df[f"mom_roc_{p}"] = momentum.roc(close, p)

        # Momentum
        df["mom_momentum_10"] = momentum.momentum(close, 10)

        # Williams %R
        df["mom_williams_r"] = momentum.williams_r(high, low, close, 14)

        # CCI
        df["mom_cci_20"] = momentum.cci(high, low, close, 20)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ATR
        for p in [7, 14, 21]:
            df[f"vol_atr_{p}"] = volatility.atr(high, low, close, p)

        # Normalized ATR (ATR / Close)
        df["vol_atr_pct"] = df["vol_atr_14"] / close

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = volatility.bollinger_bands(close, 20, 2.0)
        df["vol_bb_upper"] = bb_upper
        df["vol_bb_mid"] = bb_mid
        df["vol_bb_lower"] = bb_lower
        df["vol_bb_width"] = volatility.bollinger_bandwidth(bb_upper, bb_lower, bb_mid)
        df["vol_bb_pct_b"] = volatility.bollinger_pct_b(close, bb_upper, bb_lower)

        # Historical volatility
        for p in [10, 20, 30]:
            df[f"vol_hv_{p}"] = volatility.historical_volatility(close, p)

        # High-Low range
        df["vol_hl_range"] = high - low
        df["vol_hl_range_pct"] = (high - low) / (close + 1e-9)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        df["vol_obv"] = volume.obv(close, vol)
        df["vol_vwap_14"] = volume.vwap(high, low, close, vol, 14)
        df["vol_ratio_20"] = volume.volume_ratio(vol, 20)
        df["vol_mfi_14"] = volume.mfi(high, low, close, vol, 14)
        df["vol_ad"] = volume.accumulation_distribution(high, low, close, vol)

        # Volume z-score
        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std()
        df["vol_zscore_20"] = (vol - vol_mean) / (vol_std + 1e-9)

        # Close vs VWAP
        df["vol_close_vs_vwap"] = close / df["vol_vwap_14"] - 1

        return df

    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market structure features (support/resistance, candle patterns)."""
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]

        # Candle body and wick ratios
        body = (close - open_).abs()
        total_range = high - low + 1e-9
        df["struct_body_ratio"] = body / total_range
        df["struct_upper_wick"] = (high - close.clip(lower=open_)) / total_range
        df["struct_lower_wick"] = (open_.clip(upper=close) - low) / total_range
        df["struct_direction"] = (close > open_).astype(int)

        # Local highs/lows (rolling)
        df["struct_roll_high_20"] = high.rolling(20).max()
        df["struct_roll_low_20"] = low.rolling(20).min()
        df["struct_pct_from_high_20"] = close / df["struct_roll_high_20"] - 1
        df["struct_pct_from_low_20"] = close / df["struct_roll_low_20"] - 1

        # Consecutive up/down candles
        df["struct_up_streak"] = (
            (close > close.shift(1)).astype(int)
            .groupby((close <= close.shift(1)).cumsum())
            .cumsum()
        )

        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged return features for ML models."""
        close = df["close"]

        # Past returns (lagged, so no look-ahead bias)
        for p in [1, 3, 5, 10, 20]:
            df[f"ret_lag_{p}"] = close.pct_change(p).shift(1)  # shift(1) = past return

        # Rolling return statistics
        log_ret = np.log(close / close.shift(1))
        df["ret_log_1"] = log_ret.shift(1)  # Most recent completed candle
        df["ret_std_20"] = log_ret.rolling(20).std().shift(1)
        df["ret_skew_50"] = log_ret.rolling(50).skew().shift(1)
        df["ret_kurt_50"] = log_ret.rolling(50).kurt().shift(1)

        return df
