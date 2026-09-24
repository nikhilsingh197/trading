"""Rule-based market regime detector.

This is the primary regime detector used in production.
ML-based detectors are available as experimental alternatives.

Regimes:
    STRONG_UPTREND:   ADX > 25, close > EMA50 > EMA200, RSI > 55
    WEAK_UPTREND:     ADX 15-25, close > EMA50, RSI 45-65
    RANGING:          ADX < 20, price within Bollinger Band
    HIGH_VOLATILITY:  ATR% > 2x rolling average
    LOW_VOLATILITY:   ATR% < 0.5x rolling average
    WEAK_DOWNTREND:   ADX 15-25, close < EMA50, RSI 35-55
    STRONG_DOWNTREND: ADX > 25, close < EMA50 < EMA200, RSI < 45
    UNKNOWN:          Conflicting signals or insufficient data
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from ai_crypto_trader.core.enums import MarketRegime
from ai_crypto_trader.core.interfaces import RegimeDetectorABC
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


class RuleBasedRegimeDetector(RegimeDetectorABC):
    """Classifies market regime using technical rules."""

    def __init__(
        self,
        adx_strong_threshold: float = 25.0,
        adx_weak_threshold: float = 15.0,
        volatility_high_multiplier: float = 2.0,
        volatility_low_multiplier: float = 0.5,
        volatility_lookback: int = 50,
    ) -> None:
        self._adx_strong = adx_strong_threshold
        self._adx_weak = adx_weak_threshold
        self._vol_high_mult = volatility_high_multiplier
        self._vol_low_mult = volatility_low_multiplier
        self._vol_lookback = volatility_lookback

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Classify the regime from the latest row of df.

        Args:
            df: Feature-enriched DataFrame. Must have columns:
                trend_adx, trend_ema_50, trend_ema_200, mom_rsi_14,
                vol_atr_pct, vol_bb_width.

        Returns:
            MarketRegime enum value.
        """
        required = [
            "trend_adx", "trend_ema_50", "trend_ema_200",
            "mom_rsi_14", "vol_atr_pct",
        ]
        for col in required:
            if col not in df.columns:
                log.warning("regime_missing_feature", column=col)
                return MarketRegime.UNKNOWN

        latest = df.iloc[-1]

        try:
            return self._classify(df, latest)
        except Exception as exc:
            log.error("regime_detection_error", error=str(exc))
            return MarketRegime.UNKNOWN

    def _classify(self, df: pd.DataFrame, row: pd.Series) -> MarketRegime:
        adx = float(row.get("trend_adx", 0))
        ema50 = float(row.get("trend_ema_50", 0))
        ema200 = float(row.get("trend_ema_200", 0))
        rsi = float(row.get("mom_rsi_14", 50))
        close = float(row.get("close", 0))
        atr_pct = float(row.get("vol_atr_pct", 0))

        # Volatility regime check (takes priority over trend/range)
        if "vol_atr_pct" in df.columns and len(df) >= self._vol_lookback:
            avg_atr_pct = df["vol_atr_pct"].tail(self._vol_lookback).mean()
            if pd.notna(avg_atr_pct) and avg_atr_pct > 0:
                if atr_pct > avg_atr_pct * self._vol_high_mult:
                    return MarketRegime.HIGH_VOLATILITY
                if atr_pct < avg_atr_pct * self._vol_low_mult:
                    return MarketRegime.LOW_VOLATILITY

        # Trend classification
        price_above_ema50 = close > ema50
        price_above_ema200 = close > ema200
        ema50_above_ema200 = ema50 > ema200

        if adx > self._adx_strong:
            if price_above_ema50 and ema50_above_ema200 and rsi > 55:
                return MarketRegime.STRONG_UPTREND
            if not price_above_ema50 and not ema50_above_ema200 and rsi < 45:
                return MarketRegime.STRONG_DOWNTREND

        if self._adx_weak <= adx <= self._adx_strong:
            if price_above_ema50 and rsi > 45:
                return MarketRegime.WEAK_UPTREND
            if not price_above_ema50 and rsi < 55:
                return MarketRegime.WEAK_DOWNTREND

        if adx < 20:
            return MarketRegime.RANGING

        return MarketRegime.UNKNOWN
