"""Bollinger Bands + RSI Mean Reversion Strategy.

Designed for RANGING and LOW_VOLATILITY regimes.
Exploits statistical price extremes within stable trading ranges.

Entry Conditions (LONG):
  - Close <= Lower Bollinger Band (or Bollinger %B <= 0.05)
  - RSI(14) < 30 (oversold)
  - Regime is RANGING, LOW_VOLATILITY, or WEAK_UPTREND
  - STRICT: Never enter long in STRONG_DOWNTREND (avoids falling knives)

Entry Conditions (SHORT):
  - Close >= Upper Bollinger Band (or Bollinger %B >= 0.95)
  - RSI(14) > 70 (overbought)
  - Regime is RANGING, LOW_VOLATILITY, or WEAK_DOWNTREND
  - STRICT: Never enter short in STRONG_UPTREND (avoids shorting breakouts)

Exit Conditions:
  - Take Profit: Near middle Bollinger band (SMA 20) or 1.5x risk/reward
  - Stop Loss: 1.5x ATR beyond entry price
"""
from __future__ import annotations

import pandas as pd

from ai_crypto_trader.core.enums import MarketRegime, SignalAction
from ai_crypto_trader.core.interfaces import Signal
from ai_crypto_trader.strategies.base import BaseStrategy


class BollingerRSIMeanReversion(BaseStrategy):
    """Mean reversion strategy utilizing Bollinger Bands and RSI."""

    DEFAULT_PARAMS = {
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "atr_sl_multiplier": 1.5,
        "tp_ratio": 1.5,
    }

    def __init__(self, version_id: str = "v1", params: dict | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(version_id=version_id, name="Mean_Reversion", params=merged)

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        required_cols = [
            "vol_bb_upper", "vol_bb_mid", "vol_bb_lower",
            "mom_rsi_14", "vol_atr_14", "close",
        ]
        for col in required_cols:
            if col not in df.columns:
                return self._no_trade_signal(df, f"missing column: {col}", regime)

        if len(df) < 3:
            return self._no_trade_signal(df, "insufficient data", regime)

        # Mean reversion is strictly forbidden in strong directional trending or high volatility regimes
        if regime in {MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND, MarketRegime.HIGH_VOLATILITY}:
            return self._no_trade_signal(df, f"mean reversion blocked in regime={regime.value}", regime)

        latest = df.iloc[-1]
        close = float(latest["close"])
        bb_upper = float(latest["vol_bb_upper"])
        bb_lower = float(latest["vol_bb_lower"])
        bb_mid = float(latest["vol_bb_mid"])
        rsi = float(latest["mom_rsi_14"])
        atr = float(latest["vol_atr_14"])

        p = self._params

        # LONG: price at or below lower band and oversold
        if (
            close <= bb_lower
            and rsi <= p["rsi_oversold"]
            and regime in {MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY, MarketRegime.WEAK_UPTREND}
        ):
            stop_distance = p["atr_sl_multiplier"] * atr
            stop_loss_pct = stop_distance / close
            confidence = min(0.9, (p["rsi_oversold"] - rsi) / 20.0 + 0.6)

            return self._build_long_signal(
                df=df,
                confidence=confidence,
                stop_loss_pct=stop_loss_pct,
                tp_ratio=p["tp_ratio"],
                reason=(
                    f"Mean reversion long: close={close:.2f} <= lower_bb={bb_lower:.2f}, "
                    f"RSI={rsi:.1f} <= {p['rsi_oversold']}, regime={regime.value}"
                ),
                regime=regime,
            )

        # SHORT: price at or above upper band and overbought
        if (
            close >= bb_upper
            and rsi >= p["rsi_overbought"]
            and regime in {MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY, MarketRegime.WEAK_DOWNTREND}
        ):
            stop_distance = p["atr_sl_multiplier"] * atr
            stop_loss_pct = stop_distance / close
            confidence = min(0.9, (rsi - p["rsi_overbought"]) / 20.0 + 0.6)

            return self._build_short_signal(
                df=df,
                confidence=confidence,
                stop_loss_pct=stop_loss_pct,
                tp_ratio=p["tp_ratio"],
                reason=(
                    f"Mean reversion short: close={close:.2f} >= upper_bb={bb_upper:.2f}, "
                    f"RSI={rsi:.1f} >= {p['rsi_overbought']}, regime={regime.value}"
                ),
                regime=regime,
            )

        return self._no_trade_signal(df, "no mean reversion trigger", regime)
