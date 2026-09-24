"""Donchian Volatility Breakout Strategy.

Designed for regime transitions from LOW_VOLATILITY into STRONG_UPTREND / STRONG_DOWNTREND.
Captures explosive directional momentum following volatility compression.

Entry Conditions (LONG):
  - Close >= Donchian Upper Channel (20-period high)
  - Volume Ratio >= 1.2x (volume surge confirming breakout)
  - Bollinger Bandwidth > 20-period moving average of Bandwidth (volatility expansion)
  - Regime is NOT RANGING (avoids chop and false breakouts)

Entry Conditions (SHORT):
  - Close <= Donchian Lower Channel (20-period low)
  - Volume Ratio >= 1.2x
  - Bollinger Bandwidth expanding
  - Regime is NOT RANGING

Exit Conditions:
  - Stop Loss: 2.0x ATR
  - Take Profit: 2.5x Risk-to-Reward ratio
"""
from __future__ import annotations

import pandas as pd

from ai_crypto_trader.core.enums import MarketRegime
from ai_crypto_trader.core.interfaces import Signal
from ai_crypto_trader.strategies.base import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):
    """Channel breakout strategy confirmed by volume surge and volatility expansion."""

    DEFAULT_PARAMS = {
        "volume_ratio_min": 1.20,
        "atr_sl_multiplier": 2.0,
        "tp_ratio": 2.5,
    }

    def __init__(self, version_id: str = "v1", params: dict | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(version_id=version_id, name="Breakout", params=merged)

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        required_cols = [
            "trend_dc_upper", "trend_dc_lower", "vol_ratio_20",
            "vol_bb_width", "vol_atr_14", "close", "high", "low",
        ]
        for col in required_cols:
            if col not in df.columns:
                return self._no_trade_signal(df, f"missing column: {col}", regime)

        if len(df) < 25:
            return self._no_trade_signal(df, "insufficient data for breakout lookback", regime)

        # Breakouts in ranging markets have high false-positive rate
        if regime == MarketRegime.RANGING:
            return self._no_trade_signal(df, "breakouts disabled in ranging regime", regime)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(latest["close"])
        dc_upper = float(latest["trend_dc_upper"])
        dc_lower = float(latest["trend_dc_lower"])
        prev_dc_upper = float(prev["trend_dc_upper"])
        prev_dc_lower = float(prev["trend_dc_lower"])

        vol_ratio = float(latest.get("vol_ratio_20", 1.0))
        bb_width = float(latest["vol_bb_width"])
        mean_bb_width = float(df["vol_bb_width"].iloc[-20:].mean())
        atr = float(latest["vol_atr_14"])

        p = self._params
        vol_expansion = bb_width >= mean_bb_width
        vol_surge = vol_ratio >= p["volume_ratio_min"]

        # LONG BREAKOUT
        if close >= prev_dc_upper and vol_surge and vol_expansion:
            if regime in {MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND, MarketRegime.LOW_VOLATILITY}:
                stop_distance = p["atr_sl_multiplier"] * atr
                stop_loss_pct = stop_distance / close
                confidence = min(0.85, 0.60 + (vol_ratio - 1.0) * 0.1)

                return self._build_long_signal(
                    df=df,
                    confidence=confidence,
                    stop_loss_pct=stop_loss_pct,
                    tp_ratio=p["tp_ratio"],
                    reason=(
                        f"Donchian long breakout: close={close:.2f} >= dc_upper={prev_dc_upper:.2f}, "
                        f"vol_ratio={vol_ratio:.2f}x, bandwidth expanding={vol_expansion}, regime={regime.value}"
                    ),
                    regime=regime,
                )

        # SHORT BREAKOUT
        if close <= prev_dc_lower and vol_surge and vol_expansion:
            if regime in {MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND, MarketRegime.LOW_VOLATILITY}:
                stop_distance = p["atr_sl_multiplier"] * atr
                stop_loss_pct = stop_distance / close
                confidence = min(0.85, 0.60 + (vol_ratio - 1.0) * 0.1)

                return self._build_short_signal(
                    df=df,
                    confidence=confidence,
                    stop_loss_pct=stop_loss_pct,
                    tp_ratio=p["tp_ratio"],
                    reason=(
                        f"Donchian short breakout: close={close:.2f} <= dc_lower={prev_dc_lower:.2f}, "
                        f"vol_ratio={vol_ratio:.2f}x, bandwidth expanding={vol_expansion}, regime={regime.value}"
                    ),
                    regime=regime,
                )

        return self._no_trade_signal(df, "no breakout condition met", regime)
