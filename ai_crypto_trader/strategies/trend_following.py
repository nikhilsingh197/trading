"""EMA crossover trend-following strategy.

This strategy is intentionally simple. Complex strategies are not
automatically better — simpler strategies are easier to validate,
have fewer parameters to overfit, and degrade more gracefully.

Entry conditions (LONG):
  - EMA9 crosses above EMA21
  - ADX > 20 (confirmed trend)
  - RSI between 45 and 70 (not overbought)
  - Regime is STRONG_UPTREND or WEAK_UPTREND

Exit:
  - ATR-based stop loss (1.5x ATR below entry)
  - Take profit at 2:1 risk/reward

This strategy does NOT trade in RANGING or HIGH_VOLATILITY regimes.
"""
from __future__ import annotations

import pandas as pd

from ai_crypto_trader.core.enums import MarketRegime, SignalAction
from ai_crypto_trader.core.exceptions import InsufficientDataError
from ai_crypto_trader.core.interfaces import Signal
from ai_crypto_trader.strategies.base import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    """Simple EMA crossover trend-following strategy."""

    DEFAULT_PARAMS = {
        "fast_ema": 9,
        "slow_ema": 21,
        "adx_min": 20,
        "rsi_min": 45,
        "rsi_max": 70,
        "atr_sl_multiplier": 1.5,
        "tp_ratio": 2.0,  # Risk:reward ratio
    }

    def __init__(self, version_id: str = "v1", params: dict | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(version_id=version_id, name="EMA_Crossover", params=merged)

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        """Generate entry/no-trade signal for trend-following."""
        required_cols = [
            "trend_ema_9", "trend_ema_21", "trend_adx",
            "mom_rsi_14", "vol_atr_14",
        ]
        for col in required_cols:
            if col not in df.columns:
                return self._no_trade_signal(df, f"missing column: {col}", regime)

        if len(df) < 3:
            return self._no_trade_signal(df, "insufficient data", regime)

        # Only trade in trending regimes
        if regime in {MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY, MarketRegime.UNKNOWN}:
            return self._no_trade_signal(df, f"regime={regime.value} not tradeable", regime)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        fast_now = float(latest["trend_ema_9"])
        slow_now = float(latest["trend_ema_21"])
        fast_prev = float(prev["trend_ema_9"])
        slow_prev = float(prev["trend_ema_21"])
        adx = float(latest["trend_adx"])
        rsi = float(latest["mom_rsi_14"])
        atr = float(latest["vol_atr_14"])
        close = float(latest["close"])

        p = self._params

        # LONG: fast EMA crosses above slow EMA
        if (
            fast_prev <= slow_prev
            and fast_now > slow_now
            and adx > p["adx_min"]
            and p["rsi_min"] <= rsi <= p["rsi_max"]
            and regime in {MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND}
        ):
            stop_distance = p["atr_sl_multiplier"] * atr
            stop_loss_pct = stop_distance / close
            return self._build_long_signal(
                df=df,
                confidence=min(0.9, adx / 50),
                stop_loss_pct=stop_loss_pct,
                tp_ratio=p["tp_ratio"],
                reason=(
                    f"EMA{p['fast_ema']} crossed above EMA{p['slow_ema']}, "
                    f"ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime.value}"
                ),
                regime=regime,
            )

        # SHORT: fast EMA crosses below slow EMA
        if (
            fast_prev >= slow_prev
            and fast_now < slow_now
            and adx > p["adx_min"]
            and p["rsi_min"] <= (100 - rsi) <= p["rsi_max"]  # Mirror RSI for shorts
            and regime in {MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND}
        ):
            stop_distance = p["atr_sl_multiplier"] * atr
            stop_loss_pct = stop_distance / close
            return self._build_short_signal(
                df=df,
                confidence=min(0.9, adx / 50),
                stop_loss_pct=stop_loss_pct,
                tp_ratio=p["tp_ratio"],
                reason=(
                    f"EMA{p['fast_ema']} crossed below EMA{p['slow_ema']}, "
                    f"ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime.value}"
                ),
                regime=regime,
            )

        return self._no_trade_signal(df, "no crossover signal", regime)
