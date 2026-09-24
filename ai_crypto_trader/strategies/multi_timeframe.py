"""Multi-Timeframe Trend Confirmation Strategy.

Aligns macro trend direction with lower-timeframe pullbacks.
Core rule: Never trade against the macro trend.

Macro Trend Identification:
  - Bullish: Price > SMA 200 AND EMA 50 > EMA 200
  - Bearish: Price < SMA 200 AND EMA 50 < EMA 200

Pullback Trigger:
  - Long: In Bullish Macro, price pulls back to test EMA 21 / EMA 50 support,
          RSI pulls back to 40-55 and rebounds, ADX >= 18.
  - Short: In Bearish Macro, price rallies to test EMA 21 / EMA 50 resistance,
           RSI rebounds from 45-60, ADX >= 18.

Exit Conditions:
  - Stop Loss: 1.8x ATR below pullback swing low
  - Take Profit: 2.0x Risk-to-Reward ratio
"""
from __future__ import annotations

import pandas as pd

from ai_crypto_trader.core.enums import MarketRegime
from ai_crypto_trader.core.interfaces import Signal
from ai_crypto_trader.strategies.base import BaseStrategy


class MultiTimeframeTrendStrategy(BaseStrategy):
    """Trend-following strategy requiring macro alignment and pullback entry."""

    DEFAULT_PARAMS = {
        "adx_min": 18.0,
        "atr_sl_multiplier": 1.8,
        "tp_ratio": 2.0,
    }

    def __init__(self, version_id: str = "v1", params: dict | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(version_id=version_id, name="MTF_Trend", params=merged)

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        required_cols = [
            "trend_ema_21", "trend_ema_50", "trend_sma_200",
            "trend_adx", "mom_rsi_14", "vol_atr_14", "close", "low", "high",
        ]
        for col in required_cols:
            if col not in df.columns:
                return self._no_trade_signal(df, f"missing column: {col}", regime)

        if len(df) < 50:
            return self._no_trade_signal(df, "insufficient data for multi-timeframe lookback", regime)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(latest["close"])
        ema_21 = float(latest["trend_ema_21"])
        ema_50 = float(latest["trend_ema_50"])
        sma_200 = float(latest["trend_sma_200"])
        adx = float(latest["trend_adx"])
        rsi = float(latest["mom_rsi_14"])
        prev_rsi = float(prev["mom_rsi_14"])
        atr = float(latest["vol_atr_14"])

        p = self._params

        # Macro trend alignment
        macro_bullish = close > sma_200 and ema_50 > sma_200
        macro_bearish = close < sma_200 and ema_50 < sma_200

        # ADX trend strength filter
        trend_strong = adx >= p["adx_min"]

        # LONG PULLBACK: Macro bullish, price holding above EMA 50, touching EMA 21, RSI turning up from 40-55
        if (
            macro_bullish
            and trend_strong
            and close >= ema_50
            and abs(close - ema_21) / close <= 0.015  # Near EMA 21
            and 38.0 <= rsi <= 58.0
            and rsi >= prev_rsi  # Rebounding
            and regime in {MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND}
        ):
            stop_distance = p["atr_sl_multiplier"] * atr
            stop_loss_pct = stop_distance / close
            confidence = min(0.90, 0.65 + (adx - p["adx_min"]) * 0.01)

            return self._build_long_signal(
                df=df,
                confidence=confidence,
                stop_loss_pct=stop_loss_pct,
                tp_ratio=p["tp_ratio"],
                reason=(
                    f"MTF Bullish pullback: close={close:.2f} > sma200={sma_200:.2f}, "
                    f"ema21={ema_21:.2f}, RSI rebound={rsi:.1f}, ADX={adx:.1f}, regime={regime.value}"
                ),
                regime=regime,
            )

        # SHORT PULLBACK: Macro bearish, price below EMA 50, near EMA 21, RSI turning down from 45-62
        if (
            macro_bearish
            and trend_strong
            and close <= ema_50
            and abs(close - ema_21) / close <= 0.015
            and 42.0 <= rsi <= 62.0
            and rsi <= prev_rsi  # Turning lower
            and regime in {MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND}
        ):
            stop_distance = p["atr_sl_multiplier"] * atr
            stop_loss_pct = stop_distance / close
            confidence = min(0.90, 0.65 + (adx - p["adx_min"]) * 0.01)

            return self._build_short_signal(
                df=df,
                confidence=confidence,
                stop_loss_pct=stop_loss_pct,
                tp_ratio=p["tp_ratio"],
                reason=(
                    f"MTF Bearish rally: close={close:.2f} < sma200={sma_200:.2f}, "
                    f"ema21={ema_21:.2f}, RSI downturn={rsi:.1f}, ADX={adx:.1f}, regime={regime.value}"
                ),
                regime=regime,
            )

        return self._no_trade_signal(df, "no MTF alignment setup", regime)
