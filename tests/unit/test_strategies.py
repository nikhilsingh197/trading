"""Automated tests for Milestone 5: Rule-Based Strategies & Ensemble.

Tests:
- BollingerRSIMeanReversion (oversold/overbought triggers, regime restrictions)
- DonchianBreakoutStrategy (channel breakouts, volume filters, ranging rejection)
- MultiTimeframeTrendStrategy (macro alignment, pullback entries, counter-trend guards)
- StrategyEnsemble (regime routing, high volatility protection, conflict resolution)
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai_crypto_trader.core.enums import Direction, MarketRegime, SignalAction
from ai_crypto_trader.core.interfaces import Signal
from ai_crypto_trader.strategies.base import BaseStrategy
from ai_crypto_trader.strategies.breakout import DonchianBreakoutStrategy
from ai_crypto_trader.strategies.ensemble import StrategyEnsemble
from ai_crypto_trader.strategies.mean_reversion import BollingerRSIMeanReversion
from ai_crypto_trader.strategies.multi_timeframe import MultiTimeframeTrendStrategy


@pytest.fixture
def base_df() -> pd.DataFrame:
    """Creates a 60-row DataFrame with standard technical indicator columns."""
    n = 60
    close = np.linspace(50000, 52000, n)
    return pd.DataFrame({
        "close": close,
        "high": close + 50,
        "low": close - 50,
        "open": close - 10,
        "volume": np.full(n, 1000.0),
        "vol_bb_upper": close + 1000,
        "vol_bb_mid": close,
        "vol_bb_lower": close - 1000,
        "vol_bb_width": np.full(n, 0.04),
        "mom_rsi_14": np.full(n, 50.0),
        "vol_atr_14": np.full(n, 300.0),
        "vol_ratio_20": np.full(n, 1.0),
        "trend_dc_upper": close + 800,
        "trend_dc_lower": close - 800,
        "trend_ema_21": close - 100,
        "trend_ema_50": close - 300,
        "trend_sma_200": close - 1000,
        "trend_adx": np.full(n, 25.0),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. MEAN REVERSION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMeanReversionStrategy:
    def test_long_trigger_on_oversold_lower_band(self, base_df):
        strat = BollingerRSIMeanReversion()
        df = base_df.copy()
        # Set price below lower BB and oversold RSI
        df.loc[df.index[-1], "close"] = df.loc[df.index[-1], "vol_bb_lower"] - 10
        df.loc[df.index[-1], "mom_rsi_14"] = 22.0

        sig = strat.generate_signal(df, MarketRegime.RANGING)
        assert sig.action == SignalAction.ENTER_LONG
        assert sig.direction == Direction.LONG
        assert sig.stop_loss < sig.entry_price
        assert sig.take_profit > sig.entry_price

    def test_short_trigger_on_overbought_upper_band(self, base_df):
        strat = BollingerRSIMeanReversion()
        df = base_df.copy()
        df.loc[df.index[-1], "close"] = df.loc[df.index[-1], "vol_bb_upper"] + 10
        df.loc[df.index[-1], "mom_rsi_14"] = 78.0

        sig = strat.generate_signal(df, MarketRegime.RANGING)
        assert sig.action == SignalAction.ENTER_SHORT
        assert sig.direction == Direction.SHORT
        assert sig.stop_loss > sig.entry_price
        assert sig.take_profit < sig.entry_price

    def test_rejection_in_strong_trending_regimes(self, base_df):
        strat = BollingerRSIMeanReversion()
        df = base_df.copy()
        # Extreme oversold setup
        df.loc[df.index[-1], "close"] = df.loc[df.index[-1], "vol_bb_lower"] - 10
        df.loc[df.index[-1], "mom_rsi_14"] = 20.0

        # Must NOT buy in strong downtrend (prevents catching knives)
        sig = strat.generate_signal(df, MarketRegime.STRONG_DOWNTREND)
        assert sig.action == SignalAction.NO_TRADE


# ─────────────────────────────────────────────────────────────────────────────
# 2. BREAKOUT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBreakoutStrategy:
    def test_long_breakout_trigger(self, base_df):
        strat = DonchianBreakoutStrategy()
        df = base_df.copy()

        # Previous Donchian high
        prev_dc_upper = df.loc[df.index[-2], "trend_dc_upper"]
        # Break above with volume surge
        df.loc[df.index[-1], "close"] = prev_dc_upper + 50
        df.loc[df.index[-1], "vol_ratio_20"] = 1.6
        df.loc[df.index[-1], "vol_bb_width"] = 0.08  # Expanding

        sig = strat.generate_signal(df, MarketRegime.STRONG_UPTREND)
        assert sig.action == SignalAction.ENTER_LONG
        assert sig.confidence >= 0.60

    def test_breakout_disabled_in_ranging_markets(self, base_df):
        strat = DonchianBreakoutStrategy()
        df = base_df.copy()
        df.loc[df.index[-1], "close"] = df.loc[df.index[-2], "trend_dc_upper"] + 50
        df.loc[df.index[-1], "vol_ratio_20"] = 2.0

        # Ranging market disables breakouts (avoids chop)
        sig = strat.generate_signal(df, MarketRegime.RANGING)
        assert sig.action == SignalAction.NO_TRADE


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTI-TIMEFRAME TREND TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTimeframeTrendStrategy:
    def test_bullish_macro_pullback_entry(self, base_df):
        strat = MultiTimeframeTrendStrategy()
        df = base_df.copy()

        # Macro bullish: close > sma200 and ema50 > sma200
        close = df.loc[df.index[-1], "close"]
        df.loc[df.index[-1], "trend_sma_200"] = close - 2000
        df.loc[df.index[-1], "trend_ema_50"] = close - 500
        # Price testing EMA21
        df.loc[df.index[-1], "trend_ema_21"] = close - 10
        # RSI turning up in pullback zone
        df.loc[df.index[-2], "mom_rsi_14"] = 44.0
        df.loc[df.index[-1], "mom_rsi_14"] = 48.0
        df.loc[df.index[-1], "trend_adx"] = 24.0

        sig = strat.generate_signal(df, MarketRegime.STRONG_UPTREND)
        assert sig.action == SignalAction.ENTER_LONG
        assert sig.direction == Direction.LONG

    def test_counter_trend_block(self, base_df):
        strat = MultiTimeframeTrendStrategy()
        df = base_df.copy()
        # Macro bearish (price below SMA 200)
        close = df.loc[df.index[-1], "close"]
        df.loc[df.index[-1], "trend_sma_200"] = close + 2000

        # Even with local bullish setup, MTF forbids long trades against macro trend
        sig = strat.generate_signal(df, MarketRegime.STRONG_UPTREND)
        assert sig.action == SignalAction.NO_TRADE


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRATEGY ENSEMBLE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class MockLongStrategy(BaseStrategy):
    def __init__(self, confidence: float = 0.8):
        super().__init__("mock_long", "MockLong", {})
        self.conf = confidence

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        return self._build_long_signal(df, self.conf, 0.02, 2.0, "mock buy", regime)


class MockShortStrategy(BaseStrategy):
    def __init__(self, confidence: float = 0.85):
        super().__init__("mock_short", "MockShort", {})
        self.conf = confidence

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        return self._build_short_signal(df, self.conf, 0.02, 2.0, "mock sell", regime)


class TestStrategyEnsemble:
    def test_halt_on_high_volatility(self, base_df):
        ensemble = StrategyEnsemble()
        sig = ensemble.generate_signal(base_df, MarketRegime.HIGH_VOLATILITY)
        assert sig.action == SignalAction.NO_TRADE
        assert "HIGH_VOLATILITY" in sig.reason

    def test_conflict_resolution_emits_no_trade(self, base_df):
        # One strategy votes LONG, another votes SHORT
        conflicting_ensemble = StrategyEnsemble(
            version_id="conflict_ens",
            strategies=[MockLongStrategy(0.8), MockShortStrategy(0.85)],
        )
        sig = conflicting_ensemble.generate_signal(base_df, MarketRegime.WEAK_UPTREND)
        # Ensemble must not take ambiguous opposing risk
        assert sig.action == SignalAction.NO_TRADE
        assert "ensemble conflict" in sig.reason

    def test_concordant_selection_highest_confidence(self, base_df):
        # Two strategies vote LONG with different confidence
        strat_a = MockLongStrategy(confidence=0.70)
        strat_b = MockLongStrategy(confidence=0.88)
        agreeing_ensemble = StrategyEnsemble(
            version_id="agree_ens",
            strategies=[strat_a, strat_b],
        )
        sig = agreeing_ensemble.generate_signal(base_df, MarketRegime.WEAK_UPTREND)
        assert sig.action == SignalAction.ENTER_LONG
        assert sig.confidence == 0.88
        assert "Ensemble consensus=2/2" in sig.reason
