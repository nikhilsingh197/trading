"""Strategy Ensemble & Regime-Adaptive Dispatcher.

Combines multiple underlying strategies and routes execution dynamically
based on detected market regime:
- RANGING / LOW_VOLATILITY      -> Mean Reversion & Volatility Compression
- STRONG_UPTREND / STRONG_DOWN  -> Trend Following, MTF Pullbacks & Breakouts
- HIGH_VOLATILITY               -> Strict risk gating (Stand aside or require >= 0.85 confidence)

Conflict Resolution:
- If sub-strategies disagree on trade direction (e.g. LONG vs SHORT),
  the ensemble emits NO_TRADE to protect capital.
- If sub-strategies agree, the highest conviction signal is selected.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ai_crypto_trader.core.enums import Direction, MarketRegime, SignalAction
from ai_crypto_trader.core.interfaces import Signal, StrategyABC
from ai_crypto_trader.strategies.base import BaseStrategy
from ai_crypto_trader.strategies.breakout import DonchianBreakoutStrategy
from ai_crypto_trader.strategies.mean_reversion import BollingerRSIMeanReversion
from ai_crypto_trader.strategies.multi_timeframe import MultiTimeframeTrendStrategy
from ai_crypto_trader.strategies.trend_following import EMACrossoverStrategy


class StrategyEnsemble(BaseStrategy):
    """Regime-adaptive ensemble combining trend, mean reversion, and breakout strategies."""

    def __init__(
        self,
        version_id: str = "ens_v1",
        strategies: Optional[list[StrategyABC]] = None,
        params: dict | None = None,
    ) -> None:
        super().__init__(version_id=version_id, name="Ensemble", params=params or {})

        if strategies is not None:
            self._strategies = strategies
        else:
            self._strategies = [
                EMACrossoverStrategy(version_id=f"{version_id}_ema"),
                BollingerRSIMeanReversion(version_id=f"{version_id}_mr"),
                DonchianBreakoutStrategy(version_id=f"{version_id}_bo"),
                MultiTimeframeTrendStrategy(version_id=f"{version_id}_mtf"),
            ]

    @property
    def sub_strategies(self) -> list[StrategyABC]:
        return self._strategies

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        # High volatility defensive gate: protect capital during chaotic moves
        if regime == MarketRegime.HIGH_VOLATILITY:
            return self._no_trade_signal(df, "ensemble halted: HIGH_VOLATILITY safety gate", regime)

        # Collect signals from all eligible sub-strategies
        active_signals: list[Signal] = []

        for strat in self._strategies:
            # Regime gating for sub-strategies:
            if isinstance(strat, BollingerRSIMeanReversion):
                if regime in {MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND}:
                    continue  # Do not run mean reversion in strong trends

            if isinstance(strat, (DonchianBreakoutStrategy, EMACrossoverStrategy)):
                if regime == MarketRegime.RANGING:
                    continue  # Do not run trend/breakout in ranging markets

            sig = strat.generate_signal(df, regime)
            if sig.action != SignalAction.NO_TRADE:
                active_signals.append(sig)

        if not active_signals:
            return self._no_trade_signal(df, f"no sub-strategy active for regime={regime.value}", regime)

        # Conflict check: do we have opposing directions?
        directions = {s.direction for s in active_signals}
        if Direction.LONG in directions and Direction.SHORT in directions:
            return self._no_trade_signal(
                df,
                "ensemble conflict: opposing LONG and SHORT signals from sub-strategies",
                regime,
            )

        # Concordant signals: select the signal with highest confidence
        best_signal = max(active_signals, key=lambda s: s.confidence)

        # Annotate reason with ensemble consensus
        consensus_count = len(active_signals)
        ensemble_reason = f"[Ensemble consensus={consensus_count}/{len(self._strategies)}] {best_signal.reason}"

        return Signal(
            symbol=best_signal.symbol,
            action=best_signal.action,
            direction=best_signal.direction,
            confidence=best_signal.confidence,
            entry_price=best_signal.entry_price,
            stop_loss=best_signal.stop_loss,
            take_profit=best_signal.take_profit,
            expected_return_pct=best_signal.expected_return_pct,
            risk_pct=best_signal.risk_pct,
            regime=regime,
            reason=ensemble_reason,
            strategy_version_id=self.version_id,
            timestamp=best_signal.timestamp,
            metadata={
                "sub_signals_count": consensus_count,
                "winner_strategy": best_signal.strategy_version_id,
            },
        )
