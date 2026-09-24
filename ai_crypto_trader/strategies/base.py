"""Base strategy class with common utilities."""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import pandas as pd

from ai_crypto_trader.core.enums import Direction, MarketRegime, SignalAction
from ai_crypto_trader.core.interfaces import Signal, StrategyABC


class BaseStrategy(StrategyABC):
    """Base class for all trading strategies.

    Subclasses must implement generate_signal().
    Common utilities (stop sizing, signal construction) are provided here.
    """

    def __init__(self, version_id: str, name: str, params: dict) -> None:
        self._version_id = version_id
        self._name = name
        self._params = params

    @property
    def version_id(self) -> str:
        return self._version_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return self._params

    def _build_long_signal(
        self,
        df: pd.DataFrame,
        confidence: float,
        stop_loss_pct: float,
        tp_ratio: float,
        reason: str,
        regime: MarketRegime,
        symbol: str = "UNKNOWN",
    ) -> Signal:
        close = float(df.iloc[-1]["close"])
        stop_loss = close * (1 - stop_loss_pct)
        take_profit = close + (close - stop_loss) * tp_ratio
        return Signal(
            symbol=symbol,
            action=SignalAction.ENTER_LONG,
            direction=Direction.LONG,
            confidence=min(1.0, max(0.0, confidence)),
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_return_pct=(take_profit / close - 1) * 100,
            risk_pct=stop_loss_pct * 100,
            regime=regime,
            reason=reason,
            strategy_version_id=self._version_id,
        )

    def _build_short_signal(
        self,
        df: pd.DataFrame,
        confidence: float,
        stop_loss_pct: float,
        tp_ratio: float,
        reason: str,
        regime: MarketRegime,
        symbol: str = "UNKNOWN",
    ) -> Signal:
        close = float(df.iloc[-1]["close"])
        stop_loss = close * (1 + stop_loss_pct)
        take_profit = close - (stop_loss - close) * tp_ratio
        return Signal(
            symbol=symbol,
            action=SignalAction.ENTER_SHORT,
            direction=Direction.SHORT,
            confidence=min(1.0, max(0.0, confidence)),
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_return_pct=(close / take_profit - 1) * 100,
            risk_pct=stop_loss_pct * 100,
            regime=regime,
            reason=reason,
            strategy_version_id=self._version_id,
        )

    def _no_trade_signal(self, df: pd.DataFrame, reason: str, regime: MarketRegime) -> Signal:
        close = float(df.iloc[-1]["close"])
        return Signal(
            symbol="UNKNOWN",
            action=SignalAction.NO_TRADE,
            direction=Direction.FLAT,
            confidence=0.0,
            entry_price=close,
            stop_loss=close * 0.99,
            take_profit=close * 1.01,
            expected_return_pct=0.0,
            risk_pct=0.0,
            regime=regime,
            reason=reason,
            strategy_version_id=self._version_id,
        )
