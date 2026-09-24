"""Abstract base classes defining clean interfaces between modules.

Every concrete implementation must subclass the appropriate ABC.
This ensures modules can be swapped without changing downstream code.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncIterator

import pandas as pd

from ai_crypto_trader.core.enums import (
    Direction,
    ExitReason,
    MarketRegime,
    OrderStatus,
    OrderType,
    Side,
    SignalAction,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Candle:
    symbol: str
    timeframe: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    quality_flag: int = 0


@dataclass
class Signal:
    """Output of a strategy. Every field is required for risk management."""
    symbol: str
    action: SignalAction
    direction: Direction
    confidence: float              # 0.0 – 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    expected_return_pct: float
    risk_pct: float                # Expected risk as % of account
    regime: MarketRegime
    reason: str                    # Human-readable explanation
    strategy_version_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    strategy_id: str | None = None
    signal_id: str | None = None
    client_order_id: str | None = None


@dataclass
class OrderResult:
    client_order_id: str
    exchange_order_id: str | None
    status: OrderStatus
    filled_qty: float
    avg_fill_price: float | None
    fees: float
    error: str | None = None


@dataclass
class PositionSnapshot:
    symbol: str
    side: Direction
    entry_price: float
    quantity: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: float | None
    take_profit: float | None


@dataclass
class PortfolioState:
    total_equity: float
    available_cash: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown_pct: float
    peak_equity: float
    open_positions: list[PositionSnapshot] = field(default_factory=list)


@dataclass
class BacktestResult:
    strategy_version_id: str
    symbol: str
    timeframe: str
    period_start: datetime
    period_end: datetime
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    expectancy: float
    num_trades: int
    fees_paid: float
    extra: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base Classes
# ─────────────────────────────────────────────────────────────────────────────

class DataCollectorABC(abc.ABC):
    """Collects raw market data from an exchange."""

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def stream_candles(self, symbol: str, timeframe: str) -> AsyncIterator[Candle]: ...


class DataValidatorABC(abc.ABC):
    """Validates and cleans incoming market data."""

    @abc.abstractmethod
    def validate_candle(self, candle: Candle) -> Candle:
        """Returns validated candle or raises DataValidationError."""
        ...


class FeatureEngineABC(abc.ABC):
    """Computes feature vectors from OHLCV data."""

    @abc.abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all feature columns. Input df must not be mutated."""
        ...


class RegimeDetectorABC(abc.ABC):
    """Classifies the current market regime."""

    @abc.abstractmethod
    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Return current regime from the latest data."""
        ...


class StrategyABC(abc.ABC):
    """Generates trading signals from features and regime."""

    @property
    @abc.abstractmethod
    def version_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        """Return a Signal. Must never use future data."""
        ...


class MLModelABC(abc.ABC):
    """ML model that produces predictions from feature vectors."""

    @property
    @abc.abstractmethod
    def model_version_id(self) -> str: ...

    @abc.abstractmethod
    def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        """Return prediction dict (probability, confidence, etc.)."""
        ...

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...


class RiskEngineABC(abc.ABC):
    """Evaluates and sizes positions according to hard risk limits."""

    @abc.abstractmethod
    def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioState,
    ) -> tuple[bool, float, str]:
        """Return (approved, position_size, reason)."""
        ...


class BrokerABC(abc.ABC):
    """Submits and tracks orders (paper or live)."""

    @abc.abstractmethod
    async def submit_order(self, request: OrderRequest) -> OrderResult: ...

    @abc.abstractmethod
    async def cancel_order(self, client_order_id: str) -> bool: ...

    @abc.abstractmethod
    async def get_portfolio_state(self) -> PortfolioState: ...


class AlertSenderABC(abc.ABC):
    """Sends alert notifications."""

    @abc.abstractmethod
    async def send(self, title: str, message: str, severity: str) -> None: ...


class BacktestEngineABC(abc.ABC):
    """Runs event-driven backtests."""

    @abc.abstractmethod
    def run(
        self,
        strategy: StrategyABC,
        df: pd.DataFrame,
        initial_capital: float,
    ) -> BacktestResult: ...
