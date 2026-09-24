"""Shared type aliases and Pydantic schemas for API serialization."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CandleSchema(BaseModel):
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


class SignalSchema(BaseModel):
    symbol: str
    action: str
    direction: str
    confidence: float = Field(ge=0.0, le=1.0)
    entry_price: float
    stop_loss: float
    take_profit: float
    expected_return_pct: float
    risk_pct: float
    regime: str
    reason: str
    strategy_version_id: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class PortfolioStateSchema(BaseModel):
    total_equity: float
    available_cash: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown_pct: float
    peak_equity: float
    open_positions: list[dict[str, Any]] = Field(default_factory=list)


class BacktestResultSchema(BaseModel):
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


class RiskDecisionSchema(BaseModel):
    approved: bool
    position_size: float
    reason: str
    action_taken: str


class HealthStatusSchema(BaseModel):
    status: str  # ok / degraded / critical
    timestamp: datetime
    components: dict[str, str]
    kill_switch_active: bool
