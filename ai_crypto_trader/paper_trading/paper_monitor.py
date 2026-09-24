"""Real-time performance and risk monitor for paper trading sessions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ai_crypto_trader.config.strategy_config import PaperTradingConfig
from ai_crypto_trader.core.interfaces import PortfolioState
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.paper_trading.paper_portfolio import PaperTradeRecord

log = get_logger(__name__)


@dataclass
class PaperTradingStats:
    """Consolidated performance statistics for an active paper trading session."""
    initial_capital: float
    current_equity: float
    total_return_pct: float
    peak_equity: float
    drawdown_pct: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_fees: float
    session_start: datetime
    last_update: datetime
    status: str = "HEALTHY"  # HEALTHY, WARNING, HALTED


class PaperMonitor:
    """Monitors live paper trading performance, risk constraints, and advancement criteria."""

    def __init__(
        self,
        config: PaperTradingConfig | None = None,
        max_drawdown_limit_pct: float = 15.0,
        daily_loss_limit_pct: float = 3.0,
    ) -> None:
        self.config = config or PaperTradingConfig()
        self.max_drawdown_limit_pct = max_drawdown_limit_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct

        self.session_start = datetime.now(timezone.utc)
        self.last_update = datetime.now(timezone.utc)
        self.max_drawdown_seen = 0.0
        self.day_start_equity = 10_000.0

    def update(
        self,
        state: PortfolioState,
        trades: list[PaperTradeRecord],
    ) -> PaperTradingStats:
        """Calculate and return updated session statistics."""
        self.last_update = datetime.now(timezone.utc)

        # Track max drawdown
        if state.drawdown_pct > self.max_drawdown_seen:
            self.max_drawdown_seen = state.drawdown_pct

        # Win rate and profit factor
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)
        total_fees = sum(t.fees for t in trades)

        total_return_pct = ((state.total_equity - (state.total_equity - state.realized_pnl - state.unrealized_pnl)) / max(1.0, state.total_equity - state.realized_pnl - state.unrealized_pnl)) * 100.0

        # Risk check
        status = "HEALTHY"
        if state.drawdown_pct >= self.max_drawdown_limit_pct:
            status = "HALTED"
            log.warning("paper_monitor_max_drawdown_breach", dd=state.drawdown_pct, limit=self.max_drawdown_limit_pct)
        elif state.drawdown_pct >= self.max_drawdown_limit_pct * 0.7:
            status = "WARNING"

        return PaperTradingStats(
            initial_capital=state.total_equity - state.realized_pnl - state.unrealized_pnl,
            current_equity=state.total_equity,
            total_return_pct=total_return_pct,
            peak_equity=state.peak_equity,
            drawdown_pct=state.drawdown_pct,
            max_drawdown_pct=self.max_drawdown_seen,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_fees=total_fees,
            session_start=self.session_start,
            last_update=self.last_update,
            status=status,
        )

    def check_advancement(self, stats: PaperTradingStats) -> tuple[bool, str]:
        """Verify if session has satisfied criteria to advance to shadow / live mode."""
        if stats.max_drawdown_pct > self.max_drawdown_limit_pct:
            return False, f"Max drawdown ({stats.max_drawdown_pct:.2f}%) breached limit ({self.max_drawdown_limit_pct:.2f}%)."

        if stats.profit_factor < 1.1:
            return False, f"Profit factor ({stats.profit_factor:.2f}) < 1.1 minimum threshold."

        duration_days = (self.last_update - self.session_start).total_seconds() / 86400.0
        if duration_days < self.config.min_period_days:
            return False, f"Paper period duration ({duration_days:.1f} days) < {self.config.min_period_days} required days."

        if stats.total_trades < self.config.min_trades:
            return False, f"Total trades executed ({stats.total_trades}) < {self.config.min_trades} required trades."

        return True, "Strategy qualified for shadow / live promotion review."
