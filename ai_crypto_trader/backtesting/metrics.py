"""Backtesting performance metrics.

All metrics prioritize risk-adjusted performance over raw returns.
None of these metrics should be used in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    entry_price: float
    exit_price: float
    side: str          # 'long' or 'short'
    quantity: float
    pnl: float         # After fees
    pnl_pct: float
    fees: float
    opened_at: object  # datetime
    closed_at: object  # datetime
    exit_reason: str


@dataclass
class BacktestMetrics:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    num_trades: int = 0
    num_winners: int = 0
    num_losers: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_hold_bars: float = 0.0
    fees_paid: float = 0.0
    max_consecutive_losses: int = 0
    risk_of_ruin: float = 0.0
    exposure_pct: float = 0.0


def compute_metrics(
    trades: list[TradeRecord],
    equity_curve: pd.Series,
    initial_capital: float,
    periods_per_year: int = 365 * 24,  # Hourly crypto
) -> BacktestMetrics:
    """Compute all performance metrics from a list of trades and equity curve.

    Args:
        trades: List of completed TradeRecord objects.
        equity_curve: Time-indexed Series of portfolio equity values.
        initial_capital: Starting capital.
        periods_per_year: Number of candles per year (depends on timeframe).
    """
    m = BacktestMetrics()

    if not trades:
        return m

    m.num_trades = len(trades)
    m.fees_paid = sum(t.fees for t in trades)

    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    m.num_winners = len(winners)
    m.num_losers = len(losers)
    m.win_rate = m.num_winners / m.num_trades if m.num_trades > 0 else 0.0
    m.avg_win = np.mean(winners) if winners else 0.0
    m.avg_loss = np.mean(losers) if losers else 0.0
    m.avg_trade = np.mean(pnls)
    m.expectancy = m.win_rate * m.avg_win + (1 - m.win_rate) * m.avg_loss

    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total return
    if equity_curve is not None and len(equity_curve) > 0:
        final_equity = equity_curve.iloc[-1]
        m.total_return = (final_equity - initial_capital) / initial_capital

        # CAGR
        n_years = len(equity_curve) / periods_per_year
        if n_years > 0 and initial_capital > 0 and final_equity > 0:
            m.cagr = (final_equity / initial_capital) ** (1 / n_years) - 1

        # Max drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        m.max_drawdown = abs(float(drawdown.min()))

        # Sharpe (annualized)
        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            m.sharpe = (
                daily_returns.mean() / daily_returns.std()
            ) * np.sqrt(periods_per_year)

        # Sortino (downside deviation)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            m.sortino = (
                daily_returns.mean() / downside.std()
            ) * np.sqrt(periods_per_year)

    # Max consecutive losses
    if pnls:
        max_consec = 0
        current = 0
        for p in pnls:
            if p <= 0:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        m.max_consecutive_losses = max_consec

    # Risk of Ruin (simplified formula)
    if m.win_rate > 0 and m.win_rate < 1:
        edge = m.win_rate - (1 - m.win_rate)
        if edge > 0:
            m.risk_of_ruin = ((1 - edge) / (1 + edge)) ** 100

    return m
