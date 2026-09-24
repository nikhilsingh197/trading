"""Institutional performance & risk analytics for backtesting.

Calculates:
- Sharpe, Sortino, Calmar, Omega ratios
- Max Drawdown % and underwater duration (bars)
- Value at Risk (VaR 95%) and Conditional VaR (CVaR / Expected Shortfall)
- Fee, slippage, and funding cost breakdowns
- Profit Factor, Payoff Ratio, Win Rate, and Expectancy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class TradeDetail:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    slippage: float
    funding_paid: float
    opened_at: datetime
    closed_at: datetime
    exit_reason: str
    bars_held: int


@dataclass
class ComprehensiveMetrics:
    total_return: float = 0.0
    cagr: float = 0.0
    annualized_volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    omega_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_bars: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0
    num_trades: int = 0
    num_winners: int = 0
    num_losers: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_bars_held: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    fees_paid: float = 0.0
    slippage_paid: float = 0.0
    funding_paid: float = 0.0
    exposure_pct: float = 0.0


def calculate_comprehensive_metrics(
    trades: Sequence[TradeDetail],
    equity_series: pd.Series,
    initial_capital: float,
    periods_per_year: int = 365 * 24,  # Hourly crypto default
) -> ComprehensiveMetrics:
    """Calculate institutional-grade metrics from completed trades and portfolio equity curve."""
    m = ComprehensiveMetrics()

    if trades:
        m.num_trades = len(trades)
        pnls = [t.pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        m.num_winners = len(winners)
        m.num_losers = len(losers)
        m.win_rate = m.num_winners / m.num_trades if m.num_trades > 0 else 0.0

        m.avg_win = float(np.mean(winners)) if winners else 0.0
        m.avg_loss = float(np.mean(losers)) if losers else 0.0
        m.avg_trade_pnl = float(np.mean(pnls))
        m.avg_bars_held = float(np.mean([t.bars_held for t in trades]))

        m.fees_paid = float(sum(t.fees for t in trades))
        m.slippage_paid = float(sum(t.slippage for t in trades))
        m.funding_paid = float(sum(t.funding_paid for t in trades))

        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        m.payoff_ratio = abs(m.avg_win / m.avg_loss) if abs(m.avg_loss) > 0 else float("inf")
        m.expectancy = m.win_rate * m.avg_win + (1 - m.win_rate) * m.avg_loss

    if equity_series is not None and len(equity_series) > 1:
        final_equity = float(equity_series.iloc[-1])
        m.total_return = (final_equity - initial_capital) / initial_capital

        n_bars = len(equity_series)
        n_years = n_bars / periods_per_year
        if n_years > 0 and final_equity > 0 and initial_capital > 0:
            m.cagr = (final_equity / initial_capital) ** (1.0 / n_years) - 1.0

        # Drawdown and underwater duration
        rolling_max = equity_series.cummax()
        drawdown_series = (equity_series - rolling_max) / rolling_max
        m.max_drawdown = abs(float(drawdown_series.min()))

        # Max drawdown duration in bars
        underwater = drawdown_series < 0
        current_dd_bars = 0
        max_dd_bars = 0
        for is_uw in underwater:
            if is_uw:
                current_dd_bars += 1
                if current_dd_bars > max_dd_bars:
                    max_dd_bars = current_dd_bars
            else:
                current_dd_bars = 0
        m.max_drawdown_duration_bars = max_dd_bars

        # Return statistics
        bar_returns = equity_series.pct_change().dropna()
        if len(bar_returns) > 1:
            mean_ret = float(bar_returns.mean())
            std_ret = float(bar_returns.std())
            m.annualized_volatility = std_ret * np.sqrt(periods_per_year)

            if std_ret > 0:
                m.sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)

            # Sortino
            downside_returns = bar_returns[bar_returns < 0]
            downside_std = float(downside_returns.std()) if len(downside_returns) > 1 else 0.0
            if downside_std > 0:
                m.sortino = (mean_ret / downside_std) * np.sqrt(periods_per_year)

            # Calmar Ratio
            if m.max_drawdown > 0:
                m.calmar = m.cagr / m.max_drawdown

            # Omega Ratio (threshold = 0 return)
            gains = bar_returns[bar_returns > 0].sum()
            losses = abs(bar_returns[bar_returns < 0].sum())
            m.omega_ratio = float(gains / losses) if losses > 0 else float("inf")

            # Value at Risk (VaR 95%) and Conditional VaR (CVaR / Expected Shortfall)
            sorted_rets = np.sort(bar_returns.values)
            var_index = int(0.05 * len(sorted_rets))
            m.var_95 = abs(float(sorted_rets[var_index])) if len(sorted_rets) > 0 else 0.0
            tail_losses = sorted_rets[:var_index]
            m.cvar_95 = abs(float(np.mean(tail_losses))) if len(tail_losses) > 0 else m.var_95

    return m
