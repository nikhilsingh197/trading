"""Institutional Trading Performance Analyzer.

Computes comprehensive risk-adjusted performance metrics from trade logs and equity curves.
Metrics: Sharpe, Sortino, Calmar, Profit Factor, Expectancy, Win Rate, Drawdowns, Streaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np


@dataclass
class PerformanceSummary:
    """Consolidated performance scorecard."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    break_even_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_dollars: float
    calmar_ratio: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_fees: float
    total_slippage: float


class PerformanceAnalyzer:
    """Analyzes trade histories and equity curves for statistical edge and risk characteristics."""

    @classmethod
    def analyze(
        cls,
        trades: Sequence[dict[str, Any]],
        equity_curve: Sequence[float],
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.0,
    ) -> PerformanceSummary:
        """Calculate complete statistical performance summary."""
        total_trades = len(trades)

        if total_trades == 0:
            return cls._empty_summary(initial_capital)

        pnls = [t.get("pnl", 0.0) for t in trades]
        fees = sum(t.get("fees", 0.0) for t in trades)
        slippages = sum(t.get("slippage", 0.0) for t in trades)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        bes = [p for p in pnls if p == 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        break_even_trades = len(bes)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(pnls)

        current_equity = equity_curve[-1] if equity_curve else (initial_capital + total_pnl)
        total_return_pct = ((current_equity - initial_capital) / initial_capital) * 100.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 999.0 if gross_profit > 0 else 0.0

        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0

        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

        # Drawdown calculations from equity curve
        max_dd_pct, max_dd_dollars = cls._calculate_drawdown(equity_curve, initial_capital)

        # Sharpe and Sortino ratios
        sharpe, sortino = cls._calculate_risk_adjusted_ratios(equity_curve, risk_free_rate)

        # Calmar ratio (annualized return / max drawdown)
        calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        # Streaks
        max_wins, max_losses = cls._calculate_streaks(pnls)

        return PerformanceSummary(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            break_even_trades=break_even_trades,
            win_rate=win_rate,
            total_pnl=round(total_pnl, 2),
            total_return_pct=round(total_return_pct, 2),
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            profit_factor=round(profit_factor, 2),
            avg_trade_pnl=round(avg_trade_pnl, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            win_loss_ratio=round(win_loss_ratio, 2),
            expectancy=round(expectancy, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_drawdown_dollars=round(max_dd_dollars, 2),
            calmar_ratio=round(calmar, 2),
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            total_fees=round(fees, 2),
            total_slippage=round(slippages, 2),
        )

    @staticmethod
    def _calculate_drawdown(curve: Sequence[float], initial_capital: float) -> tuple[float, float]:
        if not curve:
            return 0.0, 0.0

        peak = max(curve[0], initial_capital)
        max_dd_pct = 0.0
        max_dd_dollars = 0.0

        for eq in curve:
            if eq > peak:
                peak = eq
            dd_dollars = peak - eq
            dd_pct = (dd_dollars / peak) * 100.0 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
            if dd_dollars > max_dd_dollars:
                max_dd_dollars = dd_dollars

        return max_dd_pct, max_dd_dollars

    @staticmethod
    def _calculate_risk_adjusted_ratios(
        curve: Sequence[float],
        rf: float = 0.0,
    ) -> tuple[float, float]:
        if len(curve) < 3:
            return 0.0, 0.0

        arr = np.array(curve)
        returns = np.diff(arr) / arr[:-1]

        if len(returns) == 0:
            return 0.0, 0.0

        excess = returns - (rf / 252.0)
        mean_ret = np.mean(excess)
        std_ret = np.std(excess)

        # Annualize (assuming 252 daily periods or 365 for 24/7 crypto)
        annual_factor = math.sqrt(365)
        sharpe = (mean_ret / std_ret * annual_factor) if std_ret > 1e-8 else 0.0

        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 1 else std_ret
        sortino = (mean_ret / downside_std * annual_factor) if downside_std > 1e-8 else 0.0

        return float(sharpe), float(sortino)

    @staticmethod
    def _calculate_streaks(pnls: Sequence[float]) -> tuple[int, int]:
        max_wins = 0
        max_losses = 0
        cur_wins = 0
        cur_losses = 0

        for p in pnls:
            if p > 0:
                cur_wins += 1
                cur_losses = 0
                if cur_wins > max_wins:
                    max_wins = cur_wins
            elif p < 0:
                cur_losses += 1
                cur_wins = 0
                if cur_losses > max_losses:
                    max_losses = cur_losses
            else:
                cur_wins = 0
                cur_losses = 0

        return max_wins, max_losses

    @classmethod
    def _empty_summary(cls, initial_capital: float) -> PerformanceSummary:
        return PerformanceSummary(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            break_even_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_return_pct=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            avg_trade_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            expectancy=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_dollars=0.0,
            calmar_ratio=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            total_fees=0.0,
            total_slippage=0.0,
        )
