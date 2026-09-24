"""Monte Carlo trade permutation & robustness analysis.

Performs bootstrap resampling of trade returns to assess:
- Distribution of expected total returns
- Maximum drawdown percentiles (5th, 50th, 95th)
- Sharpe ratio confidence intervals
- Probability of breaching risk limits or account ruin
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ai_crypto_trader.backtesting.advanced_metrics import TradeDetail


@dataclass
class MonteCarloResult:
    iterations: int
    trades_sampled: int
    median_return: float
    return_5th_pct: float
    return_95th_pct: float
    median_max_drawdown: float
    max_drawdown_95th_pct: float     # Worst-case 95th percentile drawdown
    median_sharpe: float
    sharpe_5th_pct: float
    prob_drawdown_exceeds_10pct: float
    prob_drawdown_exceeds_15pct: float
    prob_drawdown_exceeds_20pct: float


class MonteCarloSimulator:
    """Bootstraps trade sequence to measure statistical resilience."""

    def __init__(self, iterations: int = 1000, seed: int | None = 42) -> None:
        self.iterations = iterations
        self.seed = seed

    def simulate(
        self,
        trades: Sequence[TradeDetail],
        initial_capital: float = 10000.0,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation over historical trade returns."""
        if not trades:
            return MonteCarloResult(
                iterations=0,
                trades_sampled=0,
                median_return=0.0,
                return_5th_pct=0.0,
                return_95th_pct=0.0,
                median_max_drawdown=0.0,
                max_drawdown_95th_pct=0.0,
                median_sharpe=0.0,
                sharpe_5th_pct=0.0,
                prob_drawdown_exceeds_10pct=0.0,
                prob_drawdown_exceeds_15pct=0.0,
                prob_drawdown_exceeds_20pct=0.0,
            )

        if self.seed is not None:
            np.random.seed(self.seed)

        trade_pnl_pcts = np.array([t.pnl_pct for t in trades])
        n_trades = len(trade_pnl_pcts)

        sim_returns: list[float] = []
        sim_drawdowns: list[float] = []
        sim_sharpes: list[float] = []

        dd_exceed_10_count = 0
        dd_exceed_15_count = 0
        dd_exceed_20_count = 0

        for _ in range(self.iterations):
            sampled_indices = np.random.choice(n_trades, size=n_trades, replace=True)
            sampled_pcts = trade_pnl_pcts[sampled_indices]

            # Construct synthetic equity curve
            eq_curve = initial_capital * np.cumprod(1.0 + sampled_pcts)
            final_equity = eq_curve[-1]
            tot_ret = (final_equity - initial_capital) / initial_capital
            sim_returns.append(tot_ret)

            # Max drawdown for this run
            running_max = np.maximum.accumulate(np.insert(eq_curve, 0, initial_capital))
            full_curve = np.insert(eq_curve, 0, initial_capital)
            drawdowns = (full_curve - running_max) / running_max
            max_dd = abs(float(np.min(drawdowns)))
            sim_drawdowns.append(max_dd)

            if max_dd >= 0.10:
                dd_exceed_10_count += 1
            if max_dd >= 0.15:
                dd_exceed_15_count += 1
            if max_dd >= 0.20:
                dd_exceed_20_count += 1

            # Approximate trade-level Sharpe
            if len(sampled_pcts) > 1 and np.std(sampled_pcts) > 0:
                sharpe = (np.mean(sampled_pcts) / np.std(sampled_pcts)) * np.sqrt(n_trades)
                sim_sharpes.append(float(sharpe))
            else:
                sim_sharpes.append(0.0)

        return MonteCarloResult(
            iterations=self.iterations,
            trades_sampled=n_trades,
            median_return=float(np.median(sim_returns)),
            return_5th_pct=float(np.percentile(sim_returns, 5)),
            return_95th_pct=float(np.percentile(sim_returns, 95)),
            median_max_drawdown=float(np.median(sim_drawdowns)),
            max_drawdown_95th_pct=float(np.percentile(sim_drawdowns, 95)),
            median_sharpe=float(np.median(sim_sharpes)),
            sharpe_5th_pct=float(np.percentile(sim_sharpes, 5)),
            prob_drawdown_exceeds_10pct=round(dd_exceed_10_count / self.iterations, 4),
            prob_drawdown_exceeds_15pct=round(dd_exceed_15_count / self.iterations, 4),
            prob_drawdown_exceeds_20pct=round(dd_exceed_20_count / self.iterations, 4),
        )
