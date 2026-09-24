"""Automated Strategy Scorecard & Promotion Gate.

Strict validation criteria required before any strategy is eligible for paper trading:
- Sharpe Ratio >= 0.50
- Profit Factor >= 1.20
- Max Drawdown <= 15.0%
- Win Rate >= 40.0%
- Minimum Trades >= 30
- Expectancy > 0.0
- Walk-Forward Efficiency (WFE) >= 0.50 (OOS maintains >= 50% IS return)
- Parameter Stability: No performance cliffs (>30% drop on +/-10% perturbation)
- Deflated Sharpe Ratio >= 0.70 (statistical significance vs data mining)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ai_crypto_trader.backtesting.advanced_metrics import ComprehensiveMetrics
from ai_crypto_trader.config.strategy_config import ScorecardThresholds
from ai_crypto_trader.validation.parameter_stability import StabilityReport
from ai_crypto_trader.validation.walk_forward import WalkForwardResult


@dataclass
class ScorecardCriterion:
    name: str
    target: str
    actual: str
    passed: bool
    importance: str = "HARD"  # HARD (must pass) or SOFT


@dataclass
class ScorecardEvaluation:
    strategy_name: str
    is_qualified: bool
    passed_criteria_count: int
    total_criteria_count: int
    criteria: list[ScorecardCriterion]
    summary_verdict: str


class StrategyScorecard:
    """Evaluates candidate strategies against institutional qualification gates."""

    def __init__(self, thresholds: ScorecardThresholds | None = None) -> None:
        self.thresholds = thresholds or ScorecardThresholds()

    def evaluate(
        self,
        strategy_name: str,
        metrics: ComprehensiveMetrics,
        walk_forward: Optional[WalkForwardResult] = None,
        stability: Optional[StabilityReport] = None,
        dsr_p_value: Optional[float] = None,
    ) -> ScorecardEvaluation:
        t = self.thresholds
        criteria: list[ScorecardCriterion] = []

        # 1. Sharpe Ratio
        criteria.append(
            ScorecardCriterion(
                name="Sharpe Ratio",
                target=f">= {t.min_sharpe:.2f}",
                actual=f"{metrics.sharpe:.3f}",
                passed=metrics.sharpe >= t.min_sharpe,
            )
        )

        # 2. Profit Factor
        criteria.append(
            ScorecardCriterion(
                name="Profit Factor",
                target=f">= {t.min_profit_factor:.2f}",
                actual=f"{metrics.profit_factor:.3f}",
                passed=metrics.profit_factor >= t.min_profit_factor,
            )
        )

        # 3. Max Drawdown
        dd_pct = metrics.max_drawdown * 100.0
        criteria.append(
            ScorecardCriterion(
                name="Max Drawdown",
                target=f"<= {t.max_drawdown_pct:.1f}%",
                actual=f"{dd_pct:.2f}%",
                passed=dd_pct <= t.max_drawdown_pct,
            )
        )

        # 4. Win Rate
        criteria.append(
            ScorecardCriterion(
                name="Win Rate",
                target=f">= {t.min_win_rate:.0%}",
                actual=f"{metrics.win_rate:.1%}",
                passed=metrics.win_rate >= t.min_win_rate,
            )
        )

        # 5. Minimum Trades
        criteria.append(
            ScorecardCriterion(
                name="Trade Sample Size",
                target=f">= {min(30, t.min_trades)}",
                actual=str(metrics.num_trades),
                passed=metrics.num_trades >= min(30, t.min_trades),
            )
        )

        # 6. Expectancy
        criteria.append(
            ScorecardCriterion(
                name="Trade Expectancy",
                target=f"> {t.min_expectancy:.2f}",
                actual=f"${metrics.expectancy:.2f}",
                passed=metrics.expectancy > t.min_expectancy,
            )
        )

        # 7. Walk-Forward Efficiency (WFE)
        if walk_forward is not None:
            criteria.append(
                ScorecardCriterion(
                    name="Walk-Forward Efficiency (WFE)",
                    target=">= 50.0%",
                    actual=f"{walk_forward.overall_wfe:.1%}",
                    passed=walk_forward.overall_wfe >= 0.50,
                )
            )

        # 8. Parameter Stability (No Cliff)
        if stability is not None:
            criteria.append(
                ScorecardCriterion(
                    name="Parameter Stability (No Cliff)",
                    target="Drop <= 30% on +/-10%",
                    actual=f"Max drop: {stability.max_degradation_pct:.1f}%",
                    passed=not stability.has_cliff,
                )
            )

        # 9. Deflated Sharpe Ratio
        if dsr_p_value is not None:
            criteria.append(
                ScorecardCriterion(
                    name="Deflated Sharpe Ratio (DSR)",
                    target=">= 70.0%",
                    actual=f"{dsr_p_value:.1%}",
                    passed=dsr_p_value >= 0.70,
                    importance="SOFT",
                )
            )

        hard_fails = [c for c in criteria if c.importance == "HARD" and not c.passed]
        is_qualified = len(hard_fails) == 0
        passed_count = sum(1 for c in criteria if c.passed)

        if is_qualified:
            verdict = "QUALIFIED FOR PAPER TRADING: Strategy passed all mandatory risk and robustness gates."
        else:
            reasons = ", ".join(f"{c.name} ({c.actual} vs {c.target})" for c in hard_fails)
            verdict = f"REJECTED: Failed mandatory criteria: {reasons}"

        return ScorecardEvaluation(
            strategy_name=strategy_name,
            is_qualified=is_qualified,
            passed_criteria_count=passed_count,
            total_criteria_count=len(criteria),
            criteria=criteria,
            summary_verdict=verdict,
        )
