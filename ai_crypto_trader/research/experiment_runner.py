"""Autonomous Experiment Runner.

Executes backtests and validation pipelines for formulated hypotheses,
evaluating candidates against the institutional StrategyScorecard.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

import pandas as pd

from ai_crypto_trader.backtesting.advanced_metrics import ComprehensiveMetrics
from ai_crypto_trader.backtesting.engine import BacktestConfig, BacktestEngine
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.research.hypothesis_generator import Hypothesis
from ai_crypto_trader.validation.scorecard import ScorecardEvaluation, StrategyScorecard

log = get_logger(__name__)


@dataclass
class ExperimentResult:
    experiment_id: str
    hypothesis: Hypothesis
    status: str            # 'PASS', 'FAIL'
    metrics: ComprehensiveMetrics
    scorecard: ScorecardEvaluation
    is_viable_candidate: bool
    completed_at: datetime


class ExperimentRunner:
    """Runs automated experiments evaluating candidate strategy mutations against qualification gates."""

    def __init__(self, scorecard: Optional[StrategyScorecard] = None) -> None:
        self.scorecard = scorecard or StrategyScorecard()
        self.engine = BacktestEngine(BacktestConfig(initial_capital=10000.0))
        self.experiment_history: list[ExperimentResult] = []

    def run_experiment(
        self,
        hypothesis: Hypothesis,
        strategy_instance: Any,
        data: pd.DataFrame,
    ) -> ExperimentResult:
        """Execute backtest and evaluate scorecard on proposed strategy variation."""
        exp_id = f"EXP-{uuid.uuid4().hex[:8]}"
        log.info("running_experiment", id=exp_id, title=hypothesis.title)

        # Run backtest
        bt_res = self.engine.run(strategy_instance, data, initial_capital=10000.0)

        # Build metrics from backtest
        comp_metrics = ComprehensiveMetrics(
            total_return=bt_res.total_return,
            cagr=bt_res.cagr,
            sharpe=bt_res.sharpe,
            sortino=bt_res.sortino,
            max_drawdown=bt_res.max_drawdown,
            profit_factor=bt_res.profit_factor,
            win_rate=bt_res.win_rate,
            expectancy=bt_res.expectancy,
            num_trades=bt_res.num_trades,
            num_winners=int(bt_res.num_trades * bt_res.win_rate),
            num_losers=bt_res.num_trades - int(bt_res.num_trades * bt_res.win_rate),
            fees_paid=bt_res.fees_paid,
            calmar=bt_res.extra.get("calmar", 1.0),
        )

        # Evaluate against qualification scorecard
        evaluation = self.scorecard.evaluate(
            strategy_name=f"{hypothesis.base_strategy}_{hypothesis.id}",
            metrics=comp_metrics,
        )

        status = "PASS" if evaluation.is_qualified else "FAIL"

        result = ExperimentResult(
            experiment_id=exp_id,
            hypothesis=hypothesis,
            status=status,
            metrics=comp_metrics,
            scorecard=evaluation,
            is_viable_candidate=evaluation.is_qualified,
            completed_at=datetime.now(timezone.utc),
        )

        self.experiment_history.append(result)
        log.info(
            "experiment_completed",
            id=exp_id,
            status=status,
            qualified=evaluation.is_qualified,
            passed_criteria=f"{evaluation.passed_criteria_count}/{evaluation.total_criteria_count}",
        )
        return result
