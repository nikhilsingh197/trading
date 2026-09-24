"""Autonomous AI Research Agent.

Orchestrates the research feedback loop:
Performance Analysis -> Hypothesis Generation -> Experimentation -> Scorecard Qualification -> Shadow Mode Promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import pandas as pd

from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.research.champion_challenger import ChampionChallengerManager
from ai_crypto_trader.research.experiment_runner import ExperimentResult, ExperimentRunner
from ai_crypto_trader.research.hypothesis_generator import Hypothesis, HypothesisGenerator
from ai_crypto_trader.validation.scorecard import StrategyScorecard

log = get_logger(__name__)


@dataclass
class ResearchCycleResult:
    timestamp: datetime
    hypotheses_generated: int
    experiments_run: int
    passed_experiments: int
    qualified_challengers: list[str] = field(default_factory=list)


class ResearchAgent:
    """Autonomous quantitative researcher continuously proposing, testing, and qualifying strategy refinements."""

    def __init__(
        self,
        hypothesis_generator: Optional[HypothesisGenerator] = None,
        experiment_runner: Optional[ExperimentRunner] = None,
        champion_challenger: Optional[ChampionChallengerManager] = None,
    ) -> None:
        self.hypotheses = hypothesis_generator or HypothesisGenerator()
        self.runner = experiment_runner or ExperimentRunner()
        self.cc_manager = champion_challenger or ChampionChallengerManager()

    def run_research_cycle(
        self,
        strategy_name: str,
        current_params: dict[str, Any],
        recent_metrics: dict[str, Any],
        historical_data: pd.DataFrame,
        strategy_factory: Any,
    ) -> ResearchCycleResult:
        """Execute one complete research cycle from diagnostic analysis to challenger qualification."""
        now = datetime.now(timezone.utc)
        log.info("starting_research_cycle", strategy=strategy_name)

        # 1. Generate Hypotheses based on performance bottlenecks
        hyps = self.hypotheses.generate_hypotheses_from_metrics(
            strategy_name=strategy_name,
            current_params=current_params,
            metrics=recent_metrics,
        )

        passed = 0
        challengers = []

        # 2. Test each hypothesis
        for hyp in hyps:
            try:
                candidate_instance = strategy_factory(hyp.proposed_parameters)
                res: ExperimentResult = self.runner.run_experiment(
                    hypothesis=hyp,
                    strategy_instance=candidate_instance,
                    data=historical_data,
                )

                if res.is_viable_candidate:
                    passed += 1
                    challenger_name = f"{strategy_name}_{hyp.id}"
                    challengers.append(challenger_name)
                    # Register into shadow paper trading
                    self.cc_manager.register_challenger(challenger_name)
                    log.info("research_candidate_qualified_to_shadow", challenger=challenger_name)
            except Exception as e:
                log.error("experiment_execution_error", hypothesis=hyp.id, error=str(e))

        log.info(
            "research_cycle_complete",
            hypotheses=len(hyps),
            experiments=len(hyps),
            passed=passed,
            challengers=challengers,
        )

        return ResearchCycleResult(
            timestamp=now,
            hypotheses_generated=len(hyps),
            experiments_run=len(hyps),
            passed_experiments=passed,
            qualified_challengers=challengers,
        )
