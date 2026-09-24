"""Parameter stability & sensitivity testing engine.

Evaluates whether strategy performance is robust across parameter neighborhoods
or if it sits on an overfitted 'knife-edge' cliff.

Core rule:
Perturb all numeric parameters by +/-10% and +/-20%.
If performance (Sharpe) drops by more than 30% for a +/-10% perturbation,
the strategy is flagged as OVERFITTED and rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd

from ai_crypto_trader.backtesting.engine import BacktestConfig, BacktestEngine
from ai_crypto_trader.core.interfaces import StrategyABC
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class PerturbationResult:
    parameter_name: str
    baseline_value: Any
    perturbed_value: Any
    perturbation_pct: float       # e.g. +10.0, -10.0, +20.0, -20.0
    sharpe: float
    total_return: float
    degradation_pct: float        # Relative drop in Sharpe from baseline


@dataclass
class StabilityReport:
    strategy_name: str
    baseline_sharpe: float
    baseline_return: float
    results: list[PerturbationResult]
    max_degradation_pct: float
    mean_degradation_pct: float
    has_cliff: bool               # True if any +/-10% drop > 30%
    is_stable: bool               # True if passes stability criteria

    @property
    def passed(self) -> bool:
        return self.is_stable and not self.has_cliff


class ParameterStabilityTester:
    """Performs sensitivity sweeps across strategy parameter neighborhoods."""

    def __init__(
        self,
        max_allowed_drop_10pct: float = 30.0,  # Max 30% drop allowed for +/-10% perturbation
        initial_capital: float = 10000.0,
    ) -> None:
        self.max_allowed_drop_10pct = max_allowed_drop_10pct
        self.initial_capital = initial_capital
        self.backtest_engine = BacktestEngine(BacktestConfig(initial_capital=initial_capital))

    def evaluate(
        self,
        strategy_class: Type[StrategyABC],
        baseline_params: dict,
        dataset: pd.DataFrame,
    ) -> StabilityReport:
        """Run parameter sensitivity analysis on the strategy."""
        # 1. Run baseline backtest
        base_strat = strategy_class(version_id="base", params=baseline_params)
        base_res = self.backtest_engine.run(base_strat, dataset, self.initial_capital)
        base_sharpe = base_res.sharpe
        base_return = base_res.total_return

        perturbation_deltas = [-0.20, -0.10, 0.10, 0.20]
        results: list[PerturbationResult] = []
        has_cliff = False

        for param_name, orig_val in baseline_params.items():
            if not isinstance(orig_val, (int, float)) or isinstance(orig_val, bool):
                continue  # Only perturb numeric values

            for delta in perturbation_deltas:
                # Compute new value
                if isinstance(orig_val, int):
                    new_val = max(1, int(round(orig_val * (1.0 + delta))))
                    if new_val == orig_val:
                        new_val = orig_val + (1 if delta > 0 else -1)
                else:
                    new_val = float(orig_val * (1.0 + delta))

                test_params = dict(baseline_params)
                test_params[param_name] = new_val

                try:
                    test_strat = strategy_class(version_id=f"test_{param_name}", params=test_params)
                    test_res = self.backtest_engine.run(test_strat, dataset, self.initial_capital)
                    test_sharpe = test_res.sharpe
                    test_return = test_res.total_return
                except Exception as e:
                    log.warning("perturbation_run_failed", param=param_name, val=new_val, error=str(e))
                    test_sharpe = 0.0
                    test_return = -1.0

                # Compute degradation
                if base_sharpe > 0:
                    deg_pct = max(0.0, (base_sharpe - test_sharpe) / base_sharpe * 100.0)
                elif base_sharpe == 0:
                    deg_pct = 0.0 if test_sharpe >= 0 else 100.0
                else:
                    deg_pct = 0.0

                # Cliff check on +/-10%
                if abs(delta) <= 0.10 and deg_pct > self.max_allowed_drop_10pct:
                    has_cliff = True

                results.append(
                    PerturbationResult(
                        parameter_name=param_name,
                        baseline_value=orig_val,
                        perturbed_value=new_val,
                        perturbation_pct=delta * 100.0,
                        sharpe=round(test_sharpe, 3),
                        total_return=round(test_return, 4),
                        degradation_pct=round(deg_pct, 2),
                    )
                )

        max_deg = max((r.degradation_pct for r in results), default=0.0)
        mean_deg = float(np.mean([r.degradation_pct for r in results])) if results else 0.0

        is_stable = not has_cliff and mean_deg <= 40.0

        return StabilityReport(
            strategy_name=base_strat.name,
            baseline_sharpe=round(base_sharpe, 3),
            baseline_return=round(base_return, 4),
            results=results,
            max_degradation_pct=round(max_deg, 2),
            mean_degradation_pct=round(mean_deg, 2),
            has_cliff=has_cliff,
            is_stable=is_stable,
        )
