"""Champion vs. Challenger Shadow Mode Manager.

Runs candidate strategies in shadow paper trading mode concurrently with the production
Champion. Tracks outperformance, statistical significance, and generates promotion candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Optional, Sequence
import uuid

import numpy as np
from scipy import stats

from ai_crypto_trader.core.enums import StrategyStatus
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.reporting.performance_analyzer import PerformanceAnalyzer, PerformanceSummary

log = get_logger(__name__)


@dataclass
class HeadToHeadEvaluation:
    champion_name: str
    challenger_name: str
    champion_summary: PerformanceSummary
    challenger_summary: PerformanceSummary
    sharpe_diff: float
    profit_factor_diff: float
    return_diff: float
    drawdown_diff: float
    is_statistically_superior: bool
    p_value: float
    challenger_eligible: bool
    summary_verdict: str


@dataclass
class PromotionCandidate:
    candidate_id: str
    challenger_name: str
    champion_name: str
    evaluation: HeadToHeadEvaluation
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ChampionChallengerManager:
    """Manages shadow-mode parallel paper trading and statistical qualification of Challengers."""

    def __init__(
        self,
        min_shadow_trades: int = 30,
        min_shadow_days: int = 14,
    ) -> None:
        self.min_shadow_trades = min_shadow_trades
        self.min_shadow_days = min_shadow_days

        self.champion_name: str = "Production_Champion"
        self.challenger_name: Optional[str] = None
        self.challenger_start_time: Optional[datetime] = None

        self._champion_trades: list[dict[str, Any]] = []
        self._challenger_trades: list[dict[str, Any]] = []

    def set_champion(self, name: str) -> None:
        self.champion_name = name
        log.info("champion_strategy_set", name=name)

    def register_challenger(self, name: str, start_time: Optional[datetime] = None) -> None:
        self.challenger_name = name
        self.challenger_start_time = start_time or datetime.now(timezone.utc)
        self._challenger_trades.clear()
        log.info("challenger_registered_in_shadow_mode", name=name, start=self.challenger_start_time.isoformat())

    def record_trade(self, strategy_role: str, trade_dict: dict[str, Any]) -> None:
        """Record trade execution for either 'CHAMPION' or 'CHALLENGER'."""
        if strategy_role.upper() == "CHAMPION":
            self._champion_trades.append(trade_dict)
        elif strategy_role.upper() == "CHALLENGER":
            self._challenger_trades.append(trade_dict)

    def evaluate_head_to_head(self, now: Optional[datetime] = None) -> Optional[HeadToHeadEvaluation]:
        """Perform statistical head-to-head comparison between Champion and Challenger."""
        if not self.challenger_name or not self.challenger_start_time:
            return None

        current_time = now or datetime.now(timezone.utc)
        elapsed_days = (current_time - self.challenger_start_time).total_seconds() / 86400.0

        # Compute performance summaries with initial capital prepended
        champ_equity = [10000.0]
        cur = 10000.0
        for t in self._champion_trades:
            cur += t.get("pnl", 0.0)
            champ_equity.append(cur)

        chall_equity = [10000.0]
        cur = 10000.0
        for t in self._challenger_trades:
            cur += t.get("pnl", 0.0)
            chall_equity.append(cur)

        champ_summary = PerformanceAnalyzer.analyze(
            trades=self._champion_trades,
            equity_curve=champ_equity,
        )
        chall_summary = PerformanceAnalyzer.analyze(
            trades=self._challenger_trades,
            equity_curve=chall_equity,
        )

        sharpe_diff = chall_summary.sharpe_ratio - champ_summary.sharpe_ratio
        pf_diff = chall_summary.profit_factor - champ_summary.profit_factor
        ret_diff = chall_summary.total_return_pct - champ_summary.total_return_pct
        dd_diff = chall_summary.max_drawdown_pct - champ_summary.max_drawdown_pct

        # Statistical significance test on returns (Welch's t-test or Wilcoxon)
        champ_returns = [t.get("pnl_pct", 0.0) for t in self._champion_trades]
        chall_returns = [t.get("pnl_pct", 0.0) for t in self._challenger_trades]

        p_value = 1.0
        is_statistically_superior = False
        if len(champ_returns) >= 10 and len(chall_returns) >= 10:
            ttest_res = stats.ttest_ind(chall_returns, champ_returns, equal_var=False)
            p_value = float(ttest_res.pvalue) if not math.isnan(ttest_res.pvalue) else 1.0
            # Statistically superior if higher mean return and p < 0.05
            is_statistically_superior = (p_value < 0.05) and (np.mean(chall_returns) > np.mean(champ_returns))

        # Qualification gates
        trades_met = chall_summary.total_trades >= self.min_shadow_trades
        duration_met = elapsed_days >= self.min_shadow_days
        sharpe_superior = chall_summary.sharpe_ratio >= (champ_summary.sharpe_ratio + 0.10)
        drawdown_acceptable = chall_summary.max_drawdown_pct <= (champ_summary.max_drawdown_pct + 1.0)
        pf_acceptable = chall_summary.profit_factor >= champ_summary.profit_factor

        is_eligible = (
            trades_met
            and duration_met
            and sharpe_superior
            and drawdown_acceptable
            and pf_acceptable
        )

        if is_eligible:
            verdict = f"QUALIFIED: Challenger {self.challenger_name} outperformed Champion with Sharpe diff +{sharpe_diff:.2f}."
        else:
            verdict = (
                f"NOT YET QUALIFIED (trades={chall_summary.total_trades}/{self.min_shadow_trades}, "
                f"days={elapsed_days:.1f}/{self.min_shadow_days}, sharpe_diff={sharpe_diff:+.2f})"
            )

        return HeadToHeadEvaluation(
            champion_name=self.champion_name,
            challenger_name=self.challenger_name,
            champion_summary=champ_summary,
            challenger_summary=chall_summary,
            sharpe_diff=round(sharpe_diff, 2),
            profit_factor_diff=round(pf_diff, 2),
            return_diff=round(ret_diff, 2),
            drawdown_diff=round(dd_diff, 2),
            is_statistically_superior=is_statistically_superior,
            p_value=round(p_value, 4),
            challenger_eligible=is_eligible,
            summary_verdict=verdict,
        )

    def generate_promotion_candidate(self) -> Optional[PromotionCandidate]:
        """Produce a formal promotion dossier if Challenger meets all criteria."""
        evaluation = self.evaluate_head_to_head()
        if not evaluation or not evaluation.challenger_eligible:
            return None

        return PromotionCandidate(
            candidate_id=f"PROMOTE-{uuid.uuid4().hex[:8]}",
            challenger_name=evaluation.challenger_name,
            champion_name=evaluation.champion_name,
            evaluation=evaluation,
        )
