"""Unit and integration tests for autonomous research loop and promotion workflow."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from ai_crypto_trader.alerts.alert_manager import AlertManager
from ai_crypto_trader.backtesting.advanced_metrics import ComprehensiveMetrics
from ai_crypto_trader.core.enums import Direction, MarketRegime, SignalAction
from ai_crypto_trader.core.interfaces import Signal, StrategyABC
from ai_crypto_trader.research.champion_challenger import (
    ChampionChallengerManager,
    HeadToHeadEvaluation,
    PromotionCandidate,
)
from ai_crypto_trader.research.experiment_runner import ExperimentResult, ExperimentRunner
from ai_crypto_trader.research.hypothesis_generator import Hypothesis, HypothesisGenerator
from ai_crypto_trader.research.research_agent import ResearchAgent
from ai_crypto_trader.validation.approval_gate import ApprovalRequest, HumanApprovalGate
from ai_crypto_trader.validation.scorecard import ScorecardEvaluation, StrategyScorecard


# ─────────────────────────────────────────────────────────────────────────────
# 1. HypothesisGenerator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHypothesisGenerator:
    def test_generates_drawdown_curtailment_hypothesis(self):
        gen = HypothesisGenerator()
        metrics = {"max_drawdown_pct": 14.5, "win_rate": 0.52, "profit_factor": 1.6}
        hyps = gen.generate_hypotheses_from_metrics(
            strategy_name="EMA_Crossover",
            current_params={"atr_stop_multiplier": 2.0},
            metrics=metrics,
        )

        assert len(hyps) >= 1
        dd_hyp = next(h for h in hyps if h.target_metric == "DRAWDOWN")
        assert "atr_stop_multiplier" in dd_hyp.proposed_parameters
        assert dd_hyp.proposed_parameters["atr_stop_multiplier"] < 2.0
        assert dd_hyp.category == "VOLATILITY_ADAPTATION"

    def test_generates_regime_filter_hypothesis(self):
        gen = HypothesisGenerator()
        metrics = {"max_drawdown_pct": 6.0, "win_rate": 0.38, "profit_factor": 1.5}
        hyps = gen.generate_hypotheses_from_metrics(
            strategy_name="Breakout",
            current_params={"lookback": 20},
            metrics=metrics,
        )

        assert len(hyps) >= 1
        regime_hyp = next(h for h in hyps if h.target_metric == "WIN_RATE")
        assert "adx_threshold" in regime_hyp.proposed_parameters
        assert regime_hyp.category == "REGIME_FILTER"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ExperimentRunner Tests
# ─────────────────────────────────────────────────────────────────────────────

class MockStrategy(StrategyABC):
    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}

    @property
    def version_id(self) -> str:
        return "mock_v1"

    @property
    def name(self) -> str:
        return "MockStrategy"

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        return Signal(
            symbol="BTCUSDT",
            action=SignalAction.ENTER_LONG,
            direction=Direction.LONG,
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            expected_return_pct=4.0,
            risk_pct=0.5,
            regime=regime,
            reason="mock",
            strategy_version_id="mock_v1",
            timestamp=datetime.now(timezone.utc),
        )


class TestExperimentRunner:
    def test_experiment_evaluation(self):
        runner = ExperimentRunner()
        hyp = Hypothesis(
            id="HYP-001",
            title="Test Hyp",
            category="PARAMETER_TUNE",
            rationale="Test",
            base_strategy="MockStrategy",
            proposed_parameters={"param1": 10},
            target_metric="SHARPE",
            expected_improvement="None",
        )

        # Generate synthetic OHLCV data
        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        prices = [50000.0 + i * 20.0 for i in range(100)]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 50.0 for p in prices],
                "low": [p - 50.0 for p in prices],
                "close": prices,
                "volume": [10.0] * 100,
            },
            index=dates,
        )

        res = runner.run_experiment(hyp, MockStrategy(), df)
        assert res.experiment_id.startswith("EXP-")
        assert res.hypothesis == hyp
        assert res.status in ("PASS", "FAIL")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ChampionChallengerManager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChampionChallengerManager:
    def test_shadow_mode_evaluation_qualification(self):
        manager = ChampionChallengerManager(min_shadow_trades=5, min_shadow_days=0)
        manager.set_champion("Production_EMA_v1")
        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        manager.register_challenger("Challenger_ML_XGB", start_time=start_time)

        # Champion trades: modest return with variance
        champ_trades = [
            {"pnl": 20.0, "pnl_pct": 0.5, "fees": 1.0},
            {"pnl": -15.0, "pnl_pct": -0.3, "fees": 1.0},
            {"pnl": 25.0, "pnl_pct": 0.6, "fees": 1.0},
            {"pnl": -10.0, "pnl_pct": -0.2, "fees": 1.0},
            {"pnl": 15.0, "pnl_pct": 0.4, "fees": 1.0},
            {"pnl": -12.0, "pnl_pct": -0.25, "fees": 1.0},
        ]
        for t in champ_trades:
            manager.record_trade("CHAMPION", t)

        # Challenger trades: strong performance (higher Sharpe, higher returns, with variance)
        chall_trades = [
            {"pnl": 80.0, "pnl_pct": 2.0, "fees": 1.0},
            {"pnl": 110.0, "pnl_pct": 2.5, "fees": 1.0},
            {"pnl": 95.0, "pnl_pct": 2.2, "fees": 1.0},
            {"pnl": -20.0, "pnl_pct": -0.4, "fees": 1.0},
            {"pnl": 120.0, "pnl_pct": 2.8, "fees": 1.0},
            {"pnl": 105.0, "pnl_pct": 2.4, "fees": 1.0},
        ]
        for t in chall_trades:
            manager.record_trade("CHALLENGER", t)

        evaluation = manager.evaluate_head_to_head()
        assert evaluation is not None
        assert evaluation.challenger_name == "Challenger_ML_XGB"
        assert evaluation.sharpe_diff > 0
        assert evaluation.return_diff > 0
        assert evaluation.challenger_eligible is True
        assert "QUALIFIED" in evaluation.summary_verdict

        candidate = manager.generate_promotion_candidate()
        assert candidate is not None
        assert candidate.challenger_name == "Challenger_ML_XGB"
        assert candidate.champion_name == "Production_EMA_v1"


# ─────────────────────────────────────────────────────────────────────────────
# 4. HumanApprovalGate Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanApprovalGate:
    @pytest.mark.asyncio
    async def test_human_approval_promotes_champion(self):
        alert_mgr = AlertManager()
        gate = HumanApprovalGate(alert_manager=alert_mgr, operator_passcode="SECRET_KEY_123")
        gate.current_champion = "Legacy_Trend_v1"

        # Mock promotion candidate
        candidate = PromotionCandidate(
            candidate_id="CAND-001",
            challenger_name="AI_Ensemble_v2",
            champion_name="Legacy_Trend_v1",
            evaluation=MagicMock(
                sharpe_diff=0.85,
                profit_factor_diff=0.45,
                is_statistically_superior=True,
                p_value=0.012,
            ),
        )

        req = await gate.submit_candidate(candidate)
        assert req.status == "PENDING"
        assert len(gate.get_pending_requests()) == 1

        # Reject with invalid passcode
        ok, err = await gate.approve(req.request_id, operator_name="LeadTrader", passcode="WRONG")
        assert ok is False
        assert gate.current_champion == "Legacy_Trend_v1"

        # Approve with valid passcode
        ok, msg = await gate.approve(req.request_id, operator_name="LeadTrader", passcode="SECRET_KEY_123")
        assert ok is True
        assert gate.current_champion == "AI_Ensemble_v2"
        assert gate.archived_champion == "Legacy_Trend_v1"
        assert req.status == "APPROVED"

    @pytest.mark.asyncio
    async def test_human_rejection(self):
        gate = HumanApprovalGate()
        candidate = PromotionCandidate(
            candidate_id="CAND-002",
            challenger_name="Risky_Strat",
            champion_name="Safe_Champion",
            evaluation=MagicMock(sharpe_diff=0.1, profit_factor_diff=0.0, is_statistically_superior=False, p_value=0.2),
        )
        req = await gate.submit_candidate(candidate)
        ok, msg = await gate.reject(req.request_id, operator_name="RiskOfficer", reason="Too volatile")
        assert ok is True
        assert req.status == "REJECTED"
        assert gate.current_champion == "Default_Champion"

    @pytest.mark.asyncio
    async def test_champion_rollback(self):
        gate = HumanApprovalGate()
        gate.current_champion = "New_Unstable_Champion"
        gate.archived_champion = "Rock_Solid_Old_Champion"

        ok, msg = await gate.rollback(operator_name="LeadDev", reason="Degradation in live market")
        assert ok is True
        assert gate.current_champion == "Rock_Solid_Old_Champion"
        assert gate.archived_champion is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. ResearchAgent End-to-End Cycle Test
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchAgent:
    def test_run_research_cycle(self):
        agent = ResearchAgent()

        dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
        prices = [50000.0 + i * 15.0 for i in range(100)]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 40.0 for p in prices],
                "low": [p - 40.0 for p in prices],
                "close": prices,
                "volume": [10.0] * 100,
            },
            index=dates,
        )

        result = agent.run_research_cycle(
            strategy_name="EMA_Cross",
            current_params={"fast": 9, "slow": 21, "atr_stop_multiplier": 2.5},
            recent_metrics={"max_drawdown_pct": 12.0, "win_rate": 0.42, "profit_factor": 1.3},
            historical_data=df,
            strategy_factory=lambda p: MockStrategy(p),
        )

        assert result.hypotheses_generated > 0
        assert result.experiments_run == result.hypotheses_generated
