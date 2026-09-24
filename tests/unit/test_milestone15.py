"""Milestone 15 Unit Tests — Deployment Gate, Disaster Recovery & Checklist.

Covers:
- DeploymentChecklist: individual and full run_all()
- DisasterRecovery: build_snapshot, write/load, prune, emergency_shutdown
- DeploymentReadiness report structure
- Integration: kill switch + circuit breaker + DR working together
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from datetime import datetime, timezone

import pytest

from ai_crypto_trader.core.interfaces import PortfolioState, PositionSnapshot
from ai_crypto_trader.core.enums import Direction
from ai_crypto_trader.risk.kill_switch import KillSwitch
from ai_crypto_trader.risk.circuit_breaker import CircuitBreaker, CircuitBreakerState
from ai_crypto_trader.monitoring.disaster_recovery import DisasterRecovery, RecoverySnapshot
from ai_crypto_trader.validation.deployment_checklist import (
    DeploymentChecklist,
    DeploymentReadiness,
    CheckStatus,
    CheckResult,
    LivePromotionThresholds,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def default_portfolio(equity: float = 10_000.0) -> PortfolioState:
    return PortfolioState(
        total_equity=equity,
        available_cash=equity * 0.9,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        drawdown_pct=0.0,
        peak_equity=equity,
    )


def good_paper_metrics() -> dict:
    return {
        "final_equity": 10_800.0,
        "sharpe": 1.2,
        "max_drawdown": 8.0,
        "trading_days": 30,
        "num_trades": 45,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DisasterRecovery Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDisasterRecovery:

    def test_build_snapshot_fields(self, tmp_path):
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        portfolio = default_portfolio(equity=9_750.0)
        ks = KillSwitch()
        cb = CircuitBreaker()

        snap = dr.build_snapshot(portfolio_state=portfolio, circuit_breaker=cb, kill_switch=ks)
        assert isinstance(snap, RecoverySnapshot)
        assert abs(snap.total_equity - 9_750.0) < 0.01
        assert snap.circuit_breaker_state == "NORMAL"
        assert snap.kill_switch_active is False
        assert snap.kill_switch_reason == ""

    def test_snapshot_with_active_kill_switch(self, tmp_path):
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        ks = KillSwitch()
        ks._active = True
        ks._reason = "Test reason"
        portfolio = default_portfolio()

        snap = dr.build_snapshot(portfolio_state=portfolio, kill_switch=ks)
        assert snap.kill_switch_active is True
        assert snap.kill_switch_reason == "Test reason"

    def test_snapshot_with_tripped_circuit_breaker(self, tmp_path):
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        cb = CircuitBreaker()
        cb.check_metrics(daily_loss_pct=-3.0, drawdown_pct=12.0)  # Trip it
        portfolio = default_portfolio()

        snap = dr.build_snapshot(portfolio_state=portfolio, circuit_breaker=cb)
        assert snap.circuit_breaker_state == "TRIPPED"

    def test_write_and_load_roundtrip(self, tmp_path):
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        portfolio = default_portfolio(equity=9_900.0)
        snap = dr.build_snapshot(portfolio_state=portfolio)
        path = dr.write_snapshot(snap)

        assert path.exists()
        loaded = dr.load_latest_checkpoint()
        assert loaded is not None
        assert abs(loaded.total_equity - 9_900.0) < 0.01
        assert loaded.snapshot_id == snap.snapshot_id

    def test_snapshot_serialization_roundtrip(self):
        snap = RecoverySnapshot(
            snapshot_id="test_001",
            captured_at=datetime.now(timezone.utc).isoformat(),
            total_equity=12_345.67,
            available_cash=10_000.0,
            unrealized_pnl=345.67,
            realized_pnl=2_000.0,
            drawdown_pct=3.5,
            peak_equity=13_000.0,
        )
        json_str = snap.to_json()
        restored = RecoverySnapshot.from_json(json_str)
        assert restored.snapshot_id == snap.snapshot_id
        assert abs(restored.total_equity - 12_345.67) < 0.01

    def test_list_checkpoints(self, tmp_path):
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        portfolio = default_portfolio()
        snap1 = dr.build_snapshot(portfolio_state=portfolio)
        snap1 = RecoverySnapshot(**{**snap1.to_dict(), "snapshot_id": "snapshot_20260101_000001"})
        snap2 = dr.build_snapshot(portfolio_state=portfolio)
        snap2 = RecoverySnapshot(**{**snap2.to_dict(), "snapshot_id": "snapshot_20260101_000002"})
        dr.write_snapshot(snap1)
        dr.write_snapshot(snap2)

        listings = dr.list_checkpoints()
        assert len(listings) == 2
        for entry in listings:
            assert "path" in entry
            assert "snapshot_id" in entry
            assert "total_equity" in entry

    @pytest.mark.asyncio
    async def test_save_checkpoint_async(self, tmp_path):
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        portfolio = default_portfolio(equity=10_500.0)
        path = await dr.save_checkpoint(portfolio_state=portfolio)
        assert path.exists()
        loaded = dr.load_latest_checkpoint()
        assert loaded is not None
        assert abs(loaded.total_equity - 10_500.0) < 0.01

    @pytest.mark.asyncio
    async def test_emergency_shutdown_engages_kill_switch(self, tmp_path):
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        portfolio = default_portfolio(equity=8_000.0)
        ks = KillSwitch()
        cb = CircuitBreaker()

        report = await dr.emergency_shutdown(
            portfolio_state=portfolio,
            circuit_breaker=cb,
            kill_switch=ks,
            reason="Unit test emergency",
        )
        assert report.snapshot_saved
        assert ks.is_engaged
        assert report.snapshot_path is not None

    @pytest.mark.asyncio
    async def test_emergency_shutdown_report_structure(self, tmp_path):
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        report = await dr.emergency_shutdown(
            portfolio_state=default_portfolio(),
            reason="Structure test",
        )
        assert hasattr(report, "initiated_at")
        assert hasattr(report, "completed_at")
        assert hasattr(report, "snapshot_saved")
        assert hasattr(report, "clean")
        assert isinstance(report.errors, list)


# ─────────────────────────────────────────────────────────────────────────────
# DeploymentChecklist Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeploymentChecklist:

    def _make_checklist(self, tmp_path) -> DeploymentChecklist:
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        ks = KillSwitch()
        cb = CircuitBreaker()
        return DeploymentChecklist(
            kill_switch=ks,
            circuit_breaker=cb,
            disaster_recovery=dr,
        )

    @pytest.mark.asyncio
    async def test_checklist_runs_without_errors(self, tmp_path):
        checklist = self._make_checklist(tmp_path)
        report = await checklist.run_all(paper_metrics=good_paper_metrics())
        assert isinstance(report, DeploymentReadiness)
        assert report.total_checks > 0

    @pytest.mark.asyncio
    async def test_full_green_report_ready_for_live(self, tmp_path):
        """All safety components nominal + good paper metrics → ready."""
        checklist = self._make_checklist(tmp_path)
        report = await checklist.run_all(paper_metrics=good_paper_metrics())
        # RED checks only come from hard failures; with fresh CB+KS we expect 0 reds
        assert report.failed == 0, f"Unexpected RED checks: {report.blocking_issues}"
        assert report.ready_for_live is True

    @pytest.mark.asyncio
    async def test_active_kill_switch_blocks_live(self, tmp_path):
        """Pre-engaged kill switch must produce RED on CHK-12."""
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        ks = KillSwitch()
        ks._active = True
        ks._reason = "Pre-existing problem"
        cb = CircuitBreaker()

        checklist = DeploymentChecklist(
            kill_switch=ks,
            circuit_breaker=cb,
            disaster_recovery=dr,
        )
        report = await checklist.run_all(paper_metrics=good_paper_metrics())
        chk12 = next((r for r in report.results if r.check_id == "CHK-12"), None)
        assert chk12 is not None
        assert chk12.status == CheckStatus.RED
        assert not report.ready_for_live

    @pytest.mark.asyncio
    async def test_tripped_circuit_breaker_blocks_live(self, tmp_path):
        """TRIPPED circuit breaker must produce RED on CHK-02."""
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        ks = KillSwitch()
        cb = CircuitBreaker()
        cb.check_metrics(daily_loss_pct=-5.0, drawdown_pct=15.0)

        checklist = DeploymentChecklist(
            kill_switch=ks,
            circuit_breaker=cb,
            disaster_recovery=dr,
        )
        report = await checklist.run_all(paper_metrics=good_paper_metrics())
        chk02 = next((r for r in report.results if r.check_id == "CHK-02"), None)
        assert chk02 is not None
        assert chk02.status == CheckStatus.RED
        assert not report.ready_for_live

    def test_paper_equity_check_pass(self):
        """Sufficient paper equity returns GREEN."""
        checklist = DeploymentChecklist(thresholds=LivePromotionThresholds(min_paper_equity=9_000.0))
        result = checklist._chk06_paper_equity({"final_equity": 10_000.0})
        assert result.status == CheckStatus.GREEN

    def test_paper_equity_check_fail(self):
        """Insufficient paper equity returns RED."""
        checklist = DeploymentChecklist(thresholds=LivePromotionThresholds(min_paper_equity=9_000.0))
        result = checklist._chk06_paper_equity({"final_equity": 8_500.0})
        assert result.status == CheckStatus.RED

    def test_paper_sharpe_check_pass(self):
        """Sharpe above threshold returns GREEN."""
        checklist = DeploymentChecklist(thresholds=LivePromotionThresholds(min_paper_sharpe=0.5))
        result = checklist._chk07_paper_sharpe({"sharpe": 1.2})
        assert result.status == CheckStatus.GREEN

    def test_paper_sharpe_check_fail(self):
        """Sharpe below threshold returns RED."""
        checklist = DeploymentChecklist(thresholds=LivePromotionThresholds(min_paper_sharpe=0.5))
        result = checklist._chk07_paper_sharpe({"sharpe": 0.1})
        assert result.status == CheckStatus.RED

    def test_paper_drawdown_check_pass(self):
        """Drawdown below max returns GREEN."""
        checklist = DeploymentChecklist(thresholds=LivePromotionThresholds(max_paper_drawdown_pct=15.0))
        result = checklist._chk08_paper_drawdown({"max_drawdown": 8.0})
        assert result.status == CheckStatus.GREEN

    def test_paper_drawdown_check_fail(self):
        """Drawdown above max returns RED."""
        checklist = DeploymentChecklist(thresholds=LivePromotionThresholds(max_paper_drawdown_pct=15.0))
        result = checklist._chk08_paper_drawdown({"max_drawdown": 20.0})
        assert result.status == CheckStatus.RED

    def test_paper_duration_check_pass(self):
        """Sufficient days and trades returns GREEN."""
        checklist = DeploymentChecklist(
            thresholds=LivePromotionThresholds(min_paper_trading_days=14, min_paper_trades=20)
        )
        result = checklist._chk09_paper_duration({"trading_days": 30, "num_trades": 50})
        assert result.status == CheckStatus.GREEN

    def test_paper_duration_check_fail_days(self):
        """Insufficient trading days returns RED."""
        checklist = DeploymentChecklist(
            thresholds=LivePromotionThresholds(min_paper_trading_days=14, min_paper_trades=20)
        )
        result = checklist._chk09_paper_duration({"trading_days": 5, "num_trades": 50})
        assert result.status == CheckStatus.RED

    def test_checklist_skips_when_no_paper_metrics(self):
        """Missing paper_metrics should produce YELLOW (skip), not RED."""
        checklist = DeploymentChecklist()
        result = checklist._chk06_paper_equity(None)
        assert result.status == CheckStatus.YELLOW

    @pytest.mark.asyncio
    async def test_report_summary_lines_readable(self, tmp_path):
        """Summary lines must be non-empty strings."""
        checklist = self._make_checklist(tmp_path)
        report = await checklist.run_all(paper_metrics=good_paper_metrics())
        lines = report.summary_lines()
        assert len(lines) > 5
        for line in lines:
            assert isinstance(line, str)

    @pytest.mark.asyncio
    async def test_report_to_dict_serialisable(self, tmp_path):
        """to_dict() must produce a JSON-serialisable dict."""
        import json
        checklist = self._make_checklist(tmp_path)
        report = await checklist.run_all(paper_metrics=good_paper_metrics())
        d = report.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert "ready_for_live" in json_str
        assert "results" in json_str

    @pytest.mark.asyncio
    async def test_null_components_produce_yellow_not_red(self, tmp_path):
        """Checklist with no components injected should produce YELLOW (skip) not RED."""
        checklist = DeploymentChecklist()  # No components
        report = await checklist.run_all(paper_metrics=None)
        # All async checks have None components → YELLOW
        reds = [r for r in report.results if r.status == CheckStatus.RED]
        assert len(reds) == 0, f"Expected no RED from empty checklist: {[r.detail for r in reds]}"
        assert report.ready_for_live is True  # No hard failures


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMilestone15Integration:

    @pytest.mark.asyncio
    async def test_full_deployment_gate_end_to_end(self, tmp_path):
        """End-to-end: fresh system + good paper metrics → deploy gate passes."""
        ks = KillSwitch()
        cb = CircuitBreaker()
        dr = DisasterRecovery(snapshot_dir=tmp_path)

        checklist = DeploymentChecklist(
            kill_switch=ks,
            circuit_breaker=cb,
            disaster_recovery=dr,
        )

        # Simulate good paper results
        paper_metrics = {
            "final_equity": 11_200.0,
            "sharpe": 1.5,
            "max_drawdown": 6.0,
            "trading_days": 21,
            "num_trades": 60,
        }

        report = await checklist.run_all(paper_metrics=paper_metrics)
        assert report.ready_for_live is True
        assert report.failed == 0

    @pytest.mark.asyncio
    async def test_emergency_then_recovery_sequence(self, tmp_path):
        """Simulate emergency shutdown → restart → load checkpoint → reset."""
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        ks = KillSwitch()
        cb = CircuitBreaker()
        portfolio = default_portfolio(equity=9_200.0)

        # 1. Emergency shutdown
        report = await dr.emergency_shutdown(
            portfolio_state=portfolio,
            circuit_breaker=cb,
            kill_switch=ks,
            reason="Flash crash cascade",
        )
        assert report.snapshot_saved
        assert ks.is_engaged

        # 2. Restart: load checkpoint
        loaded = dr.load_latest_checkpoint()
        assert loaded is not None
        assert abs(loaded.total_equity - 9_200.0) < 0.01
        # Kill switch was activated BEFORE snapshot was saved
        assert loaded.kill_switch_active is True

        # 3. Human operator deactivates kill switch
        ks2 = KillSwitch()   # New instance simulating restart
        assert not ks2.is_engaged

        # 4. Reset circuit breaker
        cb2 = CircuitBreaker()
        assert cb2.state == CircuitBreakerState.NORMAL

    @pytest.mark.asyncio
    async def test_kill_switch_round_trip_isolated(self):
        """Kill switch activate/deactivate does not bleed between instances."""
        ks1 = KillSwitch()
        ks2 = KillSwitch()

        await ks1.activate("Test isolation")
        assert ks1.is_engaged
        assert not ks2.is_engaged  # In-memory state is isolated per instance

        await ks1.deactivate("test")
        assert not ks1.is_engaged

    @pytest.mark.asyncio
    async def test_circuit_breaker_injection_and_reset(self):
        """CB survives injection → trip → reset cycle cleanly."""
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.NORMAL

        # Level 1
        cb.check_metrics(daily_loss_pct=-1.0, drawdown_pct=3.0)
        assert cb.state == CircuitBreakerState.THROTTLED

        # Level 3 override
        cb.check_metrics(daily_loss_pct=-3.0, drawdown_pct=12.0)
        assert cb.state == CircuitBreakerState.TRIPPED

        # Human reset
        cb.reset(actor="test_operator")
        assert cb.state == CircuitBreakerState.NORMAL
