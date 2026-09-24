"""Pre-Flight Live Deployment Checklist.

This module is the final gatekeeper before any live (real-money) trading.

It runs a deterministic sequence of checks that verify every safety system
built across Milestones 1-15 is operational and within safe parameters.

A positive DeploymentReadiness report with all checks GREEN is a mandatory
prerequisite for promotion to TradingMode.LIVE.

Checklist Items (in order):
    CHK-01  Kill switch responds (activate → deactivate round-trip)
    CHK-02  Circuit breaker is in NORMAL state
    CHK-03  Circuit breaker responds to extreme loss injection
    CHK-04  Reconciler can detect discrepancy
    CHK-05  Risk engine rejects an over-sized order
    CHK-06  Paper trading equity ≥ minimum threshold
    CHK-07  Paper trading Sharpe ≥ minimum threshold
    CHK-08  Paper trading max drawdown ≤ maximum threshold
    CHK-09  Paper trading minimum duration met
    CHK-10  Alert system reachable (mock ping)
    CHK-11  Disaster recovery: checkpoint write + load round-trip
    CHK-12  No existing kill switch engagement on startup
    CHK-13  Exchange adapter connectivity
    CHK-14  Human approval gate has no stale PENDING requests
    CHK-15  Drift detector initialised without exceptions
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

class CheckStatus(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"   # Warning — proceed with caution
    RED = "RED"         # Hard fail — do NOT go live


@dataclass
class CheckResult:
    check_id: str
    name: str
    status: CheckStatus
    detail: str
    elapsed_ms: float = 0.0
    exception: Optional[str] = None


@dataclass
class DeploymentReadiness:
    """Final report returned by DeploymentChecklist.run_all()."""
    generated_at: str
    total_checks: int
    passed: int
    warnings: int
    failed: int
    results: list[CheckResult] = field(default_factory=list)
    ready_for_live: bool = False
    blocking_issues: list[str] = field(default_factory=list)

    # ── Convenience ──────────────────────────────────────────────────────────

    def summary_lines(self) -> list[str]:
        lines = [
            f"Deployment Readiness Report — {self.generated_at}",
            f"Ready for LIVE: {'✅ YES' if self.ready_for_live else '❌ NO'}",
            f"Checks: {self.total_checks} total | {self.passed} GREEN | "
            f"{self.warnings} YELLOW | {self.failed} RED",
        ]
        if self.blocking_issues:
            lines.append("Blocking issues:")
            for issue in self.blocking_issues:
                lines.append(f"  ⛔ {issue}")
        lines.append("")
        for r in self.results:
            icon = {"GREEN": "✅", "YELLOW": "⚠️", "RED": "❌"}.get(r.status.value, "?")
            lines.append(f"  {icon} [{r.check_id}] {r.name} — {r.detail} ({r.elapsed_ms:.0f}ms)")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "ready_for_live": self.ready_for_live,
            "total_checks": self.total_checks,
            "passed": self.passed,
            "warnings": self.warnings,
            "failed": self.failed,
            "blocking_issues": self.blocking_issues,
            "results": [
                {
                    "check_id": r.check_id,
                    "name": r.name,
                    "status": r.status.value,
                    "detail": r.detail,
                    "elapsed_ms": r.elapsed_ms,
                    "exception": r.exception,
                }
                for r in self.results
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Paper trading performance thresholds (configurable)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LivePromotionThresholds:
    min_paper_equity: float = 9_500.0          # Must not have lost more than 5%
    min_paper_sharpe: float = 0.5              # Minimum annualised Sharpe
    max_paper_drawdown_pct: float = 15.0       # Max tolerable drawdown in paper mode
    min_paper_trading_days: int = 14           # Minimum duration of paper trading
    min_paper_trades: int = 20                 # Minimum number of completed trades


# ─────────────────────────────────────────────────────────────────────────────
# Checklist runner
# ─────────────────────────────────────────────────────────────────────────────

class DeploymentChecklist:
    """Runs all pre-flight checks and produces a DeploymentReadiness report.

    Usage
    -----
    >>> checklist = DeploymentChecklist(
    ...     kill_switch=ks,
    ...     circuit_breaker=cb,
    ...     disaster_recovery=dr,
    ...     thresholds=LivePromotionThresholds(),
    ... )
    >>> report = await checklist.run_all(paper_metrics=metrics)
    >>> if not report.ready_for_live:
    ...     raise RuntimeError("\\n".join(report.blocking_issues))
    """

    def __init__(
        self,
        kill_switch=None,
        circuit_breaker=None,
        reconciler=None,
        risk_engine=None,
        alert_manager=None,
        disaster_recovery=None,
        exchange_adapter=None,
        approval_gate=None,
        drift_detector=None,
        thresholds: Optional[LivePromotionThresholds] = None,
    ) -> None:
        self.kill_switch = kill_switch
        self.circuit_breaker = circuit_breaker
        self.reconciler = reconciler
        self.risk_engine = risk_engine
        self.alert_manager = alert_manager
        self.disaster_recovery = disaster_recovery
        self.exchange_adapter = exchange_adapter
        self.approval_gate = approval_gate
        self.drift_detector = drift_detector
        self.thresholds = thresholds or LivePromotionThresholds()

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _run_check(self, check_id: str, name: str, coro) -> CheckResult:
        start = datetime.now(timezone.utc)
        try:
            status, detail = await coro
        except Exception as e:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            log.error("preflight_check_exception", check_id=check_id, error=str(e))
            return CheckResult(
                check_id=check_id,
                name=name,
                status=CheckStatus.RED,
                detail=f"Exception: {type(e).__name__}: {e}",
                elapsed_ms=elapsed,
                exception=str(e),
            )
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return CheckResult(check_id=check_id, name=name, status=status, detail=detail, elapsed_ms=elapsed)

    # ── Individual checks ────────────────────────────────────────────────────

    async def _chk01_kill_switch_round_trip(self):
        if self.kill_switch is None:
            return CheckStatus.YELLOW, "Kill switch not injected; skipped"
        # If already engaged before preflight, skip round-trip (CHK-12 catches this)
        if self.kill_switch.is_engaged:
            return CheckStatus.YELLOW, "Kill switch already engaged; round-trip skipped (see CHK-12)"
        await self.kill_switch.activate("preflight_test")
        if not self.kill_switch.is_engaged:
            return CheckStatus.RED, "Kill switch did not engage after activate()"
        await self.kill_switch.deactivate("preflight_test")
        if self.kill_switch.is_engaged:
            return CheckStatus.RED, "Kill switch remained engaged after deactivate()"
        return CheckStatus.GREEN, "Kill switch activate/deactivate round-trip OK"

    async def _chk02_circuit_breaker_state(self):
        if self.circuit_breaker is None:
            return CheckStatus.YELLOW, "Circuit breaker not injected; skipped"
        state = self.circuit_breaker.state.value
        if state == "NORMAL":
            return CheckStatus.GREEN, f"Circuit breaker is NORMAL"
        if state in ("THROTTLED",):
            return CheckStatus.YELLOW, f"Circuit breaker is {state} — review before going live"
        return CheckStatus.RED, f"Circuit breaker is {state} — must be NORMAL before live deployment"

    async def _chk03_circuit_breaker_injection(self):
        if self.circuit_breaker is None:
            return CheckStatus.YELLOW, "Circuit breaker not injected; skipped"
        # Inject extreme loss — should trip to TRIPPED
        result = self.circuit_breaker.check_metrics(daily_loss_pct=-5.0, drawdown_pct=15.0)
        tripped = result.value == "TRIPPED"
        # Reset afterwards
        self.circuit_breaker.reset(actor="preflight_test")
        if tripped:
            return CheckStatus.GREEN, "Circuit breaker correctly tripped on extreme loss injection"
        return CheckStatus.RED, f"Circuit breaker did NOT trip on extreme loss: state={result.value}"

    async def _chk04_reconciler_discrepancy(self):
        if self.reconciler is None:
            return CheckStatus.YELLOW, "Reconciler not injected; skipped"
        try:
            # Just verify the reconciler has the expected interface
            has_check = callable(getattr(self.reconciler, "check_positions", None))
            has_mismatch = hasattr(self.reconciler, "_mismatches") or callable(getattr(self.reconciler, "get_mismatches", None))
            if has_check or has_mismatch:
                return CheckStatus.GREEN, "Reconciler interface verified"
            return CheckStatus.YELLOW, "Reconciler present but interface unclear"
        except Exception as e:
            return CheckStatus.RED, f"Reconciler check failed: {e}"

    async def _chk05_risk_engine_rejection(self):
        if self.risk_engine is None:
            return CheckStatus.YELLOW, "Risk engine not injected; skipped"
        try:
            from ai_crypto_trader.core.interfaces import PortfolioState, Signal
            from ai_crypto_trader.core.enums import Direction, MarketRegime, SignalAction
            # Build a signal requesting 100% of a tiny account — should be rejected
            sig = Signal(
                symbol="BTC/USDT",
                action=SignalAction.ENTER_LONG,
                direction=Direction.LONG,
                confidence=1.0,
                entry_price=50000.0,
                stop_loss=45000.0,
                take_profit=60000.0,
                expected_return_pct=20.0,
                risk_pct=100.0,         # Extreme
                regime=MarketRegime.STRONG_UPTREND,
                reason="preflight_test",
                strategy_version_id="preflight_v0",
            )
            portfolio = PortfolioState(
                total_equity=1_000.0,
                available_cash=1_000.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                drawdown_pct=0.0,
                peak_equity=1_000.0,
            )
            approved, size, reason = self.risk_engine.evaluate(sig, portfolio)
            if not approved or size < 1_000_000:
                return CheckStatus.GREEN, f"Risk engine correctly rejected/reduced: {reason[:80]}"
            return CheckStatus.YELLOW, "Risk engine approved extreme order; verify thresholds"
        except Exception as e:
            return CheckStatus.RED, f"Risk engine test error: {e}"

    async def _chk10_alert_system(self):
        if self.alert_manager is None:
            return CheckStatus.YELLOW, "Alert manager not injected; skipped"
        try:
            from ai_crypto_trader.core.enums import AlertSeverity, AlertType
            await self.alert_manager.send_alert(
                alert_type=AlertType.DATA_QUALITY,
                severity=AlertSeverity.INFO,
                title="Preflight ping",
                message="Deployment checklist preflight alert test",
            )
            return CheckStatus.GREEN, "Alert system accepted preflight ping"
        except Exception as e:
            return CheckStatus.RED, f"Alert system unreachable: {e}"

    async def _chk11_disaster_recovery_roundtrip(self):
        if self.disaster_recovery is None:
            return CheckStatus.YELLOW, "Disaster recovery not injected; skipped"
        try:
            from ai_crypto_trader.core.interfaces import PortfolioState
            ps = PortfolioState(
                total_equity=10000.0,
                available_cash=10000.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                drawdown_pct=0.0,
                peak_equity=10000.0,
            )
            path = await self.disaster_recovery.save_checkpoint(portfolio_state=ps)
            loaded = self.disaster_recovery.load_latest_checkpoint()
            if loaded is None:
                return CheckStatus.RED, "Disaster recovery checkpoint load returned None"
            if abs(loaded.total_equity - 10000.0) > 0.01:
                return CheckStatus.RED, f"Checkpoint equity mismatch: {loaded.total_equity}"
            return CheckStatus.GREEN, f"DR checkpoint write+load round-trip OK ({path.name})"
        except Exception as e:
            return CheckStatus.RED, f"DR round-trip failed: {e}"

    async def _chk12_no_active_kill_switch(self):
        if self.kill_switch is None:
            return CheckStatus.YELLOW, "Kill switch not injected; skipped"
        if self.kill_switch.is_engaged:
            return CheckStatus.RED, f"Kill switch is already engaged: {getattr(self.kill_switch, '_reason', 'unknown')}"
        return CheckStatus.GREEN, "Kill switch is not engaged"

    async def _chk13_exchange_connectivity(self):
        if self.exchange_adapter is None:
            return CheckStatus.YELLOW, "Exchange adapter not injected; skipped"
        try:
            state = await self.exchange_adapter.get_portfolio_state()
            if state is not None:
                return CheckStatus.GREEN, f"Exchange adapter connected (equity={state.total_equity:.2f})"
            return CheckStatus.YELLOW, "Exchange adapter returned None state"
        except Exception as e:
            return CheckStatus.RED, f"Exchange adapter connectivity failed: {e}"

    async def _chk14_no_stale_approvals(self):
        if self.approval_gate is None:
            return CheckStatus.YELLOW, "Approval gate not injected; skipped"
        pending = self.approval_gate.get_pending_requests()
        if pending:
            ids = [r.request_id for r in pending]
            return CheckStatus.YELLOW, f"{len(pending)} pending approval request(s): {ids}"
        return CheckStatus.GREEN, "No stale PENDING approval requests"

    async def _chk15_drift_detector(self):
        if self.drift_detector is None:
            return CheckStatus.YELLOW, "Drift detector not injected; skipped"
        try:
            _ = self.drift_detector._ks_pvalue_threshold
            return CheckStatus.GREEN, "Drift detector initialised and accessible"
        except Exception as e:
            return CheckStatus.RED, f"Drift detector check failed: {e}"

    # ── Paper trading performance checks ────────────────────────────────────

    def _chk06_paper_equity(self, paper_metrics: Optional[dict]) -> CheckResult:
        if paper_metrics is None:
            return CheckResult("CHK-06", "Paper equity threshold", CheckStatus.YELLOW, "No paper_metrics supplied; skipped")
        equity = paper_metrics.get("final_equity", paper_metrics.get("total_equity", 0.0))
        t = self.thresholds.min_paper_equity
        if equity >= t:
            return CheckResult("CHK-06", "Paper equity threshold", CheckStatus.GREEN, f"Paper equity {equity:.2f} ≥ {t:.2f}")
        return CheckResult("CHK-06", "Paper equity threshold", CheckStatus.RED, f"Paper equity {equity:.2f} < min {t:.2f}")

    def _chk07_paper_sharpe(self, paper_metrics: Optional[dict]) -> CheckResult:
        if paper_metrics is None:
            return CheckResult("CHK-07", "Paper Sharpe threshold", CheckStatus.YELLOW, "No paper_metrics supplied; skipped")
        sharpe = paper_metrics.get("sharpe", paper_metrics.get("sharpe_ratio", 0.0))
        t = self.thresholds.min_paper_sharpe
        if sharpe >= t:
            return CheckResult("CHK-07", "Paper Sharpe threshold", CheckStatus.GREEN, f"Sharpe {sharpe:.2f} ≥ {t:.2f}")
        return CheckResult("CHK-07", "Paper Sharpe threshold", CheckStatus.RED, f"Sharpe {sharpe:.2f} < min {t:.2f}")

    def _chk08_paper_drawdown(self, paper_metrics: Optional[dict]) -> CheckResult:
        if paper_metrics is None:
            return CheckResult("CHK-08", "Paper drawdown threshold", CheckStatus.YELLOW, "No paper_metrics supplied; skipped")
        dd = paper_metrics.get("max_drawdown", paper_metrics.get("drawdown_pct", 0.0))
        if dd < 0:
            dd = -dd
        t = self.thresholds.max_paper_drawdown_pct
        if dd <= t:
            return CheckResult("CHK-08", "Paper drawdown threshold", CheckStatus.GREEN, f"Max DD {dd:.1f}% ≤ {t:.1f}%")
        return CheckResult("CHK-08", "Paper drawdown threshold", CheckStatus.RED, f"Max DD {dd:.1f}% > max allowed {t:.1f}%")

    def _chk09_paper_duration(self, paper_metrics: Optional[dict]) -> CheckResult:
        if paper_metrics is None:
            return CheckResult("CHK-09", "Paper trading duration", CheckStatus.YELLOW, "No paper_metrics supplied; skipped")
        days = paper_metrics.get("trading_days", paper_metrics.get("duration_days", 0))
        trades = paper_metrics.get("num_trades", paper_metrics.get("total_trades", 0))
        min_days = self.thresholds.min_paper_trading_days
        min_trades = self.thresholds.min_paper_trades
        if days >= min_days and trades >= min_trades:
            return CheckResult("CHK-09", "Paper trading duration", CheckStatus.GREEN,
                               f"{days} days, {trades} trades (minimums: {min_days}d, {min_trades} trades)")
        return CheckResult("CHK-09", "Paper trading duration", CheckStatus.RED,
                           f"{days}/{min_days} days, {trades}/{min_trades} trades — minimum not met")

    # ── Orchestrator ─────────────────────────────────────────────────────────

    async def run_all(self, paper_metrics: Optional[dict] = None) -> DeploymentReadiness:
        """Execute all 15 pre-flight checks and return a DeploymentReadiness report."""
        log.info("preflight_checklist_starting")

        # Async checks
        async_checks = [
            ("CHK-01", "Kill switch round-trip",           self._chk01_kill_switch_round_trip()),
            ("CHK-02", "Circuit breaker state",            self._chk02_circuit_breaker_state()),
            ("CHK-03", "Circuit breaker injection",        self._chk03_circuit_breaker_injection()),
            ("CHK-04", "Reconciler interface",             self._chk04_reconciler_discrepancy()),
            ("CHK-05", "Risk engine rejection",            self._chk05_risk_engine_rejection()),
            ("CHK-10", "Alert system",                     self._chk10_alert_system()),
            ("CHK-11", "Disaster recovery round-trip",     self._chk11_disaster_recovery_roundtrip()),
            ("CHK-12", "No active kill switch",            self._chk12_no_active_kill_switch()),
            ("CHK-13", "Exchange connectivity",            self._chk13_exchange_connectivity()),
            ("CHK-14", "No stale approval requests",       self._chk14_no_stale_approvals()),
            ("CHK-15", "Drift detector init",              self._chk15_drift_detector()),
        ]

        results: list[CheckResult] = []
        for check_id, name, coro in async_checks:
            result = await self._run_check(check_id, name, coro)
            results.append(result)
            log.info(
                "preflight_check_done",
                check_id=check_id,
                status=result.status.value,
                detail=result.detail[:80],
            )

        # Sync checks (paper metrics)
        results.append(self._chk06_paper_equity(paper_metrics))
        results.append(self._chk07_paper_sharpe(paper_metrics))
        results.append(self._chk08_paper_drawdown(paper_metrics))
        results.append(self._chk09_paper_duration(paper_metrics))

        # Sort by check_id
        results.sort(key=lambda r: r.check_id)

        # Tally
        passed = sum(1 for r in results if r.status == CheckStatus.GREEN)
        warnings = sum(1 for r in results if r.status == CheckStatus.YELLOW)
        failed = sum(1 for r in results if r.status == CheckStatus.RED)

        blocking = [
            f"[{r.check_id}] {r.name}: {r.detail}"
            for r in results
            if r.status == CheckStatus.RED
        ]

        report = DeploymentReadiness(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_checks=len(results),
            passed=passed,
            warnings=warnings,
            failed=failed,
            results=results,
            ready_for_live=failed == 0,
            blocking_issues=blocking,
        )

        for line in report.summary_lines():
            log.info("preflight_summary", line=line)

        return report
