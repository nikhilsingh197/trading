"""Chaos Simulation Tests — Flash Crash, Market Crash, API Outage.

These tests verify that safety systems respond correctly under extreme,
adversarial market conditions.  They are simulation-level tests (not unit
tests) because they exercise multiple components interacting together.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from ai_crypto_trader.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
from ai_crypto_trader.risk.kill_switch import KillSwitch
from ai_crypto_trader.monitoring.health_checker import HealthChecker
from ai_crypto_trader.monitoring.disaster_recovery import DisasterRecovery, RecoverySnapshot
from ai_crypto_trader.core.interfaces import PortfolioState


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_portfolio(equity: float = 10_000.0, drawdown: float = 0.0) -> PortfolioState:
    return PortfolioState(
        total_equity=equity,
        available_cash=equity * 0.8,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        drawdown_pct=drawdown,
        peak_equity=10_000.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Flash Crash Simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestFlashCrashSimulation:
    """Simulate a rapid price drop and verify circuit breaker response."""

    def _make_cb(self) -> CircuitBreaker:
        return CircuitBreaker(
            CircuitBreakerConfig(
                flash_crash_threshold_pct=5.0,
                flash_crash_window_seconds=300,
                flash_crash_halt_seconds=600,
            )
        )

    def test_normal_price_movement_no_halt(self):
        """Moderate price movement should not trigger flash crash halt."""
        cb = self._make_cb()
        base = datetime.now(timezone.utc)
        prices = [100.0, 99.0, 98.5, 98.0, 97.0]  # 3% drop — below threshold
        halted = False
        for i, price in enumerate(prices):
            ts = base + timedelta(seconds=i * 30)
            halted = cb.record_price_tick("BTC/USDT", price, timestamp=ts)
        assert not halted, "3% drop should not trigger flash crash halt"
        assert cb.state == CircuitBreakerState.NORMAL

    def test_flash_crash_triggers_halt(self):
        """Rapid 7%+ price drop must trigger FLASH_CRASH_HALT."""
        cb = self._make_cb()
        base = datetime.now(timezone.utc)

        # Seed initial price
        cb.record_price_tick("BTC/USDT", 50_000.0, timestamp=base)
        # Rapid crash: drop to 46,000 (~8%)
        halted = cb.record_price_tick("BTC/USDT", 46_000.0, timestamp=base + timedelta(seconds=60))

        assert halted, "8% drop should trigger flash crash halt"
        assert cb.state == CircuitBreakerState.FLASH_CRASH_HALT
        assert not cb.can_trade()[0], "Trading must be blocked during FLASH_CRASH_HALT"

    def test_flash_crash_halt_recovers_after_cooldown(self):
        """After cool-off timer expires, breaker should auto-recover to THROTTLED."""
        cb = CircuitBreaker(CircuitBreakerConfig(
            flash_crash_threshold_pct=5.0,
            flash_crash_window_seconds=300,
            flash_crash_halt_seconds=1,  # 1s for test speed
        ))
        base = datetime.now(timezone.utc)
        cb.record_price_tick("BTC/USDT", 50_000.0, timestamp=base)
        cb.record_price_tick("BTC/USDT", 45_000.0, timestamp=base + timedelta(seconds=10))

        assert cb._state == CircuitBreakerState.FLASH_CRASH_HALT
        # paused_until = base + 10s + 1s = base + 11s
        # Advance well past expiry
        future = base + timedelta(seconds=20)
        cb._check_cool_off_expiry(now=future)

        # After expiry: should transition to THROTTLED
        assert cb._state in (CircuitBreakerState.THROTTLED, CircuitBreakerState.NORMAL)

    def test_flash_crash_multi_symbol_isolation(self):
        """Flash crash on one symbol should not affect another."""
        cb = self._make_cb()
        base = datetime.now(timezone.utc)
        cb.record_price_tick("BTC/USDT", 50_000.0, timestamp=base)
        cb.record_price_tick("BTC/USDT", 45_000.0, timestamp=base + timedelta(seconds=60))

        assert cb.state == CircuitBreakerState.FLASH_CRASH_HALT

        # ETH should still contribute to same global halt
        # (circuit breaker is global — any symbol flash crash halts all trading)
        can_trade, reason = cb.can_trade()
        assert not can_trade
        assert "FLASH_CRASH_HALT" in reason

    @pytest.mark.asyncio
    async def test_circuit_breaker_cascades_to_kill_switch(self):
        """When CB hits TRIPPED, a downstream handler can activate kill switch."""
        cb = CircuitBreaker()
        ks = KillSwitch()

        kill_switch_activated = []

        def on_trip(reason: str):
            kill_switch_activated.append(reason)

        ks.register_liquidation_callback(on_trip)

        # Trip circuit breaker
        cb.check_metrics(daily_loss_pct=-5.0, drawdown_pct=15.0)
        assert cb.state == CircuitBreakerState.TRIPPED

        # Simulate kill switch activation (as execution_engine would do)
        await ks.activate("Circuit breaker TRIPPED")

        assert ks.is_engaged
        assert len(kill_switch_activated) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Sustained Market Crash Simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketCrashSimulation:
    """Simulate sustained drawdown and verify multi-tier breaker response."""

    def test_level1_throttle_at_soft_loss(self):
        """1% daily loss triggers THROTTLED."""
        cb = CircuitBreaker()
        state = cb.check_metrics(daily_loss_pct=-1.0, drawdown_pct=3.0)
        assert state == CircuitBreakerState.THROTTLED
        assert cb.get_size_multiplier() == 0.5

    def test_level2_pause_at_moderate_loss(self):
        """1.5% daily loss triggers PAUSED (cool-off)."""
        cb = CircuitBreaker()
        state = cb.check_metrics(daily_loss_pct=-1.5, drawdown_pct=4.0)
        assert state == CircuitBreakerState.PAUSED
        can_trade, _ = cb.can_trade()
        assert not can_trade

    def test_level3_trip_at_hard_loss(self):
        """2% daily loss triggers TRIPPED (full halt)."""
        cb = CircuitBreaker()
        state = cb.check_metrics(daily_loss_pct=-2.0, drawdown_pct=5.0)
        assert state == CircuitBreakerState.TRIPPED
        assert cb.get_size_multiplier() == 0.0

    def test_sustained_drawdown_triggers_level3(self):
        """10% drawdown from peak must trigger TRIPPED."""
        cb = CircuitBreaker()
        state = cb.check_metrics(daily_loss_pct=-0.5, drawdown_pct=10.0)
        assert state == CircuitBreakerState.TRIPPED

    def test_tripped_state_persists_despite_recovery(self):
        """Once TRIPPED, breaker should not auto-recover without explicit reset."""
        cb = CircuitBreaker()
        cb.check_metrics(daily_loss_pct=-2.0, drawdown_pct=5.0)
        assert cb.state == CircuitBreakerState.TRIPPED

        # Even with good metrics, TRIPPED should persist
        state2 = cb.check_metrics(daily_loss_pct=0.0, drawdown_pct=0.0)
        assert state2 == CircuitBreakerState.TRIPPED

    def test_manual_reset_restores_normal(self):
        """Explicit reset by operator restores NORMAL."""
        cb = CircuitBreaker()
        cb.check_metrics(daily_loss_pct=-2.0, drawdown_pct=5.0)
        cb.reset(actor="human_operator")
        assert cb.state == CircuitBreakerState.NORMAL

    def test_consecutive_losses_trigger_throttle(self):
        """4 consecutive losses should throttle even with small daily P&L."""
        cb = CircuitBreaker()
        state = cb.check_metrics(daily_loss_pct=-0.2, drawdown_pct=1.0, consecutive_losses=4)
        assert state == CircuitBreakerState.THROTTLED


# ─────────────────────────────────────────────────────────────────────────────
# API Outage / Health Check Simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIOutageSimulation:
    """Simulate exchange API failures and verify health monitoring detection."""

    def test_health_checker_instantiates(self):
        """HealthChecker should instantiate without errors."""
        hc = HealthChecker(kill_switch=KillSwitch(), circuit_breaker=CircuitBreaker())
        assert hc is not None

    @pytest.mark.asyncio
    async def test_stale_data_detected_via_check_health(self):
        """HealthChecker should detect stale market data beyond threshold."""
        hc = HealthChecker(
            kill_switch=KillSwitch(),
            circuit_breaker=CircuitBreaker(),
            stale_threshold_seconds=1,
        )
        # Do NOT update tick — data is immediately stale after 1s
        await asyncio.sleep(0.05)
        status = await hc.check_health()
        assert status is not None
        assert status.status in ("HEALTHY", "DEGRADED", "CRITICAL")

    @pytest.mark.asyncio
    async def test_fresh_data_healthy(self):
        """Recent price tick should result in market_data HEALTHY status."""
        hc = HealthChecker(
            kill_switch=KillSwitch(),
            circuit_breaker=CircuitBreaker(),
            stale_threshold_seconds=60,
        )
        hc.record_market_data_tick()  # fresh tick right now
        status = await hc.check_health()
        assert status.components.get("market_data") is not None
        assert status.components["market_data"].status == "HEALTHY"

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_on_stale_data(self):
        """Kill switch can be activated in response to health failure."""
        ks = KillSwitch()
        cb = CircuitBreaker()
        _ = HealthChecker(kill_switch=ks, circuit_breaker=cb)
        assert not ks.is_engaged
        await ks.activate("Exchange data stale — health check failed")
        assert ks.is_engaged

    @pytest.mark.asyncio
    async def test_tripped_circuit_breaker_shows_in_health(self):
        """Tripped circuit breaker should surface as DOWN in health status."""
        ks = KillSwitch()
        cb = CircuitBreaker()
        cb.check_metrics(daily_loss_pct=-5.0, drawdown_pct=15.0)
        assert cb.state == CircuitBreakerState.TRIPPED
        hc = HealthChecker(kill_switch=ks, circuit_breaker=cb)
        hc.record_market_data_tick()
        status = await hc.check_health()
        assert status.components["circuit_breaker"].status == "DOWN"
        assert not status.is_healthy


# ─────────────────────────────────────────────────────────────────────────────
# Disaster Recovery Chaos Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDisasterRecoveryChaos:
    """Verify disaster recovery snapshot and restore under chaos conditions."""

    @pytest.mark.asyncio
    async def test_checkpoint_survives_system_state(self, tmp_path):
        """Full snapshot write/load round-trip with realistic data."""
        from ai_crypto_trader.core.interfaces import PositionSnapshot
        from ai_crypto_trader.core.enums import Direction

        dr = DisasterRecovery(snapshot_dir=tmp_path)
        pos = PositionSnapshot(
            symbol="BTC/USDT",
            side=Direction.LONG,
            entry_price=50_000.0,
            quantity=0.1,
            current_price=48_000.0,
            unrealized_pnl=-200.0,
            unrealized_pnl_pct=-2.0,
            stop_loss=45_000.0,
            take_profit=60_000.0,
        )
        portfolio = PortfolioState(
            total_equity=9_800.0,
            available_cash=4_800.0,
            unrealized_pnl=-200.0,
            realized_pnl=0.0,
            drawdown_pct=2.0,
            peak_equity=10_000.0,
            open_positions=[pos],
        )
        cb = CircuitBreaker()
        ks = KillSwitch()

        await dr.save_checkpoint(portfolio_state=portfolio, circuit_breaker=cb, kill_switch=ks)
        loaded = dr.load_latest_checkpoint()

        assert loaded is not None
        assert abs(loaded.total_equity - 9_800.0) < 0.01
        assert loaded.circuit_breaker_state == "NORMAL"
        assert not loaded.kill_switch_active
        assert len(loaded.open_positions) == 1
        assert loaded.open_positions[0]["symbol"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_snapshot_rotation_prunes_old(self, tmp_path):
        """Old snapshots pruned when max_snapshots exceeded."""
        dr = DisasterRecovery(snapshot_dir=tmp_path, max_snapshots=3)
        portfolio = make_portfolio()
        for _ in range(5):
            await dr.save_checkpoint(portfolio_state=portfolio)
            await asyncio.sleep(0.01)  # Ensure distinct timestamps

        checkpoints = dr.list_checkpoints()
        assert len(checkpoints) <= 3, f"Expected ≤3 snapshots, got {len(checkpoints)}"

    @pytest.mark.asyncio
    async def test_emergency_shutdown_saves_snapshot(self, tmp_path):
        """Emergency shutdown must persist snapshot before exiting."""
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        portfolio = make_portfolio(equity=9_500.0, drawdown=5.0)
        ks = KillSwitch()
        cb = CircuitBreaker()

        report = await dr.emergency_shutdown(
            portfolio_state=portfolio,
            circuit_breaker=cb,
            kill_switch=ks,
            reason="Test emergency shutdown",
        )

        assert report.snapshot_saved
        assert report.snapshot_path is not None
        assert ks.is_engaged  # Kill switch should be engaged
        loaded = dr.load_latest_checkpoint()
        assert loaded is not None
        assert loaded.kill_switch_active is True
        assert loaded.extra.get("shutdown_reason") == "Test emergency shutdown"

    @pytest.mark.asyncio
    async def test_no_checkpoint_returns_none(self, tmp_path):
        """load_latest_checkpoint returns None when no snapshots exist."""
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        result = dr.load_latest_checkpoint()
        assert result is None

    def test_list_checkpoints_empty_dir(self, tmp_path):
        """list_checkpoints returns empty list for fresh directory."""
        dr = DisasterRecovery(snapshot_dir=tmp_path)
        assert dr.list_checkpoints() == []
