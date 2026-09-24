"""Comprehensive unit tests for institutional risk engine components.

Covers:
- PositionSizer: Fixed Fractional, Volatility Parity (ATR), Kelly Criterion, Portfolio Heat, Leverage Caps
- CircuitBreaker: Multi-tier thresholds (NORMAL, THROTTLED, PAUSED, TRIPPED, FLASH_CRASH_HALT), Cool-off, Reset
- KillSwitch: In-memory/Redis checks, Emergency Liquidation callbacks (sync & async)
- RiskEngine: Full decision tree, limits enforcement, defensive sizing dampeners, regime adjustments
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from ai_crypto_trader.config.risk_config import RiskConfig
from ai_crypto_trader.core.enums import Direction, MarketRegime, RiskAction, SignalAction
from ai_crypto_trader.core.interfaces import PortfolioState, PositionSnapshot, Signal
from ai_crypto_trader.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)
from ai_crypto_trader.risk.kill_switch import KillSwitch
from ai_crypto_trader.risk.position_sizer import PositionSizer
from ai_crypto_trader.risk.risk_engine import RiskEngine


@pytest.fixture
def risk_config():
    return RiskConfig(
        max_trade_risk_pct=0.5,           # 0.5% max risk per trade
        daily_loss_limit_pct=2.0,         # 2.0% daily loss limit
        max_drawdown_halt_pct=10.0,       # 10.0% max drawdown halt
        max_drawdown_reduce_pct=5.0,      # 5.0% drawdown halving
        max_open_positions=3,             # Max 3 open positions
        max_leverage=3.0,                 # Max 3x leverage
        max_position_size_pct=30.0,       # Max 30% account equity in 1 asset
        max_correlated_exposure_pct=60.0, # Max 60% correlated crypto exposure
        max_trades_per_day=20,            # Max 20 trades per day
        max_consecutive_losses=4,         # Max 4 losses in a row
    )


@pytest.fixture
def sizer():
    return PositionSizer(
        max_trade_risk_pct=0.5,
        max_position_size_pct=30.0,
        max_leverage=3.0,
    )


@pytest.fixture
def sample_signal():
    return Signal(
        symbol="BTCUSDT",
        action=SignalAction.ENTER_LONG,
        direction=Direction.LONG,
        confidence=0.80,
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=53000.0,
        expected_return_pct=6.0,
        risk_pct=0.5,
        regime=MarketRegime.STRONG_UPTREND,
        reason="trend_following_breakout",
        strategy_version_id="strat_v1",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def clean_portfolio():
    return PortfolioState(
        total_equity=10000.0,
        available_cash=10000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        drawdown_pct=0.0,
        peak_equity=10000.0,
        open_positions=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Position Sizer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizer:
    def test_fixed_fractional_basic(self, sizer):
        result = sizer.fixed_fractional(
            equity=10000.0,
            entry=50000.0,
            stop_loss=49000.0,
        )
        # 0.5% of 10000 = $50 risk amount
        # Stop distance = 1000 -> quantity = 50 / 1000 = 0.05
        assert abs(result.quantity - 0.05) < 1e-6
        assert result.position_value == 2500.0
        assert abs(result.risk_amount - 50.0) < 1e-6
        assert abs(result.risk_pct - 0.5) < 1e-6

    def test_fixed_fractional_respects_max_position(self, sizer):
        # Tiny stop distance yields gigantic raw quantity -> must be capped at 30% of equity ($3000)
        result = sizer.fixed_fractional(
            equity=10000.0,
            entry=50000.0,
            stop_loss=49999.0,
        )
        max_qty = (10000 * 0.30) / 50000.0  # 0.06
        assert result.quantity <= max_qty + 1e-9

    def test_fixed_fractional_zero_stop_distance(self, sizer):
        result = sizer.fixed_fractional(
            equity=10000.0,
            entry=50000.0,
            stop_loss=50000.0,
        )
        assert result.quantity == 0.0
        assert result.position_value == 0.0

    def test_volatility_parity_basic(self, sizer):
        # ATR = 500, multiplier = 2.0 -> stop_distance = 1000
        result = sizer.volatility_parity(
            equity=10000.0,
            entry=50000.0,
            atr=500.0,
            atr_multiplier=2.0,
            risk_pct=0.5,
        )
        # 0.5% risk = $50 -> quantity = 50 / 1000 = 0.05
        assert abs(result.quantity - 0.05) < 1e-6
        assert result.stop_distance == 1000.0
        assert "volatility_parity" in result.method

    def test_volatility_parity_higher_atr_reduces_size(self, sizer):
        # ATR = 500 -> stop = 1000 -> qty = 50 / 1000 = 0.05 (value = $2500 < $3000 cap)
        res_low_vol = sizer.volatility_parity(equity=10000.0, entry=50000.0, atr=500.0)
        # ATR = 1000 -> stop = 2000 -> qty = 50 / 2000 = 0.025 (value = $1250 < $3000 cap)
        res_high_vol = sizer.volatility_parity(equity=10000.0, entry=50000.0, atr=1000.0)

        assert res_high_vol.quantity < res_low_vol.quantity
        assert abs(res_high_vol.quantity * 2 - res_low_vol.quantity) < 1e-6

    def test_kelly_positive_edge(self, sizer):
        result = sizer.kelly(
            equity=10000.0,
            entry=50000.0,
            stop_loss=49000.0,
            win_rate=0.55,
            avg_win_loss_ratio=1.5,
        )
        assert result.quantity > 0.0
        assert result.risk_pct <= 0.5  # Enforces hard risk cap

    def test_kelly_negative_edge_returns_zero(self, sizer):
        result = sizer.kelly(
            equity=10000.0,
            entry=50000.0,
            stop_loss=49000.0,
            win_rate=0.30,
            avg_win_loss_ratio=1.0,
        )
        assert result.quantity == 0.0

    def test_compute_portfolio_heat(self, sizer):
        positions = [
            PositionSnapshot(
                symbol="BTCUSDT",
                side=Direction.LONG,
                entry_price=50000.0,
                quantity=0.05,
                current_price=50100.0,
                unrealized_pnl=5.0,
                unrealized_pnl_pct=0.1,
                stop_loss=49000.0,  # $50 risk
                take_profit=53000.0,
            ),
            PositionSnapshot(
                symbol="ETHUSDT",
                side=Direction.LONG,
                entry_price=3000.0,
                quantity=1.0,
                current_price=3020.0,
                unrealized_pnl=20.0,
                unrealized_pnl_pct=0.67,
                stop_loss=2950.0,   # $50 risk
                take_profit=3200.0,
            ),
        ]
        heat = sizer.compute_portfolio_heat(positions, equity=10000.0)
        # Total risk = $100 on $10000 equity = 1.0% heat
        assert abs(heat - 1.0) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 2. Circuit Breaker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_circuit_breaker_starts_normal(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.NORMAL
        can_trade, reason = cb.can_trade()
        assert can_trade is True
        assert cb.get_size_multiplier() == 1.0

    def test_level1_triggers_throttled(self):
        cb = CircuitBreaker()
        # Daily loss 1.2% (above 1.0% level 1, below 1.5% level 2)
        state = cb.check_metrics(daily_loss_pct=-1.2, drawdown_pct=2.0)
        assert state == CircuitBreakerState.THROTTLED
        can_trade, _ = cb.can_trade()
        assert can_trade is True
        assert cb.get_size_multiplier() == 0.5  # Position size halved

    def test_level2_triggers_paused_with_cool_off(self):
        now = datetime.now(timezone.utc)
        cb = CircuitBreaker(CircuitBreakerConfig(cool_off_seconds=1800))
        # Daily loss 1.6% triggers Level 2
        state = cb.check_metrics(daily_loss_pct=-1.6, drawdown_pct=4.0, now=now)
        assert state == CircuitBreakerState.PAUSED
        can_trade, reason = cb.can_trade()
        assert can_trade is False
        assert "PAUSED" in reason
        assert cb.get_size_multiplier() == 0.0

        # Before cool-off expires: still paused
        mid_time = now + timedelta(seconds=900)
        cb._check_cool_off_expiry(mid_time)
        assert cb.state == CircuitBreakerState.PAUSED

        # After cool-off expires (1800s): resumes into THROTTLED
        after_time = now + timedelta(seconds=1801)
        cb._check_cool_off_expiry(after_time)
        assert cb.state == CircuitBreakerState.THROTTLED
        can_trade, _ = cb.can_trade()
        assert can_trade is True
        assert cb.get_size_multiplier() == 0.5

    def test_level3_triggers_hard_tripped(self):
        cb = CircuitBreaker()
        # Drawdown 11.0% triggers Level 3 TRIPPED
        state = cb.check_metrics(daily_loss_pct=0.0, drawdown_pct=11.0)
        assert state == CircuitBreakerState.TRIPPED
        can_trade, reason = cb.can_trade()
        assert can_trade is False
        assert "TRIPPED" in reason

    def test_flash_crash_detection(self):
        now = datetime.now(timezone.utc)
        cb = CircuitBreaker(CircuitBreakerConfig(flash_crash_threshold_pct=5.0, flash_crash_window_seconds=600))
        # Price drops from 50000 to 47000 (6% drop in 1 minute)
        cb.record_price_tick("BTCUSDT", 50000.0, now)
        tripped = cb.record_price_tick("BTCUSDT", 47000.0, now + timedelta(seconds=60))
        assert tripped is True
        assert cb.state == CircuitBreakerState.FLASH_CRASH_HALT
        can_trade, _ = cb.can_trade()
        assert can_trade is False

    def test_manual_reset(self):
        cb = CircuitBreaker()
        cb.check_metrics(daily_loss_pct=-3.0, drawdown_pct=12.0)
        assert cb.state == CircuitBreakerState.TRIPPED

        cb.reset(actor="risk_officer")
        assert cb.state == CircuitBreakerState.NORMAL
        can_trade, _ = cb.can_trade()
        assert can_trade is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kill Switch Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_kill_switch_starts_inactive(self):
        ks = KillSwitch()
        assert not await ks.is_active()
        assert not ks.is_engaged

    @pytest.mark.asyncio
    async def test_activate_and_deactivate(self):
        ks = KillSwitch()
        await ks.activate("emergency drawdown test")
        assert await ks.is_active()
        assert ks.is_engaged
        status = await ks.status()
        assert status["active"] is True
        assert status["reason"] == "emergency drawdown test"

        await ks.deactivate(actor="operator")
        assert not await ks.is_active()
        assert not ks.is_engaged

    @pytest.mark.asyncio
    async def test_emergency_liquidation_callbacks(self):
        ks = KillSwitch()
        sync_called = []
        async_called = []

        def sync_callback(reason: str):
            sync_called.append(reason)

        async def async_callback(reason: str):
            async_called.append(reason)

        ks.register_liquidation_callback(sync_callback)
        ks.register_liquidation_callback(async_callback)

        await ks.activate("flash crash liquidation")
        assert len(sync_called) == 1
        assert sync_called[0] == "flash crash liquidation"
        assert len(async_called) == 1
        assert async_called[0] == "flash crash liquidation"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Institutional Risk Engine Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskEngine:
    def test_risk_engine_approves_valid_signal(self, risk_config, sample_signal, clean_portfolio):
        engine = RiskEngine(config=risk_config)
        approved, qty, reason = engine.evaluate(sample_signal, clean_portfolio)
        assert approved is True
        assert qty > 0.0
        assert "Approved" in reason

    def test_rejection_when_kill_switch_active(self, risk_config, sample_signal, clean_portfolio):
        ks = KillSwitch()
        ks._active = True
        ks._reason = "Manual halt"
        engine = RiskEngine(config=risk_config, kill_switch=ks)

        decision = engine.evaluate_signal(sample_signal, clean_portfolio)
        assert decision.approved is False
        assert decision.action == RiskAction.KILL_SWITCH
        assert "Kill switch" in decision.reason

    def test_rejection_when_circuit_breaker_paused_or_tripped(self, risk_config, sample_signal, clean_portfolio):
        cb = CircuitBreaker()
        cb.check_metrics(daily_loss_pct=-2.5, drawdown_pct=11.0)  # Hard trip
        engine = RiskEngine(config=risk_config, circuit_breaker=cb)

        decision = engine.evaluate_signal(sample_signal, clean_portfolio)
        assert decision.approved is False
        assert decision.action == RiskAction.REJECT
        assert "Breaker" in decision.reason

    def test_rejection_on_daily_loss_limit(self, risk_config, sample_signal, clean_portfolio):
        engine = RiskEngine(config=risk_config)
        # Record loss exceeding 2.0% limit
        engine.record_trade_closed("BTCUSDT", position_value=2000.0, pnl_pct=-2.5)

        decision = engine.evaluate_signal(sample_signal, clean_portfolio)
        assert decision.approved is False
        assert "Daily loss limit" in decision.reason

    def test_rejection_on_max_drawdown_halt(self, risk_config, sample_signal):
        engine = RiskEngine(config=risk_config)
        portfolio_in_drawdown = PortfolioState(
            total_equity=8900.0,
            available_cash=8900.0,
            unrealized_pnl=0.0,
            realized_pnl=-1100.0,
            drawdown_pct=11.0,  # Above 10.0% limit
            peak_equity=10000.0,
        )
        decision = engine.evaluate_signal(sample_signal, portfolio_in_drawdown)
        assert decision.approved is False
        assert "drawdown halt" in decision.reason

    def test_drawdown_reduce_halves_position_size(self, risk_config, sample_signal, clean_portfolio):
        engine = RiskEngine(config=risk_config)
        # Normal baseline evaluation
        normal_dec = engine.evaluate_signal(sample_signal, clean_portfolio)

        # Portfolio with 6% drawdown (between 5% reduce and 10% halt)
        reduced_portfolio = PortfolioState(
            total_equity=9400.0,
            available_cash=9400.0,
            unrealized_pnl=0.0,
            realized_pnl=-600.0,
            drawdown_pct=6.0,
            peak_equity=10000.0,
        )
        reduced_dec = engine.evaluate_signal(sample_signal, reduced_portfolio)

        assert reduced_dec.approved is True
        assert reduced_dec.action == RiskAction.REDUCE_SIZE
        # Quantity should be approximately half of normal (adjusted for equity difference)
        assert reduced_dec.quantity < normal_dec.quantity * 0.6

    def test_rejection_on_max_open_positions(self, risk_config, sample_signal, clean_portfolio):
        engine = RiskEngine(config=risk_config)
        # Record 3 open positions (limit is 3)
        engine.record_trade_opened("BTCUSDT", 1000.0)
        engine.record_trade_opened("ETHUSDT", 1000.0)
        engine.record_trade_opened("SOLUSDT", 1000.0)

        decision = engine.evaluate_signal(sample_signal, clean_portfolio)
        assert decision.approved is False
        assert "Max open positions" in decision.reason

    def test_rejection_on_consecutive_losses(self, risk_config, sample_signal, clean_portfolio):
        engine = RiskEngine(config=risk_config)
        # 4 consecutive losses (limit is 4)
        for _ in range(4):
            engine.record_trade_closed("BTCUSDT", 500.0, pnl_pct=-0.1)

        decision = engine.evaluate_signal(sample_signal, clean_portfolio)
        assert decision.approved is False
        assert "Max consecutive losses" in decision.reason

    def test_asset_exposure_cap(self, risk_config, clean_portfolio):
        engine = RiskEngine(config=risk_config)
        # Signal with very small stop loss (could command large position value)
        signal = Signal(
            symbol="BTCUSDT",
            action=SignalAction.ENTER_LONG,
            direction=Direction.LONG,
            confidence=0.85,
            entry_price=50000.0,
            stop_loss=49950.0,  # $50 stop distance
            take_profit=51000.0,
            expected_return_pct=2.0,
            risk_pct=0.5,
            regime=MarketRegime.STRONG_UPTREND,
            reason="tight_stop_breakout",
            strategy_version_id="s1",
            timestamp=datetime.now(timezone.utc),
        )
        decision = engine.evaluate_signal(signal, clean_portfolio)
        assert decision.approved is True
        # Position value must not exceed max_position_size_pct (30% of $10,000 = $3,000)
        assert decision.position_value <= 3001.0

    def test_high_volatility_regime_dampens_size(self, risk_config, sample_signal, clean_portfolio):
        engine = RiskEngine(config=risk_config)
        normal_dec = engine.evaluate_signal(sample_signal, clean_portfolio)

        # High volatility signal
        from dataclasses import replace
        high_vol_signal = replace(sample_signal, regime=MarketRegime.HIGH_VOLATILITY)
        high_vol_dec = engine.evaluate_signal(high_vol_signal, clean_portfolio)

        assert high_vol_dec.approved is True
        assert high_vol_dec.action == RiskAction.REDUCE_SIZE
        # High volatility applies 0.70x multiplier
        assert abs(high_vol_dec.quantity - (normal_dec.quantity * 0.70)) < 1e-5

    def test_volatility_parity_sizing_mode(self, risk_config, sample_signal, clean_portfolio):
        engine = RiskEngine(config=risk_config)
        decision = engine.evaluate_signal(
            sample_signal,
            clean_portfolio,
            sizing_method="volatility_parity",
            atr=800.0,
        )
        assert decision.approved is True
        assert "volatility_parity" in decision.reason
        assert decision.quantity > 0.0

    def test_record_price_tick_flash_crash(self, risk_config, sample_signal, clean_portfolio):
        engine = RiskEngine(config=risk_config)
        now = datetime.now(timezone.utc)
        # Normal ticks
        engine.record_price_tick("BTCUSDT", 50000.0, now)
        # Crash tick: 10% drop
        tripped = engine.record_price_tick("BTCUSDT", 45000.0, now + timedelta(seconds=30))
        assert tripped is True

        decision = engine.evaluate_signal(sample_signal, clean_portfolio)
        assert decision.approved is False
        assert "FLASH_CRASH" in decision.reason
