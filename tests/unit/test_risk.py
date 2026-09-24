"""Unit tests for risk management components."""
from __future__ import annotations

import pytest

from ai_crypto_trader.core.enums import Direction, MarketRegime, SignalAction
from ai_crypto_trader.core.interfaces import PortfolioState, Signal
from ai_crypto_trader.risk.position_sizer import PositionSizer


@pytest.fixture
def sizer():
    return PositionSizer(
        max_trade_risk_pct=0.5,   # 0.5% max risk
        max_position_size_pct=30.0,  # 30% max position
    )


@pytest.fixture
def sample_signal():
    from datetime import datetime, timezone
    return Signal(
        symbol="BTCUSDT",
        action=SignalAction.ENTER_LONG,
        direction=Direction.LONG,
        confidence=0.75,
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        expected_return_pct=4.0,
        risk_pct=2.0,
        regime=MarketRegime.STRONG_UPTREND,
        reason="test",
        strategy_version_id="v1",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def portfolio():
    return PortfolioState(
        total_equity=10000.0,
        available_cash=10000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        drawdown_pct=0.0,
        peak_equity=10000.0,
    )


class TestPositionSizer:
    def test_fixed_fractional_basic(self, sizer):
        result = sizer.fixed_fractional(
            equity=10000.0,
            entry=50000.0,
            stop_loss=49000.0,
        )
        # 0.5% of 10000 = 50 at risk
        # Stop distance = 1000
        # Expected qty = 50 / 1000 = 0.05
        assert abs(result.quantity - 0.05) < 1e-6
        assert result.risk_pct <= 0.5  # Should not exceed max_trade_risk

    def test_fixed_fractional_respects_max_position(self, sizer):
        # Large risk should be capped by max position size
        result = sizer.fixed_fractional(
            equity=10000.0,
            entry=50000.0,
            stop_loss=49999.0,  # Tiny stop = huge qty, must be capped
        )
        max_qty = (10000 * 0.30) / 50000  # 30% of equity
        assert result.quantity <= max_qty + 1e-9

    def test_zero_stop_returns_zero(self, sizer):
        result = sizer.fixed_fractional(
            equity=10000.0,
            entry=50000.0,
            stop_loss=50000.0,  # Zero stop distance
        )
        assert result.quantity == 0.0

    def test_kelly_positive_edge(self, sizer):
        result = sizer.kelly(
            equity=10000.0,
            entry=50000.0,
            stop_loss=49000.0,
            win_rate=0.55,
            avg_win_loss_ratio=1.5,
        )
        assert result.quantity > 0
        assert result.risk_pct <= 0.5

    def test_kelly_negative_edge_returns_zero(self, sizer):
        result = sizer.kelly(
            equity=10000.0,
            entry=50000.0,
            stop_loss=49000.0,
            win_rate=0.3,   # Negative edge
            avg_win_loss_ratio=1.0,
        )
        assert result.quantity == 0.0


class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_kill_switch_starts_inactive(self):
        from ai_crypto_trader.risk.kill_switch import KillSwitch
        ks = KillSwitch(redis_client=None)  # In-memory mode
        assert not await ks.is_active()

    @pytest.mark.asyncio
    async def test_activate_deactivate(self):
        from ai_crypto_trader.risk.kill_switch import KillSwitch
        ks = KillSwitch(redis_client=None)
        await ks.activate("test reason")
        assert await ks.is_active()
        await ks.deactivate(actor="test")
        assert not await ks.is_active()

    @pytest.mark.asyncio
    async def test_kill_switch_status(self):
        from ai_crypto_trader.risk.kill_switch import KillSwitch
        ks = KillSwitch(redis_client=None)
        await ks.activate("drawdown exceeded")
        status = await ks.status()
        assert status["active"] is True
        assert status["reason"] == "drawdown exceeded"
