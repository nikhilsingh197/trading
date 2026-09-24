"""Unit and integration tests for exchange connectivity and execution engine."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import pytest

from ai_crypto_trader.alerts.alert_manager import AlertManager
from ai_crypto_trader.config.risk_config import RiskConfig
from ai_crypto_trader.core.enums import Direction, MarketRegime, OrderStatus, OrderType, Side, SignalAction
from ai_crypto_trader.core.interfaces import OrderRequest, PortfolioState, PositionSnapshot, Signal
from ai_crypto_trader.exchange.binance_adapter import BinanceAdapter
from ai_crypto_trader.execution.execution_engine import ExecutionEngine
from ai_crypto_trader.execution.order_manager import OrderManager
from ai_crypto_trader.execution.reconciler import Reconciler
from ai_crypto_trader.risk.kill_switch import KillSwitch
from ai_crypto_trader.risk.risk_engine import RiskEngine


@pytest.fixture
def binance_sandbox():
    adapter = BinanceAdapter(
        api_key="",
        api_secret="",
        testnet=True,
        initial_sandbox_balance=10000.0,
    )
    return adapter


@pytest.fixture
def risk_config():
    return RiskConfig(
        max_trade_risk_pct=0.5,
        daily_loss_limit_pct=2.0,
        max_drawdown_halt_pct=10.0,
        max_drawdown_reduce_pct=5.0,
        max_open_positions=3,
        max_leverage=3.0,
        max_position_size_pct=30.0,
        max_correlated_exposure_pct=60.0,
        max_trades_per_day=20,
        max_consecutive_losses=4,
    )


@pytest.fixture
def test_signal():
    return Signal(
        symbol="BTCUSDT",
        action=SignalAction.ENTER_LONG,
        direction=Direction.LONG,
        confidence=0.82,
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=53000.0,
        expected_return_pct=6.0,
        risk_pct=0.5,
        regime=MarketRegime.STRONG_UPTREND,
        reason="momentum_trend_breakout",
        strategy_version_id="strat_v1",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def test_portfolio():
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
# 1. BinanceAdapter Sandbox Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBinanceAdapter:
    @pytest.mark.asyncio
    async def test_connect_and_ticker(self, binance_sandbox):
        await binance_sandbox.connect()
        ticker = await binance_sandbox.fetch_ticker("BTCUSDT")
        assert ticker["last"] == 50000.0
        assert ticker["bid"] < ticker["ask"]

    @pytest.mark.asyncio
    async def test_order_book_fetch(self, binance_sandbox):
        ob = await binance_sandbox.fetch_order_book("BTCUSDT", limit=5)
        assert "bids" in ob
        assert "asks" in ob
        assert len(ob["bids"]) == 5
        assert len(ob["asks"]) == 5

    @pytest.mark.asyncio
    async def test_create_order_fills_and_adjusts_balance(self, binance_sandbox):
        res = await binance_sandbox.create_order(
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.04,
        )
        assert res.status == OrderStatus.FILLED
        assert res.filled_qty == 0.04
        assert res.avg_fill_price == 50000.0
        assert res.fees > 0.0

        # Position is recorded
        positions = await binance_sandbox.fetch_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTCUSDT"
        assert positions[0].quantity == 0.04
        assert positions[0].side == Direction.LONG

        # Balance is updated
        bal = await binance_sandbox.fetch_balance()
        assert bal["used"] == 2000.0  # 0.04 * 50000
        assert bal["free"] < 10000.0

    @pytest.mark.asyncio
    async def test_cancel_order(self, binance_sandbox):
        # Manually inject an open pending order in sandbox
        from ai_crypto_trader.core.interfaces import OrderResult
        cl_id = "TEST-ORDER-1"
        binance_sandbox._sandbox_orders[cl_id] = OrderResult(
            client_order_id=cl_id,
            exchange_order_id="EXT-1",
            status=OrderStatus.SUBMITTED,
            filled_qty=0.0,
            avg_fill_price=None,
            fees=0.0,
        )

        open_orders = await binance_sandbox.fetch_open_orders()
        assert len(open_orders) == 1

        success = await binance_sandbox.cancel_order(cl_id)
        assert success is True

        open_after = await binance_sandbox.fetch_open_orders()
        assert len(open_after) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. OrderManager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderManager:
    @pytest.mark.asyncio
    async def test_submit_bracket_orders(self, binance_sandbox):
        om = OrderManager(binance_sandbox)
        entry_req = OrderRequest(
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.05,
        )

        entry_res, sl_res, tp_res = await om.submit_bracket(
            entry_request=entry_req,
            stop_loss_price=49000.0,
            take_profit_price=53000.0,
        )

        assert entry_res.status == OrderStatus.FILLED
        assert sl_res is not None
        assert tp_res is not None
        assert "ACT-SL" in sl_res.client_order_id
        assert "ACT-TP" in tp_res.client_order_id

        assert len(om._brackets) == 1
        bracket = om._brackets[entry_res.client_order_id]
        assert bracket.stop_loss_order_id == sl_res.client_order_id
        assert bracket.take_profit_order_id == tp_res.client_order_id

    @pytest.mark.asyncio
    async def test_bracket_oco_resolution_on_sl_hit(self, binance_sandbox):
        om = OrderManager(binance_sandbox)
        entry_req = OrderRequest(
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.05,
        )
        entry_res, sl_res, tp_res = await om.submit_bracket(
            entry_request=entry_req,
            stop_loss_price=49000.0,
            take_profit_price=53000.0,
        )

        # Simulate SL triggered
        await om.handle_exit_fill(sl_res.client_order_id)

        bracket = om._brackets[entry_res.client_order_id]
        assert bracket.status == "CLOSED"
        # TP order should be cancelled in exchange
        tp_order = om.get_order(tp_res.client_order_id)
        assert tp_order.status == OrderStatus.CANCELLED


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reconciler Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReconciler:
    @pytest.mark.asyncio
    async def test_reconcile_positions_in_sync(self, binance_sandbox):
        om = OrderManager(binance_sandbox)
        reconciler = Reconciler(exchange=binance_sandbox, order_manager=om)

        # Both internal and exchange have zero positions
        report = await reconciler.reconcile_positions(internal_positions={})
        assert report.is_synchronized is True
        assert len(report.discrepancies) == 0

    @pytest.mark.asyncio
    async def test_reconcile_detects_position_mismatch_and_triggers_kill_switch(self, binance_sandbox):
        om = OrderManager(binance_sandbox)
        ks = KillSwitch()
        reconciler = Reconciler(
            exchange=binance_sandbox,
            order_manager=om,
            kill_switch=ks,
            auto_kill_on_critical=True,
        )

        # Internal thinks we have 0.10 BTC
        internal_pos = {
            "BTCUSDT": PositionSnapshot(
                symbol="BTCUSDT",
                side=Direction.LONG,
                entry_price=50000.0,
                quantity=0.10,
                current_price=50000.0,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                stop_loss=None,
                take_profit=None,
            )
        }
        # Exchange has 0 positions (discrepancy!)
        report = await reconciler.reconcile_positions(internal_positions=internal_pos)

        assert report.is_synchronized is False
        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].severity == "CRITICAL"
        assert "mismatch" in report.discrepancies[0].message
        # Kill switch engaged automatically!
        assert ks.is_engaged is True


# ─────────────────────────────────────────────────────────────────────────────
# 4. ExecutionEngine End-to-End Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionEngine:
    @pytest.mark.asyncio
    async def test_execute_approved_signal(self, binance_sandbox, risk_config, test_signal, test_portfolio):
        ks = KillSwitch()
        re = RiskEngine(config=risk_config, kill_switch=ks)
        engine = ExecutionEngine(exchange=binance_sandbox, risk_engine=re, kill_switch=ks)

        success, order_res, msg = await engine.execute_signal(test_signal, test_portfolio)
        assert success is True
        assert order_res is not None
        assert order_res.status == OrderStatus.FILLED
        assert "BTCUSDT" in engine.active_positions
        assert engine.active_positions["BTCUSDT"].quantity == order_res.filled_qty

    @pytest.mark.asyncio
    async def test_execute_signal_blocked_by_kill_switch(self, binance_sandbox, risk_config, test_signal, test_portfolio):
        ks = KillSwitch()
        await ks.activate("Manual test halt")
        re = RiskEngine(config=risk_config, kill_switch=ks)
        engine = ExecutionEngine(exchange=binance_sandbox, risk_engine=re, kill_switch=ks)

        success, order_res, msg = await engine.execute_signal(test_signal, test_portfolio)
        assert success is False
        assert order_res is None
        assert "Kill switch" in msg

    @pytest.mark.asyncio
    async def test_close_position(self, binance_sandbox, risk_config, test_signal, test_portfolio):
        ks = KillSwitch()
        re = RiskEngine(config=risk_config, kill_switch=ks)
        engine = ExecutionEngine(exchange=binance_sandbox, risk_engine=re, kill_switch=ks)

        await engine.execute_signal(test_signal, test_portfolio)
        assert "BTCUSDT" in engine.active_positions

        # Close position
        close_res = await engine.close_position("BTCUSDT", reason="TAKE_PROFIT")
        assert close_res is not None
        assert "BTCUSDT" not in engine.active_positions

    @pytest.mark.asyncio
    async def test_emergency_flatten_all_via_kill_switch(self, binance_sandbox, risk_config, test_signal, test_portfolio):
        ks = KillSwitch()
        re = RiskEngine(config=risk_config, kill_switch=ks)
        engine = ExecutionEngine(exchange=binance_sandbox, risk_engine=re, kill_switch=ks)

        # Open a position
        await engine.execute_signal(test_signal, test_portfolio)
        assert len(engine.active_positions) == 1

        # Trip the kill switch (which automatically calls registered liquidation callbacks)
        await ks.activate("Market crash triggered")

        # Active positions should be flattened!
        assert len(engine.active_positions) == 0
