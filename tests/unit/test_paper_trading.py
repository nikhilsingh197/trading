"""Comprehensive unit tests for Milestone 8: Paper Trading Engine.

Tests:
1. OrderMatcher: Market, Limit, Stop-Loss, Take-Profit execution with slippage and fees.
2. PaperPortfolio: Cash accounting, long/short positions, unrealized/realized PnL, drawdown.
3. PaperBroker: Order routing, OCO bracket orders, BrokerABC interface conformance.
4. PaperMonitor: Session metrics, drawdown breach alerting, advancement qualification gate.
5. PaperSession: End-to-end candle replay loop and database persistence.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ai_crypto_trader.core.enums import Direction, OrderStatus, OrderType, Side
from ai_crypto_trader.core.interfaces import Candle, OrderRequest
from ai_crypto_trader.database.models import Base
from ai_crypto_trader.database.repositories.trading_repo import PortfolioSnapshotRepository, TradeRepository
from ai_crypto_trader.paper_trading.order_matcher import OrderMatcher
from ai_crypto_trader.paper_trading.paper_broker import PaperBroker
from ai_crypto_trader.paper_trading.paper_monitor import PaperMonitor
from ai_crypto_trader.paper_trading.paper_portfolio import PaperPortfolio
from ai_crypto_trader.paper_trading.paper_session import PaperSession
from ai_crypto_trader.strategies.trend_following import EMACrossoverStrategy


# ── 1. Order Matcher Tests ───────────────────────────────────────────────────

def test_order_matcher_market_orders():
    """Verify market buy and sell executions apply spread, slippage, and taker fees."""
    matcher = OrderMatcher(maker_fee=0.0002, taker_fee=0.0006, base_slippage=0.0005)

    req_buy = OrderRequest(symbol="BTCUSDT", side=Side.BUY, order_type=OrderType.MARKET, quantity=0.1)
    res_buy = matcher.execute_market_order(req_buy, bid=50000.0, ask=50010.0, volume=100.0)

    assert res_buy.status == OrderStatus.FILLED
    assert res_buy.filled_qty == 0.1
    # Buy fill price is above ask
    assert res_buy.avg_fill_price > 50010.0
    assert res_buy.fees > 0.0

    req_sell = OrderRequest(symbol="BTCUSDT", side=Side.SELL, order_type=OrderType.MARKET, quantity=0.1)
    res_sell = matcher.execute_market_order(req_sell, bid=50000.0, ask=50010.0, volume=100.0)

    assert res_sell.status == OrderStatus.FILLED
    # Sell fill price is below bid
    assert res_sell.avg_fill_price < 50000.0
    assert res_sell.fees > 0.0


def test_order_matcher_limit_and_stop_orders():
    """Verify limit and stop orders trigger only when price breaches threshold."""
    matcher = OrderMatcher()

    # Limit Buy at 49,000
    matcher.register_passive_order(
        OrderRequest(
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=49000.0,
            client_order_id="l_buy",
        )
    )

    # Bar 1: Low is 49,500 -> Does NOT fill
    fills1 = matcher.process_price_bar("BTCUSDT", high=51000.0, low=49500.0, close=50000.0)
    assert len(fills1) == 0

    # Bar 2: Low is 48,800 -> FILLS
    fills2 = matcher.process_price_bar("BTCUSDT", high=50500.0, low=48800.0, close=49200.0)
    assert len(fills2) == 1
    assert fills2[0].client_order_id == "l_buy"
    assert fills2[0].avg_fill_price == 49000.0

    # Stop-Loss Sell at 48,000
    matcher.register_passive_order(
        OrderRequest(
            symbol="BTCUSDT",
            side=Side.SELL,
            order_type=OrderType.STOP_MARKET,
            quantity=0.1,
            stop_price=48000.0,
            client_order_id="sl_sell",
        )
    )

    # Bar 3: Low touches 47,500 -> Stop triggers
    fills3 = matcher.process_price_bar("BTCUSDT", high=49000.0, low=47500.0, close=47800.0)
    assert len(fills3) == 1
    assert fills3[0].client_order_id == "sl_sell"
    assert fills3[0].avg_fill_price <= 48000.0


# ── 2. Paper Portfolio Tests ─────────────────────────────────────────────────

def test_paper_portfolio_long_lifecycle():
    """Verify opening, holding, mark-to-market, and closing a long position."""
    portfolio = PaperPortfolio(initial_capital=10_000.0)

    # Open Long: 0.1 BTC @ 50,000 (Cost = $5,000 + $3 fee)
    trade = portfolio.on_order_fill(
        symbol="BTCUSDT",
        side=Side.BUY,
        quantity=0.1,
        fill_price=50000.0,
        fee=3.0,
    )
    assert trade is None  # Opening trade returns None
    assert portfolio.available_cash == pytest.approx(4997.0)

    pos = portfolio.get_position("BTCUSDT")
    assert pos is not None
    assert pos.side == Direction.LONG
    assert pos.quantity == 0.1

    # Price rises to 55,000
    portfolio.update_price("BTCUSDT", 55000.0)
    assert pos.unrealized_pnl == pytest.approx(500.0)  # (55000 - 50000) * 0.1
    assert portfolio.total_equity == pytest.approx(10497.0)
    assert portfolio.drawdown_pct == 0.0

    # Close position at 54,000 (Fee = $3.24)
    closed_trade = portfolio.on_order_fill(
        symbol="BTCUSDT",
        side=Side.SELL,
        quantity=0.1,
        fill_price=54000.0,
        fee=3.24,
    )
    assert closed_trade is not None
    assert closed_trade.pnl == pytest.approx(400.0 - 3.24)
    assert "BTCUSDT" not in portfolio.positions
    assert portfolio.realized_pnl > 390.0


def test_paper_portfolio_short_lifecycle():
    """Verify opening, holding, and closing a short position."""
    portfolio = PaperPortfolio(initial_capital=10_000.0)

    # Open Short: 0.1 BTC @ 50,000 (Margin = $5,000 + $3 fee)
    portfolio.on_order_fill(
        symbol="BTCUSDT",
        side=Side.SELL,
        quantity=0.1,
        fill_price=50000.0,
        fee=3.0,
    )

    pos = portfolio.get_position("BTCUSDT")
    assert pos.side == Direction.SHORT

    # Price drops to 46,000 (Gain for short)
    portfolio.update_price("BTCUSDT", 46000.0)
    assert pos.unrealized_pnl == pytest.approx(400.0)

    # Buy to close at 46,000
    closed_trade = portfolio.on_order_fill(
        symbol="BTCUSDT",
        side=Side.BUY,
        quantity=0.1,
        fill_price=46000.0,
        fee=2.76,
    )
    assert closed_trade.pnl == pytest.approx(400.0 - 2.76)


def test_paper_portfolio_drawdown_calculation():
    """Verify peak equity tracking and drawdown percentage."""
    portfolio = PaperPortfolio(initial_capital=10_000.0)

    portfolio.on_order_fill("BTCUSDT", Side.BUY, 0.1, 50000.0, 0.0)
    portfolio.update_price("BTCUSDT", 60000.0)  # Equity = $11,000 (New Peak)
    assert portfolio.peak_equity == 11000.0
    assert portfolio.drawdown_pct == 0.0

    portfolio.update_price("BTCUSDT", 45000.0)  # Equity = $9,500
    # DD = (11,000 - 9,500) / 11,000 = 1,500 / 11,000 = ~13.64%
    assert portfolio.drawdown_pct == pytest.approx(13.636, rel=1e-2)


# ── 3. Paper Broker & OCO Bracket Tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_paper_broker_market_order_and_oco_bracket():
    """Verify broker routes market orders and sets up OCO bracket stops."""
    broker = PaperBroker(initial_capital=10_000.0)

    # Prime price
    candle = Candle(
        symbol="BTCUSDT",
        timeframe="1h",
        open_time=datetime.now(timezone.utc),
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50000.0,
        volume=100.0,
        close_time=datetime.now(timezone.utc),
    )
    broker.on_candle(candle)

    # Submit Long entry with SL at 48,000 and TP at 54,000
    req = OrderRequest(
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        stop_price=48000.0,  # SL
        price=54000.0,       # TP
    )
    res = await broker.submit_order(req)
    assert res.status == OrderStatus.FILLED

    # Check active bracket orders
    assert len(broker.matcher.active_orders) == 2
    assert len(broker.oco_pairs) == 2

    # Incoming candle touches TP at 54,500
    tp_candle = Candle(
        symbol="BTCUSDT",
        timeframe="1h",
        open_time=datetime.now(timezone.utc),
        open=52000.0,
        high=55000.0,
        low=51800.0,
        close=54200.0,
        volume=100.0,
        close_time=datetime.now(timezone.utc),
    )
    trades = broker.on_candle(tp_candle)

    # TP filled, trade recorded, and SL automatically cancelled via OCO!
    assert len(trades) == 1
    assert trades[0].exit_reason == "TAKE_PROFIT"
    assert len(broker.matcher.active_orders) == 0  # Partner SL was cancelled
    assert len(broker.oco_pairs) == 0


@pytest.mark.asyncio
async def test_paper_broker_insufficient_cash():
    """Verify broker rejects orders when account lacks sufficient cash."""
    broker = PaperBroker(initial_capital=100.0)
    broker.latest_prices["BTCUSDT"] = 50000.0

    # Try to buy 1 BTC ($50,000) with only $100 cash
    req = OrderRequest(symbol="BTCUSDT", side=Side.BUY, order_type=OrderType.MARKET, quantity=1.0)
    res = await broker.submit_order(req)

    assert res.status == OrderStatus.REJECTED
    assert "Insufficient available cash" in (res.error or "")


# ── 4. Paper Monitor Tests ────────────────────────────────────────────────────

def test_paper_monitor_metrics_and_risk_halt():
    """Verify paper monitor aggregates stats and trips circuit breaker on max drawdown breach."""
    monitor = PaperMonitor(max_drawdown_limit_pct=15.0)
    broker = PaperBroker(initial_capital=10_000.0)

    # Normal state
    state = broker.portfolio.get_state()
    stats = monitor.update(state, [])
    assert stats.status == "HEALTHY"
    assert stats.total_trades == 0

    # Simulate heavy drawdown (16%)
    broker.portfolio.peak_equity = 10_000.0
    broker.portfolio.available_cash = 8_400.0  # 16% loss

    breached_state = broker.portfolio.get_state()
    stats_breached = monitor.update(breached_state, [])
    assert stats_breached.status == "HALTED"

    eligible, reason = monitor.check_advancement(stats_breached)
    assert not eligible
    assert "breached limit" in reason


# ── 5. Paper Session End-to-End Replay & DB Persistence ───────────────────────

@pytest.mark.asyncio
async def test_paper_session_replay_and_db_persistence():
    """Verify full paper session runs on a sequence of candles and persists state to SQLite."""
    # Build 40 synthetic candles
    candles: list[Candle] = []
    base_price = 50000.0
    for i in range(40):
        p = base_price + (i * 20.0)
        c = Candle(
            symbol="BTCUSDT",
            timeframe="1h",
            open_time=datetime(2026, 1, 1, i % 24, tzinfo=timezone.utc),
            open=p,
            high=p + 30.0,
            low=p - 30.0,
            close=p + 10.0,
            volume=50.0,
            close_time=datetime(2026, 1, 1, i % 24, tzinfo=timezone.utc),
        )
        candles.append(c)

    strat = EMACrossoverStrategy("paper_ema", {"fast_ema": 5, "slow_ema": 12})
    session = PaperSession(
        strategy=strat,
        symbol="BTCUSDT",
        timeframe="1h",
        initial_capital=10_000.0,
    )

    stats = await session.run_replay(candles)
    assert stats.current_equity > 0
    assert stats.status in ("HEALTHY", "WARNING")

    # DB Persistence
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as db_sess:
        await session.persist_to_db(db_sess)

        snap_repo = PortfolioSnapshotRepository(db_sess)
        snap = await snap_repo.get_latest_snapshot("paper")
        assert snap is not None
        assert float(snap.total_equity) > 0.0

    await engine.dispose()
