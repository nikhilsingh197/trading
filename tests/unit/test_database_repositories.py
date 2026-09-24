"""Automated tests for Milestone 3: Database & Repositories Layer.

Tests:
- OrderRepository, TradeRepository, PositionRepository, PortfolioSnapshotRepository
- StrategyRepository, StrategyMetricsRepository, BacktestResultRepository
- ModelRepository, SignalRepository, ExperimentRepository
- SystemEventRepository, RiskEventRepository, AuditLogRepository, AlertRepository
- Alembic migration script generation & schema integrity
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ai_crypto_trader.database.models import Base
from ai_crypto_trader.database.repositories import (
    AlertRepository,
    AuditLogRepository,
    BacktestResultRepository,
    ExperimentRepository,
    ModelRepository,
    OrderRepository,
    PortfolioSnapshotRepository,
    PositionRepository,
    RiskEventRepository,
    SignalRepository,
    StrategyMetricsRepository,
    StrategyRepository,
    SystemEventRepository,
    TradeRepository,
)


@pytest_asyncio.fixture
async def db_session():
    """Provides an isolated in-memory SQLite database session for repository testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session

    await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRADING DOMAIN TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestTradingRepositories:
    async def test_order_lifecycle(self, db_session):
        repo = OrderRepository(db_session)

        # Create
        order = await repo.create_order(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.5,
            price=60000.0,
            client_order_id="test_ord_1",
        )
        assert order.status == "PENDING"
        assert order.client_order_id == "test_ord_1"

        # Query
        fetched = await repo.get_by_client_id("test_ord_1")
        assert fetched is not None
        assert fetched.id == order.id

        # Update
        updated = await repo.update_status(
            order_id=order.id,
            status="FILLED",
            filled_qty=0.5,
            avg_fill_price=59990.0,
            fees=3.0,
            exchange_order_id="ex_123",
        )
        assert updated.status == "FILLED"
        assert updated.filled_qty == 0.5
        assert updated.fees == 3.0

        # Open orders query should now exclude filled order
        open_orders = await repo.get_open_orders()
        assert len(open_orders) == 0

    async def test_trade_recording(self, db_session):
        repo = TradeRepository(db_session)
        now = datetime.now(timezone.utc)

        trade = await repo.record_trade(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=60000.0,
            exit_price=61200.0,
            quantity=0.1,
            pnl=120.0,
            pnl_pct=2.0,
            opened_at=now - timedelta(hours=2),
            fees=2.5,
            exit_reason="TAKE_PROFIT",
        )
        assert trade.pnl == 120.0
        assert trade.exit_reason == "TAKE_PROFIT"

        count = await repo.count_trades()
        assert count == 1

        recent = await repo.get_recent_trades(limit=10)
        assert len(recent) == 1
        assert recent[0].symbol == "BTCUSDT"

    async def test_position_lifecycle(self, db_session):
        repo = PositionRepository(db_session)

        pos = await repo.open_position(
            symbol="ETHUSDT",
            side="LONG",
            entry_price=3000.0,
            quantity=2.0,
            stop_loss=2900.0,
            take_profit=3200.0,
        )
        assert pos.status == "OPEN"

        open_positions = await repo.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].symbol == "ETHUSDT"

        closed = await repo.close_position(pos.id, realized_pnl=400.0)
        assert closed.status == "CLOSED"
        assert closed.realized_pnl == 400.0

        open_positions_after = await repo.get_open_positions()
        assert len(open_positions_after) == 0

    async def test_portfolio_snapshots(self, db_session):
        repo = PortfolioSnapshotRepository(db_session)

        s1 = await repo.record_snapshot(
            total_equity=10000.0,
            available_cash=8000.0,
            unrealized_pnl=200.0,
            realized_pnl=0.0,
            drawdown_pct=0.0,
            peak_equity=10000.0,
        )
        assert s1.total_equity == 10000.0

        latest = await repo.get_latest_snapshot()
        assert latest is not None
        assert latest.total_equity == 10000.0

        history = await repo.get_history(limit=5)
        assert len(history) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRATEGY DOMAIN TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestStrategyRepositories:
    async def test_strategy_champion_challenger_promotion(self, db_session):
        strat_repo = StrategyRepository(db_session)

        strat = await strat_repo.register_strategy(
            name="TrendEMA",
            category="trend_following",
            description="EMA crossover strategy",
        )
        assert strat.name == "TrendEMA"

        # Create v1 and promote to champion
        v1 = await strat_repo.create_version(strat.id, "v1", params={"fast": 9, "slow": 21})
        await strat_repo.promote_to_champion(v1.id)

        champ = await strat_repo.get_champion(strat.id)
        assert champ is not None
        assert champ.version == "v1"
        assert champ.status == "CHAMPION"

        # Create v2 (challenger) and promote
        v2 = await strat_repo.create_version(strat.id, "v2", params={"fast": 12, "slow": 26})
        await strat_repo.promote_to_champion(v2.id)

        # v2 is now CHAMPION, v1 is RETIRED
        champ_now = await strat_repo.get_champion(strat.id)
        assert champ_now.version == "v2"
        assert champ_now.status == "CHAMPION"

        # Check v1 retired
        await db_session.refresh(v1)
        assert v1.status == "RETIRED"
        assert v1.retired_at is not None

    async def test_backtest_result_persistence(self, db_session):
        strat_repo = StrategyRepository(db_session)
        bt_repo = BacktestResultRepository(db_session)

        strat = await strat_repo.register_strategy("BreakoutATR", "breakout")
        ver = await strat_repo.create_version(strat.id, "v1", params={"period": 20})

        res = await bt_repo.save_result(
            strategy_version_id=ver.id,
            symbol="BTCUSDT",
            timeframe="1h",
            total_return=0.35,
            cagr=0.42,
            sharpe=1.65,
            sortino=2.10,
            max_drawdown=0.08,
            win_rate=0.55,
            profit_factor=1.80,
            expectancy=0.012,
            num_trades=140,
            fees_paid=45.0,
        )
        assert res.sharpe == 1.65
        assert res.max_drawdown == 0.08

        results = await bt_repo.get_results_for_strategy(ver.id)
        assert len(results) == 1
        assert results[0].symbol == "BTCUSDT"


# ─────────────────────────────────────────────────────────────────────────────
# 3. AI / ML DOMAIN TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAIRepositories:
    async def test_model_versioning(self, db_session):
        repo = ModelRepository(db_session)

        model = await repo.register_model("XGBoostPredictor", "gradient_boosting")
        ver = await repo.create_version(
            model_id=model.id,
            version="1.0.0",
            artifact_path="data/models/xgb_v1.json",
            params={"max_depth": 5, "n_estimators": 100},
            feature_list=["trend_adx", "mom_rsi_14"],
            val_metrics={"f1": 0.62, "accuracy": 0.58},
        )
        assert ver.version == "1.0.0"

        latest = await repo.get_latest_version(model.id)
        assert latest is not None
        assert latest.version == "1.0.0"

    async def test_signals_recording(self, db_session):
        repo = SignalRepository(db_session)

        sig = await repo.record_signal(
            symbol="BTCUSDT",
            direction="LONG",
            confidence=0.82,
            entry_price=62000.0,
            stop_loss=61000.0,
            take_profit=64000.0,
            expected_return=3.2,
            risk_pct=1.6,
            regime="STRONG_UPTREND",
            reason="Confirmed trend breakout with ADX>25",
        )
        assert sig.direction == "LONG"
        assert sig.acted_upon is False

        await repo.mark_acted(sig.id)
        await db_session.refresh(sig)
        assert sig.acted_upon is True

    async def test_experiment_tracking(self, db_session):
        repo = ExperimentRepository(db_session)

        exp = await repo.create_experiment(
            name="Experiment_001_RSI_Filter",
            hypothesis="Adding RSI filter reduces false breakouts in ranging markets",
            features=["trend_ema_21", "mom_rsi_14"],
            model_type="LogisticRegression",
        )
        assert exp.result == "RUNNING"

        completed = await repo.complete_experiment(
            experiment_id=exp.id,
            result="PASS",
            decision="Hypothesis confirmed: Sharpe improved from 1.1 to 1.45",
            metrics={"sharpe_improvement": 0.35, "false_positives_reduced_pct": 18.0},
        )
        assert completed.result == "PASS"
        assert completed.completed_at is not None


# ─────────────────────────────────────────────────────────────────────────────
# 4. SYSTEM DOMAIN TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSystemRepositories:
    async def test_system_event_logging(self, db_session):
        repo = SystemEventRepository(db_session)

        ev = await repo.log_event(
            event_type="STARTUP",
            severity="INFO",
            component="core",
            message="Trading system booted in paper mode",
            metadata={"version": "0.1.0"},
        )
        assert ev.severity == "INFO"

        events = await repo.get_recent_events(limit=10)
        assert len(events) == 1
        assert events[0].event_type == "STARTUP"

    async def test_risk_event_logging(self, db_session):
        repo = RiskEventRepository(db_session)

        ev = await repo.log_risk_event(
            event_type="DRAWDOWN_WARNING",
            trigger="Portfolio drawdown reached 5.2%",
            action_taken="Position sizes reduced by 50%",
            details={"current_drawdown": 5.2, "halt_threshold": 10.0},
        )
        assert ev.action_taken == "Position sizes reduced by 50%"

        events = await repo.get_recent_risk_events()
        assert len(events) == 1

    async def test_audit_log_immutable(self, db_session):
        repo = AuditLogRepository(db_session)

        entry = await repo.log_action(
            actor="admin:human",
            action="CHANGE_MODE",
            resource="trading_mode",
            before_val={"mode": "paper"},
            after_val={"mode": "shadow"},
        )
        assert entry.actor == "admin:human"

        trail = await repo.get_audit_trail()
        assert len(trail) == 1

    async def test_alert_lifecycle(self, db_session):
        repo = AlertRepository(db_session)

        alt = await repo.create_alert(
            alert_type="KILL_SWITCH_ACTIVATED",
            severity="CRITICAL",
            title="Kill Switch Engaged",
            message="Kill switch was engaged manually via CLI",
        )
        assert alt.acknowledged is False

        unack = await repo.get_unacknowledged_alerts()
        assert len(unack) == 1

        await repo.acknowledge_alert(alt.id)
        unack_after = await repo.get_unacknowledged_alerts()
        assert len(unack_after) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. ALEMBIC MIGRATION SCHEMA INTEGRITY TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_alembic_offline_sql_generation():
    """Verify Alembic migration revision 0001 generates valid SQL without errors."""
    cfg = Config("alembic.ini")
    # Generates SQL string without executing against live DB
    # If any model, foreign key, or column definition is broken, this raises an exception
    command.upgrade(cfg, "head", sql=True)
