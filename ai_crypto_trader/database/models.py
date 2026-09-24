"""SQLAlchemy ORM models (declarative, async-compatible).

Supports PostgreSQL / TimescaleDB in production and SQLite in testing.
TimescaleDB hypertables are created via Alembic migrations.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, Index,
    Integer, JSON, Numeric, SmallInteger, String, Text,
    UniqueConstraint, ForeignKey, Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func

# Dialect-portable types:
# On PostgreSQL: native JSONB, native UUID, BIGSERIAL
# On SQLite / other: standard JSON, CHAR(36) UUID, 64-bit autoincrement Integer
JSONType = JSON().with_variant(JSONB, "postgresql")
UUIDType = Uuid(as_uuid=True)
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Market Data
# ─────────────────────────────────────────────────────────────────────────────

class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "open_time", name="uq_candle"),
        Index("ix_candle_symbol_tf_time", "symbol", "timeframe", "open_time"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(5), nullable=False)
    open_time = Column(DateTime(timezone=True), nullable=False)
    open = Column(Numeric(20, 8), nullable=False)
    high = Column(Numeric(20, 8), nullable=False)
    low = Column(Numeric(20, 8), nullable=False)
    close = Column(Numeric(20, 8), nullable=False)
    volume = Column(Numeric(30, 8), nullable=False)
    close_time = Column(DateTime(timezone=True), nullable=False)
    quote_volume = Column(Numeric(30, 8))
    trade_count = Column(Integer)
    source = Column(String(50), nullable=False, default="binance")
    quality_flag = Column(SmallInteger, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FundingRate(Base):
    __tablename__ = "funding_rates"
    __table_args__ = (
        UniqueConstraint("symbol", "funding_time", name="uq_funding_rate"),
    )

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    funding_time = Column(DateTime(timezone=True), nullable=False)
    funding_rate = Column(Numeric(20, 10), nullable=False)
    predicted_rate = Column(Numeric(20, 10))
    source = Column(String(50), nullable=False)


class OpenInterest(Base):
    __tablename__ = "open_interest"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    open_interest = Column(Numeric(30, 8), nullable=False)
    source = Column(String(50))


# ─────────────────────────────────────────────────────────────────────────────
# Trading
# ─────────────────────────────────────────────────────────────────────────────

class Order(Base):
    __tablename__ = "orders"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    client_order_id = Column(String(100), unique=True)
    exchange_order_id = Column(String(100))
    symbol = Column(String(20), nullable=False)
    side = Column(String(5), nullable=False)
    order_type = Column(String(20), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8))
    stop_price = Column(Numeric(20, 8))
    status = Column(String(20), nullable=False, default="PENDING")
    filled_qty = Column(Numeric(20, 8), default=0)
    avg_fill_price = Column(Numeric(20, 8))
    fees = Column(Numeric(20, 8), default=0)
    environment = Column(String(10), nullable=False)
    strategy_id = Column(UUIDType)
    signal_id = Column(UUIDType)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    executions = relationship("Execution", back_populates="order")


class Execution(Base):
    __tablename__ = "executions"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    order_id = Column(UUIDType, ForeignKey("orders.id"), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    fee = Column(Numeric(20, 8), nullable=False, default=0)
    fee_asset = Column(String(10))
    executed_at = Column(DateTime(timezone=True), server_default=func.now())

    order = relationship("Order", back_populates="executions")


class Position(Base):
    __tablename__ = "positions"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    side = Column(String(5), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    unrealized_pnl = Column(Numeric(20, 8), default=0)
    realized_pnl = Column(Numeric(20, 8), default=0)
    stop_loss = Column(Numeric(20, 8))
    take_profit = Column(Numeric(20, 8))
    leverage = Column(SmallInteger, default=1)
    environment = Column(String(10), nullable=False)
    strategy_id = Column(UUIDType)
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True))
    status = Column(String(10), nullable=False, default="OPEN")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    side = Column(String(5), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    exit_price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    pnl = Column(Numeric(20, 8), nullable=False)
    pnl_pct = Column(Numeric(10, 6), nullable=False)
    fees = Column(Numeric(20, 8), nullable=False, default=0)
    slippage = Column(Numeric(20, 8), default=0)
    duration_seconds = Column(Integer)
    exit_reason = Column(String(30))
    environment = Column(String(10), nullable=False)
    strategy_id = Column(UUIDType)
    regime = Column(String(30))
    opened_at = Column(DateTime(timezone=True), nullable=False)
    closed_at = Column(DateTime(timezone=True), server_default=func.now())


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False)
    total_equity = Column(Numeric(20, 8), nullable=False)
    available_cash = Column(Numeric(20, 8), nullable=False)
    unrealized_pnl = Column(Numeric(20, 8), nullable=False, default=0)
    realized_pnl = Column(Numeric(20, 8), nullable=False, default=0)
    drawdown_pct = Column(Numeric(10, 6), nullable=False, default=0)
    peak_equity = Column(Numeric(20, 8), nullable=False)
    environment = Column(String(10), nullable=False)


# ─────────────────────────────────────────────────────────────────────────────
# AI / ML
# ─────────────────────────────────────────────────────────────────────────────

class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    versions = relationship("MLModelVersion", back_populates="model")


class MLModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "version", name="uq_model_version"),
    )

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    model_id = Column(UUIDType, ForeignKey("ml_models.id"), nullable=False)
    version = Column(String(20), nullable=False)
    artifact_path = Column(Text, nullable=False)
    params = Column(JSONType)
    feature_list = Column(JSONType)
    target = Column(String(100))
    train_start = Column(DateTime(timezone=True))
    train_end = Column(DateTime(timezone=True))
    val_metrics = Column(JSONType)
    test_metrics = Column(JSONType)
    status = Column(String(20), nullable=False, default="CANDIDATE")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    model = relationship("MLModel", back_populates="versions")


class Signal(Base):
    __tablename__ = "signals"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(5), nullable=False)
    confidence = Column(Numeric(5, 4))
    entry_price = Column(Numeric(20, 8))
    stop_loss = Column(Numeric(20, 8))
    take_profit = Column(Numeric(20, 8))
    expected_return = Column(Numeric(10, 6))
    risk_pct = Column(Numeric(10, 6))
    regime = Column(String(30))
    reason = Column(Text)
    strategy_version_id = Column(UUIDType)
    ts = Column(DateTime(timezone=True), server_default=func.now())
    acted_upon = Column(Boolean, nullable=False, default=False)


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    hypothesis = Column(Text)
    dataset_hash = Column(String(64))
    features = Column(JSONType)
    strategy_params = Column(JSONType)
    model_type = Column(String(50))
    metrics = Column(JSONType)
    result = Column(String(20))  # PASS/FAIL/INCONCLUSIVE
    decision = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    category = Column(String(50), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    versions = relationship("StrategyVersion", back_populates="strategy")


class StrategyVersion(Base):
    __tablename__ = "strategy_versions"
    __table_args__ = (
        UniqueConstraint("strategy_id", "version", name="uq_strategy_version"),
    )

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUIDType, ForeignKey("strategies.id"), nullable=False)
    version = Column(String(20), nullable=False)
    params = Column(JSONType, nullable=False)
    model_version_id = Column(UUIDType)
    status = Column(String(20), nullable=False, default="CANDIDATE")
    promoted_at = Column(DateTime(timezone=True))
    retired_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    strategy = relationship("Strategy", back_populates="versions")


class StrategyMetrics(Base):
    __tablename__ = "strategy_metrics"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    strategy_version_id = Column(UUIDType, ForeignKey("strategy_versions.id"), nullable=False)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    environment = Column(String(10), nullable=False)
    total_trades = Column(Integer)
    win_rate = Column(Numeric(5, 4))
    profit_factor = Column(Numeric(10, 4))
    sharpe = Column(Numeric(10, 4))
    sortino = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 6))
    expectancy = Column(Numeric(10, 6))
    total_return = Column(Numeric(10, 6))
    computed_at = Column(DateTime(timezone=True), server_default=func.now())


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    strategy_version_id = Column(UUIDType, ForeignKey("strategy_versions.id"), nullable=False)
    experiment_id = Column(UUIDType)
    symbol = Column(String(20))
    timeframe = Column(String(5))
    train_start = Column(DateTime(timezone=True))
    train_end = Column(DateTime(timezone=True))
    test_start = Column(DateTime(timezone=True))
    test_end = Column(DateTime(timezone=True))
    total_return = Column(Numeric(10, 6))
    cagr = Column(Numeric(10, 6))
    sharpe = Column(Numeric(10, 4))
    sortino = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 6))
    win_rate = Column(Numeric(5, 4))
    profit_factor = Column(Numeric(10, 4))
    expectancy = Column(Numeric(10, 6))
    num_trades = Column(Integer)
    fees_paid = Column(Numeric(20, 8))
    params_used = Column(JSONType)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ─────────────────────────────────────────────────────────────────────────────
# System
# ─────────────────────────────────────────────────────────────────────────────

class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(10), nullable=False)
    component = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    event_metadata = Column("metadata", JSONType)
    ts = Column(DateTime(timezone=True), server_default=func.now())


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)
    trigger = Column(String(100), nullable=False)
    action_taken = Column(String(50), nullable=False)
    details = Column(JSONType)
    ts = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(BigIntPK, primary_key=True, autoincrement=True)
    actor = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    resource = Column(String(100))
    before_val = Column(JSONType)
    after_val = Column(JSONType)
    ts = Column(DateTime(timezone=True), server_default=func.now())


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(10), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    sent_to = Column(JSONType)
    acknowledged = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
