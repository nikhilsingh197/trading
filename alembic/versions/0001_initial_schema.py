"""0001_initial_schema

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-09-24

Initial migration creating all platform tables across Market Data, Trading,
AI/ML, Strategy, and System domains, with conditional TimescaleDB hypertable setup.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. Candles ────────────────────────────────────────────────────────────
    op.create_table(
        "candles",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timeframe", sa.String(5), nullable=False),
        sa.Column("open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(20, 8), nullable=False),
        sa.Column("high", sa.Numeric(20, 8), nullable=False),
        sa.Column("low", sa.Numeric(20, 8), nullable=False),
        sa.Column("close", sa.Numeric(20, 8), nullable=False),
        sa.Column("volume", sa.Numeric(30, 8), nullable=False),
        sa.Column("close_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quote_volume", sa.Numeric(30, 8), nullable=True),
        sa.Column("trade_count", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="binance"),
        sa.Column("quality_flag", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("symbol", "timeframe", "open_time", name="uq_candle"),
    )
    op.create_index("ix_candle_symbol_tf_time", "candles", ["symbol", "timeframe", "open_time"])

    # ── 2. Funding Rates ──────────────────────────────────────────────────────
    op.create_table(
        "funding_rates",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("funding_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("funding_rate", sa.Numeric(20, 10), nullable=False),
        sa.Column("predicted_rate", sa.Numeric(20, 10), nullable=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.UniqueConstraint("symbol", "funding_time", name="uq_funding_rate"),
    )

    # ── 3. Open Interest ──────────────────────────────────────────────────────
    op.create_table(
        "open_interest",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open_interest", sa.Numeric(30, 8), nullable=False),
        sa.Column("source", sa.String(50), nullable=True),
    )

    # ── 4. Orders ─────────────────────────────────────────────────────────────
    op.create_table(
        "orders",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("client_order_id", sa.String(100), unique=True, nullable=True),
        sa.Column("exchange_order_id", sa.String(100), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(5), nullable=False),
        sa.Column("order_type", sa.String(20), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("price", sa.Numeric(20, 8), nullable=True),
        sa.Column("stop_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("filled_qty", sa.Numeric(20, 8), server_default="0"),
        sa.Column("avg_fill_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("fees", sa.Numeric(20, 8), server_default="0"),
        sa.Column("environment", sa.String(10), nullable=False),
        sa.Column("strategy_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("signal_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 5. Executions ─────────────────────────────────────────────────────────
    op.create_table(
        "executions",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("order_id", sa.Uuid(as_uuid=True), sa.ForeignKey("orders.id"), nullable=False),
        sa.Column("price", sa.Numeric(20, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("fee", sa.Numeric(20, 8), server_default="0"),
        sa.Column("fee_asset", sa.String(10), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 6. Positions ──────────────────────────────────────────────────────────
    op.create_table(
        "positions",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(5), nullable=False),
        sa.Column("entry_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(20, 8), server_default="0"),
        sa.Column("realized_pnl", sa.Numeric(20, 8), server_default="0"),
        sa.Column("stop_loss", sa.Numeric(20, 8), nullable=True),
        sa.Column("take_profit", sa.Numeric(20, 8), nullable=True),
        sa.Column("leverage", sa.SmallInteger(), server_default="1"),
        sa.Column("environment", sa.String(10), nullable=False),
        sa.Column("strategy_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(10), nullable=False, server_default="OPEN"),
    )

    # ── 7. Trades ─────────────────────────────────────────────────────────────
    op.create_table(
        "trades",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(5), nullable=False),
        sa.Column("entry_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("exit_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("pnl", sa.Numeric(20, 8), nullable=False),
        sa.Column("pnl_pct", sa.Numeric(10, 6), nullable=False),
        sa.Column("fees", sa.Numeric(20, 8), server_default="0"),
        sa.Column("slippage", sa.Numeric(20, 8), server_default="0"),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("exit_reason", sa.String(30), nullable=True),
        sa.Column("environment", sa.String(10), nullable=False),
        sa.Column("strategy_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("regime", sa.String(30), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 8. Portfolio Snapshots ────────────────────────────────────────────────
    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_equity", sa.Numeric(20, 8), nullable=False),
        sa.Column("available_cash", sa.Numeric(20, 8), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(20, 8), server_default="0"),
        sa.Column("realized_pnl", sa.Numeric(20, 8), server_default="0"),
        sa.Column("drawdown_pct", sa.Numeric(10, 6), server_default="0"),
        sa.Column("peak_equity", sa.Numeric(20, 8), nullable=False),
        sa.Column("environment", sa.String(10), nullable=False),
    )

    # ── 9. ML Models ──────────────────────────────────────────────────────────
    op.create_table(
        "ml_models",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 10. Model Versions ────────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("model_id", sa.Uuid(as_uuid=True), sa.ForeignKey("ml_models.id"), nullable=False),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("params", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("feature_list", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("target", sa.String(100), nullable=True),
        sa.Column("train_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("train_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("val_metrics", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("test_metrics", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="CANDIDATE"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("model_id", "version", name="uq_model_version"),
    )

    # ── 11. Signals ───────────────────────────────────────────────────────────
    op.create_table(
        "signals",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("direction", sa.String(5), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("entry_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("stop_loss", sa.Numeric(20, 8), nullable=True),
        sa.Column("take_profit", sa.Numeric(20, 8), nullable=True),
        sa.Column("expected_return", sa.Numeric(10, 6), nullable=True),
        sa.Column("risk_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("regime", sa.String(30), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("strategy_version_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("acted_upon", sa.Boolean(), nullable=False, server_default="0"),
    )

    # ── 12. Experiments ───────────────────────────────────────────────────────
    op.create_table(
        "experiments",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("dataset_hash", sa.String(64), nullable=True),
        sa.Column("features", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("strategy_params", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("model_type", sa.String(50), nullable=True),
        sa.Column("metrics", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("result", sa.String(20), nullable=True),
        sa.Column("decision", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ── 13. Strategies ────────────────────────────────────────────────────────
    op.create_table(
        "strategies",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 14. Strategy Versions ─────────────────────────────────────────────────
    op.create_table(
        "strategy_versions",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("strategy_id", sa.Uuid(as_uuid=True), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("params", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.Column("model_version_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="CANDIDATE"),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("strategy_id", "version", name="uq_strategy_version"),
    )

    # ── 15. Strategy Metrics ──────────────────────────────────────────────────
    op.create_table(
        "strategy_metrics",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("strategy_version_id", sa.Uuid(as_uuid=True), sa.ForeignKey("strategy_versions.id"), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("environment", sa.String(10), nullable=False),
        sa.Column("total_trades", sa.Integer(), nullable=True),
        sa.Column("win_rate", sa.Numeric(5, 4), nullable=True),
        sa.Column("profit_factor", sa.Numeric(10, 4), nullable=True),
        sa.Column("sharpe", sa.Numeric(10, 4), nullable=True),
        sa.Column("sortino", sa.Numeric(10, 4), nullable=True),
        sa.Column("max_drawdown", sa.Numeric(10, 6), nullable=True),
        sa.Column("expectancy", sa.Numeric(10, 6), nullable=True),
        sa.Column("total_return", sa.Numeric(10, 6), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 16. Backtest Results ──────────────────────────────────────────────────
    op.create_table(
        "backtest_results",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("strategy_version_id", sa.Uuid(as_uuid=True), sa.ForeignKey("strategy_versions.id"), nullable=False),
        sa.Column("experiment_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=True),
        sa.Column("timeframe", sa.String(5), nullable=True),
        sa.Column("train_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("train_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("test_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("test_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_return", sa.Numeric(10, 6), nullable=True),
        sa.Column("cagr", sa.Numeric(10, 6), nullable=True),
        sa.Column("sharpe", sa.Numeric(10, 4), nullable=True),
        sa.Column("sortino", sa.Numeric(10, 4), nullable=True),
        sa.Column("max_drawdown", sa.Numeric(10, 6), nullable=True),
        sa.Column("win_rate", sa.Numeric(5, 4), nullable=True),
        sa.Column("profit_factor", sa.Numeric(10, 4), nullable=True),
        sa.Column("expectancy", sa.Numeric(10, 6), nullable=True),
        sa.Column("num_trades", sa.Integer(), nullable=True),
        sa.Column("fees_paid", sa.Numeric(20, 8), nullable=True),
        sa.Column("params_used", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 17. System Events ─────────────────────────────────────────────────────
    op.create_table(
        "system_events",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(10), nullable=False),
        sa.Column("component", sa.String(50), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 18. Risk Events ───────────────────────────────────────────────────────
    op.create_table(
        "risk_events",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("trigger", sa.String(100), nullable=False),
        sa.Column("action_taken", sa.String(50), nullable=False),
        sa.Column("details", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 19. Audit Logs ────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("actor", sa.String(100), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource", sa.String(100), nullable=True),
        sa.Column("before_val", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("after_val", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── 20. Alerts ────────────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("alert_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(10), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("sent_to", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("acknowledged", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── TimescaleDB Hypertables (PostgreSQL only) ─────────────────────────────
    bind = op.get_bind()
    if bind and bind.dialect.name == "postgresql":
        try:
            op.execute("SELECT create_hypertable('candles', 'open_time', if_not_exists => TRUE);")
            op.execute("SELECT create_hypertable('portfolio_snapshots', 'ts', if_not_exists => TRUE);")
            op.execute("SELECT create_hypertable('open_interest', 'ts', if_not_exists => TRUE);")
            op.execute("SELECT create_hypertable('system_events', 'ts', if_not_exists => TRUE);")
        except Exception:
            pass  # Extension might not be loaded in plain postgres test instances


def downgrade() -> None:
    tables = [
        "alerts",
        "audit_logs",
        "risk_events",
        "system_events",
        "backtest_results",
        "strategy_metrics",
        "strategy_versions",
        "strategies",
        "experiments",
        "signals",
        "model_versions",
        "ml_models",
        "portfolio_snapshots",
        "trades",
        "positions",
        "executions",
        "orders",
        "open_interest",
        "funding_rates",
        "candles",
    ]
    for table in tables:
        op.drop_table(table)
