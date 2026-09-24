"""Prometheus Metrics Exporter and Production Performance Monitor.

Instrumented for real-time scraping via Prometheus and Grafana dashboards.
Tracks portfolio equity, drawdowns, execution counts, feature drift p-values, and health status.
"""
from __future__ import annotations

from typing import Any, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from ai_crypto_trader.core.interfaces import PortfolioState
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


class PrometheusPerformanceMonitor:
    """Central Prometheus metrics registry for institutional trading platform observability."""

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry or CollectorRegistry()

        # ── Portfolio Metrics ──────────────────────────────────────────────────
        self.gauge_equity = Gauge(
            "act_portfolio_equity_usd",
            "Total portfolio equity in USD",
            registry=self.registry,
        )
        self.gauge_cash = Gauge(
            "act_portfolio_available_cash_usd",
            "Available unencumbered cash in USD",
            registry=self.registry,
        )
        self.gauge_unrealized_pnl = Gauge(
            "act_portfolio_unrealized_pnl_usd",
            "Aggregate unrealized PnL in USD across open positions",
            registry=self.registry,
        )
        self.gauge_drawdown = Gauge(
            "act_portfolio_drawdown_pct",
            "Current drawdown percentage from peak equity",
            registry=self.registry,
        )
        self.gauge_open_positions = Gauge(
            "act_active_positions_count",
            "Number of currently open perpetual/margin positions",
            registry=self.registry,
        )

        # ── Trade & Execution Metrics ──────────────────────────────────────────
        self.counter_trades = Counter(
            "act_trades_total",
            "Total executed trades",
            ["symbol", "side", "status"],
            registry=self.registry,
        )
        self.counter_fees = Counter(
            "act_fees_paid_total_usd",
            "Total cumulative trading fees paid in USD",
            registry=self.registry,
        )
        self.histogram_order_latency = Histogram(
            "act_order_submission_duration_seconds",
            "Latency of order submission to exchange API",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )

        # ── ML & Drift Metrics ─────────────────────────────────────────────────
        self.gauge_feature_pvalue = Gauge(
            "act_feature_ks_pvalue",
            "Two-sample KS-test p-value for feature drift detection",
            ["feature"],
            registry=self.registry,
        )
        self.gauge_model_confidence = Gauge(
            "act_model_prediction_confidence",
            "Rolling average model prediction confidence score",
            registry=self.registry,
        )

        # ── System Health Metric ───────────────────────────────────────────────
        self.gauge_health = Gauge(
            "act_system_health_status",
            "System health status (1 = Healthy, 0 = Degraded/Down)",
            registry=self.registry,
        )
        self.gauge_health.set(1.0)

    def update_portfolio(self, state: PortfolioState) -> None:
        """Update portfolio gauges from current PortfolioState."""
        self.gauge_equity.set(state.total_equity)
        self.gauge_cash.set(state.available_cash)
        self.gauge_unrealized_pnl.set(state.unrealized_pnl)
        self.gauge_drawdown.set(state.drawdown_pct)
        self.gauge_open_positions.set(len(state.open_positions))

    def record_trade(self, symbol: str, side: str, status: str, fees: float = 0.0) -> None:
        """Record trade execution metrics."""
        self.counter_trades.labels(symbol=symbol, side=side, status=status).inc()
        if fees > 0:
            self.counter_fees.inc(fees)

    def update_drift_metrics(self, feature_pvalues: dict[str, float], confidence: float) -> None:
        """Update ML drift and model confidence gauges."""
        for feat, pval in feature_pvalues.items():
            self.gauge_feature_pvalue.labels(feature=feat).set(pval)
        self.gauge_model_confidence.set(confidence)

    def update_health(self, is_healthy: bool) -> None:
        """Set health gauge."""
        self.gauge_health.set(1.0 if is_healthy else 0.0)

    def export_metrics(self) -> str:
        """Render metrics in Prometheus plaintext exposition format."""
        return generate_latest(self.registry).decode("utf-8")
