"""Unit and integration tests for monitoring, drift detection, and Prometheus metrics."""
from __future__ import annotations

import asyncio
import time
import numpy as np
import pytest
from starlette.testclient import TestClient

from ai_crypto_trader.alerts.alert_manager import AlertManager
from ai_crypto_trader.api.main import app
from ai_crypto_trader.core.interfaces import PortfolioState, PositionSnapshot
from ai_crypto_trader.exchange.binance_adapter import BinanceAdapter
from ai_crypto_trader.monitoring.drift_detector import DriftDetector
from ai_crypto_trader.monitoring.health_checker import HealthChecker
from ai_crypto_trader.monitoring.performance_monitor import PrometheusPerformanceMonitor
from ai_crypto_trader.risk.circuit_breaker import CircuitBreaker
from ai_crypto_trader.risk.kill_switch import KillSwitch


# ─────────────────────────────────────────────────────────────────────────────
# 1. DriftDetector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftDetector:
    @pytest.mark.asyncio
    async def test_no_drift_when_distributions_match(self):
        detector = DriftDetector()
        np.random.seed(42)
        base_feature = np.random.normal(loc=0.0, scale=1.0, size=200).tolist()
        detector.set_baselines(
            baseline_features={"rsi_14": base_feature},
            baseline_confidence=0.75,
            baseline_win_rate=0.55,
        )

        # Stream similar samples
        for _ in range(50):
            val = float(np.random.normal(loc=0.0, scale=1.0))
            detector.record_prediction({"rsi_14": val}, confidence=0.74)

        report = await detector.check_drift()
        assert report.has_drift is False
        assert report.recommended_action == "CONTINUE"
        assert len(report.feature_drifts) == 1
        assert report.feature_drifts[0].is_drifted is False

    @pytest.mark.asyncio
    async def test_detects_injected_covariate_feature_drift(self):
        alert_mgr = AlertManager()
        detector = DriftDetector(alert_manager=alert_mgr)
        np.random.seed(42)
        base_feature = np.random.normal(loc=50.0, scale=5.0, size=200).tolist()
        detector.set_baselines(baseline_features={"rsi_14": base_feature})

        # Inject major distribution shift (mean shifts from 50 to 80)
        for _ in range(50):
            shifted_val = float(np.random.normal(loc=80.0, scale=5.0))
            detector.record_prediction({"rsi_14": shifted_val}, confidence=0.72)

        report = await detector.check_drift()
        assert report.has_drift is True
        assert report.recommended_action == "TRIGGER_RETRAIN"
        assert report.feature_drifts[0].is_drifted is True
        assert report.feature_drifts[0].p_value < 0.05
        # Alert was dispatched
        assert len(alert_mgr._history) == 1
        assert "Drift Detected" in alert_mgr._history[0].title

    @pytest.mark.asyncio
    async def test_detects_confidence_decay(self):
        detector = DriftDetector(confidence_decline_threshold_pct=15.0)
        detector.set_baselines(baseline_features={}, baseline_confidence=0.80)

        # Stream decaying confidence: 0.50 (37.5% drop from 0.80)
        for _ in range(40):
            detector.record_prediction({}, confidence=0.50)

        report = await detector.check_drift()
        assert report.has_drift is True
        assert report.recommended_action == "REDUCE_SIZE"
        assert report.confidence_decline_pct > 15.0

    @pytest.mark.asyncio
    async def test_detects_win_rate_and_negative_expectancy_drift(self):
        detector = DriftDetector(win_rate_decline_threshold_pct=20.0)
        detector.set_baselines(
            baseline_features={},
            baseline_win_rate=0.60,
            baseline_expectancy=50.0,
        )

        # Stream 20 consecutive losing trades
        for _ in range(20):
            detector.record_trade_result(-100.0)

        report = await detector.check_drift()
        assert report.has_drift is True
        assert report.recommended_action == "DEMOTE_TO_PAPER"
        assert report.rolling_expectancy < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. HealthChecker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthChecker:
    @pytest.mark.asyncio
    async def test_health_checker_healthy_state(self):
        exchange = BinanceAdapter()
        ks = KillSwitch()
        cb = CircuitBreaker()
        checker = HealthChecker(exchange=exchange, kill_switch=ks, circuit_breaker=cb)

        # Update data tick to now
        checker.record_market_data_tick(time.time())

        status = await checker.check_health()
        assert status.is_healthy is True
        assert status.status == "HEALTHY"
        assert status.components["market_data"].status == "HEALTHY"
        assert status.components["exchange"].status == "HEALTHY"
        assert status.components["kill_switch"].status == "HEALTHY"
        assert status.components["circuit_breaker"].status == "HEALTHY"

    @pytest.mark.asyncio
    async def test_health_checker_detects_stale_data(self):
        checker = HealthChecker(stale_threshold_seconds=1.0)
        # Set tick 5 seconds in the past
        checker.record_market_data_tick(time.time() - 5.0)

        status = await checker.check_health()
        assert status.is_healthy is False
        assert status.status in ("DEGRADED", "CRITICAL")
        assert status.components["market_data"].status in ("DEGRADED", "DOWN")

    @pytest.mark.asyncio
    async def test_health_checker_detects_kill_switch(self):
        ks = KillSwitch()
        await ks.activate("Emergency stop")
        checker = HealthChecker(kill_switch=ks)
        checker.record_market_data_tick(time.time())

        status = await checker.check_health()
        assert status.is_healthy is False
        assert status.components["kill_switch"].status == "DEGRADED"


# ─────────────────────────────────────────────────────────────────────────────
# 3. PrometheusPerformanceMonitor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPrometheusMonitor:
    def test_metrics_collection_and_export(self):
        monitor = PrometheusPerformanceMonitor()

        # Update portfolio
        portfolio = PortfolioState(
            total_equity=12500.0,
            available_cash=8500.0,
            unrealized_pnl=450.0,
            realized_pnl=2050.0,
            drawdown_pct=2.4,
            peak_equity=12800.0,
            open_positions=[],
        )
        monitor.update_portfolio(portfolio)

        # Record trade
        monitor.record_trade("BTCUSDT", "BUY", "FILLED", fees=3.20)

        # Record drift
        monitor.update_drift_metrics({"rsi_14": 0.35}, confidence=0.78)

        # Export metrics text
        metrics_text = monitor.export_metrics()

        assert "act_portfolio_equity_usd 12500.0" in metrics_text
        assert "act_portfolio_available_cash_usd 8500.0" in metrics_text
        assert "act_portfolio_drawdown_pct 2.4" in metrics_text
        assert 'act_trades_total{side="BUY",status="FILLED",symbol="BTCUSDT"} 1.0' in metrics_text
        assert "3.2" in metrics_text
        assert 'act_feature_ks_pvalue{feature="rsi_14"} 0.35' in metrics_text
        assert "act_model_prediction_confidence 0.78" in metrics_text

    def test_api_metrics_endpoint(self):
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "act_portfolio_equity_usd" in response.text
