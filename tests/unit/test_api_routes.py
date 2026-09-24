"""Unit and integration tests for FastAPI REST and WebSocket endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_crypto_trader.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_system_status(client):
    res = client.get("/api/v1/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "environment" in data
    assert "trading_mode" in data


def test_risk_limits(client):
    res = client.get("/api/v1/risk-limits")
    assert res.status_code == 200
    data = res.json()
    assert "max_trade_risk_pct" in data
    assert "daily_loss_limit_pct" in data
    assert "max_drawdown_halt_pct" in data


def test_portfolio_endpoints(client):
    res = client.get("/api/v1/portfolio")
    assert res.status_code == 200
    data = res.json()
    assert data["total_equity"] > 0
    assert data["available_cash"] > 0
    assert "drawdown_pct" in data

    hist_res = client.get("/api/v1/portfolio/history")
    assert hist_res.status_code == 200
    assert len(hist_res.json()["snapshots"]) > 0


def test_positions_endpoints(client):
    res = client.get("/api/v1/positions")
    assert res.status_code == 200
    positions = res.json()
    assert isinstance(positions, list)
    assert len(positions) > 0
    assert positions[0]["symbol"] == "BTCUSDT"

    close_res = client.post("/api/v1/positions/BTCUSDT/close")
    assert close_res.status_code == 200
    assert close_res.json()["status"] == "success"


def test_trades_endpoint(client):
    res = client.get("/api/v1/trades")
    assert res.status_code == 200
    data = res.json()
    assert "trades" in data
    assert len(data["trades"]) > 0


def test_strategies_endpoint(client):
    res = client.get("/api/v1/strategies")
    assert res.status_code == 200
    strats = res.json()["strategies"]
    names = [s["name"] for s in strats]
    assert "ML_Signal" in names
    assert "EMA_Crossover" in names


def test_models_endpoint(client):
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    assert "models" in res.json()


def test_dashboard_html_view(client):
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert "AI CRYPTO TRADER" in res.text
