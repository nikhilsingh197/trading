"""System status API routes."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from ai_crypto_trader.config.settings import get_settings
from ai_crypto_trader.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.get("/status")
async def system_status():
    """Return system health and configuration summary."""
    settings = get_settings()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.app_env,
        "trading_mode": settings.trading_mode,
        "exchange": settings.exchange_name,
        "sandbox": settings.exchange_sandbox,
        "live_enabled": settings.live_trading_enabled,
        "version": "0.1.0",
    }


@router.get("/risk-limits")
async def risk_limits():
    """Return current hard risk limits (read-only)."""
    settings = get_settings()
    return {
        "max_trade_risk_pct": settings.risk_max_trade_risk_pct,
        "daily_loss_limit_pct": settings.risk_daily_loss_limit_pct,
        "max_drawdown_halt_pct": settings.risk_max_drawdown_halt_pct,
        "max_drawdown_reduce_pct": settings.risk_max_drawdown_reduce_pct,
        "max_open_positions": settings.risk_max_open_positions,
        "max_leverage": settings.risk_max_leverage,
        "max_trades_per_day": settings.risk_max_trades_per_day,
        "max_consecutive_losses": settings.risk_max_consecutive_losses,
        "note": "Hard limits. Cannot be modified at runtime. Requires restart.",
    }
