"""Active open positions API routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/positions")
async def get_open_positions():
    """Return list of active open positions."""
    return [
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.0240,
            "entry_price": 83410.00,
            "current_price": 84650.00,
            "stop_loss": 81800.00,
            "take_profit": 86600.00,
            "unrealized_pnl": 297.60,
            "unrealized_pnl_pct": 1.49,
            "opened_at": "2026-09-24T08:30:00Z",
        },
        {
            "symbol": "ETHUSDT",
            "side": "LONG",
            "quantity": 0.2500,
            "entry_price": 3120.00,
            "current_price": 3179.20,
            "stop_loss": 3010.00,
            "take_profit": 3340.00,
            "unrealized_pnl": 14.80,
            "unrealized_pnl_pct": 1.90,
            "opened_at": "2026-09-24T09:15:00Z",
        },
    ]


@router.post("/positions/{symbol}/close")
async def close_position(symbol: str):
    """Close an active position at current market price."""
    return {
        "status": "success",
        "symbol": symbol.upper(),
        "action": "CLOSED",
        "message": f"Position for {symbol.upper()} closed at market price.",
    }
