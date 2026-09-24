"""Trade blotter and history API routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/trades")
async def get_recent_trades(limit: int = 50, symbol: str | None = None):
    """Return completed trades blotter."""
    trades = [
        {
            "id": "trade_001",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 80997.50,
            "exit_price": 86654.88,
            "quantity": 0.020,
            "pnl": 113.87,
            "pnl_pct": 6.92,
            "fees": 1.04,
            "exit_reason": "TAKE_PROFIT",
            "closed_at": "2026-09-24T11:00:00Z",
        },
        {
            "id": "trade_002",
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "entry_price": 86413.20,
            "exit_price": 83166.37,
            "quantity": 0.019,
            "pnl": 61.57,
            "pnl_pct": 3.70,
            "fees": 0.98,
            "exit_reason": "TAKE_PROFIT",
            "closed_at": "2026-09-24T09:00:00Z",
        },
        {
            "id": "trade_003",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 85702.40,
            "exit_price": 83180.92,
            "quantity": 0.020,
            "pnl": -50.41,
            "pnl_pct": -3.00,
            "fees": 1.01,
            "exit_reason": "STOP_LOSS",
            "closed_at": "2026-09-24T07:00:00Z",
        },
    ]
    if symbol:
        trades = [t for t in trades if t["symbol"] == symbol.upper()]
    return {"trades": trades[:limit], "total": len(trades)}
