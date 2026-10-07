"""Active open positions API routes."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/positions")
async def get_open_positions(request: Request):
    """Return list of active open positions from live ML session."""
    trader = getattr(request.app.state, "live_trader", None)
    if trader:
        broker = trader.session.broker
        positions = broker.portfolio.get_all_positions()
        res = []
        for pos in positions:
            current_price = broker.latest_prices.get(pos.symbol, pos.entry_price)
            unrealized_pnl = (
                (current_price - pos.entry_price) * pos.quantity
                if pos.side.value == "LONG"
                else (pos.entry_price - current_price) * pos.quantity
            )
            cost = pos.entry_price * pos.quantity
            pnl_pct = (unrealized_pnl / cost) * 100 if cost > 0 else 0.0

            res.append({
                "symbol": pos.symbol,
                "side": pos.side.value if hasattr(pos.side, "value") else str(pos.side),
                "quantity": round(float(pos.quantity), 4),
                "entry_price": round(float(pos.entry_price), 2),
                "current_price": round(float(current_price), 2),
                "stop_loss": round(float(pos.stop_loss), 2) if pos.stop_loss else None,
                "take_profit": round(float(pos.take_profit), 2) if pos.take_profit else None,
                "unrealized_pnl": round(float(unrealized_pnl), 2),
                "unrealized_pnl_pct": round(float(pnl_pct), 2),
                "opened_at": pos.opened_at.isoformat() if hasattr(pos.opened_at, "isoformat") else str(pos.opened_at),
            })
        return res

    return []


@router.post("/positions/{symbol}/close")
async def close_position(symbol: str, request: Request):
    """Close an active position at current market price."""
    trader = getattr(request.app.state, "live_trader", None)
    if trader:
        trader.session.broker.portfolio.close_position(symbol.upper(), exit_reason="MANUAL")
    return {
        "status": "success",
        "symbol": symbol.upper(),
        "action": "CLOSED",
        "message": f"Position for {symbol.upper()} closed.",
    }
