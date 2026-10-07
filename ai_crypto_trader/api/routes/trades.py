"""Trade blotter and history API routes."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/trades")
async def get_recent_trades(request: Request, limit: int = 50, symbol: str | None = None):
    """Return completed trades blotter from the active live ML trading session."""
    trader = getattr(request.app.state, "live_trader", None)
    
    if trader and trader.session.broker.portfolio.trades_history:
        trades = []
        for t in reversed(trader.session.broker.portfolio.trades_history):
            if symbol and t.symbol.upper() != symbol.upper():
                continue
            trades.append({
                "id": str(t.trade_id),
                "symbol": t.symbol,
                "side": t.side.value if hasattr(t.side, "value") else str(t.side),
                "entry_price": round(float(t.entry_price), 2),
                "exit_price": round(float(t.exit_price), 2),
                "quantity": round(float(t.quantity), 4),
                "pnl": round(float(t.pnl), 2),
                "pnl_pct": round(float(t.pnl_pct), 2),
                "fees": round(float(t.fees), 2),
                "exit_reason": t.exit_reason.value if hasattr(t.exit_reason, "value") else str(t.exit_reason),
                "closed_at": t.exit_time.isoformat() if hasattr(t.exit_time, "isoformat") else str(t.exit_time),
            })
        return {"trades": trades[:limit], "total": len(trades)}

    # Fallback if starting up
    return {"trades": [], "total": 0}
