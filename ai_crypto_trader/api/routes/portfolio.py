"""Portfolio status and performance API routes."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/portfolio")
async def get_portfolio_summary(request: Request):
    """Return latest portfolio equity, cash, and performance overview from live session."""
    trader = getattr(request.app.state, "live_trader", None)
    if trader:
        p = trader.session.broker.portfolio
        state = await trader.session.broker.get_portfolio_state()
        return {
            "total_equity": round(float(state.total_equity), 2),
            "available_cash": round(float(state.available_cash), 2),
            "margin_used": round(float(state.total_equity - state.available_cash), 2),
            "unrealized_pnl": round(float(state.unrealized_pnl), 2),
            "unrealized_pnl_pct": round(float((state.unrealized_pnl / 10000.0) * 100), 2),
            "realized_pnl": round(float(p.realized_pnl), 2),
            "peak_equity": round(float(p.peak_equity), 2),
            "drawdown_pct": round(float(p.drawdown_pct), 2),
            "environment": "paper",
            "open_positions_count": len(state.open_positions),
            "total_trades_count": len(p.trades_history),
            "win_rate": round(float(p.win_rate), 3),
            "profit_factor": round(float(p.profit_factor), 2),
        }

    return {
        "total_equity": 10000.0,
        "available_cash": 10000.0,
        "margin_used": 0.0,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": 0.0,
        "realized_pnl": 0.0,
        "peak_equity": 10000.0,
        "drawdown_pct": 0.0,
        "environment": "paper",
        "open_positions_count": 0,
        "total_trades_count": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
    }


@router.get("/portfolio/history")
async def get_portfolio_history(request: Request, limit: int = 100):
    """Return historical equity curve snapshots."""
    trader = getattr(request.app.state, "live_trader", None)
    if trader and trader.session.broker.portfolio.trades_history:
        snapshots = []
        running_equity = 10000.0
        peak = 10000.0
        for t in trader.session.broker.portfolio.trades_history:
            running_equity += t.pnl
            if running_equity > peak:
                peak = running_equity
            dd = max(0.0, (peak - running_equity) / peak * 100.0)
            snapshots.append({
                "ts": t.exit_time.isoformat() if hasattr(t.exit_time, "isoformat") else str(t.exit_time),
                "total_equity": round(running_equity, 2),
                "drawdown_pct": round(dd, 2),
            })
        return {"snapshots": snapshots[-limit:]}

    return {"snapshots": []}
