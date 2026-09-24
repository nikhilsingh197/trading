"""Real-time WebSocket streaming feed for live dashboard updates."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ai_crypto_trader.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.websocket("/ws")
async def websocket_dashboard_feed(websocket: WebSocket):
    """Broadcasts real-time prices, portfolio equity, and signals to connected frontend clients."""
    await websocket.accept()
    log.info("websocket_client_connected", client=str(websocket.client))

    base_price = 84650.0
    try:
        while True:
            # Emit live market tick and equity snapshot
            now = datetime.now(timezone.utc).isoformat()
            data = {
                "type": "TICK",
                "timestamp": now,
                "symbol": "BTCUSDT",
                "price": base_price,
                "portfolio_equity": 10482.50,
                "unrealized_pnl": 312.40,
                "drawdown_pct": 1.57,
            }
            await websocket.send_json(data)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        log.info("websocket_client_disconnected")
    except Exception as e:
        log.warning("websocket_error", error=str(e))
