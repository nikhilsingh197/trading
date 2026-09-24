"""Real-time WebSocket streaming feed for live dashboard updates."""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ai_crypto_trader.core.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.websocket("/ws")
async def websocket_dashboard_feed(websocket: WebSocket):
    """Broadcasts dynamic real-time market ticks, candle updates, and portfolio equity."""
    await websocket.accept()
    log.info("websocket_client_connected", client=str(websocket.client))

    btc_price = 84650.00
    eth_price = 3179.20
    available_cash = 8214.30
    btc_entry = 83410.00
    btc_qty = 0.0240
    eth_entry = 3120.00
    eth_qty = 0.2500
    cost_basis = btc_entry * btc_qty + eth_entry * eth_qty  # 2781.84
    peak_equity = 10650.00

    # Current bar state
    bar_open = btc_price
    bar_high = btc_price
    bar_low = btc_price
    bar_volume = 12.4
    tick_count = 0

    try:
        while True:
            # Stochastic price movement (geometric random walk with slight mean reversion)
            btc_drift = (84700.0 - btc_price) * 0.001
            btc_shock = random.normalvariate(0, 1) * 18.50
            btc_price = round(max(50000.0, btc_price + btc_drift + btc_shock), 2)

            eth_drift = (3185.0 - eth_price) * 0.001
            eth_shock = random.normalvariate(0, 1) * 1.20
            eth_price = round(max(1000.0, eth_price + eth_drift + eth_shock), 2)

            # Update current candle bar
            bar_high = max(bar_high, btc_price)
            bar_low = min(bar_low, btc_price)
            bar_volume = round(bar_volume + random.uniform(0.05, 0.40), 4)
            tick_count += 1

            new_bar_closed = False
            # Every 30 ticks (~15 seconds), form a new candle bar
            if tick_count >= 30:
                new_bar_closed = True
                bar_open = btc_price
                bar_high = btc_price
                bar_low = btc_price
                bar_volume = round(random.uniform(2.0, 5.0), 4)
                tick_count = 0

            # Dynamic PnL calculation
            btc_pnl = (btc_price - btc_entry) * btc_qty
            eth_pnl = (eth_price - eth_entry) * eth_qty
            total_unrealized = round(btc_pnl + eth_pnl, 2)

            total_equity = round(available_cash + cost_basis + total_unrealized, 2)
            if total_equity > peak_equity:
                peak_equity = total_equity
            drawdown_pct = round(max(0.0, (peak_equity - total_equity) / peak_equity * 100.0), 2)

            now = datetime.now(timezone.utc).isoformat()
            data = {
                "type": "TICK",
                "timestamp": now,
                "symbol": "BTCUSDT",
                "price": btc_price,
                "bid": round(btc_price * 0.9999, 2),
                "ask": round(btc_price * 1.0001, 2),
                "eth_price": eth_price,
                "portfolio_equity": total_equity,
                "available_cash": available_cash,
                "unrealized_pnl": total_unrealized,
                "btc_pnl": round(btc_pnl, 2),
                "eth_pnl": round(eth_pnl, 2),
                "drawdown_pct": drawdown_pct,
                "peak_equity": peak_equity,
                "candle": {
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": btc_price,
                    "volume": bar_volume,
                    "new_bar": new_bar_closed,
                },
            }

            await websocket.send_json(data)
            await asyncio.sleep(0.5)  # 500ms tick frequency for smooth, live updates

    except WebSocketDisconnect:
        log.info("websocket_client_disconnected")
    except Exception as e:
        log.warning("websocket_error", error=str(e))
