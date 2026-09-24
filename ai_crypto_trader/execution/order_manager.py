"""Institutional Order Manager.

Maintains client-side order state machines, deterministic order IDs,
and OCO (One-Cancels-the-Other) bracket order lifecycle management.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from ai_crypto_trader.core.enums import OrderStatus, OrderType, Side
from ai_crypto_trader.core.interfaces import OrderRequest, OrderResult
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.exchange.base import ExchangeAdapterABC

log = get_logger(__name__)


@dataclass
class BracketInfo:
    """Links an active position to its opposing stop-loss and take-profit orders."""
    entry_order_id: str
    symbol: str
    quantity: float
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    status: str = "ACTIVE"  # ACTIVE, CLOSED


class OrderManager:
    """Manages order submission, tracking, and bracket OCO order pairs."""

    def __init__(self, exchange: ExchangeAdapterABC) -> None:
        self.exchange = exchange
        self._orders: dict[str, OrderResult] = {}
        self._brackets: dict[str, BracketInfo] = {}  # keyed by entry_order_id
        self._lock = asyncio.Lock()

    def generate_order_id(self, prefix: str = "ACT") -> str:
        """Create a collision-resistant deterministic client order ID."""
        ts = int(datetime.now(timezone.utc).timestamp())
        uid = uuid.uuid4().hex[:8]
        return f"{prefix}-{ts}-{uid}"

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submit a single order through the exchange adapter and record internal state."""
        cl_id = request.client_order_id or self.generate_order_id()
        request.client_order_id = cl_id

        log.info(
            "order_submitting",
            symbol=request.symbol,
            side=request.side.value,
            type=request.order_type.value,
            qty=request.quantity,
            price=request.price,
            client_order_id=cl_id,
        )

        result = await self.exchange.submit_order(request)

        async with self._lock:
            self._orders[cl_id] = result

        return result

    async def submit_bracket(
        self,
        entry_request: OrderRequest,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> tuple[OrderResult, Optional[OrderResult], Optional[OrderResult]]:
        """Submit main entry order and attach conditional SL and TP exit orders upon fill."""
        entry_res = await self.submit_order(entry_request)

        if entry_res.status != OrderStatus.FILLED:
            # If not filled immediately, return without placing exits yet
            return entry_res, None, None

        # Exit side is the inverse of entry
        exit_side = Side.SELL if entry_request.side == Side.BUY else Side.BUY

        sl_id = self.generate_order_id(prefix="ACT-SL")
        sl_req = OrderRequest(
            symbol=entry_request.symbol,
            side=exit_side,
            order_type=OrderType.STOP_MARKET,
            quantity=entry_res.filled_qty,
            stop_price=stop_loss_price,
            client_order_id=sl_id,
        )
        sl_res = await self.submit_order(sl_req)

        tp_id = self.generate_order_id(prefix="ACT-TP")
        tp_req = OrderRequest(
            symbol=entry_request.symbol,
            side=exit_side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=entry_res.filled_qty,
            stop_price=take_profit_price,
            client_order_id=tp_id,
        )
        tp_res = await self.submit_order(tp_req)

        bracket = BracketInfo(
            entry_order_id=entry_res.client_order_id,
            symbol=entry_request.symbol,
            quantity=entry_res.filled_qty,
            stop_loss_order_id=sl_id,
            take_profit_order_id=tp_id,
        )

        async with self._lock:
            self._brackets[entry_res.client_order_id] = bracket

        log.info(
            "bracket_orders_attached",
            symbol=entry_request.symbol,
            entry_id=entry_res.client_order_id,
            sl_id=sl_id,
            tp_id=tp_id,
        )
        return entry_res, sl_res, tp_res

    async def handle_exit_fill(self, filled_order_id: str) -> None:
        """One-Cancels-the-Other (OCO) resolution when one bracket leg fills."""
        async with self._lock:
            for bracket in self._brackets.values():
                if bracket.status != "ACTIVE":
                    continue

                if filled_order_id == bracket.stop_loss_order_id:
                    # SL hit -> Cancel TP
                    log.info("bracket_sl_hit_cancelling_tp", sl_id=filled_order_id, tp_id=bracket.take_profit_order_id)
                    bracket.status = "CLOSED"
                    if bracket.take_profit_order_id:
                        await self.exchange.cancel_order(bracket.take_profit_order_id, bracket.symbol)
                    break
                elif filled_order_id == bracket.take_profit_order_id:
                    # TP hit -> Cancel SL
                    log.info("bracket_tp_hit_cancelling_sl", tp_id=filled_order_id, sl_id=bracket.stop_loss_order_id)
                    bracket.status = "CLOSED"
                    if bracket.stop_loss_order_id:
                        await self.exchange.cancel_order(bracket.stop_loss_order_id, bracket.symbol)
                    break

    async def cancel_order(self, client_order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an open order."""
        success = await self.exchange.cancel_order(client_order_id, symbol)
        if success and client_order_id in self._orders:
            self._orders[client_order_id].status = OrderStatus.CANCELLED
        return success

    async def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        count = await self.exchange.cancel_all_orders(symbol)
        for o in self._orders.values():
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                if symbol is None or getattr(o, "symbol", "") == symbol:
                    o.status = OrderStatus.CANCELLED
        return count

    def get_order(self, client_order_id: str) -> Optional[OrderResult]:
        return self._orders.get(client_order_id)

    def list_open_orders(self) -> list[OrderResult]:
        return [o for o in self._orders.values() if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)]

    def list_all_orders(self) -> list[OrderResult]:
        return list(self._orders.values())
