"""Simulated order matcher with realistic fills, slippage, and maker/taker fees."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_crypto_trader.core.constants import DEFAULT_MAKER_FEE, DEFAULT_SLIPPAGE, DEFAULT_TAKER_FEE
from ai_crypto_trader.core.enums import OrderStatus, OrderType, Side
from ai_crypto_trader.core.interfaces import OrderRequest, OrderResult
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class PaperOrder:
    """Internal representation of an order in the paper matching engine."""
    client_order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    fees: float = 0.0
    strategy_id: str | None = None
    signal_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OrderMatcher:
    """Matches paper orders against real-time market data (ticks or candles)."""

    def __init__(
        self,
        maker_fee: float = DEFAULT_MAKER_FEE,
        taker_fee: float = DEFAULT_TAKER_FEE,
        base_slippage: float = DEFAULT_SLIPPAGE,
    ) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.base_slippage = base_slippage
        self.active_orders: dict[str, PaperOrder] = {}

    def _compute_slippage(self, quantity: float, bar_volume: float) -> float:
        """Dynamic slippage: base + impact penalty based on volume participation."""
        if bar_volume <= 0:
            return self.base_slippage
        participation = quantity / bar_volume
        impact = 0.1 * (participation ** 0.5)
        return min(0.02, self.base_slippage + impact)  # Cap slippage at 2%

    def execute_market_order(
        self,
        request: OrderRequest,
        bid: float,
        ask: float,
        volume: float = 1000.0,
    ) -> OrderResult:
        """Immediately executes a market order against the prevailing bid/ask."""
        client_id = request.client_order_id or f"ord_{uuid.uuid4().hex[:12]}"
        slippage = self._compute_slippage(request.quantity, volume)

        if request.side == Side.BUY:
            fill_price = ask * (1.0 + slippage)
        else:
            fill_price = bid * (1.0 - slippage)

        fee = fill_price * request.quantity * self.taker_fee

        order = PaperOrder(
            client_order_id=client_id,
            symbol=request.symbol,
            side=request.side,
            order_type=OrderType.MARKET,
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            status=OrderStatus.FILLED,
            filled_qty=request.quantity,
            avg_fill_price=fill_price,
            fees=fee,
            strategy_id=request.strategy_id,
            signal_id=request.signal_id,
        )

        log.debug(
            "market_order_filled",
            client_id=client_id,
            side=request.side.value,
            qty=request.quantity,
            fill_price=fill_price,
            fee=fee,
        )

        return OrderResult(
            client_order_id=client_id,
            exchange_order_id=f"sim_{client_id}",
            status=OrderStatus.FILLED,
            filled_qty=request.quantity,
            avg_fill_price=fill_price,
            fees=fee,
        )

    def register_passive_order(self, request: OrderRequest) -> OrderResult:
        """Register a limit or stop order into the matching book."""
        client_id = request.client_order_id or f"ord_{uuid.uuid4().hex[:12]}"

        order = PaperOrder(
            client_order_id=client_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            status=OrderStatus.SUBMITTED,
            strategy_id=request.strategy_id,
            signal_id=request.signal_id,
        )
        self.active_orders[client_id] = order

        log.debug(
            "passive_order_registered",
            client_id=client_id,
            order_type=request.order_type.value,
            side=request.side.value,
            price=request.price,
            stop_price=request.stop_price,
        )

        return OrderResult(
            client_order_id=client_id,
            exchange_order_id=f"sim_{client_id}",
            status=OrderStatus.SUBMITTED,
            filled_qty=0.0,
            avg_fill_price=None,
            fees=0.0,
        )

    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel a pending order."""
        if client_order_id in self.active_orders:
            order = self.active_orders.pop(client_order_id)
            order.status = OrderStatus.CANCELLED
            log.debug("order_cancelled", client_id=client_order_id)
            return True
        return False

    def process_price_bar(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        volume: float = 1000.0,
    ) -> list[OrderResult]:
        """Check all active orders for execution against a high-low-close bar."""
        filled_results: list[OrderResult] = []
        orders_to_remove: list[str] = []

        for client_id, order in self.active_orders.items():
            if order.symbol != symbol:
                continue

            filled = False
            fill_price: float = 0.0
            fee: float = 0.0

            # 1. Limit orders (Maker fees)
            if order.order_type == OrderType.LIMIT and order.price is not None:
                if order.side == Side.BUY and low <= order.price:
                    fill_price = min(order.price, high)
                    fee = fill_price * order.quantity * self.maker_fee
                    filled = True
                elif order.side == Side.SELL and high >= order.price:
                    fill_price = max(order.price, low)
                    fee = fill_price * order.quantity * self.maker_fee
                    filled = True

            # 2. Stop-loss orders (Taker fees + slippage)
            elif order.order_type == OrderType.STOP_MARKET and order.stop_price is not None:
                slippage = self._compute_slippage(order.quantity, volume)
                if order.side == Side.SELL and low <= order.stop_price:
                    fill_price = order.stop_price * (1.0 - slippage)
                    fee = fill_price * order.quantity * self.taker_fee
                    filled = True
                elif order.side == Side.BUY and high >= order.stop_price:
                    fill_price = order.stop_price * (1.0 + slippage)
                    fee = fill_price * order.quantity * self.taker_fee
                    filled = True

            # 3. Take-profit orders (Taker fees)
            elif order.order_type == OrderType.TAKE_PROFIT_MARKET and order.stop_price is not None:
                if order.side == Side.SELL and high >= order.stop_price:
                    fill_price = order.stop_price
                    fee = fill_price * order.quantity * self.taker_fee
                    filled = True
                elif order.side == Side.BUY and low <= order.stop_price:
                    fill_price = order.stop_price
                    fee = fill_price * order.quantity * self.taker_fee
                    filled = True

            if filled:
                order.status = OrderStatus.FILLED
                order.filled_qty = order.quantity
                order.avg_fill_price = fill_price
                order.fees = fee
                orders_to_remove.append(client_id)

                filled_results.append(
                    OrderResult(
                        client_order_id=client_id,
                        exchange_order_id=f"sim_{client_id}",
                        status=OrderStatus.FILLED,
                        filled_qty=order.quantity,
                        avg_fill_price=fill_price,
                        fees=fee,
                    )
                )

        for client_id in orders_to_remove:
            del self.active_orders[client_id]

        return filled_results
