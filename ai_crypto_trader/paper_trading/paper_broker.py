"""Paper broker implementing BrokerABC with simulated order routing and fill processing."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ai_crypto_trader.core.enums import Direction, OrderStatus, OrderType, Side
from ai_crypto_trader.core.interfaces import BrokerABC, Candle, OrderRequest, OrderResult, PortfolioState
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.paper_trading.order_matcher import OrderMatcher
from ai_crypto_trader.paper_trading.paper_portfolio import PaperPortfolio, PaperTradeRecord

log = get_logger(__name__)


class PaperBroker(BrokerABC):
    """Paper trading broker providing simulated exchange execution and portfolio tracking."""

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0006,
        base_slippage: float = 0.0005,
    ) -> None:
        self.portfolio = PaperPortfolio(initial_capital=initial_capital)
        self.matcher = OrderMatcher(
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            base_slippage=base_slippage,
        )
        self.latest_prices: dict[str, float] = {}
        # Tracks OCO (One-Cancels-the-Other) paired SL and TP orders: order_id -> partner_order_id
        self.oco_pairs: dict[str, str] = {}

    def get_price(self, symbol: str) -> float | None:
        return self.latest_prices.get(symbol)

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order for simulated execution."""
        symbol = request.symbol
        current_price = self.latest_prices.get(symbol)

        if request.order_type == OrderType.MARKET:
            if current_price is None:
                return OrderResult(
                    client_order_id=request.client_order_id or "err",
                    exchange_order_id=None,
                    status=OrderStatus.REJECTED,
                    filled_qty=0.0,
                    avg_fill_price=None,
                    fees=0.0,
                    error=f"No price data available for {symbol}",
                )

            # Bid/ask spread approximation: 0.02%
            bid = current_price * 0.9999
            ask = current_price * 1.0001

            # Estimate cost
            est_fee = current_price * request.quantity * self.matcher.taker_fee
            # Check cash if opening position
            existing = self.portfolio.get_position(symbol)
            is_opening = (
                existing is None
                or (existing.side == Direction.LONG and request.side == Side.BUY)
                or (existing.side == Direction.SHORT and request.side == Side.SELL)
            )
            if is_opening and not self.portfolio.can_open_position(request.side, ask, request.quantity, est_fee):
                return OrderResult(
                    client_order_id=request.client_order_id or "err",
                    exchange_order_id=None,
                    status=OrderStatus.REJECTED,
                    filled_qty=0.0,
                    avg_fill_price=None,
                    fees=0.0,
                    error="Insufficient available cash in paper account",
                )

            # Execute market fill
            result = self.matcher.execute_market_order(request, bid=bid, ask=ask)
            if result.status == OrderStatus.FILLED:
                self.portfolio.on_order_fill(
                    symbol=symbol,
                    side=request.side,
                    quantity=result.filled_qty,
                    fill_price=result.avg_fill_price,  # type: ignore
                    fee=result.fees,
                    stop_loss=request.stop_price if request.stop_price else None,
                    strategy_id=request.strategy_id,
                )

                # If SL or TP were specified on the market entry request, place protective orders
                pos = self.portfolio.get_position(symbol)
                if pos is not None and (request.price is not None or request.stop_price is not None):
                    self._place_bracket_orders(request, pos)

            return result

        else:
            # Passive order (LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET)
            return self.matcher.register_passive_order(request)

    def _place_bracket_orders(self, entry_req: OrderRequest, pos: Any) -> None:
        """Place protective Stop-Loss and Take-Profit orders for an opened position."""
        exit_side = Side.SELL if pos.side == Direction.LONG else Side.BUY

        sl_id = f"sl_{uuid.uuid4().hex[:10]}"
        tp_id = f"tp_{uuid.uuid4().hex[:10]}"

        # Stop-loss
        if entry_req.stop_price is not None:
            self.matcher.register_passive_order(
                OrderRequest(
                    symbol=pos.symbol,
                    side=exit_side,
                    order_type=OrderType.STOP_MARKET,
                    quantity=pos.quantity,
                    stop_price=entry_req.stop_price,
                    client_order_id=sl_id,
                    strategy_id=entry_req.strategy_id,
                )
            )

        # Take-profit (entry_req.price can carry target TP price)
        if entry_req.price is not None:
            self.matcher.register_passive_order(
                OrderRequest(
                    symbol=pos.symbol,
                    side=exit_side,
                    order_type=OrderType.TAKE_PROFIT_MARKET,
                    quantity=pos.quantity,
                    stop_price=entry_req.price,
                    client_order_id=tp_id,
                    strategy_id=entry_req.strategy_id,
                )
            )

        # Link OCO pair
        if entry_req.stop_price and entry_req.price:
            self.oco_pairs[sl_id] = tp_id
            self.oco_pairs[tp_id] = sl_id

    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel a pending paper order."""
        return self.matcher.cancel_order(client_order_id)

    async def get_portfolio_state(self) -> PortfolioState:
        """Return current snapshot of paper portfolio."""
        return self.portfolio.get_state()

    def on_candle(self, candle: Candle) -> list[PaperTradeRecord]:
        """Process incoming market candle, updating market prices and evaluating fills."""
        symbol = candle.symbol
        close = float(candle.close)
        high = float(candle.high)
        low = float(candle.low)
        vol = float(candle.volume)

        self.latest_prices[symbol] = close
        self.portfolio.update_price(symbol, close)

        # Match pending orders against candle range
        filled_orders = self.matcher.process_price_bar(
            symbol=symbol,
            high=high,
            low=low,
            close=close,
            volume=vol,
        )

        completed_trades: list[PaperTradeRecord] = []
        for order_res in filled_orders:
            # Check OCO partner cancellation
            partner_id = self.oco_pairs.pop(order_res.client_order_id, None)
            if partner_id:
                self.matcher.cancel_order(partner_id)
                self.oco_pairs.pop(partner_id, None)

            # Determine side from original order
            is_sl = order_res.client_order_id.startswith("sl_")
            is_tp = order_res.client_order_id.startswith("tp_")
            exit_reason = "STOP_LOSS" if is_sl else ("TAKE_PROFIT" if is_tp else "LIMIT")

            # Determine side from position
            pos = self.portfolio.get_position(symbol)
            if pos:
                side = Side.SELL if pos.side == Direction.LONG else Side.BUY
            else:
                side = Side.BUY

            trade = self.portfolio.on_order_fill(
                symbol=symbol,
                side=side,
                quantity=order_res.filled_qty,
                fill_price=order_res.avg_fill_price,  # type: ignore
                fee=order_res.fees,
                exit_reason=exit_reason,
                timestamp=candle.open_time,
            )
            if trade:
                completed_trades.append(trade)

        return completed_trades
