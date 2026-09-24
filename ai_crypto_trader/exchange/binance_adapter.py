"""Binance Exchange Adapter using CCXT.

Supports Binance Spot and USDT-M Futures with unified order submission,
order book queries, position snapshots, balance checks, and testnet sandboxing.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

import ccxt

from ai_crypto_trader.core.enums import Direction, OrderStatus, OrderType, Side
from ai_crypto_trader.core.exceptions import (
    ExchangeConnectionError,
    ExchangeRateLimitError,
    OrderNotFoundError,
    OrderSubmissionError,
)
from ai_crypto_trader.core.interfaces import OrderRequest, OrderResult, PortfolioState, PositionSnapshot
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.exchange.base import ExchangeAdapterABC

log = get_logger(__name__)


class BinanceAdapter(ExchangeAdapterABC):
    """Institutional adapter connecting to Binance Spot / Futures API via CCXT."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        is_futures: bool = True,
        initial_sandbox_balance: float = 10000.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.testnet = testnet
        self.is_futures = is_futures
        self.initial_balance = initial_sandbox_balance

        self._connected = False
        self._ccxt_client: Optional[ccxt.binance] = None

        # In-memory sandbox storage (active when credentials not provided or offline)
        self._sandbox_balance = initial_sandbox_balance
        self._sandbox_positions: dict[str, PositionSnapshot] = {}
        self._sandbox_orders: dict[str, OrderResult] = {}
        self._sandbox_tickers: dict[str, float] = {
            "BTC/USDT": 50000.0,
            "BTCUSDT": 50000.0,
            "ETH/USDT": 3000.0,
            "ETHUSDT": 3000.0,
            "SOL/USDT": 150.0,
            "SOLUSDT": 150.0,
        }

    @property
    def is_live_exchange(self) -> bool:
        return bool(self.api_key and self.api_secret and self._connected)

    async def connect(self) -> None:
        """Initialize CCXT client and verify connectivity."""
        if self.api_key and self.api_secret:
            options: dict[str, Any] = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future" if self.is_futures else "spot",
                },
            }
            client = ccxt.binance(options)
            if self.testnet:
                client.set_sandbox_mode(True)

            try:
                await asyncio.to_thread(client.load_markets)
                self._ccxt_client = client
                self._connected = True
                log.info(
                    "binance_connected",
                    testnet=self.testnet,
                    futures=self.is_futures,
                )
                return
            except Exception as e:
                log.warning("binance_connection_fallback_to_sandbox", error=str(e))

        # Fallback to local sandbox engine
        self._connected = True
        log.info("binance_sandbox_mode_initialized", initial_balance=self.initial_balance)

    async def disconnect(self) -> None:
        """Close connection pools."""
        self._connected = False
        self._ccxt_client = None
        log.info("binance_disconnected")

    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """Fetch market price ticker."""
        normalized = self._normalize_symbol(symbol)
        if self.is_live_exchange and self._ccxt_client:
            try:
                t = await asyncio.to_thread(self._ccxt_client.fetch_ticker, normalized)
                last_price = float(t.get("last") or t.get("close") or 0.0)
                return {
                    "bid": float(t.get("bid") or last_price * 0.9999),
                    "ask": float(t.get("ask") or last_price * 1.0001),
                    "last": last_price,
                    "mark": float(t.get("last") or last_price),
                }
            except Exception as e:
                self._handle_ccxt_error(e)

        # Sandbox ticker
        price = self._sandbox_tickers.get(normalized, self._sandbox_tickers.get(symbol, 50000.0))
        return {
            "bid": round(price * 0.9999, 2),
            "ask": round(price * 1.0001, 2),
            "last": price,
            "mark": price,
        }

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """Fetch order book snapshot."""
        normalized = self._normalize_symbol(symbol)
        if self.is_live_exchange and self._ccxt_client:
            try:
                return await asyncio.to_thread(self._ccxt_client.fetch_order_book, normalized, limit)
            except Exception as e:
                self._handle_ccxt_error(e)

        ticker = await self.fetch_ticker(symbol)
        mid = ticker["last"]
        bids = [[round(mid * (1 - 0.0001 * i), 2), round(1.0 + 0.2 * i, 4)] for i in range(1, limit + 1)]
        asks = [[round(mid * (1 + 0.0001 * i), 2), round(1.0 + 0.2 * i, 4)] for i in range(1, limit + 1)]
        return {"symbol": symbol, "bids": bids, "asks": asks, "timestamp": datetime.now(timezone.utc).timestamp()}

    async def create_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """Submit an order to the exchange or execute via local sandbox."""
        cl_id = client_order_id or f"ACT-{uuid.uuid4().hex[:12]}"
        normalized = self._normalize_symbol(symbol)

        if self.is_live_exchange and self._ccxt_client:
            try:
                ccxt_side = side.value.lower()
                ccxt_type = order_type.value.lower()
                params = {"clientOrderId": cl_id}
                if stop_price:
                    params["stopPrice"] = stop_price

                res = await asyncio.to_thread(
                    self._ccxt_client.create_order,
                    normalized,
                    ccxt_type,
                    ccxt_side,
                    quantity,
                    price,
                    params,
                )
                return self._map_ccxt_order_to_result(res, cl_id)
            except Exception as e:
                self._handle_ccxt_error(e)

        # Sandbox Order Execution
        ticker = await self.fetch_ticker(symbol)
        market_price = ticker["last"]
        fill_price = price if (order_type == OrderType.LIMIT and price) else market_price

        # Fee (0.04% taker)
        fee = round(quantity * fill_price * 0.0004, 4)

        result = OrderResult(
            client_order_id=cl_id,
            exchange_order_id=f"BINANCE-SIM-{uuid.uuid4().hex[:8]}",
            status=OrderStatus.FILLED,
            filled_qty=quantity,
            avg_fill_price=fill_price,
            fees=fee,
        )
        self._sandbox_orders[cl_id] = result

        # Update Sandbox Position & Balance
        direction = Direction.LONG if side == Side.BUY else Direction.SHORT
        self._update_sandbox_position(symbol, direction, quantity, fill_price, fee)

        log.info(
            "sandbox_order_filled",
            symbol=symbol,
            side=side.value,
            qty=quantity,
            price=fill_price,
            fee=fee,
            client_order_id=cl_id,
        )
        return result

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        """BrokerABC implementation."""
        return await self.create_order(
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            client_order_id=request.client_order_id,
        )

    async def cancel_order(self, client_order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an open order."""
        if self.is_live_exchange and self._ccxt_client and symbol:
            try:
                normalized = self._normalize_symbol(symbol)
                await asyncio.to_thread(self._ccxt_client.cancel_order, client_order_id, normalized)
                return True
            except Exception as e:
                log.warning("cancel_order_failed", id=client_order_id, error=str(e))
                return False

        if client_order_id in self._sandbox_orders:
            order = self._sandbox_orders[client_order_id]
            order.status = OrderStatus.CANCELLED
            log.info("sandbox_order_cancelled", id=client_order_id)
            return True
        return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        cancelled_count = 0
        if self.is_live_exchange and self._ccxt_client:
            try:
                normalized = self._normalize_symbol(symbol) if symbol else None
                res = await asyncio.to_thread(self._ccxt_client.cancel_all_orders, normalized)
                return len(res) if isinstance(res, list) else 1
            except Exception as e:
                log.warning("cancel_all_orders_exception", error=str(e))

        for ord_id, ord_res in list(self._sandbox_orders.items()):
            if ord_res.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                ord_res.status = OrderStatus.CANCELLED
                cancelled_count += 1
        return cancelled_count

    async def fetch_order(self, client_order_id: str, symbol: Optional[str] = None) -> OrderResult:
        """Fetch details of an order."""
        if self.is_live_exchange and self._ccxt_client and symbol:
            try:
                normalized = self._normalize_symbol(symbol)
                res = await asyncio.to_thread(self._ccxt_client.fetch_order, client_order_id, normalized)
                return self._map_ccxt_order_to_result(res, client_order_id)
            except Exception as e:
                self._handle_ccxt_error(e)

        if client_order_id in self._sandbox_orders:
            return self._sandbox_orders[client_order_id]

        raise OrderNotFoundError(f"Order {client_order_id} not found")

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[OrderResult]:
        """Fetch all currently open orders."""
        if self.is_live_exchange and self._ccxt_client:
            try:
                normalized = self._normalize_symbol(symbol) if symbol else None
                orders = await asyncio.to_thread(self._ccxt_client.fetch_open_orders, normalized)
                return [self._map_ccxt_order_to_result(o, o.get("clientOrderId", "")) for o in orders]
            except Exception as e:
                self._handle_ccxt_error(e)

        return [o for o in self._sandbox_orders.values() if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)]

    async def fetch_positions(self, symbols: Optional[list[str]] = None) -> list[PositionSnapshot]:
        """Fetch open positions."""
        if self.is_live_exchange and self._ccxt_client:
            try:
                positions = await asyncio.to_thread(self._ccxt_client.fetch_positions, symbols)
                results = []
                for p in positions:
                    contracts = float(p.get("contracts") or p.get("amount") or 0.0)
                    if abs(contracts) > 0:
                        side = Direction.LONG if contracts > 0 else Direction.SHORT
                        entry_price = float(p.get("entryPrice") or 0.0)
                        mark_price = float(p.get("markPrice") or entry_price)
                        unrealized = float(p.get("unrealizedPnl") or 0.0)
                        pct = (unrealized / (entry_price * abs(contracts))) * 100.0 if entry_price > 0 else 0.0
                        results.append(
                            PositionSnapshot(
                                symbol=p.get("symbol", ""),
                                side=side,
                                entry_price=entry_price,
                                quantity=abs(contracts),
                                current_price=mark_price,
                                unrealized_pnl=unrealized,
                                unrealized_pnl_pct=pct,
                                stop_loss=None,
                                take_profit=None,
                            )
                        )
                return results
            except Exception as e:
                self._handle_ccxt_error(e)

        return list(self._sandbox_positions.values())

    async def fetch_balance(self) -> dict[str, float]:
        """Fetch cash and margin balances."""
        if self.is_live_exchange and self._ccxt_client:
            try:
                b = await asyncio.to_thread(self._ccxt_client.fetch_balance)
                usdt = b.get("USDT", {})
                free = float(usdt.get("free", 0.0))
                used = float(usdt.get("used", 0.0))
                total = float(usdt.get("total", free + used))
                return {"free": free, "used": used, "total": total}
            except Exception as e:
                self._handle_ccxt_error(e)

        used_margin = sum(p.quantity * p.entry_price for p in self._sandbox_positions.values())
        return {
            "free": max(0.0, self._sandbox_balance - used_margin),
            "used": used_margin,
            "total": self._sandbox_balance,
        }

    async def get_portfolio_state(self) -> PortfolioState:
        """BrokerABC implementation: aggregate current portfolio state."""
        bal = await self.fetch_balance()
        positions = await self.fetch_positions()

        unrealized = sum(p.unrealized_pnl for p in positions)
        total_equity = bal["total"] + unrealized

        return PortfolioState(
            total_equity=round(total_equity, 2),
            available_cash=round(bal["free"], 2),
            unrealized_pnl=round(unrealized, 2),
            realized_pnl=0.0,
            drawdown_pct=0.0,
            peak_equity=round(total_equity, 2),
            open_positions=positions,
        )

    def set_sandbox_ticker(self, symbol: str, price: float) -> None:
        """Testing utility to inject market prices in sandbox mode."""
        self._sandbox_tickers[symbol] = price
        self._sandbox_tickers[self._normalize_symbol(symbol)] = price

        # Update unrealized PnL on positions
        if symbol in self._sandbox_positions:
            pos = self._sandbox_positions[symbol]
            pos.current_price = price
            pnl = (price - pos.entry_price) * pos.quantity if pos.side == Direction.LONG else (pos.entry_price - price) * pos.quantity
            pos.unrealized_pnl = round(pnl, 2)
            pos.unrealized_pnl_pct = round((pnl / (pos.entry_price * pos.quantity)) * 100.0, 2)

    def _update_sandbox_position(
        self,
        symbol: str,
        side: Direction,
        quantity: float,
        price: float,
        fee: float,
    ) -> None:
        self._sandbox_balance -= fee
        if symbol in self._sandbox_positions:
            existing = self._sandbox_positions[symbol]
            if existing.side == side:
                # Add to position
                new_qty = existing.quantity + quantity
                avg_entry = ((existing.entry_price * existing.quantity) + (price * quantity)) / new_qty
                existing.quantity = new_qty
                existing.entry_price = round(avg_entry, 2)
                existing.current_price = price
            else:
                # Close or flip
                if quantity >= existing.quantity:
                    del self._sandbox_positions[symbol]
                else:
                    existing.quantity -= quantity
        else:
            self._sandbox_positions[symbol] = PositionSnapshot(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=quantity,
                current_price=price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                stop_loss=None,
                take_profit=None,
            )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = symbol.upper().replace("-", "").replace("_", "")
        if "/" not in s:
            if s.endswith("USDT"):
                return f"{s[:-4]}/USDT"
            elif s.endswith("USD"):
                return f"{s[:-3]}/USD"
        return s

    @staticmethod
    def _map_ccxt_order_to_result(order_dict: dict[str, Any], cl_id: str) -> OrderResult:
        status_raw = str(order_dict.get("status", "")).lower()
        mapping = {
            "open": OrderStatus.SUBMITTED,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        status = mapping.get(status_raw, OrderStatus.SUBMITTED)

        filled = float(order_dict.get("filled", 0.0))
        price = float(order_dict.get("price") or order_dict.get("average") or 0.0)
        fee = float(order_dict.get("fee", {}).get("cost", 0.0)) if order_dict.get("fee") else 0.0

        return OrderResult(
            client_order_id=cl_id or str(order_dict.get("clientOrderId", "")),
            exchange_order_id=str(order_dict.get("id", "")),
            status=status,
            filled_qty=filled,
            avg_fill_price=price if price > 0 else None,
            fees=fee,
        )

    @staticmethod
    def _handle_ccxt_error(e: Exception) -> None:
        if isinstance(e, ccxt.RateLimitExceeded):
            raise ExchangeRateLimitError(f"Binance rate limit breached: {e}")
        elif isinstance(e, ccxt.InsufficientFunds):
            raise OrderSubmissionError(f"Insufficient funds: {e}")
        elif isinstance(e, ccxt.InvalidOrder):
            raise OrderSubmissionError(f"Invalid order payload: {e}")
        elif isinstance(e, ccxt.OrderNotFound):
            raise OrderNotFoundError(f"Order not found: {e}")
        elif isinstance(e, (ccxt.NetworkError, ccxt.ExchangeNotAvailable)):
            raise ExchangeConnectionError(f"Exchange connection lost: {e}")
        else:
            raise OrderSubmissionError(f"Exchange error: {e}")
