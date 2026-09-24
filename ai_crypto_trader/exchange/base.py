"""Base abstraction for exchange connectivity."""
from __future__ import annotations

import abc
from typing import Any, Optional

from ai_crypto_trader.core.enums import Direction, OrderStatus, OrderType, Side
from ai_crypto_trader.core.interfaces import BrokerABC, OrderRequest, OrderResult, PortfolioState, PositionSnapshot


class ExchangeAdapterABC(BrokerABC, abc.ABC):
    """Abstract interface for cryptocurrency exchange adapters (CCXT / Native REST & WebSocket)."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish exchange API connection and validate credentials."""
        ...

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close API client and underlying HTTP sessions."""
        ...

    @abc.abstractmethod
    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """Fetch current market ticker (bid, ask, last, mark price)."""
        ...

    @abc.abstractmethod
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """Fetch top of order book snapshot."""
        ...

    @abc.abstractmethod
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
        """Submit an order to the exchange."""
        ...

    @abc.abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders for a specific symbol or globally."""
        ...

    @abc.abstractmethod
    async def fetch_order(self, client_order_id: str, symbol: Optional[str] = None) -> OrderResult:
        """Query status of a specific order."""
        ...

    @abc.abstractmethod
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[OrderResult]:
        """Fetch all currently open/active orders."""
        ...

    @abc.abstractmethod
    async def fetch_positions(self, symbols: Optional[list[str]] = None) -> list[PositionSnapshot]:
        """Fetch open perpetual/margin positions."""
        ...

    @abc.abstractmethod
    async def fetch_balance(self) -> dict[str, float]:
        """Fetch cash and margin balances (free, used, total)."""
        ...
