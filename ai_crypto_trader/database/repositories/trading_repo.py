"""Trading domain database repositories.

Handles Orders, Trades, Positions, and Portfolio Snapshots.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID

from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.database.models import (
    Execution as ExecutionModel,
    Order as OrderModel,
    Position as PositionModel,
    PortfolioSnapshot as PortfolioSnapshotModel,
    Trade as TradeModel,
)

log = get_logger(__name__)


class OrderRepository:
    """Async repository for Orders and Executions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        environment: str = "paper",
        strategy_id: UUID | None = None,
        signal_id: UUID | None = None,
        client_order_id: str | None = None,
    ) -> OrderModel:
        """Create and persist a new order."""
        client_id = client_order_id or f"ord_{uuid.uuid4().hex[:16]}"
        order = OrderModel(
            client_order_id=client_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status="PENDING",
            environment=environment,
            strategy_id=strategy_id,
            signal_id=signal_id,
        )
        self._session.add(order)
        await self._session.flush()
        return order

    async def get_by_client_id(self, client_order_id: str) -> OrderModel | None:
        stmt = select(OrderModel).where(OrderModel.client_order_id == client_order_id)
        res = await self._session.execute(stmt)
        return res.scalars().first()

    async def update_status(
        self,
        order_id: UUID,
        status: str,
        filled_qty: float = 0.0,
        avg_fill_price: float | None = None,
        fees: float = 0.0,
        exchange_order_id: str | None = None,
    ) -> OrderModel | None:
        stmt = (
            update(OrderModel)
            .where(OrderModel.id == order_id)
            .values(
                status=status,
                filled_qty=filled_qty,
                avg_fill_price=avg_fill_price,
                fees=fees,
                exchange_order_id=exchange_order_id,
                updated_at=datetime.now(timezone.utc),
            )
            .returning(OrderModel)
        )
        res = await self._session.execute(stmt)
        await self._session.flush()
        return res.scalars().first()

    async def get_open_orders(self, symbol: str | None = None, environment: str = "paper") -> list[OrderModel]:
        stmt = select(OrderModel).where(
            OrderModel.status.in_(["PENDING", "SUBMITTED", "PARTIAL"]),
            OrderModel.environment == environment,
        )
        if symbol:
            stmt = stmt.where(OrderModel.symbol == symbol)
        res = await self._session.execute(stmt.order_by(OrderModel.created_at.desc()))
        return list(res.scalars().all())


class TradeRepository:
    """Async repository for completed Trades."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        opened_at: datetime,
        environment: str = "paper",
        fees: float = 0.0,
        slippage: float = 0.0,
        exit_reason: str = "SIGNAL",
        strategy_id: UUID | None = None,
        regime: str | None = None,
        duration_seconds: int | None = None,
    ) -> TradeModel:
        trade = TradeModel(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=fees,
            slippage=slippage,
            opened_at=opened_at,
            environment=environment,
            exit_reason=exit_reason,
            strategy_id=strategy_id,
            regime=regime,
            duration_seconds=duration_seconds,
        )
        self._session.add(trade)
        await self._session.flush()
        return trade

    async def get_recent_trades(
        self,
        limit: int = 50,
        environment: str = "paper",
        symbol: str | None = None,
    ) -> list[TradeModel]:
        stmt = select(TradeModel).where(TradeModel.environment == environment)
        if symbol:
            stmt = stmt.where(TradeModel.symbol == symbol)
        stmt = stmt.order_by(TradeModel.closed_at.desc()).limit(limit)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def count_trades(self, environment: str = "paper") -> int:
        stmt = select(func.count(TradeModel.id)).where(TradeModel.environment == environment)
        res = await self._session.execute(stmt)
        return res.scalar() or 0


class PositionRepository:
    """Async repository for active & closed positions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        environment: str = "paper",
        stop_loss: float | None = None,
        take_profit: float | None = None,
        leverage: int = 1,
        strategy_id: UUID | None = None,
    ) -> PositionModel:
        pos = PositionModel(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            environment=environment,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            strategy_id=strategy_id,
            status="OPEN",
        )
        self._session.add(pos)
        await self._session.flush()
        return pos

    async def close_position(self, position_id: UUID, realized_pnl: float) -> PositionModel | None:
        stmt = (
            update(PositionModel)
            .where(PositionModel.id == position_id)
            .values(
                status="CLOSED",
                realized_pnl=realized_pnl,
                closed_at=datetime.now(timezone.utc),
            )
            .returning(PositionModel)
        )
        res = await self._session.execute(stmt)
        await self._session.flush()
        return res.scalars().first()

    async def get_open_positions(self, environment: str = "paper") -> list[PositionModel]:
        stmt = (
            select(PositionModel)
            .where(PositionModel.status == "OPEN", PositionModel.environment == environment)
            .order_by(PositionModel.opened_at.desc())
        )
        res = await self._session.execute(stmt)
        return list(res.scalars().all())


class PortfolioSnapshotRepository:
    """Async repository for equity and drawdown snapshots."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_snapshot(
        self,
        total_equity: float,
        available_cash: float,
        unrealized_pnl: float,
        realized_pnl: float,
        drawdown_pct: float,
        peak_equity: float,
        environment: str = "paper",
        ts: datetime | None = None,
    ) -> PortfolioSnapshotModel:
        snapshot = PortfolioSnapshotModel(
            ts=ts or datetime.now(timezone.utc),
            total_equity=total_equity,
            available_cash=available_cash,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            drawdown_pct=drawdown_pct,
            peak_equity=peak_equity,
            environment=environment,
        )
        self._session.add(snapshot)
        await self._session.flush()
        return snapshot

    async def get_latest_snapshot(self, environment: str = "paper") -> PortfolioSnapshotModel | None:
        stmt = (
            select(PortfolioSnapshotModel)
            .where(PortfolioSnapshotModel.environment == environment)
            .order_by(PortfolioSnapshotModel.ts.desc())
            .limit(1)
        )
        res = await self._session.execute(stmt)
        return res.scalars().first()

    async def get_history(self, limit: int = 100, environment: str = "paper") -> list[PortfolioSnapshotModel]:
        stmt = (
            select(PortfolioSnapshotModel)
            .where(PortfolioSnapshotModel.environment == environment)
            .order_by(PortfolioSnapshotModel.ts.desc())
            .limit(limit)
        )
        res = await self._session.execute(stmt)
        return list(res.scalars().all())
