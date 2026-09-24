"""Strategy domain database repositories.

Handles Strategies, Strategy Versions, Strategy Metrics, and Backtest Results.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.database.models import (
    BacktestResult as BacktestResultModel,
    Strategy as StrategyModel,
    StrategyMetrics as StrategyMetricsModel,
    StrategyVersion as StrategyVersionModel,
)

log = get_logger(__name__)


class StrategyRepository:
    """Async repository for Strategies and immutable strategy versions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def register_strategy(self, name: str, category: str, description: str | None = None) -> StrategyModel:
        stmt = select(StrategyModel).where(StrategyModel.name == name)
        existing = (await self._session.execute(stmt)).scalars().first()
        if existing:
            return existing

        strat = StrategyModel(name=name, category=category, description=description)
        self._session.add(strat)
        await self._session.flush()
        return strat

    async def create_version(
        self,
        strategy_id: UUID,
        version: str,
        params: dict,
        model_version_id: UUID | None = None,
        status: str = "CANDIDATE",
    ) -> StrategyVersionModel:
        ver = StrategyVersionModel(
            strategy_id=strategy_id,
            version=version,
            params=params,
            model_version_id=model_version_id,
            status=status,
        )
        self._session.add(ver)
        await self._session.flush()
        return ver

    async def promote_to_champion(self, version_id: UUID) -> None:
        """Promote a challenger/paper version to CHAMPION and retire existing champion."""
        stmt_ver = select(StrategyVersionModel).where(StrategyVersionModel.id == version_id)
        target = (await self._session.execute(stmt_ver)).scalars().first()
        if not target:
            raise ValueError(f"StrategyVersion {version_id} not found")

        # Demote previous champion of this strategy
        demote_stmt = (
            update(StrategyVersionModel)
            .where(
                StrategyVersionModel.strategy_id == target.strategy_id,
                StrategyVersionModel.status == "CHAMPION",
            )
            .values(status="RETIRED", retired_at=datetime.now(timezone.utc))
        )
        await self._session.execute(demote_stmt)

        # Promote target
        target.status = "CHAMPION"
        target.promoted_at = datetime.now(timezone.utc)
        await self._session.flush()
        log.info("strategy_promoted_to_champion", version_id=str(version_id))

    async def get_champion(self, strategy_id: UUID) -> StrategyVersionModel | None:
        stmt = (
            select(StrategyVersionModel)
            .where(
                StrategyVersionModel.strategy_id == strategy_id,
                StrategyVersionModel.status == "CHAMPION",
            )
            .limit(1)
        )
        res = await self._session.execute(stmt)
        return res.scalars().first()


class BacktestResultRepository:
    """Async repository for saving and querying backtest runs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_result(
        self,
        strategy_version_id: UUID,
        symbol: str,
        timeframe: str,
        total_return: float,
        cagr: float,
        sharpe: float,
        sortino: float,
        max_drawdown: float,
        win_rate: float,
        profit_factor: float,
        expectancy: float,
        num_trades: int,
        fees_paid: float,
        params_used: dict | None = None,
        experiment_id: UUID | None = None,
        train_start: datetime | None = None,
        train_end: datetime | None = None,
        test_start: datetime | None = None,
        test_end: datetime | None = None,
    ) -> BacktestResultModel:
        res = BacktestResultModel(
            strategy_version_id=strategy_version_id,
            experiment_id=experiment_id,
            symbol=symbol,
            timeframe=timeframe,
            total_return=total_return,
            cagr=cagr,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            num_trades=num_trades,
            fees_paid=fees_paid,
            params_used=params_used or {},
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        self._session.add(res)
        await self._session.flush()
        return res

    async def get_results_for_strategy(self, strategy_version_id: UUID) -> list[BacktestResultModel]:
        stmt = (
            select(BacktestResultModel)
            .where(BacktestResultModel.strategy_version_id == strategy_version_id)
            .order_by(BacktestResultModel.created_at.desc())
        )
        res = await self._session.execute(stmt)
        return list(res.scalars().all())


class StrategyMetricsRepository:
    """Async repository for rolling strategy performance snapshots."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_metrics(
        self,
        strategy_version_id: UUID,
        period_start: datetime,
        period_end: datetime,
        environment: str,
        total_trades: int,
        win_rate: float,
        profit_factor: float,
        sharpe: float,
        sortino: float,
        max_drawdown: float,
        expectancy: float,
        total_return: float,
    ) -> StrategyMetricsModel:
        m = StrategyMetricsModel(
            strategy_version_id=strategy_version_id,
            period_start=period_start,
            period_end=period_end,
            environment=environment,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_drawdown,
            expectancy=expectancy,
            total_return=total_return,
        )
        self._session.add(m)
        await self._session.flush()
        return m
