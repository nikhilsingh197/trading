"""AI and Machine Learning domain database repositories.

Handles ML Models, Model Versions, Signals, and Research Experiments.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.database.models import (
    Experiment as ExperimentModel,
    MLModel as MLModelModel,
    MLModelVersion as MLModelVersionModel,
    Signal as SignalModel,
)

log = get_logger(__name__)


class ModelRepository:
    """Async repository for ML Models and immutable model versions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def register_model(self, name: str, model_type: str, description: str | None = None) -> MLModelModel:
        stmt = select(MLModelModel).where(MLModelModel.name == name)
        existing = (await self._session.execute(stmt)).scalars().first()
        if existing:
            return existing

        model = MLModelModel(name=name, model_type=model_type, description=description)
        self._session.add(model)
        await self._session.flush()
        return model

    async def create_version(
        self,
        model_id: UUID,
        version: str,
        artifact_path: str,
        params: dict | None = None,
        feature_list: list[str] | None = None,
        target: str | None = None,
        val_metrics: dict | None = None,
        test_metrics: dict | None = None,
        status: str = "CANDIDATE",
    ) -> MLModelVersionModel:
        ver = MLModelVersionModel(
            model_id=model_id,
            version=version,
            artifact_path=artifact_path,
            params=params or {},
            feature_list=feature_list or [],
            target=target,
            val_metrics=val_metrics or {},
            test_metrics=test_metrics or {},
            status=status,
        )
        self._session.add(ver)
        await self._session.flush()
        return ver

    async def get_latest_version(self, model_id: UUID) -> MLModelVersionModel | None:
        stmt = (
            select(MLModelVersionModel)
            .where(MLModelVersionModel.model_id == model_id)
            .order_by(MLModelVersionModel.created_at.desc())
            .limit(1)
        )
        res = await self._session.execute(stmt)
        return res.scalars().first()


class SignalRepository:
    """Async repository for generated signals."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        expected_return: float,
        risk_pct: float,
        regime: str,
        reason: str,
        strategy_version_id: UUID | None = None,
    ) -> SignalModel:
        sig = SignalModel(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_return=expected_return,
            risk_pct=risk_pct,
            regime=regime,
            reason=reason,
            strategy_version_id=strategy_version_id,
        )
        self._session.add(sig)
        await self._session.flush()
        return sig

    async def mark_acted(self, signal_id: UUID) -> None:
        stmt = update(SignalModel).where(SignalModel.id == signal_id).values(acted_upon=True)
        await self._session.execute(stmt)
        await self._session.flush()

    async def get_recent_signals(self, limit: int = 50, symbol: str | None = None) -> list[SignalModel]:
        stmt = select(SignalModel)
        if symbol:
            stmt = stmt.where(SignalModel.symbol == symbol)
        stmt = stmt.order_by(SignalModel.ts.desc()).limit(limit)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())


class ExperimentRepository:
    """Async repository for tracking AI research experiments."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_experiment(
        self,
        name: str,
        hypothesis: str,
        features: list[str] | None = None,
        strategy_params: dict | None = None,
        model_type: str | None = None,
    ) -> ExperimentModel:
        exp = ExperimentModel(
            name=name,
            hypothesis=hypothesis,
            features=features or [],
            strategy_params=strategy_params or {},
            model_type=model_type,
            result="RUNNING",
        )
        self._session.add(exp)
        await self._session.flush()
        return exp

    async def complete_experiment(
        self,
        experiment_id: UUID,
        result: str,
        decision: str,
        metrics: dict | None = None,
    ) -> ExperimentModel | None:
        stmt = (
            update(ExperimentModel)
            .where(ExperimentModel.id == experiment_id)
            .values(
                result=result,
                decision=decision,
                metrics=metrics or {},
                completed_at=datetime.now(timezone.utc),
            )
            .returning(ExperimentModel)
        )
        res = await self._session.execute(stmt)
        await self._session.flush()
        return res.scalars().first()

    async def list_experiments(self, limit: int = 50) -> list[ExperimentModel]:
        stmt = select(ExperimentModel).order_by(ExperimentModel.created_at.desc()).limit(limit)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())
