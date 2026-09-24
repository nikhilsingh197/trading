"""System domain database repositories.

Handles System Events, Risk Events, Audit Logs, and Alerts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.database.models import (
    Alert as AlertModel,
    AuditLog as AuditLogModel,
    RiskEvent as RiskEventModel,
    SystemEvent as SystemEventModel,
)

log = get_logger(__name__)


class SystemEventRepository:
    """Async repository for operational system events."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log_event(
        self,
        event_type: str,
        severity: str,
        component: str,
        message: str,
        metadata: dict | None = None,
    ) -> SystemEventModel:
        ev = SystemEventModel(
            event_type=event_type,
            severity=severity,
            component=component,
            message=message,
            event_metadata=metadata or {},
        )
        self._session.add(ev)
        await self._session.flush()
        return ev

    async def get_recent_events(self, limit: int = 50, severity: str | None = None) -> list[SystemEventModel]:
        stmt = select(SystemEventModel)
        if severity:
            stmt = stmt.where(SystemEventModel.severity == severity)
        stmt = stmt.order_by(SystemEventModel.ts.desc()).limit(limit)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())


class RiskEventRepository:
    """Async repository for risk limit breaches and safety triggers."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log_risk_event(
        self,
        event_type: str,
        trigger: str,
        action_taken: str,
        details: dict | None = None,
    ) -> RiskEventModel:
        ev = RiskEventModel(
            event_type=event_type,
            trigger=trigger,
            action_taken=action_taken,
            details=details or {},
        )
        self._session.add(ev)
        await self._session.flush()
        return ev

    async def get_recent_risk_events(self, limit: int = 50) -> list[RiskEventModel]:
        stmt = select(RiskEventModel).order_by(RiskEventModel.ts.desc()).limit(limit)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())


class AuditLogRepository:
    """Async repository for immutable audit logs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log_action(
        self,
        actor: str,
        action: str,
        resource: str | None = None,
        before_val: dict | None = None,
        after_val: dict | None = None,
    ) -> AuditLogModel:
        entry = AuditLogModel(
            actor=actor,
            action=action,
            resource=resource,
            before_val=before_val or {},
            after_val=after_val or {},
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def get_audit_trail(self, limit: int = 100) -> list[AuditLogModel]:
        stmt = select(AuditLogModel).order_by(AuditLogModel.ts.desc()).limit(limit)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())


class AlertRepository:
    """Async repository for system and trading alerts."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        sent_to: dict | None = None,
    ) -> AlertModel:
        alt = AlertModel(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            sent_to=sent_to or {},
            acknowledged=False,
        )
        self._session.add(alt)
        await self._session.flush()
        return alt

    async def get_unacknowledged_alerts(self) -> list[AlertModel]:
        stmt = (
            select(AlertModel)
            .where(AlertModel.acknowledged == False)
            .order_by(AlertModel.created_at.desc())
        )
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def acknowledge_alert(self, alert_id: UUID) -> None:
        stmt = update(AlertModel).where(AlertModel.id == alert_id).values(acknowledged=True)
        await self._session.execute(stmt)
        await self._session.flush()
