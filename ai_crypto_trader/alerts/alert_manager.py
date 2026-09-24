"""Centralized Alert & Incident Manager.

Dispatches alerts across Telegram, Email, and Database logs with severity-based
routing, debouncing (deduplication), and acknowledgement tracking.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import uuid

from ai_crypto_trader.alerts.email_sender import EmailSender
from ai_crypto_trader.alerts.telegram_sender import TelegramSender
from ai_crypto_trader.config.settings import Settings, get_settings
from ai_crypto_trader.core.enums import AlertSeverity, AlertType
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class AlertRecord:
    """Immutable audit record of a triggered platform alert."""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    metadata: dict[str, Any]
    sent_to: list[str]
    acknowledged: bool
    created_at: datetime


class AlertManager:
    """Institutional alert manager with debouncing, multi-channel dispatch, and acknowledgement."""

    def __init__(
        self,
        telegram_sender: Optional[TelegramSender] = None,
        email_sender: Optional[EmailSender] = None,
        settings: Optional[Settings] = None,
        debounce_seconds: float = 60.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.telegram = telegram_sender or TelegramSender(
            token=self.settings.alert_telegram_token,
            chat_id=self.settings.alert_telegram_chat_id,
        )
        self.email = email_sender or EmailSender(
            host=self.settings.alert_email_host,
            port=self.settings.alert_email_port,
            user=self.settings.alert_email_user,
            password=self.settings.alert_email_password,
            to_email=self.settings.alert_email_to,
        )
        self.debounce_seconds = debounce_seconds

        # Internal state
        self._history: list[AlertRecord] = []
        self._last_sent: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def send_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        metadata: Optional[dict[str, Any]] = None,
        force: bool = False,
    ) -> AlertRecord:
        """Trigger and route an alert to appropriate channels based on severity."""
        meta = metadata or {}
        now = datetime.now(timezone.utc)
        debounce_key = f"{alert_type.value}:{title}"

        # Deduplication check (unless severity is CRITICAL or force=True)
        if not force and severity != AlertSeverity.CRITICAL:
            if debounce_key in self._last_sent:
                elapsed = (now - self._last_sent[debounce_key]).total_seconds()
                if elapsed < self.debounce_seconds:
                    log.info(
                        "alert_debounced_suppressed",
                        alert_type=alert_type.value,
                        title=title,
                        elapsed=elapsed,
                    )
                    # Return latest matching record
                    for item in reversed(self._history):
                        if item.title == title and item.alert_type == alert_type:
                            return item

        self._last_sent[debounce_key] = now
        alert_id = str(uuid.uuid4())
        channels_sent: list[str] = []

        # Structured audit log
        log.warning(
            "ALERT_TRIGGERED",
            alert_id=alert_id,
            alert_type=alert_type.value,
            severity=severity.value,
            title=title,
            metadata=meta,
        )

        tasks = []
        # Routing logic
        # 1. Telegram: WARNING, ERROR, CRITICAL
        if severity in (AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL):
            channels_sent.append("telegram")
            tasks.append(self.telegram.send(title, message, severity.value))

        # 2. Email: ERROR, CRITICAL
        if severity in (AlertSeverity.ERROR, AlertSeverity.CRITICAL):
            channels_sent.append("email")
            tasks.append(self.email.send(title, message, severity.value))

        # Execute channel dispatches concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        record = AlertRecord(
            id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            metadata=meta,
            sent_to=channels_sent,
            acknowledged=False,
            created_at=now,
        )

        async with self._lock:
            self._history.append(record)

        return record

    def get_recent_alerts(
        self,
        limit: int = 50,
        severity: Optional[AlertSeverity] = None,
        unacknowledged_only: bool = False,
    ) -> list[AlertRecord]:
        """Fetch alert history with optional filtering."""
        filtered = self._history
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        if unacknowledged_only:
            filtered = [a for a in filtered if not a.acknowledged]

        return list(reversed(filtered[-limit:]))

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as reviewed/acknowledged by an operator."""
        for alert in self._history:
            if alert.id == alert_id:
                alert.acknowledged = True
                log.info("alert_acknowledged", alert_id=alert_id, title=alert.title)
                return True
        return False

    def clear_history(self) -> None:
        """Clear memory alert history."""
        self._history.clear()
        self._last_sent.clear()
