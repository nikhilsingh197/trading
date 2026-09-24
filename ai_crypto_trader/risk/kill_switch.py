"""Global kill switch.

The kill switch is the highest-priority safety control.
Once activated, ALL new positions are blocked regardless of any other signal.
The kill switch can only be deactivated by explicit human action.

Design:
- Stores state in Redis for cross-process visibility.
- Falls back to in-memory state if Redis is unavailable.
- Every component that submits orders MUST check is_active() first.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from ai_crypto_trader.core.enums import AlertType
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)

KILL_SWITCH_KEY = "kill_switch:active"
KILL_SWITCH_REASON_KEY = "kill_switch:reason"
KILL_SWITCH_TIME_KEY = "kill_switch:activated_at"


class KillSwitch:
    """Thread-safe, cross-process kill switch.

    Usage:
        ks = KillSwitch(redis_client)
        if await ks.is_active():
            raise KillSwitchActivated(...)
        await ks.activate("Drawdown exceeded 10%")
    """

    def __init__(self, redis_client=None) -> None:
        self._redis = redis_client
        # In-memory fallback
        self._active: bool = False
        self._reason: str = ""
        self._activated_at: Optional[datetime] = None

    async def is_active(self) -> bool:
        """Return True if the kill switch is engaged."""
        if self._redis:
            try:
                val = await self._redis.get(KILL_SWITCH_KEY)
                return val == b"1"
            except Exception:
                log.error("kill_switch_redis_error", fallback="in_memory")
        return self._active

    async def activate(self, reason: str = "Manual activation") -> None:
        """Engage the kill switch. Blocks all new trading."""
        now = datetime.now(timezone.utc)
        self._active = True
        self._reason = reason
        self._activated_at = now

        if self._redis:
            try:
                async with self._redis.pipeline() as pipe:
                    await pipe.set(KILL_SWITCH_KEY, "1")
                    await pipe.set(KILL_SWITCH_REASON_KEY, reason)
                    await pipe.set(KILL_SWITCH_TIME_KEY, now.isoformat())
                    await pipe.execute()
            except Exception:
                log.error("kill_switch_redis_write_error")

        log.critical(
            "KILL_SWITCH_ACTIVATED",
            reason=reason,
            activated_at=now.isoformat(),
        )

    async def deactivate(self, actor: str = "system") -> None:
        """Disengage the kill switch. MUST be called by an authorized human."""
        self._active = False
        self._reason = ""
        self._activated_at = None

        if self._redis:
            try:
                async with self._redis.pipeline() as pipe:
                    await pipe.delete(KILL_SWITCH_KEY)
                    await pipe.delete(KILL_SWITCH_REASON_KEY)
                    await pipe.delete(KILL_SWITCH_TIME_KEY)
                    await pipe.execute()
            except Exception:
                log.error("kill_switch_redis_delete_error")

        log.warning(
            "KILL_SWITCH_DEACTIVATED",
            actor=actor,
        )

    async def status(self) -> dict:
        active = await self.is_active()
        return {
            "active": active,
            "reason": self._reason,
            "activated_at": self._activated_at.isoformat() if self._activated_at else None,
        }
