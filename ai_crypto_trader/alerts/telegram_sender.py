"""Telegram notification sender.

Delivers real-time trade, risk, and platform incident alerts to operator Telegram chats.
Includes token bucket rate-limiting, exponential backoff retries, and markdown formatting.
Falls back to mock mode if credentials are empty or during unit tests.
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from ai_crypto_trader.core.interfaces import AlertSenderABC
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)

SEVERITY_EMOJIS = {
    "INFO": "ℹ️",
    "WARNING": "⚠️",
    "ERROR": "🚨",
    "CRITICAL": "🔥",
}


class TelegramSender(AlertSenderABC):
    """Asynchronous Telegram bot notification dispatcher."""

    def __init__(
        self,
        token: str = "",
        chat_id: str = "",
        max_rate_per_minute: int = 20,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.token = token.strip()
        self.chat_id = chat_id.strip()
        self.max_rate_per_minute = max_rate_per_minute
        self.timeout = timeout_seconds

        # Rate limiting state
        self._sent_timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

        # In-memory history for testing and verification
        self.sent_messages: list[dict[str, Any]] = []

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(self, title: str, message: str, severity: str = "INFO") -> None:
        """Send formatted alert complying with AlertSenderABC interface."""
        emoji = SEVERITY_EMOJIS.get(severity.upper(), "📢")
        formatted_text = (
            f"{emoji} *[{severity.upper()}] {title}*\n\n"
            f"{message}\n\n"
            f"_Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_"
        )
        await self.send_message(formatted_text, parse_mode="Markdown")

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Deliver raw or formatted text message with rate-limiting and retries."""
        async with self._lock:
            await self._enforce_rate_limit()

            record = {
                "text": text,
                "parse_mode": parse_mode,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "configured": self.is_configured,
            }
            self.sent_messages.append(record)

            if not self.is_configured:
                log.info("telegram_alert_mock_recorded", text_preview=text[:80])
                return True

            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }

            for attempt in range(1, 4):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(url, json=payload)
                        if response.status_code == 200:
                            log.info("telegram_alert_sent", attempt=attempt)
                            return True
                        elif response.status_code == 429:
                            retry_after = response.json().get("parameters", {}).get("retry_after", 2)
                            log.warning("telegram_rate_limited", retry_after=retry_after)
                            await asyncio.sleep(retry_after)
                        else:
                            log.error("telegram_send_failed", status=response.status_code, body=response.text)
                except Exception as e:
                    log.warning("telegram_send_exception", attempt=attempt, error=str(e))
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

            log.error("telegram_alert_failed_all_retries")
            return False

    async def _enforce_rate_limit(self) -> None:
        """Sliding-window rate limiter ensuring compliance with Telegram API limits."""
        now = asyncio.get_event_loop().time()
        one_minute_ago = now - 60.0

        while self._sent_timestamps and self._sent_timestamps[0] < one_minute_ago:
            self._sent_timestamps.popleft()

        if len(self._sent_timestamps) >= self.max_rate_per_minute:
            sleep_duration = 60.0 - (now - self._sent_timestamps[0]) + 0.1
            if sleep_duration > 0:
                log.warning("telegram_rate_limiter_throttling", sleep_seconds=sleep_duration)
                await asyncio.sleep(sleep_duration)

        self._sent_timestamps.append(asyncio.get_event_loop().time())
