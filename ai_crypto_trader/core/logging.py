"""Structured logging configuration.

Uses structlog for JSON-structured logging. Secrets are
never included in log output by design.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


# Fields that must never appear in logs
_REDACTED_FIELDS = frozenset({
    "api_key", "api_secret", "secret", "password", "token",
    "exchange_api_key", "exchange_api_secret", "alert_telegram_token",
    "alert_email_password", "app_secret_key",
})


def _redact_secrets(
    logger: Any,
    method: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor that redacts sensitive fields."""
    for key in list(event_dict.keys()):
        if key.lower() in _REDACTED_FIELDS:
            event_dict[key] = "***REDACTED***"
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog + stdlib logging for the entire application.

    Args:
        level: Log level string (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        json_output: If True, output JSON lines; otherwise human-readable.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure stdlib root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Silence noisy third-party loggers
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_secrets,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
        )

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)
