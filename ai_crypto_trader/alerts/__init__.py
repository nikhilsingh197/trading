"""Alerts and incident notifications package."""
from ai_crypto_trader.alerts.alert_manager import AlertManager, AlertRecord
from ai_crypto_trader.alerts.email_sender import EmailSender
from ai_crypto_trader.alerts.telegram_sender import TelegramSender

__all__ = [
    "AlertManager",
    "AlertRecord",
    "EmailSender",
    "TelegramSender",
]
