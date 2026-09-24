"""Unit tests for alerts, incident management, and performance reporting."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import pytest

from ai_crypto_trader.alerts.alert_manager import AlertManager, AlertRecord
from ai_crypto_trader.alerts.email_sender import EmailSender
from ai_crypto_trader.alerts.telegram_sender import TelegramSender
from ai_crypto_trader.core.enums import AlertSeverity, AlertType
from ai_crypto_trader.reporting.daily_report import DailyReportGenerator
from ai_crypto_trader.reporting.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceSummary,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TelegramSender Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTelegramSender:
    @pytest.mark.asyncio
    async def test_telegram_mock_send(self):
        sender = TelegramSender(token="", chat_id="")
        assert not sender.is_configured
        success = await sender.send_message("Test message")
        assert success is True
        assert len(sender.sent_messages) == 1
        assert "Test message" in sender.sent_messages[0]["text"]

    @pytest.mark.asyncio
    async def test_telegram_send_formatted(self):
        sender = TelegramSender(token="", chat_id="")
        await sender.send(title="Risk Breach", message="Loss > 2%", severity="CRITICAL")
        assert len(sender.sent_messages) == 1
        msg = sender.sent_messages[0]["text"]
        assert "CRITICAL" in msg
        assert "Risk Breach" in msg
        assert "Loss > 2%" in msg

    @pytest.mark.asyncio
    async def test_telegram_rate_limiting(self):
        # Allow only 2 per minute for testing
        sender = TelegramSender(token="", chat_id="", max_rate_per_minute=2)
        await sender.send_message("msg 1")
        await sender.send_message("msg 2")
        assert len(sender.sent_messages) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. EmailSender Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmailSender:
    @pytest.mark.asyncio
    async def test_email_mock_send(self):
        sender = EmailSender(user="", password="", to_email="")
        assert not sender.is_configured
        success = await sender.send_email(subject="Test Alert", plain_text="Body content")
        assert success is True
        assert len(sender.sent_emails) == 1
        assert sender.sent_emails[0]["subject"] == "Test Alert"

    @pytest.mark.asyncio
    async def test_email_send_formatted_html(self):
        sender = EmailSender(user="", password="", to_email="trader@fund.com")
        await sender.send(title="Daily Halt", message="Drawdown triggered", severity="ERROR")
        assert len(sender.sent_emails) == 1
        record = sender.sent_emails[0]
        assert "[ERROR]" in record["subject"]
        assert "html_content" in record
        assert "Daily Halt" in record["html_content"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. AlertManager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertManager:
    @pytest.mark.asyncio
    async def test_alert_manager_dispatch_by_severity(self):
        telegram = TelegramSender()
        email = EmailSender()
        mgr = AlertManager(telegram_sender=telegram, email_sender=email, debounce_seconds=60.0)

        # 1. INFO: should NOT route to telegram or email
        r_info = await mgr.send_alert(
            alert_type=AlertType.TRADE_OPENED,
            severity=AlertSeverity.INFO,
            title="Trade Opened",
            message="BTC 0.05",
        )
        assert "telegram" not in r_info.sent_to
        assert "email" not in r_info.sent_to
        assert len(telegram.sent_messages) == 0

        # 2. WARNING: routes to Telegram only
        r_warn = await mgr.send_alert(
            alert_type=AlertType.DAILY_LOSS_LIMIT,
            severity=AlertSeverity.WARNING,
            title="Loss Warning",
            message="Loss approaching limit",
        )
        assert "telegram" in r_warn.sent_to
        assert "email" not in r_warn.sent_to
        assert len(telegram.sent_messages) == 1

        # 3. ERROR: routes to Telegram AND Email
        r_err = await mgr.send_alert(
            alert_type=AlertType.EXCHANGE_FAILURE,
            severity=AlertSeverity.ERROR,
            title="API Outage",
            message="Order reject 502",
        )
        assert "telegram" in r_err.sent_to
        assert "email" in r_err.sent_to
        assert len(email.sent_emails) == 1

        # 4. CRITICAL: routes to Telegram AND Email
        r_crit = await mgr.send_alert(
            alert_type=AlertType.KILL_SWITCH_ACTIVATED,
            severity=AlertSeverity.CRITICAL,
            title="Kill Switch Active",
            message="System stopped",
        )
        assert "telegram" in r_crit.sent_to
        assert "email" in r_crit.sent_to

    @pytest.mark.asyncio
    async def test_alert_debouncing(self):
        telegram = TelegramSender()
        mgr = AlertManager(telegram_sender=telegram, debounce_seconds=30.0)

        # First alert fires
        r1 = await mgr.send_alert(
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            title="Spike Detected",
            message="Price spike 5%",
        )
        assert len(telegram.sent_messages) == 1

        # Identical alert immediately after -> should be debounced
        r2 = await mgr.send_alert(
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            title="Spike Detected",
            message="Price spike 5%",
        )
        assert len(telegram.sent_messages) == 1
        assert r1.id == r2.id

        # Forced alert bypasses debounce
        r3 = await mgr.send_alert(
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            title="Spike Detected",
            message="Price spike 5%",
            force=True,
        )
        assert len(telegram.sent_messages) == 2
        assert r3.id != r1.id

    @pytest.mark.asyncio
    async def test_acknowledgement_and_filtering(self):
        mgr = AlertManager()
        r1 = await mgr.send_alert(
            alert_type=AlertType.STOP_LOSS,
            severity=AlertSeverity.INFO,
            title="SL Hit",
            message="Closed at stop",
        )
        assert r1.acknowledged is False

        unack = mgr.get_recent_alerts(unacknowledged_only=True)
        assert len(unack) == 1

        success = mgr.acknowledge_alert(r1.id)
        assert success is True

        unack_after = mgr.get_recent_alerts(unacknowledged_only=True)
        assert len(unack_after) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. PerformanceAnalyzer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceAnalyzer:
    def test_empty_trade_analysis(self):
        summary = PerformanceAnalyzer.analyze(trades=[], equity_curve=[10000.0])
        assert summary.total_trades == 0
        assert summary.win_rate == 0.0
        assert summary.profit_factor == 0.0

    def test_complete_trade_analysis(self):
        trades = [
            {"pnl": 150.0, "fees": 2.5, "slippage": 1.0, "pnl_pct": 1.5},
            {"pnl": -50.0, "fees": 2.5, "slippage": 0.5, "pnl_pct": -0.5},
            {"pnl": 200.0, "fees": 3.0, "slippage": 1.2, "pnl_pct": 2.0},
            {"pnl": -80.0, "fees": 2.0, "slippage": 0.8, "pnl_pct": -0.8},
            {"pnl": 100.0, "fees": 2.0, "slippage": 0.5, "pnl_pct": 1.0},
        ]
        equity_curve = [10000.0, 10150.0, 10100.0, 10300.0, 10220.0, 10320.0]

        summary = PerformanceAnalyzer.analyze(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=10000.0,
        )

        assert summary.total_trades == 5
        assert summary.winning_trades == 3
        assert summary.losing_trades == 2
        assert summary.win_rate == 0.60
        assert summary.total_pnl == 320.0
        assert summary.total_return_pct == 3.20
        assert summary.gross_profit == 450.0
        assert summary.gross_loss == 130.0
        assert abs(summary.profit_factor - (450.0 / 130.0)) < 0.05
        assert summary.total_fees == 12.0
        assert summary.max_consecutive_wins == 1
        assert summary.max_drawdown_pct > 0.0
        assert summary.sharpe_ratio > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. DailyReportGenerator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyReportGenerator:
    def test_daily_report_file_generation(self, tmp_path):
        gen = DailyReportGenerator(output_dir=tmp_path)

        sample_trades = [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.02,
                "entry_price": 50000.0,
                "exit_price": 51500.0,
                "pnl": 30.0,
                "pnl_pct": 3.0,
                "fees": 1.5,
                "slippage": 0.5,
                "exit_reason": "TAKE_PROFIT",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        equity_curve = [10000.0, 10030.0]

        report = gen.generate_report(
            report_date=date(2026, 9, 24),
            initial_capital=10000.0,
            trades=sample_trades,
            equity_curve=equity_curve,
        )

        md_path = Path(report["md_path"])
        html_path = Path(report["html_path"])
        json_path = Path(report["json_path"])

        assert md_path.exists()
        assert html_path.exists()
        assert json_path.exists()

        md_text = md_path.read_text(encoding="utf-8")
        assert "Institutional Daily Trading Report" in md_text
        assert "BTCUSDT" in md_text
        assert "+$30.00" in md_text

        html_text = html_path.read_text(encoding="utf-8")
        assert "Daily Performance Report" in html_text
        assert "BTCUSDT" in html_text
        assert "$10,030.00" in html_text

    @pytest.mark.asyncio
    async def test_dispatch_summary(self, tmp_path):
        telegram = TelegramSender()
        email = EmailSender()
        mgr = AlertManager(telegram_sender=telegram, email_sender=email)
        gen = DailyReportGenerator(output_dir=tmp_path, alert_manager=mgr)

        report = gen.generate_report(
            report_date=date(2026, 9, 24),
            initial_capital=10000.0,
            trades=[{"pnl": 50.0, "fees": 1.0, "slippage": 0.0}],
            equity_curve=[10000.0, 10050.0],
        )

        await gen.dispatch_summary(report)
        assert len(telegram.sent_messages) == 0  # INFO does not send to telegram by default
        assert len(mgr._history) == 1
        assert "Daily Performance Report" in mgr._history[0].title
