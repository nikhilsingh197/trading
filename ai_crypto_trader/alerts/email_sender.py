"""Email notification sender.

Delivers formatted HTML and plain text alerts for critical platform events,
daily performance reports, and risk breaches.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from typing import Any, Optional

from ai_crypto_trader.core.interfaces import AlertSenderABC
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)

SEVERITY_COLORS = {
    "INFO": "#0284c7",      # Sky blue
    "WARNING": "#d97706",   # Amber
    "ERROR": "#dc2626",     # Red
    "CRITICAL": "#991b1b",  # Dark red / Crimson
}


class EmailSender(AlertSenderABC):
    """SMTP email notification dispatcher with responsive HTML templates."""

    def __init__(
        self,
        host: str = "smtp.gmail.com",
        port: int = 587,
        user: str = "",
        password: str = "",
        to_email: str = "",
        use_tls: bool = True,
    ) -> None:
        self.host = host.strip()
        self.port = port
        self.user = user.strip()
        self.password = password.strip()
        self.to_email = to_email.strip()
        self.use_tls = use_tls

        # Test/audit tracking
        self.sent_emails: list[dict[str, Any]] = []

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.user and self.password and self.to_email)

    async def send(self, title: str, message: str, severity: str = "INFO") -> None:
        """Deliver an email alert complying with AlertSenderABC."""
        subject = f"[AI Crypto Trader] [{severity.upper()}] {title}"
        html_body = self._generate_html_template(title, message, severity)
        plain_body = f"[{severity.upper()}] {title}\n\n{message}\n\nTimestamp: {datetime.now(timezone.utc).isoformat()}"

        await self.send_email(
            subject=subject,
            plain_text=plain_body,
            html_content=html_body,
        )

    async def send_email(
        self,
        subject: str,
        plain_text: str,
        html_content: Optional[str] = None,
    ) -> bool:
        """Send email via SMTP with thread-pool offload."""
        record = {
            "subject": subject,
            "plain_text": plain_text,
            "html_content": html_content,
            "to": self.to_email,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configured": self.is_configured,
        }
        self.sent_emails.append(record)

        if not self.is_configured:
            log.info("email_alert_mock_recorded", subject=subject)
            return True

        return await asyncio.to_thread(self._send_smtp_sync, subject, plain_text, html_content)

    def _send_smtp_sync(self, subject: str, plain_text: str, html_content: Optional[str]) -> bool:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.user
            msg["To"] = self.to_email
            msg["Date"] = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

            msg.attach(MIMEText(plain_text, "plain", "utf-8"))
            if html_content:
                msg.attach(MIMEText(html_content, "html", "utf-8"))

            with smtplib.SMTP(self.host, self.port, timeout=10.0) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.user, self.password)
                server.sendmail(self.user, [self.to_email], msg.as_string())

            log.info("email_alert_sent", to=self.to_email, subject=subject)
            return True
        except Exception as e:
            log.error("email_send_failed", error=str(e), host=self.host, port=self.port)
            return False

    def _generate_html_template(self, title: str, message: str, severity: str) -> str:
        color = SEVERITY_COLORS.get(severity.upper(), "#4b5563")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        formatted_message = message.replace("\n", "<br>")

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
            .card {{ background-color: #1e293b; border-radius: 12px; border: 1px solid #334155; max-width: 600px; margin: 0 auto; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); }}
            .header {{ background-color: {color}; color: #ffffff; padding: 18px 24px; font-weight: bold; font-size: 16px; letter-spacing: 0.5px; text-transform: uppercase; }}
            .body {{ padding: 24px; line-height: 1.6; font-size: 14px; color: #cbd5e1; }}
            .title {{ font-size: 18px; font-weight: bold; color: #f8fafc; margin-bottom: 12px; }}
            .footer {{ background-color: #0f172a; padding: 14px 24px; font-size: 11px; color: #64748b; border-top: 1px solid #334155; font-family: monospace; }}
          </style>
        </head>
        <body>
          <div class="card">
            <div class="header">SYSTEM ALERT &bull; {severity.upper()}</div>
            <div class="body">
              <div class="title">{title}</div>
              <div>{formatted_message}</div>
            </div>
            <div class="footer">
              AI CRYPTO TRADER &bull; Generated at {ts}
            </div>
          </div>
        </body>
        </html>
        """
