"""Automated Daily Performance Report Generator.

Compiles daily trading activities, risk events, execution quality, and equity curves
into styled HTML dashboards, Markdown summaries, and JSON artifacts.
Dispatches end-of-day reports to operator channels.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from ai_crypto_trader.alerts.alert_manager import AlertManager
from ai_crypto_trader.config.settings import get_settings
from ai_crypto_trader.core.enums import AlertSeverity, AlertType
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.reporting.performance_analyzer import PerformanceAnalyzer, PerformanceSummary

log = get_logger(__name__)


class DailyReportGenerator:
    """Generates institutional end-of-day reports and delivers them to operators."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        settings = get_settings()
        self.output_dir = output_dir or settings.report_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alert_manager = alert_manager

    def generate_report(
        self,
        report_date: Optional[date] = None,
        initial_capital: float = 10000.0,
        trades: Sequence[dict[str, Any]] = (),
        equity_curve: Sequence[float] = (),
        system_events: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Compile complete report data, generate Markdown + HTML, and persist artifacts."""
        target_date = report_date or datetime.now(timezone.utc).date()
        date_str = target_date.isoformat()

        # Compute statistical summary
        metrics: PerformanceSummary = PerformanceAnalyzer.analyze(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
        )

        report_data = {
            "report_date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "initial_capital": initial_capital,
            "current_equity": equity_curve[-1] if equity_curve else (initial_capital + metrics.total_pnl),
            "metrics": asdict(metrics),
            "trade_count": len(trades),
            "trades": list(trades),
            "event_count": len(system_events),
            "events": list(system_events),
        }

        # 1. Render Markdown
        markdown_content = self.render_markdown(report_data)
        md_file = self.output_dir / f"daily_report_{date_str}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # 2. Render HTML
        html_content = self.render_html(report_data)
        html_file = self.output_dir / f"daily_report_{date_str}.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 3. Render JSON
        json_file = self.output_dir / f"daily_report_{date_str}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        log.info(
            "daily_report_generated",
            date=date_str,
            html_file=str(html_file),
            md_file=str(md_file),
        )

        report_data["html_path"] = str(html_file)
        report_data["md_path"] = str(md_file)
        report_data["json_path"] = str(json_file)

        return report_data

    async def dispatch_summary(self, report_data: dict[str, Any]) -> None:
        """Send daily report summary via Telegram and Email using AlertManager."""
        if not self.alert_manager:
            log.warning("alert_manager_not_configured_for_reporting")
            return

        date_str = report_data["report_date"]
        m = report_data["metrics"]
        pnl = m["total_pnl"]
        pnl_sign = "+" if pnl >= 0 else ""
        ret_pct = m["total_return_pct"]

        title = f"Daily Performance Report — {date_str}"
        summary_msg = (
            f"📊 *Summary for {date_str}*\n"
            f"• Net P&L: *{pnl_sign}${pnl:,.2f}* ({pnl_sign}{ret_pct:.2f}%)\n"
            f"• Portfolio Equity: *${report_data['current_equity']:,.2f}*\n"
            f"• Total Trades: *{m['total_trades']}* ({m['winning_trades']}W / {m['losing_trades']}L / {m['break_even_trades']}BE)\n"
            f"• Win Rate: *{m['win_rate'] * 100:.1f}%* | Profit Factor: *{m['profit_factor']:.2f}*\n"
            f"• Sharpe Ratio: *{m['sharpe_ratio']:.2f}* | Max DD: *{m['max_drawdown_pct']:.2f}%*\n"
            f"• Total Fees Paid: *${m['total_fees']:,.2f}*\n\n"
            f"Full HTML/MD report archived at `{report_data.get('html_path')}`"
        )

        severity = AlertSeverity.INFO if pnl >= 0 else AlertSeverity.WARNING
        await self.alert_manager.send_alert(
            alert_type=AlertType.STRATEGY_CANDIDATE,
            severity=severity,
            title=title,
            message=summary_msg,
            metadata={"date": date_str, "pnl": pnl, "equity": report_data["current_equity"]},
            force=True,
        )

    def render_markdown(self, data: dict[str, Any]) -> str:
        """Render high-level markdown summary."""
        m = data["metrics"]
        pnl_sign = "+" if m["total_pnl"] >= 0 else ""

        lines = [
            f"# Institutional Daily Trading Report — {data['report_date']}",
            f"*Generated at: {data['generated_at']}*",
            "",
            "## 1. Executive Summary",
            f"- **Starting Capital**: ${data['initial_capital']:,.2f}",
            f"- **Closing Equity**: ${data['current_equity']:,.2f}",
            f"- **Net P&L**: {pnl_sign}${m['total_pnl']:,.2f} ({pnl_sign}{m['total_return_pct']:.2f}%)",
            f"- **Max Drawdown**: {m['max_drawdown_pct']:.2f}% (${m['max_drawdown_dollars']:,.2f})",
            f"- **Sharpe Ratio**: {m['sharpe_ratio']:.2f} | **Sortino Ratio**: {m['sortino_ratio']:.2f}",
            f"- **Profit Factor**: {m['profit_factor']:.2f} | **Calmar Ratio**: {m['calmar_ratio']:.2f}",
            "",
            "## 2. Trade Execution Statistics",
            f"- **Total Completed Trades**: {m['total_trades']}",
            f"- **Win Rate**: {m['win_rate'] * 100:.1f}% ({m['winning_trades']}W - {m['losing_trades']}L - {m['break_even_trades']}BE)",
            f"- **Average Trade P&L**: ${m['avg_trade_pnl']:,.2f}",
            f"- **Average Win**: ${m['avg_win']:,.2f} | **Average Loss**: ${m['avg_loss']:,.2f}",
            f"- **Win/Loss Ratio**: {m['win_loss_ratio']:.2f} | **Trade Expectancy**: ${m['expectancy']:,.2f}",
            f"- **Max Consecutive Wins**: {m['max_consecutive_wins']} | **Losses**: {m['max_consecutive_losses']}",
            f"- **Fees Paid**: ${m['total_fees']:,.2f} | **Slippage Incurred**: ${m['total_slippage']:,.2f}",
            "",
            "## 3. Completed Trades Log",
            "| Time | Symbol | Side | Qty | Entry | Exit | PnL ($) | Return (%) | Reason |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        if not data["trades"]:
            lines.append("| — | No trades executed during this session | — | — | — | — | — | — | — |")
        else:
            for t in data["trades"][:50]:
                p = t.get("pnl", 0.0)
                sign = "+" if p >= 0 else ""
                lines.append(
                    f"| {t.get('closed_at', t.get('timestamp', '—'))[:19]} | "
                    f"{t.get('symbol', '—')} | {t.get('side', '—')} | {t.get('quantity', 0.0):.4f} | "
                    f"${t.get('entry_price', 0.0):,.2f} | ${t.get('exit_price', 0.0):,.2f} | "
                    f"{sign}${p:,.2f} | {sign}{t.get('pnl_pct', 0.0):.2f}% | {t.get('exit_reason', '—')} |"
                )

        return "\n".join(lines)

    def render_html(self, data: dict[str, Any]) -> str:
        """Render beautiful standalone HTML dashboard report."""
        m = data["metrics"]
        pnl = m["total_pnl"]
        pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
        pnl_sign = "+" if pnl >= 0 else ""

        # Trade rows
        trade_rows_html = ""
        if not data["trades"]:
            trade_rows_html = '<tr><td colspan="9" style="text-align: center; padding: 24px; color: #64748b;">No trades executed during this session</td></tr>'
        else:
            for t in data["trades"][:100]:
                t_pnl = t.get("pnl", 0.0)
                c = "#10b981" if t_pnl >= 0 else "#ef4444"
                s = "+" if t_pnl >= 0 else ""
                trade_rows_html += f"""
                <tr style="border-bottom: 1px solid #334155;">
                  <td style="padding: 10px; font-family: monospace;">{t.get('closed_at', t.get('timestamp', '—'))[:19]}</td>
                  <td style="padding: 10px; font-weight: bold;">{t.get('symbol', '—')}</td>
                  <td style="padding: 10px;"><span style="padding: 2px 6px; border-radius: 4px; background: rgba(56, 189, 248, 0.1); color: #38bdf8;">{t.get('side', '—')}</span></td>
                  <td style="padding: 10px; font-family: monospace;">{t.get('quantity', 0.0):.4f}</td>
                  <td style="padding: 10px; font-family: monospace;">${t.get('entry_price', 0.0):,.2f}</td>
                  <td style="padding: 10px; font-family: monospace;">${t.get('exit_price', 0.0):,.2f}</td>
                  <td style="padding: 10px; font-family: monospace; font-weight: bold; color: {c};">{s}${t_pnl:,.2f}</td>
                  <td style="padding: 10px; font-family: monospace; color: {c};">{s}{t.get('pnl_pct', 0.0):.2f}%</td>
                  <td style="padding: 10px; font-size: 12px; color: #94a3b8;">{t.get('exit_reason', '—')}</td>
                </tr>
                """

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>Daily Trading Report — {data['report_date']}</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #090d16; color: #f8fafc; margin: 0; padding: 30px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 20px; margin-bottom: 24px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 30px; }}
            .card {{ background-color: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }}
            .card-title {{ font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px; letter-spacing: 0.5px; }}
            .card-value {{ font-size: 20px; font-weight: bold; font-family: monospace; }}
            .section-title {{ font-size: 16px; font-weight: bold; margin: 28px 0 14px 0; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
            table {{ width: 100%; border-collapse: collapse; background-color: #0f172a; border-radius: 10px; overflow: hidden; border: 1px solid #1e293b; font-size: 13px; }}
            th {{ background-color: #1e293b; color: #94a3b8; text-align: left; padding: 12px 10px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <div>
                <h1 style="margin: 0; font-size: 24px; color: #f8fafc;">Daily Performance Report</h1>
                <p style="margin: 4px 0 0 0; font-size: 12px; color: #64748b;">AI CRYPTO TRADER &bull; Session Date: {data['report_date']}</p>
              </div>
              <div style="text-align: right;">
                <div style="font-size: 12px; color: #64748b;">Generated At</div>
                <div style="font-family: monospace; font-size: 13px; color: #38bdf8;">{data['generated_at'][:19]} UTC</div>
              </div>
            </div>

            <!-- KEY METRICS GRID -->
            <div class="grid">
              <div class="card">
                <div class="card-title">Portfolio Equity</div>
                <div class="card-value">${data['current_equity']:,.2f}</div>
              </div>
              <div class="card">
                <div class="card-title">Session Net P&L</div>
                <div class="card-value" style="color: {pnl_color};">{pnl_sign}${pnl:,.2f}</div>
              </div>
              <div class="card">
                <div class="card-title">Total Return</div>
                <div class="card-value" style="color: {pnl_color};">{pnl_sign}{m['total_return_pct']:.2f}%</div>
              </div>
              <div class="card">
                <div class="card-title">Win Rate</div>
                <div class="card-value">{m['win_rate'] * 100:.1f}%</div>
              </div>
              <div class="card">
                <div class="card-title">Profit Factor</div>
                <div class="card-value">{m['profit_factor']:.2f}</div>
              </div>
              <div class="card">
                <div class="card-title">Sharpe Ratio</div>
                <div class="card-value">{m['sharpe_ratio']:.2f}</div>
              </div>
              <div class="card">
                <div class="card-title">Max Drawdown</div>
                <div class="card-value" style="color: #f59e0b;">{m['max_drawdown_pct']:.2f}%</div>
              </div>
              <div class="card">
                <div class="card-title">Total Fees</div>
                <div class="card-value">${m['total_fees']:,.2f}</div>
              </div>
            </div>

            <!-- TRADE LOG TABLE -->
            <div class="section-title">Trade Executions ({data['trade_count']})</div>
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Quantity</th>
                  <th>Entry Price</th>
                  <th>Exit Price</th>
                  <th>Net P&L</th>
                  <th>Return</th>
                  <th>Exit Reason</th>
                </tr>
              </thead>
              <tbody>
                {trade_rows_html}
              </tbody>
            </table>
          </div>
        </body>
        </html>
        """
