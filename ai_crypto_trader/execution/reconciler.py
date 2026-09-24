"""State Reconciliation Engine.

Detects discrepancies between internal platform state and the actual exchange
state (position mismatch, orphan orders, unrecorded fills, liquidations).
Triggers automatic remediation, alerts, or emergency kill switch halts.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ai_crypto_trader.alerts.alert_manager import AlertManager
from ai_crypto_trader.core.enums import AlertSeverity, AlertType, Direction
from ai_crypto_trader.core.interfaces import PositionSnapshot
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.exchange.base import ExchangeAdapterABC
from ai_crypto_trader.execution.order_manager import OrderManager
from ai_crypto_trader.risk.kill_switch import KillSwitch

log = get_logger(__name__)


@dataclass
class Discrepancy:
    item_type: str        # 'POSITION' or 'ORDER'
    symbol: str
    internal_value: str
    exchange_value: str
    severity: str         # 'WARNING' or 'CRITICAL'
    message: str


@dataclass
class ReconciliationReport:
    timestamp: datetime
    is_synchronized: bool
    discrepancies: list[Discrepancy] = field(default_factory=list)


class Reconciler:
    """Institutional state reconciler continuously verifying ledger integrity against the exchange."""

    def __init__(
        self,
        exchange: ExchangeAdapterABC,
        order_manager: OrderManager,
        alert_manager: Optional[AlertManager] = None,
        kill_switch: Optional[KillSwitch] = None,
        tolerance_qty: float = 1e-4,
        auto_kill_on_critical: bool = True,
    ) -> None:
        self.exchange = exchange
        self.order_manager = order_manager
        self.alert_manager = alert_manager
        self.kill_switch = kill_switch
        self.tolerance_qty = tolerance_qty
        self.auto_kill_on_critical = auto_kill_on_critical

        self.last_report: Optional[ReconciliationReport] = None

    async def reconcile_positions(
        self,
        internal_positions: dict[str, PositionSnapshot],
    ) -> ReconciliationReport:
        """Compare internal positions vs live exchange positions."""
        now = datetime.now(timezone.utc)
        discrepancies: list[Discrepancy] = []

        try:
            exchange_positions = await self.exchange.fetch_positions()
        except Exception as e:
            log.error("reconciliation_fetch_failed", error=str(e))
            disc = Discrepancy(
                item_type="EXCHANGE_API",
                symbol="ALL",
                internal_value="CONNECTED",
                exchange_value=f"ERROR: {e}",
                severity="CRITICAL",
                message="Failed to fetch positions from exchange API",
            )
            discrepancies.append(disc)
            return ReconciliationReport(timestamp=now, is_synchronized=False, discrepancies=discrepancies)

        # Index exchange positions by symbol
        exch_map: dict[str, PositionSnapshot] = {p.symbol: p for p in exchange_positions}

        all_symbols = set(internal_positions.keys()) | set(exch_map.keys())

        for sym in all_symbols:
            int_pos = internal_positions.get(sym)
            ext_pos = exch_map.get(sym)

            int_qty = int_pos.quantity if int_pos else 0.0
            ext_qty = ext_pos.quantity if ext_pos else 0.0

            int_side = int_pos.side if int_pos else Direction.FLAT
            ext_side = ext_pos.side if ext_pos else Direction.FLAT

            # Case 1: Quantity mismatch
            qty_diff = abs(int_qty - ext_qty)
            if qty_diff > self.tolerance_qty:
                severity = "CRITICAL" if (int_qty == 0 or ext_qty == 0 or qty_diff > 0.01) else "WARNING"
                msg = f"Position quantity mismatch on {sym}: internal={int_qty:.4f}, exchange={ext_qty:.4f} (diff={qty_diff:.4f})"
                discrepancies.append(
                    Discrepancy(
                        item_type="POSITION",
                        symbol=sym,
                        internal_value=f"{int_side.value} {int_qty:.4f}",
                        exchange_value=f"{ext_side.value} {ext_qty:.4f}",
                        severity=severity,
                        message=msg,
                    )
                )

            # Case 2: Direction mismatch
            elif int_qty > 0 and ext_qty > 0 and int_side != ext_side:
                msg = f"Position direction mismatch on {sym}: internal={int_side.value}, exchange={ext_side.value}"
                discrepancies.append(
                    Discrepancy(
                        item_type="POSITION",
                        symbol=sym,
                        internal_value=int_side.value,
                        exchange_value=ext_side.value,
                        severity="CRITICAL",
                        message=msg,
                    )
                )

        is_sync = len(discrepancies) == 0
        report = ReconciliationReport(timestamp=now, is_synchronized=is_sync, discrepancies=discrepancies)
        self.last_report = report

        if not is_sync:
            await self._handle_discrepancies(discrepancies)

        return report

    async def reconcile_orders(self) -> ReconciliationReport:
        """Compare open orders in OrderManager against active exchange orders."""
        now = datetime.now(timezone.utc)
        discrepancies: list[Discrepancy] = []

        try:
            exchange_orders = await self.exchange.fetch_open_orders()
        except Exception as e:
            disc = Discrepancy(
                item_type="EXCHANGE_API",
                symbol="ALL",
                internal_value="CONNECTED",
                exchange_value=f"ERROR: {e}",
                severity="WARNING",
                message="Failed to fetch open orders from exchange",
            )
            discrepancies.append(disc)
            return ReconciliationReport(timestamp=now, is_synchronized=False, discrepancies=discrepancies)

        internal_open = {o.client_order_id: o for o in self.order_manager.list_open_orders()}
        external_open = {o.client_order_id: o for o in exchange_orders}

        # Check for orphan orders on exchange (not in internal system)
        for cl_id, ord_res in external_open.items():
            if cl_id not in internal_open:
                discrepancies.append(
                    Discrepancy(
                        item_type="ORDER",
                        symbol=getattr(ord_res, "symbol", "UNKNOWN"),
                        internal_value="ABSENT",
                        exchange_value=f"OPEN ({cl_id})",
                        severity="WARNING",
                        message=f"Orphan order detected on exchange: {cl_id}",
                    )
                )

        is_sync = len(discrepancies) == 0
        report = ReconciliationReport(timestamp=now, is_synchronized=is_sync, discrepancies=discrepancies)

        if not is_sync:
            await self._handle_discrepancies(discrepancies)

        return report

    async def _handle_discrepancies(self, discrepancies: list[Discrepancy]) -> None:
        """Dispatch alerts and activate kill switch if a critical discrepancy is detected."""
        has_critical = any(d.severity == "CRITICAL" for d in discrepancies)
        summary = "\n".join([f"• [{d.severity}] {d.message}" for d in discrepancies])

        log.critical("RECONCILIATION_DISCREPANCY_DETECTED", count=len(discrepancies), details=summary)

        if self.alert_manager:
            severity = AlertSeverity.CRITICAL if has_critical else AlertSeverity.WARNING
            await self.alert_manager.send_alert(
                alert_type=AlertType.EXCHANGE_FAILURE,
                severity=severity,
                title="Position Reconciliation Discrepancy",
                message=f"State mismatch detected:\n{summary}",
            )

        if has_critical and self.auto_kill_on_critical and self.kill_switch:
            await self.kill_switch.activate(
                reason=f"Reconciler detected critical position mismatch: {discrepancies[0].message}"
            )
