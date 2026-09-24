"""Institutional Execution Engine.

The central execution orchestrator executing signals vetted by the RiskEngine,
managing order submission, tracking fills, managing brackets, and handling emergency liquidations.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from ai_crypto_trader.alerts.alert_manager import AlertManager
from ai_crypto_trader.core.enums import AlertSeverity, AlertType, Direction, OrderStatus, OrderType, Side
from ai_crypto_trader.core.interfaces import OrderRequest, OrderResult, PortfolioState, PositionSnapshot, Signal
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.exchange.base import ExchangeAdapterABC
from ai_crypto_trader.execution.order_manager import OrderManager
from ai_crypto_trader.execution.reconciler import Reconciler
from ai_crypto_trader.risk.kill_switch import KillSwitch
from ai_crypto_trader.risk.risk_engine import RiskEngine

log = get_logger(__name__)


class ExecutionEngine:
    """Unified execution pipeline coordinating Risk, Order Management, Exchange Adapters, and Emergency Liquidation."""

    def __init__(
        self,
        exchange: ExchangeAdapterABC,
        risk_engine: RiskEngine,
        order_manager: Optional[OrderManager] = None,
        reconciler: Optional[Reconciler] = None,
        alert_manager: Optional[AlertManager] = None,
        kill_switch: Optional[KillSwitch] = None,
    ) -> None:
        self.exchange = exchange
        self.risk_engine = risk_engine
        self.order_manager = order_manager or OrderManager(exchange)
        self.alert_manager = alert_manager
        self.kill_switch = kill_switch or risk_engine.kill_switch
        self.reconciler = reconciler or Reconciler(
            exchange=exchange,
            order_manager=self.order_manager,
            alert_manager=self.alert_manager,
            kill_switch=self.kill_switch,
        )

        # Register emergency liquidation with the kill switch
        if self.kill_switch:
            self.kill_switch.register_liquidation_callback(self.emergency_flatten_all)

        # In-memory tracking of active positions managed by this engine
        self._active_positions: dict[str, PositionSnapshot] = {}

    @property
    def active_positions(self) -> dict[str, PositionSnapshot]:
        return self._active_positions

    async def execute_signal(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        use_bracket: bool = True,
    ) -> tuple[bool, Optional[OrderResult], str]:
        """Verify signal with RiskEngine and submit approved orders to the exchange."""
        # 1. Kill Switch Check
        if self.kill_switch and self.kill_switch.is_engaged:
            return False, None, "Kill switch is active; order execution blocked"

        # 2. Risk Gatekeeper Evaluation
        decision = self.risk_engine.evaluate_signal(signal, portfolio)
        if not decision.approved:
            log.warning("signal_rejected_by_risk", symbol=signal.symbol, reason=decision.reason)
            return False, None, f"Risk rejected: {decision.reason}"

        # 3. Construct Order Request
        order_side = Side.BUY if signal.direction == Direction.LONG else Side.SELL
        entry_request = OrderRequest(
            symbol=signal.symbol,
            side=order_side,
            order_type=OrderType.MARKET,
            quantity=decision.quantity,
            price=signal.entry_price,
            strategy_id=signal.strategy_version_id,
        )

        # 4. Submit Order (Bracket or Plain Market)
        if use_bracket and signal.stop_loss and signal.take_profit:
            entry_res, sl_res, tp_res = await self.order_manager.submit_bracket(
                entry_request=entry_request,
                stop_loss_price=signal.stop_loss,
                take_profit_price=signal.take_profit,
            )
            order_res = entry_res
        else:
            order_res = await self.order_manager.submit_order(entry_request)

        if order_res.status not in (OrderStatus.FILLED, OrderStatus.SUBMITTED):
            return False, order_res, f"Order status {order_res.status.value}: {order_res.error}"

        # 5. Record Opened Position in Risk State
        fill_price = order_res.avg_fill_price or signal.entry_price
        position_value = order_res.filled_qty * fill_price
        self.risk_engine.record_trade_opened(signal.symbol, position_value)

        # 6. Update internal active positions
        self._active_positions[signal.symbol] = PositionSnapshot(
            symbol=signal.symbol,
            side=signal.direction,
            entry_price=fill_price,
            quantity=order_res.filled_qty,
            current_price=fill_price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        # 7. Dispatch Alert
        if self.alert_manager:
            await self.alert_manager.send_alert(
                alert_type=AlertType.TRADE_OPENED,
                severity=AlertSeverity.INFO,
                title=f"Trade Opened: {signal.symbol} {signal.direction.value}",
                message=(
                    f"Executed {order_side.value} {order_res.filled_qty:.4f} {signal.symbol} @ ${fill_price:,.2f}\n"
                    f"Stop Loss: ${signal.stop_loss or 0:,.2f} | Take Profit: ${signal.take_profit or 0:,.2f}\n"
                    f"Strategy: {signal.strategy_version_id}"
                ),
                metadata={"symbol": signal.symbol, "qty": order_res.filled_qty, "price": fill_price},
            )

        log.info(
            "trade_executed_successfully",
            symbol=signal.symbol,
            direction=signal.direction.value,
            qty=order_res.filled_qty,
            price=fill_price,
        )
        return True, order_res, f"Executed {signal.direction.value} {order_res.filled_qty:.4f} {signal.symbol}"

    async def close_position(
        self,
        symbol: str,
        reason: str = "MANUAL",
    ) -> Optional[OrderResult]:
        """Close an active position via market order and reconcile state."""
        if symbol not in self._active_positions:
            log.warning("close_position_not_found", symbol=symbol)
            return None

        pos = self._active_positions[symbol]
        close_side = Side.SELL if pos.side == Direction.LONG else Side.BUY

        close_req = OrderRequest(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
        )

        res = await self.order_manager.submit_order(close_req)
        exit_price = res.avg_fill_price or pos.current_price

        # Calculate PnL
        if pos.side == Direction.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity - res.fees
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity - res.fees

        pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100.0 if pos.entry_price > 0 else 0.0

        # Update Risk Engine state
        self.risk_engine.record_trade_closed(symbol, pos.quantity * pos.entry_price, pnl_pct)

        # Remove from internal tracker
        del self._active_positions[symbol]

        # Dispatch alert
        if self.alert_manager:
            severity = AlertSeverity.INFO if pnl >= 0 else AlertSeverity.WARNING
            pnl_sign = "+" if pnl >= 0 else ""
            await self.alert_manager.send_alert(
                alert_type=AlertType.TRADE_CLOSED,
                severity=severity,
                title=f"Trade Closed: {symbol} ({reason})",
                message=(
                    f"Closed {pos.side.value} {pos.quantity:.4f} {symbol}\n"
                    f"Entry: ${pos.entry_price:,.2f} -> Exit: ${exit_price:,.2f}\n"
                    f"Net P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                    f"Reason: {reason}"
                ),
                metadata={"symbol": symbol, "pnl": pnl, "pnl_pct": pnl_pct},
            )

        log.info("position_closed", symbol=symbol, pnl=pnl, exit_price=exit_price, reason=reason)
        return res

    async def emergency_flatten_all(self, reason: str = "Emergency liquidation") -> dict[str, Any]:
        """Cancel all open orders and market-close all active positions across the exchange."""
        log.critical("EMERGENCY_FLATTEN_ALL_TRIGGERED", reason=reason)

        # 1. Cancel all open orders
        cancelled_orders = await self.order_manager.cancel_all()

        # 2. Market-close all active positions
        positions_to_close = list(self._active_positions.keys())
        closed_positions = []

        for sym in positions_to_close:
            try:
                res = await self.close_position(sym, reason=f"KILL_SWITCH: {reason}")
                if res:
                    closed_positions.append(sym)
            except Exception as e:
                log.error("emergency_close_failed", symbol=sym, error=str(e))

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "orders_cancelled": cancelled_orders,
            "positions_closed": closed_positions,
        }

        if self.alert_manager:
            await self.alert_manager.send_alert(
                alert_type=AlertType.KILL_SWITCH_ACTIVATED,
                severity=AlertSeverity.CRITICAL,
                title="EMERGENCY FLATTEN COMPLETE",
                message=(
                    f"Emergency liquidation executed.\n"
                    f"Reason: {reason}\n"
                    f"Orders Cancelled: {cancelled_orders}\n"
                    f"Positions Closed: {', '.join(closed_positions) if closed_positions else 'None'}"
                ),
                metadata=summary,
                force=True,
            )

        return summary
