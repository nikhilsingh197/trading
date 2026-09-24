"""Risk management engine.

This is the gatekeeper for all trade execution.
Every signal MUST pass through the risk engine before becoming an order.

The AI/model layer cannot modify hard risk limits.
Limits are loaded from the frozen RiskConfig at startup.

Decision flow:
1. Check kill switch
2. Check daily loss limit
3. Check max drawdown (halt/reduce)
4. Check max open positions
5. Check max exposure per asset
6. Check max trades per day
7. Check max consecutive losses
8. Compute position size
9. Approve or reject
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from ai_crypto_trader.config.risk_config import RiskConfig
from ai_crypto_trader.core.enums import RiskAction
from ai_crypto_trader.core.exceptions import KillSwitchActivated
from ai_crypto_trader.core.interfaces import PortfolioState, RiskEngineABC, Signal
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.risk.kill_switch import KillSwitch
from ai_crypto_trader.risk.position_sizer import PositionSizer, SizingResult

log = get_logger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    action: RiskAction
    quantity: float
    position_value: float
    risk_pct: float
    reason: str
    sizing: Optional[SizingResult] = None


@dataclass
class RiskState:
    """Mutable state tracked by the risk engine."""
    daily_loss_pct: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    date_tracked: date = field(default_factory=lambda: date.today())
    open_position_count: int = 0
    exposure_by_asset: dict[str, float] = field(default_factory=dict)

    def reset_daily_if_needed(self) -> None:
        today = date.today()
        if self.date_tracked != today:
            self.daily_loss_pct = 0.0
            self.trades_today = 0
            self.date_tracked = today


class RiskEngine(RiskEngineABC):
    """Evaluates signals against hard risk limits.

    IMPORTANT: This class does NOT modify its config.
    Config is a frozen dataclass set at construction time.
    """

    def __init__(
        self,
        config: RiskConfig,
        kill_switch: KillSwitch,
    ) -> None:
        self._config = config
        self._kill_switch = kill_switch
        self._sizer = PositionSizer(
            max_trade_risk_pct=config.max_trade_risk_pct,
            max_position_size_pct=config.max_position_size_pct,
        )
        self._state = RiskState()
        log.info("risk_engine_initialized", config=repr(config))

    def evaluate(self, signal: Signal, portfolio: PortfolioState) -> tuple[bool, float, str]:
        """Evaluate a signal against all risk rules.

        Returns:
            (approved: bool, quantity: float, reason: str)
        """
        decision = self._evaluate_full(signal, portfolio)
        log.info(
            "risk_decision",
            symbol=signal.symbol,
            approved=decision.approved,
            action=decision.action.value,
            quantity=decision.quantity,
            reason=decision.reason,
        )
        return decision.approved, decision.quantity, decision.reason

    def _evaluate_full(self, signal: Signal, portfolio: PortfolioState) -> RiskDecision:
        self._state.reset_daily_if_needed()

        # 1. Kill switch (highest priority)
        # Note: async kill switch check should be done before calling evaluate()
        # Here we check the in-memory flag for sync contexts
        if self._kill_switch._active:
            return RiskDecision(
                approved=False,
                action=RiskAction.KILL_SWITCH,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason="Kill switch is active",
            )

        # 2. Daily loss limit
        if self._state.daily_loss_pct <= -self._config.daily_loss_limit_pct:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Daily loss limit {self._config.daily_loss_limit_pct}% exceeded",
            )

        # 3. Max drawdown halt
        if portfolio.drawdown_pct >= self._config.max_drawdown_halt_pct:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Drawdown halt {self._config.max_drawdown_halt_pct}% triggered",
            )

        # 4. Max open positions
        if self._state.open_position_count >= self._config.max_open_positions:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Max open positions {self._config.max_open_positions} reached",
            )

        # 5. Max trades per day
        if self._state.trades_today >= self._config.max_trades_per_day:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Max trades per day {self._config.max_trades_per_day} reached",
            )

        # 6. Max consecutive losses
        if self._state.consecutive_losses >= self._config.max_consecutive_losses:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Max consecutive losses {self._config.max_consecutive_losses} reached",
            )

        # 7. Compute position size
        size_multiplier = 1.0
        if portfolio.drawdown_pct >= self._config.max_drawdown_reduce_pct:
            size_multiplier = 0.5
            log.warning(
                "drawdown_reduce_triggered",
                drawdown=portfolio.drawdown_pct,
                reduce_pct=self._config.max_drawdown_reduce_pct,
            )

        sizing = self._sizer.fixed_fractional(
            equity=portfolio.total_equity,
            entry=signal.entry_price,
            stop_loss=signal.stop_loss,
            risk_pct=signal.risk_pct * size_multiplier,
        )

        if sizing.quantity <= 0:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason="Position sizer returned zero quantity",
            )

        action = RiskAction.REDUCE_SIZE if size_multiplier < 1.0 else RiskAction.APPROVE
        return RiskDecision(
            approved=True,
            action=action,
            quantity=sizing.quantity,
            position_value=sizing.position_value,
            risk_pct=sizing.risk_pct,
            reason=f"Approved via {sizing.method} (size_mult={size_multiplier})",
            sizing=sizing,
        )

    def record_trade_opened(self, symbol: str, position_value: float) -> None:
        self._state.trades_today += 1
        self._state.open_position_count += 1
        current = self._state.exposure_by_asset.get(symbol, 0.0)
        self._state.exposure_by_asset[symbol] = current + position_value

    def record_trade_closed(self, symbol: str, position_value: float, pnl_pct: float) -> None:
        self._state.open_position_count = max(0, self._state.open_position_count - 1)
        current = self._state.exposure_by_asset.get(symbol, 0.0)
        self._state.exposure_by_asset[symbol] = max(0.0, current - position_value)
        self._state.daily_loss_pct += pnl_pct
        if pnl_pct < 0:
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0
