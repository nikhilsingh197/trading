"""Institutional Risk Management Engine.

This is the gatekeeper for all trade execution.
Every trading signal MUST pass through the risk engine before becoming an order.

The AI/model layer CANNOT modify hard risk limits at runtime.
Limits are defined in the immutable frozen RiskConfig at startup.

Comprehensive Decision Flow:
1. Kill switch verification (hard stop)
2. Multi-tier circuit breaker status check (TRIPPED, PAUSED, FLASH_CRASH_HALT)
3. Daily loss limit check
4. Max drawdown halt check
5. Max open positions check
6. Max trades per day check
7. Max consecutive losses check
8. Portfolio leverage limit check
9. Asset exposure cap check
10. Correlated crypto exposure cap check
11. Market regime and volatility scaling
12. Multi-algorithm position sizing (Fixed Fractional, Volatility Parity, Kelly)
13. Apply circuit breaker & drawdown size dampeners
14. Cash and margin feasibility verification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional, Sequence

from ai_crypto_trader.config.risk_config import RiskConfig
from ai_crypto_trader.core.enums import Direction, MarketRegime, RiskAction, SignalAction
from ai_crypto_trader.core.interfaces import PortfolioState, PositionSnapshot, RiskEngineABC, Signal
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
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
    """Internal mutable state tracked by the risk engine across the trading session."""
    daily_loss_pct: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    date_tracked: date = field(default_factory=lambda: date.today())
    open_position_count: int = 0
    exposure_by_asset: dict[str, float] = field(default_factory=dict)

    def reset_daily_if_needed(self, now: Optional[datetime] = None) -> None:
        current_date = (now or datetime.now(timezone.utc)).date() if isinstance(now, datetime) else date.today()
        if self.date_tracked != current_date:
            log.info("risk_state_daily_reset", previous_date=str(self.date_tracked), new_date=str(current_date))
            self.daily_loss_pct = 0.0
            self.trades_today = 0
            self.date_tracked = current_date


class RiskEngine(RiskEngineABC):
    """Institutional risk engine enforcing absolute portfolio safeguards."""

    def __init__(
        self,
        config: RiskConfig,
        kill_switch: Optional[KillSwitch] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self._config = config
        self._kill_switch = kill_switch or KillSwitch()
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            CircuitBreakerConfig(
                level1_daily_loss_pct=config.daily_loss_limit_pct * 0.5,
                level2_daily_loss_pct=config.daily_loss_limit_pct * 0.75,
                level3_daily_loss_pct=config.daily_loss_limit_pct,
                level1_drawdown_pct=config.max_drawdown_reduce_pct,
                level2_drawdown_pct=(config.max_drawdown_reduce_pct + config.max_drawdown_halt_pct) / 2.0,
                level3_drawdown_pct=config.max_drawdown_halt_pct,
                max_consecutive_losses=config.max_consecutive_losses,
            )
        )
        self._sizer = PositionSizer(
            max_trade_risk_pct=config.max_trade_risk_pct,
            max_position_size_pct=config.max_position_size_pct,
            max_leverage=config.max_leverage,
        )
        self._state = RiskState()
        log.info("risk_engine_initialized", config=repr(config))

    @property
    def config(self) -> RiskConfig:
        return self._config

    @property
    def kill_switch(self) -> KillSwitch:
        return self._kill_switch

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    @property
    def state(self) -> RiskState:
        return self._state

    def evaluate(self, signal: Signal, portfolio: PortfolioState) -> tuple[bool, float, str]:
        """Evaluate a signal against all risk rules.

        Returns:
            (approved: bool, quantity: float, reason: str)
        """
        decision = self.evaluate_signal(signal, portfolio)
        log.info(
            "risk_decision",
            symbol=signal.symbol,
            approved=decision.approved,
            action=decision.action.value,
            quantity=decision.quantity,
            reason=decision.reason,
        )
        return decision.approved, decision.quantity, decision.reason

    def evaluate_signal(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        sizing_method: str = "fixed_fractional",
        atr: Optional[float] = None,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> RiskDecision:
        """Full institutional evaluation producing a detailed RiskDecision object."""
        self._state.reset_daily_if_needed()

        # Update circuit breaker with current live metrics
        self._circuit_breaker.check_metrics(
            daily_loss_pct=self._state.daily_loss_pct,
            drawdown_pct=portfolio.drawdown_pct,
            consecutive_losses=self._state.consecutive_losses,
        )

        # 1. Kill Switch Check (Highest Priority)
        if self._kill_switch.is_engaged:
            return RiskDecision(
                approved=False,
                action=RiskAction.KILL_SWITCH,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Kill switch is active ({self._kill_switch._reason})",
            )

        # 2. Hard Daily Loss Limit
        if self._state.daily_loss_pct <= -self._config.daily_loss_limit_pct:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Daily loss limit {self._config.daily_loss_limit_pct}% exceeded (current {self._state.daily_loss_pct:.2f}%)",
            )

        # 3. Hard Max Drawdown Halt
        if portfolio.drawdown_pct >= self._config.max_drawdown_halt_pct:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Max drawdown halt {self._config.max_drawdown_halt_pct}% triggered (current {portfolio.drawdown_pct:.2f}%)",
            )

        # 4. Circuit Breaker Check (Covers flash crash, intraday pauses, cool-offs)
        can_trade, breaker_reason = self._circuit_breaker.can_trade()
        if not can_trade:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=breaker_reason,
            )

        # 5. Open Position Count
        active_positions = len(portfolio.open_positions) if portfolio.open_positions else self._state.open_position_count
        if active_positions >= self._config.max_open_positions:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Max open positions limit ({self._config.max_open_positions}) reached",
            )

        # 6. Max Trades Per Day
        if self._state.trades_today >= self._config.max_trades_per_day:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Max trades per day ({self._config.max_trades_per_day}) reached",
            )

        # 7. Max Consecutive Losses
        if self._state.consecutive_losses >= self._config.max_consecutive_losses:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Max consecutive losses ({self._config.max_consecutive_losses}) reached",
            )

        # 8. Sizing Multipliers (Defensive Reductions)
        size_multiplier = 1.0

        # Drawdown reduction rule
        if portfolio.drawdown_pct >= self._config.max_drawdown_reduce_pct:
            size_multiplier *= 0.5
            log.warning(
                "drawdown_size_reduction_applied",
                drawdown=portfolio.drawdown_pct,
                threshold=self._config.max_drawdown_reduce_pct,
            )

        # Circuit breaker multiplier (e.g. 0.5 if THROTTLED)
        breaker_multiplier = self._circuit_breaker.get_size_multiplier()
        size_multiplier *= breaker_multiplier

        # Regime & Market Volatility scaling
        regime_mult = self._get_regime_multiplier(signal.regime, signal.direction)
        size_multiplier *= regime_mult

        if size_multiplier <= 0:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason="Sizing multiplier throttled to 0 by defensive controls",
            )

        # 9. Compute Base Position Size
        target_risk_pct = signal.risk_pct * size_multiplier
        if sizing_method == "volatility_parity" and atr and atr > 0:
            sizing = self._sizer.volatility_parity(
                equity=portfolio.total_equity,
                entry=signal.entry_price,
                atr=atr,
                risk_pct=target_risk_pct,
            )
        elif sizing_method == "kelly" and win_rate and win_loss_ratio:
            sizing = self._sizer.kelly(
                equity=portfolio.total_equity,
                entry=signal.entry_price,
                stop_loss=signal.stop_loss,
                win_rate=win_rate,
                avg_win_loss_ratio=win_loss_ratio,
            )
        else:
            sizing = self._sizer.fixed_fractional(
                equity=portfolio.total_equity,
                entry=signal.entry_price,
                stop_loss=signal.stop_loss,
                risk_pct=target_risk_pct,
            )

        if sizing.quantity <= 0:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Position sizer returned zero quantity via {sizing.method}",
            )

        # 10. Check Asset Exposure Cap
        current_asset_exposure = self._state.exposure_by_asset.get(signal.symbol, 0.0)
        max_allowed_asset_value = portfolio.total_equity * (self._config.max_position_size_pct / 100.0)
        if (current_asset_exposure + sizing.position_value) > max_allowed_asset_value * 1.01:
            # Scale down to allowable remaining room
            allowed_room = max(0.0, max_allowed_asset_value - current_asset_exposure)
            if allowed_room <= 0:
                return RiskDecision(
                    approved=False,
                    action=RiskAction.REJECT,
                    quantity=0.0,
                    position_value=0.0,
                    risk_pct=0.0,
                    reason=f"Max asset exposure {self._config.max_position_size_pct}% on {signal.symbol} already reached",
                )
            scaled_qty = allowed_room / signal.entry_price
            sizing = SizingResult(
                quantity=scaled_qty,
                position_value=scaled_qty * signal.entry_price,
                risk_amount=scaled_qty * sizing.stop_distance,
                risk_pct=(scaled_qty * sizing.stop_distance / portfolio.total_equity) * 100.0,
                stop_distance=sizing.stop_distance,
                method=f"{sizing.method}_capped_by_asset_exposure",
            )

        # 11. Check Correlated Exposure
        total_open_value = sum(self._state.exposure_by_asset.values())
        max_correlated_value = portfolio.total_equity * (self._config.max_correlated_exposure_pct / 100.0)
        if (total_open_value + sizing.position_value) > max_correlated_value * 1.01:
            allowed_corr_room = max(0.0, max_correlated_value - total_open_value)
            if allowed_corr_room <= 0:
                return RiskDecision(
                    approved=False,
                    action=RiskAction.REJECT,
                    quantity=0.0,
                    position_value=0.0,
                    risk_pct=0.0,
                    reason=f"Max correlated exposure {self._config.max_correlated_exposure_pct}% exceeded",
                )
            scaled_qty = allowed_corr_room / signal.entry_price
            sizing = SizingResult(
                quantity=scaled_qty,
                position_value=scaled_qty * signal.entry_price,
                risk_amount=scaled_qty * sizing.stop_distance,
                risk_pct=(scaled_qty * sizing.stop_distance / portfolio.total_equity) * 100.0,
                stop_distance=sizing.stop_distance,
                method=f"{sizing.method}_capped_by_correlated_exposure",
            )

        # 12. Check Portfolio Leverage Cap
        total_post_trade_exposure = total_open_value + sizing.position_value
        current_leverage = total_post_trade_exposure / portfolio.total_equity
        if current_leverage > self._config.max_leverage * 1.01:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Leverage limit {self._config.max_leverage}x exceeded (post-trade: {current_leverage:.2f}x)",
            )

        # 13. Cash Availability Check
        margin_required = sizing.position_value / max(1.0, self._config.max_leverage)
        if margin_required > portfolio.available_cash:
            return RiskDecision(
                approved=False,
                action=RiskAction.REJECT,
                quantity=0.0,
                position_value=0.0,
                risk_pct=0.0,
                reason=f"Insufficient available cash (${portfolio.available_cash:.2f} vs margin req ${margin_required:.2f})",
            )

        action = RiskAction.REDUCE_SIZE if size_multiplier < 0.99 else RiskAction.APPROVE
        reason = f"Approved via {sizing.method} (multiplier={size_multiplier:.2f})"
        return RiskDecision(
            approved=True,
            action=action,
            quantity=sizing.quantity,
            position_value=sizing.position_value,
            risk_pct=sizing.risk_pct,
            reason=reason,
            sizing=sizing,
        )

    def record_price_tick(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> bool:
        """Feed price ticks into flash crash circuit breaker."""
        return self._circuit_breaker.record_price_tick(symbol, price, timestamp)

    def record_trade_opened(self, symbol: str, position_value: float) -> None:
        """Update internal risk state when an order fill opens a position."""
        self._state.trades_today += 1
        self._state.open_position_count += 1
        current = self._state.exposure_by_asset.get(symbol, 0.0)
        self._state.exposure_by_asset[symbol] = current + position_value
        log.info(
            "risk_trade_opened_recorded",
            symbol=symbol,
            position_value=position_value,
            trades_today=self._state.trades_today,
            open_positions=self._state.open_position_count,
        )

    def record_trade_closed(self, symbol: str, position_value: float, pnl_pct: float) -> None:
        """Update internal risk state when a position closes."""
        self._state.open_position_count = max(0, self._state.open_position_count - 1)
        current = self._state.exposure_by_asset.get(symbol, 0.0)
        self._state.exposure_by_asset[symbol] = max(0.0, current - position_value)
        self._state.daily_loss_pct += pnl_pct

        if pnl_pct < 0:
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0

        log.info(
            "risk_trade_closed_recorded",
            symbol=symbol,
            pnl_pct=pnl_pct,
            daily_loss_pct=self._state.daily_loss_pct,
            consecutive_losses=self._state.consecutive_losses,
        )

    def _get_regime_multiplier(self, regime: MarketRegime, direction: Direction) -> float:
        """Dynamically dampen position size under adversarial volatility or counter-trend regimes."""
        if regime == MarketRegime.HIGH_VOLATILITY:
            return 0.70  # Scale down 30% during volatile conditions
        if regime == MarketRegime.STRONG_DOWNTREND and direction == Direction.LONG:
            return 0.50  # Counter-trend long penalized
        if regime == MarketRegime.STRONG_UPTREND and direction == Direction.SHORT:
            return 0.50  # Counter-trend short penalized
        return 1.0
