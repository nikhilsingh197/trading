"""Multi-tier institutional circuit breaker.

Provides cascading defensive halts to safeguard capital during adverse
market conditions, sudden regime shifts, or liquidity flash crashes.

Tiers:
- NORMAL: Standard execution.
- THROTTLED (Level 1): Soft threshold breached (e.g. daily loss >= 1.0%
  or drawdown >= 5.0%). Position sizes scaled down by 50%.
- PAUSED (Level 2): Moderate threshold breached (e.g. daily loss >= 1.5%).
  Trading paused for a cool-off window (default: 60 minutes).
- TRIPPED (Level 3): Hard threshold breached (e.g. daily loss >= 2.0%
  or drawdown >= 10.0%). All trading halted until UTC midnight or manual reset.
- FLASH_CRASH_HALT: Rapid price drop (e.g. > 7% in 15 mins) detected on any tracked asset.
  Haults new orders to prevent executing into toxic order books.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


class CircuitBreakerState(str, Enum):
    NORMAL = "NORMAL"
    THROTTLED = "THROTTLED"        # Level 1: 50% size reduction
    PAUSED = "PAUSED"              # Level 2: Intraday cool-off pause
    TRIPPED = "TRIPPED"            # Level 3: Hard daily shutdown
    FLASH_CRASH_HALT = "FLASH_CRASH_HALT"  # Severe market anomaly halt


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configurable thresholds for cascading circuit breaker trips."""
    level1_daily_loss_pct: float = 1.0       # Triggers THROTTLED
    level2_daily_loss_pct: float = 1.5       # Triggers PAUSED
    level3_daily_loss_pct: float = 2.0       # Triggers TRIPPED
    level1_drawdown_pct: float = 5.0         # Triggers THROTTLED
    level2_drawdown_pct: float = 8.0         # Triggers PAUSED
    level3_drawdown_pct: float = 10.0        # Triggers TRIPPED
    cool_off_seconds: int = 3600             # 1 hour cool-off for Level 2
    flash_crash_threshold_pct: float = 7.0   # % drop in window
    flash_crash_window_seconds: int = 900    # 15 minutes rolling window
    flash_crash_halt_seconds: int = 1800     # 30 minutes halt after crash
    max_consecutive_losses: int = 4          # Consecutive losses before throttle


@dataclass
class PriceTick:
    timestamp: datetime
    price: float


class CircuitBreaker:
    """Autonomous circuit breaker monitoring trading performance and market sanity."""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self._state: CircuitBreakerState = CircuitBreakerState.NORMAL
        self._reason: str = "System normal"
        self._paused_until: Optional[datetime] = None
        self._last_state_change: datetime = datetime.now(timezone.utc)
        self._price_windows: dict[str, deque[PriceTick]] = {}

    @property
    def state(self) -> CircuitBreakerState:
        self._check_cool_off_expiry()
        return self._state

    @property
    def reason(self) -> str:
        return self._reason

    def can_trade(self) -> tuple[bool, str]:
        """Check if circuit breaker permits new trade entries.

        Returns:
            (can_trade: bool, reason: str)
        """
        current_state = self.state
        if current_state == CircuitBreakerState.NORMAL:
            return True, "Breaker NORMAL: Trading permitted"
        if current_state == CircuitBreakerState.THROTTLED:
            return True, f"Breaker THROTTLED: Trading permitted at 50% size ({self._reason})"
        if current_state == CircuitBreakerState.PAUSED:
            remaining = int((self._paused_until - datetime.now(timezone.utc)).total_seconds()) if self._paused_until else 0
            return False, f"Breaker PAUSED: In cool-off for {max(0, remaining)}s ({self._reason})"
        if current_state == CircuitBreakerState.TRIPPED:
            return False, f"Breaker TRIPPED: Hard daily shutdown ({self._reason})"
        if current_state == CircuitBreakerState.FLASH_CRASH_HALT:
            remaining = int((self._paused_until - datetime.now(timezone.utc)).total_seconds()) if self._paused_until else 0
            return False, f"Breaker FLASH_CRASH_HALT: Anomaly cool-off {max(0, remaining)}s ({self._reason})"
        return False, f"Breaker in unknown state: {current_state}"

    def get_size_multiplier(self) -> float:
        """Return position size multiplier based on breaker state."""
        current_state = self.state
        if current_state == CircuitBreakerState.NORMAL:
            return 1.0
        if current_state == CircuitBreakerState.THROTTLED:
            return 0.5
        return 0.0

    def check_metrics(
        self,
        daily_loss_pct: float,
        drawdown_pct: float,
        consecutive_losses: int = 0,
        now: Optional[datetime] = None,
    ) -> CircuitBreakerState:
        """Evaluate account performance against multi-tier thresholds."""
        ts = now or datetime.now(timezone.utc)
        self._check_cool_off_expiry(ts)

        # Do not override hard TRIPPED state automatically during same trading day
        if self._state == CircuitBreakerState.TRIPPED:
            return self._state

        # Flash crash halt takes precedence until its timer expires
        if self._state == CircuitBreakerState.FLASH_CRASH_HALT:
            return self._state

        loss = abs(daily_loss_pct) if daily_loss_pct < 0 else 0.0

        # Level 3: Hard trip
        if loss >= self.config.level3_daily_loss_pct or drawdown_pct >= self.config.level3_drawdown_pct:
            reason = (
                f"Level 3 breach: daily loss {loss:.2f}% (limit {self.config.level3_daily_loss_pct}%) "
                f"or drawdown {drawdown_pct:.2f}% (limit {self.config.level3_drawdown_pct}%)"
            )
            self._transition_to(CircuitBreakerState.TRIPPED, reason, ts)
            return self._state

        # Level 2: Pause / cool-off
        if loss >= self.config.level2_daily_loss_pct or drawdown_pct >= self.config.level2_drawdown_pct:
            reason = (
                f"Level 2 breach: daily loss {loss:.2f}% (limit {self.config.level2_daily_loss_pct}%) "
                f"or drawdown {drawdown_pct:.2f}% (limit {self.config.level2_drawdown_pct}%)"
            )
            self._transition_to(
                CircuitBreakerState.PAUSED,
                reason,
                ts,
                paused_until=ts + timedelta(seconds=self.config.cool_off_seconds),
            )
            return self._state

        # Level 1: Throttled / soft reduction
        if (
            loss >= self.config.level1_daily_loss_pct
            or drawdown_pct >= self.config.level1_drawdown_pct
            or consecutive_losses >= self.config.max_consecutive_losses
        ):
            reason = (
                f"Level 1 breach: daily loss {loss:.2f}%, drawdown {drawdown_pct:.2f}%, "
                f"or consec losses {consecutive_losses}"
            )
            self._transition_to(CircuitBreakerState.THROTTLED, reason, ts)
            return self._state

        # If previously throttled and metrics recovered under limits, return to normal
        if self._state == CircuitBreakerState.THROTTLED:
            self._transition_to(CircuitBreakerState.NORMAL, "Metrics recovered below soft limits", ts)

        return self._state

    def record_price_tick(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Inspect rapid price changes to detect flash crashes.

        Returns:
            True if flash crash detected and breaker engaged, False otherwise.
        """
        now = timestamp or datetime.now(timezone.utc)
        if symbol not in self._price_windows:
            self._price_windows[symbol] = deque()

        window = self._price_windows[symbol]
        window.append(PriceTick(timestamp=now, price=price))

        cutoff = now - timedelta(seconds=self.config.flash_crash_window_seconds)
        while window and window[0].timestamp < cutoff:
            window.popleft()

        if len(window) < 2:
            return False

        max_price = max(t.price for t in window)
        current_price = price
        drop_pct = ((max_price - current_price) / max_price) * 100.0

        if drop_pct >= self.config.flash_crash_threshold_pct:
            reason = (
                f"Flash crash detected on {symbol}: dropped {drop_pct:.2f}% in "
                f"{self.config.flash_crash_window_seconds}s (limit {self.config.flash_crash_threshold_pct}%)"
            )
            self._transition_to(
                CircuitBreakerState.FLASH_CRASH_HALT,
                reason,
                now,
                paused_until=now + timedelta(seconds=self.config.flash_crash_halt_seconds),
            )
            return True

        return False

    def reset(self, actor: str = "system", now: Optional[datetime] = None) -> None:
        """Manual or automated end-of-day breaker reset."""
        ts = now or datetime.now(timezone.utc)
        log.warning("circuit_breaker_reset", actor=actor, previous_state=self._state.value)
        self._transition_to(CircuitBreakerState.NORMAL, f"Manual reset by {actor}", ts)
        self._paused_until = None

    def _transition_to(
        self,
        new_state: CircuitBreakerState,
        reason: str,
        timestamp: datetime,
        paused_until: Optional[datetime] = None,
    ) -> None:
        if self._state == new_state and self._reason == reason:
            return

        old_state = self._state
        self._state = new_state
        self._reason = reason
        self._last_state_change = timestamp
        self._paused_until = paused_until

        if new_state in (CircuitBreakerState.TRIPPED, CircuitBreakerState.FLASH_CRASH_HALT):
            log.critical(
                "CIRCUIT_BREAKER_CRITICAL_HALT",
                old_state=old_state.value,
                new_state=new_state.value,
                reason=reason,
                paused_until=paused_until.isoformat() if paused_until else None,
            )
        elif new_state in (CircuitBreakerState.PAUSED, CircuitBreakerState.THROTTLED):
            log.warning(
                "CIRCUIT_BREAKER_DEFENSIVE_MODE",
                old_state=old_state.value,
                new_state=new_state.value,
                reason=reason,
                paused_until=paused_until.isoformat() if paused_until else None,
            )
        else:
            log.info("circuit_breaker_state_restored", state=new_state.value)

    def _check_cool_off_expiry(self, now: Optional[datetime] = None) -> None:
        ts = now or datetime.now(timezone.utc)
        if self._state in (CircuitBreakerState.PAUSED, CircuitBreakerState.FLASH_CRASH_HALT):
            if self._paused_until and ts >= self._paused_until:
                self._transition_to(
                    CircuitBreakerState.THROTTLED,
                    "Cool-off period elapsed; resuming in THROTTLED mode",
                    ts,
                )
                self._paused_until = None
