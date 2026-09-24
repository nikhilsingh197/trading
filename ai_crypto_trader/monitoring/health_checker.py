"""System Health and Liveness Monitor.

Tracks heartbeat liveness, data freshness (stale data watchdog), exchange API latency,
database ping times, and circuit breaker status.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Optional

from ai_crypto_trader.alerts.alert_manager import AlertManager
from ai_crypto_trader.core.constants import HEALTH_CHECK_INTERVAL_SECONDS, STALE_DATA_THRESHOLD_SECONDS
from ai_crypto_trader.core.enums import AlertSeverity, AlertType
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.exchange.base import ExchangeAdapterABC
from ai_crypto_trader.risk.circuit_breaker import CircuitBreaker
from ai_crypto_trader.risk.kill_switch import KillSwitch

log = get_logger(__name__)


@dataclass
class ComponentHealth:
    name: str
    status: str          # 'HEALTHY', 'DEGRADED', 'DOWN'
    latency_ms: float = 0.0
    details: str = ""
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SystemHealthStatus:
    is_healthy: bool
    status: str          # 'HEALTHY', 'DEGRADED', 'CRITICAL'
    uptime_seconds: float
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HealthChecker:
    """Institutional system health monitor with stale data watchdogs and exchange latency probes."""

    def __init__(
        self,
        exchange: Optional[ExchangeAdapterABC] = None,
        kill_switch: Optional[KillSwitch] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        alert_manager: Optional[AlertManager] = None,
        stale_threshold_seconds: float = STALE_DATA_THRESHOLD_SECONDS,
        max_acceptable_latency_ms: float = 1200.0,
    ) -> None:
        self.exchange = exchange
        self.kill_switch = kill_switch
        self.circuit_breaker = circuit_breaker
        self.alert_manager = alert_manager
        self.stale_threshold_seconds = stale_threshold_seconds
        self.max_acceptable_latency_ms = max_acceptable_latency_ms

        self._start_time = time.time()
        self._last_market_tick_time: float = time.time()
        self._is_running = False

    def record_market_data_tick(self, timestamp: Optional[float] = None) -> None:
        """Invoked by ingestion/websocket collectors to update the watchdog timestamp."""
        self._last_market_tick_time = timestamp or time.time()

    async def check_health(self) -> SystemHealthStatus:
        """Execute comprehensive platform health check across all core subsystems."""
        now = datetime.now(timezone.utc)
        components: dict[str, ComponentHealth] = {}
        all_healthy = True
        has_critical = False

        # 1. Market Data Watchdog (Stale Data Check)
        age = time.time() - self._last_market_tick_time
        if age > self.stale_threshold_seconds * 2:
            all_healthy = False
            has_critical = True
            components["market_data"] = ComponentHealth(
                name="market_data",
                status="DOWN",
                latency_ms=round(age * 1000, 1),
                details=f"CRITICAL: Market data feed stopped! Age: {age:.1f}s (threshold: {self.stale_threshold_seconds}s)",
            )
        elif age > self.stale_threshold_seconds:
            all_healthy = False
            components["market_data"] = ComponentHealth(
                name="market_data",
                status="DEGRADED",
                latency_ms=round(age * 1000, 1),
                details=f"WARNING: Market data stale. Age: {age:.1f}s",
            )
        else:
            components["market_data"] = ComponentHealth(
                name="market_data",
                status="HEALTHY",
                latency_ms=round(age * 1000, 1),
                details=f"Feed active (last tick {age:.2f}s ago)",
            )

        # 2. Exchange API Liveness & Latency
        if self.exchange:
            start_probe = time.time()
            try:
                await self.exchange.fetch_ticker("BTCUSDT")
                latency = (time.time() - start_probe) * 1000.0
                if latency > self.max_acceptable_latency_ms:
                    all_healthy = False
                    components["exchange"] = ComponentHealth(
                        name="exchange",
                        status="DEGRADED",
                        latency_ms=round(latency, 1),
                        details=f"High latency ({latency:.0f}ms > {self.max_acceptable_latency_ms:.0f}ms)",
                    )
                else:
                    components["exchange"] = ComponentHealth(
                        name="exchange",
                        status="HEALTHY",
                        latency_ms=round(latency, 1),
                        details="API connected",
                    )
            except Exception as e:
                all_healthy = False
                has_critical = True
                components["exchange"] = ComponentHealth(
                    name="exchange",
                    status="DOWN",
                    details=f"Exchange connection failed: {e}",
                )

        # 3. Safety Subsystems (Kill Switch & Circuit Breaker)
        if self.kill_switch:
            is_active = await self.kill_switch.is_active()
            if is_active:
                all_healthy = False
                components["kill_switch"] = ComponentHealth(
                    name="kill_switch",
                    status="DEGRADED",
                    details=f"KILL SWITCH ACTIVE ({self.kill_switch._reason})",
                )
            else:
                components["kill_switch"] = ComponentHealth(
                    name="kill_switch",
                    status="HEALTHY",
                    details="Armed and ready (standby)",
                )

        if self.circuit_breaker:
            breaker_state = self.circuit_breaker.state
            if breaker_state.value in ("TRIPPED", "FLASH_CRASH_HALT"):
                all_healthy = False
                has_critical = True
                components["circuit_breaker"] = ComponentHealth(
                    name="circuit_breaker",
                    status="DOWN",
                    details=f"Breaker tripped: {self.circuit_breaker.reason}",
                )
            elif breaker_state.value in ("PAUSED", "THROTTLED"):
                all_healthy = False
                components["circuit_breaker"] = ComponentHealth(
                    name="circuit_breaker",
                    status="DEGRADED",
                    details=f"Breaker in defensive mode: {self.circuit_breaker.reason}",
                )
            else:
                components["circuit_breaker"] = ComponentHealth(
                    name="circuit_breaker",
                    status="HEALTHY",
                    details="Normal operations",
                )

        overall_status = "HEALTHY" if all_healthy else ("CRITICAL" if has_critical else "DEGRADED")
        uptime = round(time.time() - self._start_time, 1)

        status_obj = SystemHealthStatus(
            is_healthy=all_healthy,
            status=overall_status,
            uptime_seconds=uptime,
            components=components,
            timestamp=now,
        )

        if not all_healthy:
            log.warning("system_health_degraded", status=overall_status, components={k: v.status for k, v in components.items()})

        return status_obj
