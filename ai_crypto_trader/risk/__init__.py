"""Risk management package.

Implements institutional-grade safeguards, multi-tier circuit breakers,
Kelly/volatility position sizing, and global kill switch.
"""
from ai_crypto_trader.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)
from ai_crypto_trader.risk.kill_switch import KillSwitch
from ai_crypto_trader.risk.position_sizer import PositionSizer, SizingResult
from ai_crypto_trader.risk.risk_engine import RiskDecision, RiskEngine, RiskState

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "KillSwitch",
    "PositionSizer",
    "RiskDecision",
    "RiskEngine",
    "RiskState",
    "SizingResult",
]
