"""Platform-wide enumerations."""
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PAPER = "paper"
    SHADOW = "shadow"   # paper runs in parallel with live champion
    LIVE = "live"


class TradingMode(str, Enum):
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class MarketRegime(str, Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    UNKNOWN = "UNKNOWN"


class SignalAction(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


class StrategyStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    BACKTESTING = "BACKTESTING"
    PAPER = "PAPER"
    CHAMPION = "CHAMPION"
    CHALLENGER = "CHALLENGER"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class ModelStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"
    FAILED = "FAILED"


class RiskAction(str, Enum):
    APPROVE = "APPROVE"
    REDUCE_SIZE = "REDUCE_SIZE"
    REJECT = "REJECT"
    KILL_SWITCH = "KILL_SWITCH"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    TRADE_OPENED = "TRADE_OPENED"
    TRADE_CLOSED = "TRADE_CLOSED"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    STRATEGY_DEGRADATION = "STRATEGY_DEGRADATION"
    API_FAILURE = "API_FAILURE"
    EXCHANGE_FAILURE = "EXCHANGE_FAILURE"
    MODEL_RETRAIN = "MODEL_RETRAIN"
    STRATEGY_CANDIDATE = "STRATEGY_CANDIDATE"
    STRATEGY_PROMOTED = "STRATEGY_PROMOTED"
    STRATEGY_ROLLBACK = "STRATEGY_ROLLBACK"
    KILL_SWITCH_ACTIVATED = "KILL_SWITCH_ACTIVATED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    DATA_QUALITY = "DATA_QUALITY"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        return mapping[self.value]


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    SIGNAL = "SIGNAL"
    MANUAL = "MANUAL"
    KILL_SWITCH = "KILL_SWITCH"
    RISK_LIMIT = "RISK_LIMIT"
    DRIFT = "DRIFT"
    TIMEOUT = "TIMEOUT"
