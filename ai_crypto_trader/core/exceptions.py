"""Platform-wide custom exceptions."""


class TradingPlatformError(Exception):
    """Base exception for the trading platform."""


# ── Data Exceptions ───────────────────────────────────────────────────────────
class DataValidationError(TradingPlatformError):
    """Raised when incoming market data fails validation."""


class DataGapError(TradingPlatformError):
    """Raised when a gap is detected in the candle series."""


class StaleDataError(TradingPlatformError):
    """Raised when market data is too old to be trusted."""


class DuplicateCandleError(TradingPlatformError):
    """Raised when a duplicate candle timestamp is detected."""


# ── Exchange Exceptions ───────────────────────────────────────────────────────
class ExchangeConnectionError(TradingPlatformError):
    """Raised when the exchange connection is unavailable."""


class ExchangeRateLimitError(TradingPlatformError):
    """Raised when an exchange rate limit is hit."""


class OrderSubmissionError(TradingPlatformError):
    """Raised when an order cannot be submitted."""


class OrderNotFoundError(TradingPlatformError):
    """Raised when an expected order does not exist."""


class PositionReconciliationError(TradingPlatformError):
    """Raised when internal state disagrees with exchange position."""


# ── Risk Exceptions ───────────────────────────────────────────────────────────
class RiskLimitExceeded(TradingPlatformError):
    """Raised when a risk limit is breached."""


class KillSwitchActivated(TradingPlatformError):
    """Raised when the kill switch is engaged — stops all trading."""


class DailyLossLimitExceeded(RiskLimitExceeded):
    """Daily loss limit has been hit."""


class DrawdownLimitExceeded(RiskLimitExceeded):
    """Maximum drawdown threshold breached."""


# ── Strategy Exceptions ───────────────────────────────────────────────────────
class StrategyNotFoundError(TradingPlatformError):
    """Raised when a requested strategy does not exist."""


class ValidationPipelineError(TradingPlatformError):
    """Raised when a strategy fails the validation pipeline."""


class InsufficientDataError(TradingPlatformError):
    """Raised when there is not enough data to compute a signal or feature."""


# ── Model Exceptions ──────────────────────────────────────────────────────────
class ModelNotFoundError(TradingPlatformError):
    """Raised when a requested ML model version does not exist."""


class ModelPredictionError(TradingPlatformError):
    """Raised when a model cannot produce a prediction."""


# ── Feature Exceptions ────────────────────────────────────────────────────────
class LookAheadBiasError(TradingPlatformError):
    """Raised when a look-ahead bias is detected during feature computation."""


class FeatureComputationError(TradingPlatformError):
    """Raised when a feature cannot be computed from given data."""


# ── Configuration Exceptions ─────────────────────────────────────────────────
class ConfigurationError(TradingPlatformError):
    """Raised for invalid or missing configuration."""


# ── Database Exceptions ───────────────────────────────────────────────────────
class DatabaseError(TradingPlatformError):
    """Raised for unrecoverable database errors."""
