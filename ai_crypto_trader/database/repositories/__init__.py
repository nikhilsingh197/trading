"""Database repositories package exporting all domain CRUD interfaces."""
from ai_crypto_trader.database.repositories.ai_repo import (
    ExperimentRepository,
    ModelRepository,
    SignalRepository,
)
from ai_crypto_trader.database.repositories.candle_repo import CandleRepository
from ai_crypto_trader.database.repositories.strategy_repo import (
    BacktestResultRepository,
    StrategyMetricsRepository,
    StrategyRepository,
)
from ai_crypto_trader.database.repositories.system_repo import (
    AlertRepository,
    AuditLogRepository,
    RiskEventRepository,
    SystemEventRepository,
)
from ai_crypto_trader.database.repositories.trading_repo import (
    OrderRepository,
    PortfolioSnapshotRepository,
    PositionRepository,
    TradeRepository,
)

__all__ = [
    "CandleRepository",
    "OrderRepository",
    "TradeRepository",
    "PositionRepository",
    "PortfolioSnapshotRepository",
    "ModelRepository",
    "SignalRepository",
    "ExperimentRepository",
    "StrategyRepository",
    "StrategyMetricsRepository",
    "BacktestResultRepository",
    "SystemEventRepository",
    "RiskEventRepository",
    "AuditLogRepository",
    "AlertRepository",
]
