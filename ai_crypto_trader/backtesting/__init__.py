"""Backtesting engine package.

Provides both single-asset and multi-asset event-driven simulation engines,
institutional risk analytics, dynamic cost modeling, and Monte Carlo resampling.
"""
from ai_crypto_trader.backtesting.advanced_metrics import (
    ComprehensiveMetrics,
    TradeDetail,
    calculate_comprehensive_metrics,
)
from ai_crypto_trader.backtesting.cost_model import CostModel, CostModelConfig, ExecutionFill
from ai_crypto_trader.backtesting.engine import BacktestConfig, BacktestEngine
from ai_crypto_trader.backtesting.metrics import BacktestMetrics, TradeRecord, compute_metrics
from ai_crypto_trader.backtesting.monte_carlo import MonteCarloResult, MonteCarloSimulator
from ai_crypto_trader.backtesting.portfolio_engine import (
    MultiAssetBacktestEngine,
    MultiAssetBacktestResult,
    PortfolioEngineConfig,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "MultiAssetBacktestEngine",
    "PortfolioEngineConfig",
    "MultiAssetBacktestResult",
    "CostModel",
    "CostModelConfig",
    "ExecutionFill",
    "ComprehensiveMetrics",
    "TradeDetail",
    "calculate_comprehensive_metrics",
    "MonteCarloSimulator",
    "MonteCarloResult",
    "compute_metrics",
    "TradeRecord",
    "BacktestMetrics",
]
