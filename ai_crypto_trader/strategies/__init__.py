"""Strategies package exporting all rule-based and ensemble trading strategies."""
from ai_crypto_trader.strategies.base import BaseStrategy
from ai_crypto_trader.strategies.breakout import DonchianBreakoutStrategy
from ai_crypto_trader.strategies.ensemble import StrategyEnsemble
from ai_crypto_trader.strategies.mean_reversion import BollingerRSIMeanReversion
from ai_crypto_trader.strategies.multi_timeframe import MultiTimeframeTrendStrategy
from ai_crypto_trader.strategies.trend_following import EMACrossoverStrategy

__all__ = [
    "BaseStrategy",
    "EMACrossoverStrategy",
    "BollingerRSIMeanReversion",
    "DonchianBreakoutStrategy",
    "MultiTimeframeTrendStrategy",
    "StrategyEnsemble",
]
