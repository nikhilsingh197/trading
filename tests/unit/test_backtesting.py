"""Unit tests for the backtesting engine."""
from __future__ import annotations

import pytest
import pandas as pd

from ai_crypto_trader.backtesting.engine import BacktestConfig, BacktestEngine
from ai_crypto_trader.backtesting.metrics import BacktestMetrics, TradeRecord, compute_metrics
from datetime import datetime, timezone


class TestBacktestMetrics:
    def _make_trades(self, pnls: list[float]) -> list[TradeRecord]:
        trades = []
        for p in pnls:
            trades.append(
                TradeRecord(
                    entry_price=50000.0,
                    exit_price=50000.0 + p,
                    side="long",
                    quantity=0.01,
                    pnl=p,
                    pnl_pct=p / 50000,
                    fees=0.5,
                    opened_at=datetime.now(timezone.utc),
                    closed_at=datetime.now(timezone.utc),
                    exit_reason="TP",
                )
            )
        return trades

    def test_win_rate_calculation(self):
        trades = self._make_trades([100, -50, 100, -50, 100])
        equity = pd.Series([10000, 10100, 10050, 10150, 10100, 10200])
        m = compute_metrics(trades, equity, 10000)
        assert m.win_rate == pytest.approx(3/5, rel=1e-3)

    def test_profit_factor(self):
        trades = self._make_trades([100, 100, -50])
        equity = pd.Series([10000, 10100, 10200, 10150])
        m = compute_metrics(trades, equity, 10000)
        assert m.profit_factor == pytest.approx(200/50, rel=1e-3)

    def test_empty_trades_returns_defaults(self):
        m = compute_metrics([], pd.Series([10000]), 10000)
        assert m.num_trades == 0
        assert m.sharpe == 0.0

    def test_max_consecutive_losses(self):
        trades = self._make_trades([100, -50, -50, -50, 100, -50])
        equity = pd.Series([10000, 10100, 10050, 10000, 9950, 10050, 10000])
        m = compute_metrics(trades, equity, 10000)
        assert m.max_consecutive_losses == 3


class TestBacktestEngine:
    def test_rejects_target_columns(self, feature_df):
        """Engine must raise if target columns are present (leakage guard)."""
        from ai_crypto_trader.features.target_generator import TargetConfig, TargetGenerator
        from ai_crypto_trader.strategies.trend_following import EMACrossoverStrategy

        df_with_targets = TargetGenerator().add_targets(feature_df, TargetConfig())
        engine = BacktestEngine()
        strategy = EMACrossoverStrategy(version_id="test_v1")

        with pytest.raises(ValueError, match="target columns"):
            engine.run(strategy, df_with_targets)

    def test_basic_run_completes(self, feature_df):
        """Backtest runs without error on clean feature DataFrame."""
        from ai_crypto_trader.strategies.trend_following import EMACrossoverStrategy

        engine = BacktestEngine(BacktestConfig(initial_capital=10000))
        strategy = EMACrossoverStrategy(version_id="test_v1")
        result = engine.run(strategy, feature_df)

        assert result is not None
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe, float)
        assert result.num_trades >= 0
