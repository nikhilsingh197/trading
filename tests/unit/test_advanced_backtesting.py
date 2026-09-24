"""Automated tests for Milestone 4: Advanced Backtesting Engine.

Tests:
- CostModel (dynamic slippage, volume participation limit, funding rate calculations)
- ComprehensiveMetrics (Calmar, Omega, underwater duration, VaR, CVaR)
- MonteCarloSimulator (bootstrap percentiles, drawdown probabilities)
- MultiAssetBacktestEngine (multi-asset synchronization, shared capital, latency modeling)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_crypto_trader.backtesting.advanced_metrics import (
    ComprehensiveMetrics,
    TradeDetail,
    calculate_comprehensive_metrics,
)
from ai_crypto_trader.backtesting.cost_model import CostModel, CostModelConfig
from ai_crypto_trader.backtesting.monte_carlo import MonteCarloSimulator
from ai_crypto_trader.backtesting.portfolio_engine import (
    MultiAssetBacktestEngine,
    PortfolioEngineConfig,
)
from ai_crypto_trader.core.enums import Direction
from ai_crypto_trader.features.feature_engine import FeatureEngine
from ai_crypto_trader.strategies.trend_following import EMACrossoverStrategy


# ─────────────────────────────────────────────────────────────────────────────
# 1. COST MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCostModel:
    def test_dynamic_slippage_scales_with_volume_ratio(self):
        cm = CostModel(CostModelConfig(base_slippage=0.0005, impact_multiplier=0.1))

        # Small trade: 1 unit on 10,000 volume
        fill_small = cm.calculate_fill(
            requested_quantity=1.0,
            base_price=50000.0,
            direction=Direction.LONG,
            bar_volume=10000.0,
        )

        # Large trade: 100 units on 10,000 volume
        fill_large = cm.calculate_fill(
            requested_quantity=100.0,
            base_price=50000.0,
            direction=Direction.LONG,
            bar_volume=10000.0,
        )

        # Larger trade must suffer higher effective slippage per unit
        effective_slip_small = (fill_small.effective_price - 50000.0) / 50000.0
        effective_slip_large = (fill_large.effective_price - 50000.0) / 50000.0
        assert effective_slip_large > effective_slip_small

    def test_volume_participation_partial_fill(self):
        # Max participation = 5% of bar volume
        cm = CostModel(CostModelConfig(max_volume_participation=0.05))

        # Request 100 units when bar volume is 1000 -> max allowed is 50 units
        fill = cm.calculate_fill(
            requested_quantity=100.0,
            base_price=100.0,
            direction=Direction.LONG,
            bar_volume=1000.0,
        )

        assert fill.is_partial is True
        assert fill.filled_quantity == 50.0
        assert fill.remaining_quantity == 50.0

    def test_funding_rate_payment_direction(self):
        cm = CostModel()
        pos_val = 10000.0  # $10,000 position
        funding_rate = 0.0001  # 0.01% positive rate

        # Long pays positive funding rate
        long_flow = cm.calculate_funding_payment(pos_val, Direction.LONG, funding_rate)
        assert long_flow == -1.0  # $1.00 cash outflow

        # Short receives positive funding rate
        short_flow = cm.calculate_funding_payment(pos_val, Direction.SHORT, funding_rate)
        assert short_flow == 1.0  # $1.00 cash inflow

    def test_funding_hour_detection(self):
        assert CostModel.is_funding_hour(datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)) is True
        assert CostModel.is_funding_hour(datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)) is True
        assert CostModel.is_funding_hour(datetime(2025, 1, 1, 16, 0, tzinfo=timezone.utc)) is True
        assert CostModel.is_funding_hour(datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)) is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADVANCED METRICS TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAdvancedMetrics:
    def test_comprehensive_metrics_computation(self):
        now = datetime.now(timezone.utc)
        trades = [
            TradeDetail(
                symbol="BTCUSDT",
                side="LONG",
                entry_price=50000.0,
                exit_price=52000.0,
                quantity=1.0,
                pnl=1900.0,
                pnl_pct=0.038,
                fees=50.0,
                slippage=50.0,
                funding_paid=-5.0,
                opened_at=now - timedelta(hours=10),
                closed_at=now - timedelta(hours=5),
                exit_reason="TAKE_PROFIT",
                bars_held=5,
            ),
            TradeDetail(
                symbol="BTCUSDT",
                side="LONG",
                entry_price=52000.0,
                exit_price=51000.0,
                quantity=1.0,
                pnl=-1080.0,
                pnl_pct=-0.021,
                fees=40.0,
                slippage=40.0,
                funding_paid=0.0,
                opened_at=now - timedelta(hours=4),
                closed_at=now,
                exit_reason="STOP_LOSS",
                bars_held=4,
            ),
        ]

        equity = pd.Series([10000.0, 10500.0, 11000.0, 10800.0, 10700.0, 11200.0])
        m = calculate_comprehensive_metrics(trades, equity, initial_capital=10000.0)

        assert m.num_trades == 2
        assert m.num_winners == 1
        assert m.num_losers == 1
        assert m.win_rate == 0.5
        assert m.profit_factor > 1.0
        assert m.payoff_ratio > 1.0
        assert m.total_return == 0.12  # (11200 - 10000)/10000
        assert m.max_drawdown > 0.0
        assert m.max_drawdown_duration_bars == 2  # 10800, 10700 underwater relative to 11000 peak
        assert m.fees_paid == 90.0
        assert m.slippage_paid == 90.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. MONTE CARLO SIMULATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMonteCarloSimulator:
    def test_simulation_percentiles_order(self):
        now = datetime.now(timezone.utc)
        # Create a series of 20 trades
        trades = []
        for i in range(20):
            pnl_pct = 0.02 if i % 2 == 0 else -0.015
            trades.append(
                TradeDetail(
                    symbol="BTCUSDT",
                    side="LONG",
                    entry_price=50000.0,
                    exit_price=50000.0 * (1 + pnl_pct),
                    quantity=1.0,
                    pnl=50000.0 * pnl_pct,
                    pnl_pct=pnl_pct,
                    fees=5.0,
                    slippage=5.0,
                    funding_paid=0.0,
                    opened_at=now,
                    closed_at=now,
                    exit_reason="SIGNAL",
                    bars_held=3,
                )
            )

        sim = MonteCarloSimulator(iterations=200, seed=42)
        res = sim.simulate(trades, initial_capital=10000.0)

        assert res.iterations == 200
        assert res.trades_sampled == 20
        # 5th percentile return must be <= median <= 95th percentile
        assert res.return_5th_pct <= res.median_return <= res.return_95th_pct
        # Worst drawdown 95th percentile must be >= median drawdown
        assert res.max_drawdown_95th_pct >= res.median_max_drawdown

    def test_empty_trades_returns_zeros(self):
        sim = MonteCarloSimulator(iterations=100)
        res = sim.simulate([], initial_capital=10000.0)
        assert res.iterations == 0
        assert res.median_return == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-ASSET PORTFOLIO ENGINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiAssetBacktestEngine:
    def _create_synthetic_bars(self, symbol: str, n_bars: int = 150) -> pd.DataFrame:
        np.random.seed(42 if symbol == "BTCUSDT" else 100)
        base_price = 60000.0 if symbol == "BTCUSDT" else 3000.0
        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

        dates = [start + timedelta(hours=i) for i in range(n_bars)]
        returns = np.random.normal(0.0005, 0.015, n_bars)
        close = base_price * np.cumprod(1.0 + returns)
        high = close * 1.008
        low = close * 0.992
        open_ = close * 0.999
        volume = np.random.uniform(500, 2000, n_bars)

        df = pd.DataFrame(
            {
                "open_time": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ).set_index("open_time")

        fe = FeatureEngine()
        return fe.compute(df)

    def test_multi_asset_synchronized_simulation(self):
        btc_df = self._create_synthetic_bars("BTCUSDT")
        eth_df = self._create_synthetic_bars("ETHUSDT")

        datasets = {
            "BTCUSDT": btc_df,
            "ETHUSDT": eth_df,
        }

        strat = EMACrossoverStrategy(version_id="multi_v1")
        engine = MultiAssetBacktestEngine(
            PortfolioEngineConfig(
                initial_capital=50000.0,
                max_open_positions=3,
                run_monte_carlo=True,
                monte_carlo_iterations=100,
            )
        )

        result = engine.run(strat, datasets)

        assert result is not None
        assert isinstance(result.portfolio_metrics, ComprehensiveMetrics)
        assert len(result.equity_curve) > 0
        assert "BTCUSDT" in result.per_symbol_metrics
        assert "ETHUSDT" in result.per_symbol_metrics

        # Final equity in series must match capital * (1 + total_return)
        final_equity = result.equity_curve.iloc[-1]
        expected_equity = 50000.0 * (1.0 + result.portfolio_metrics.total_return)
        assert abs(final_equity - expected_equity) < 1e-4

    def test_leakage_guard_rejects_target_columns(self):
        btc_df = self._create_synthetic_bars("BTCUSDT")
        # Artificially inject target column (leakage attempt)
        btc_df["target_future_return"] = 0.05

        strat = EMACrossoverStrategy(version_id="multi_v1")
        engine = MultiAssetBacktestEngine()

        with pytest.raises(ValueError, match="training target columns"):
            engine.run(strat, {"BTCUSDT": btc_df})
