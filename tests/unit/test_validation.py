"""Automated tests for Milestone 6: Walk-Forward Validation & Overfitting Prevention.

Tests:
- Deflated Sharpe Ratio (DSR) & multiple testing corrections
- Parameter Stability Tester (cliff detection, smooth parameter neighborhoods)
- Walk-Forward Validation (rolling windows, non-overlapping OOS, WFE ratio)
- Strategy Scorecard (qualification criteria, hard limit rejections)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_crypto_trader.backtesting.advanced_metrics import ComprehensiveMetrics
from ai_crypto_trader.core.enums import Direction, MarketRegime, SignalAction
from ai_crypto_trader.core.interfaces import Signal
from ai_crypto_trader.features.feature_engine import FeatureEngine
from ai_crypto_trader.strategies.base import BaseStrategy
from ai_crypto_trader.strategies.trend_following import EMACrossoverStrategy
from ai_crypto_trader.validation.deflated_sharpe import (
    calculate_deflated_sharpe,
    expected_max_sharpe,
)
from ai_crypto_trader.validation.parameter_stability import ParameterStabilityTester
from ai_crypto_trader.validation.scorecard import StrategyScorecard
from ai_crypto_trader.validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardValidator,
)


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    """Generate 250 bars of trending synthetic price data."""
    np.random.seed(42)
    n = 250
    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    dates = [start + timedelta(hours=i) for i in range(n)]

    returns = np.random.normal(0.001, 0.015, n)
    close = 50000.0 * np.cumprod(1.0 + returns)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.998
    volume = np.random.uniform(500, 2000, n)

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


# ─────────────────────────────────────────────────────────────────────────────
# 1. DEFLATED SHARPE RATIO TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDeflatedSharpe:
    def test_expected_max_sharpe_increases_with_trials(self):
        # As more strategies are tried, chance of seeing a high random Sharpe increases
        s_10 = expected_max_sharpe(num_trials=10, variance_trials=0.5)
        s_100 = expected_max_sharpe(num_trials=100, variance_trials=0.5)
        s_1000 = expected_max_sharpe(num_trials=1000, variance_trials=0.5)

        assert 0.0 < s_10 < s_100 < s_1000

    def test_dsr_penalizes_multiple_trials(self):
        returns = [0.01, -0.005, 0.015, -0.002, 0.02, 0.01, -0.003, 0.018] * 10
        obs_sharpe = 2.0

        # With only 2 trials, statistical confidence is high
        dsr_low_trials = calculate_deflated_sharpe(obs_sharpe, returns, num_trials=2)
        # With 500 trials, data snooping penalty is severe
        dsr_high_trials = calculate_deflated_sharpe(obs_sharpe, returns, num_trials=500)

        assert dsr_low_trials > dsr_high_trials

    def test_zero_or_negative_sharpe_returns_zero(self):
        returns = [-0.01, -0.02, 0.005] * 5
        assert calculate_deflated_sharpe(observed_sharpe=-0.5, returns=returns) == 0.0
        assert calculate_deflated_sharpe(observed_sharpe=0.0, returns=returns) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. PARAMETER STABILITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterStability:
    def test_stability_evaluation_runs(self, synthetic_ohlcv_df):
        tester = ParameterStabilityTester(max_allowed_drop_10pct=30.0)
        params = {"fast_period": 9, "slow_period": 21}

        report = tester.evaluate(EMACrossoverStrategy, params, synthetic_ohlcv_df)

        assert report is not None
        assert report.strategy_name == "EMA_Crossover"
        assert len(report.results) > 0
        # Check perturbation delta coverage
        perturbations = {r.perturbation_pct for r in report.results}
        assert -10.0 in perturbations
        assert 10.0 in perturbations


# ─────────────────────────────────────────────────────────────────────────────
# 3. WALK-FORWARD VALIDATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardValidation:
    def test_walk_forward_rolling_windows(self, synthetic_ohlcv_df):
        validator = WalkForwardValidator(
            WalkForwardConfig(num_windows=3, is_ratio=0.70, min_wfe=0.50)
        )
        params = {"fast_period": 9, "slow_period": 21}

        result = validator.validate(EMACrossoverStrategy, params, synthetic_ohlcv_df)

        assert result is not None
        assert result.num_windows >= 2
        # Check window sequence
        for i in range(len(result.windows) - 1):
            curr_w = result.windows[i]
            next_w = result.windows[i + 1]
            assert curr_w.window_id < next_w.window_id
            assert curr_w.is_start <= next_w.is_start

    def test_insufficient_bars_raises_error(self, synthetic_ohlcv_df):
        validator = WalkForwardValidator()
        short_df = synthetic_ohlcv_df.iloc[:50]  # Less than 100 bars

        with pytest.raises(ValueError, match="Insufficient bars"):
            validator.validate(EMACrossoverStrategy, {"fast_period": 9}, short_df)


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRATEGY SCORECARD TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyScorecard:
    def test_scorecard_full_qualification(self):
        scorecard = StrategyScorecard()
        metrics = ComprehensiveMetrics(
            sharpe=1.85,
            profit_factor=2.10,
            max_drawdown=0.08,     # 8% (passes <= 15%)
            win_rate=0.58,         # 58% (passes >= 40%)
            num_trades=65,         # passes >= 30
            expectancy=150.0,      # passes > 0
        )

        eval_res = scorecard.evaluate("TestStrategy", metrics)

        assert eval_res.is_qualified is True
        assert "QUALIFIED FOR PAPER TRADING" in eval_res.summary_verdict

    def test_scorecard_rejection_on_excessive_drawdown(self):
        scorecard = StrategyScorecard()
        metrics = ComprehensiveMetrics(
            sharpe=1.85,
            profit_factor=2.10,
            max_drawdown=0.22,     # 22% (FAILS <= 15% limit)
            win_rate=0.58,
            num_trades=65,
            expectancy=150.0,
        )

        eval_res = scorecard.evaluate("RiskyStrategy", metrics)

        assert eval_res.is_qualified is False
        assert "REJECTED" in eval_res.summary_verdict
        assert "Max Drawdown" in eval_res.summary_verdict
