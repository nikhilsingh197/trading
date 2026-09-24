"""Unit tests for feature engine."""
from __future__ import annotations

import pandas as pd
import pytest

from ai_crypto_trader.features.feature_engine import FeatureEngine
from ai_crypto_trader.features.target_generator import TargetConfig, TargetGenerator


class TestFeatureEngine:
    def test_compute_returns_dataframe(self, sample_ohlcv_df):
        engine = FeatureEngine()
        result = engine.compute(sample_ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_original_not_mutated(self, sample_ohlcv_df):
        original_cols = set(sample_ohlcv_df.columns)
        engine = FeatureEngine()
        engine.compute(sample_ohlcv_df)
        assert set(sample_ohlcv_df.columns) == original_cols

    def test_adds_trend_features(self, sample_ohlcv_df):
        engine = FeatureEngine()
        result = engine.compute(sample_ohlcv_df)
        assert "trend_ema_9" in result.columns
        assert "trend_adx" in result.columns
        assert "trend_macd" in result.columns

    def test_adds_momentum_features(self, sample_ohlcv_df):
        engine = FeatureEngine()
        result = engine.compute(sample_ohlcv_df)
        assert "mom_rsi_14" in result.columns
        assert "mom_stoch_k" in result.columns

    def test_adds_volatility_features(self, sample_ohlcv_df):
        engine = FeatureEngine()
        result = engine.compute(sample_ohlcv_df)
        assert "vol_atr_14" in result.columns
        assert "vol_bb_width" in result.columns

    def test_rsi_range_valid(self, sample_ohlcv_df):
        engine = FeatureEngine()
        result = engine.compute(sample_ohlcv_df)
        rsi = result["mom_rsi_14"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_missing_column_raises(self):
        engine = FeatureEngine()
        bad_df = pd.DataFrame({"open": [1, 2], "close": [1, 2]})
        with pytest.raises(ValueError, match="missing columns"):
            engine.compute(bad_df)


class TestTargetGenerator:
    def test_target_columns_added(self, feature_df):
        gen = TargetGenerator()
        result = gen.add_targets(feature_df, TargetConfig(horizon=10))
        assert "target_future_return" in result.columns
        assert "target_direction" in result.columns
        assert "target_tp_before_sl" in result.columns

    def test_target_uses_future_data(self, feature_df):
        """Verify targets are shifted correctly."""
        gen = TargetGenerator()
        result = gen.add_targets(feature_df, TargetConfig(horizon=5))
        # Last 5 rows should be NaN for tp_before_sl
        assert result["target_tp_before_sl"].iloc[-1] != result["target_tp_before_sl"].iloc[-1]  # NaN

    def test_direction_is_binary(self, feature_df):
        gen = TargetGenerator()
        result = gen.add_targets(feature_df, TargetConfig(horizon=10))
        valid = result["target_direction"].dropna()
        assert set(valid.unique()).issubset({0, 1})
