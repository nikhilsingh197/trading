"""Unit tests for regime detection."""
from __future__ import annotations

import pytest

from ai_crypto_trader.core.enums import MarketRegime
from ai_crypto_trader.features.feature_engine import FeatureEngine
from ai_crypto_trader.regimes.rule_based import RuleBasedRegimeDetector


class TestRuleBasedRegimeDetector:
    def test_returns_regime_enum(self, feature_df):
        detector = RuleBasedRegimeDetector()
        regime = detector.detect(feature_df)
        assert isinstance(regime, MarketRegime)

    def test_missing_columns_returns_unknown(self, sample_ohlcv_df):
        """Without features, detector returns UNKNOWN (not an error)."""
        detector = RuleBasedRegimeDetector()
        regime = detector.detect(sample_ohlcv_df)  # No feature columns
        assert regime == MarketRegime.UNKNOWN

    def test_all_possible_regimes_are_valid(self):
        valid = set(MarketRegime)
        for r in valid:
            assert isinstance(r, MarketRegime)
