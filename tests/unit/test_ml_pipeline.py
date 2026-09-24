"""Comprehensive unit tests for Milestone 7: ML Research Pipeline.

Verifies:
1. Feature selection (variance threshold, collinearity reduction, top-K).
2. Purged & Embargoed Cross-Validation (zero leakage, purging window, embargo window).
3. Model Zoo (LogisticRegression, RandomForest, XGBoost, LightGBM save/load/predict).
4. Model Registry (discovery, factory instantiation, auto-loading).
5. MLSignalStrategy (signal generation, ATR stops, backtest engine integration).
6. End-to-end ModelTrainer and DB persistence.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ai_crypto_trader.backtesting.engine import BacktestConfig, BacktestEngine
from ai_crypto_trader.core.enums import Direction, MarketRegime, SignalAction
from ai_crypto_trader.database.models import Base
from ai_crypto_trader.database.repositories.ai_repo import ModelRepository
from ai_crypto_trader.features.feature_engine import FeatureEngine
from ai_crypto_trader.features.feature_selector import FeatureSelector
from ai_crypto_trader.features.target_generator import TargetConfig, TargetGenerator
from ai_crypto_trader.models.base import BaseMLModel
from ai_crypto_trader.models.lightgbm_model import LightGBMModel
from ai_crypto_trader.models.logistic_regression import LogisticRegressionModel
from ai_crypto_trader.models.random_forest import RandomForestModel
from ai_crypto_trader.models.registry import ModelRegistry
from ai_crypto_trader.models.xgboost_model import XGBoostModel
from ai_crypto_trader.strategies.ml_strategy import MLSignalStrategy
from ai_crypto_trader.training.cross_validator import PurgedKFold, PurgedTimeSeriesSplit
from ai_crypto_trader.training.metrics import evaluate_classification, simulate_model_returns
from ai_crypto_trader.training.trainer import ModelTrainer


# ── Synthetic Dataset Fixture ──────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv_features():
    """Generates 200 synthetic bars with realistic trend, noise, and computed features."""
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")

    returns = np.random.normal(0.0005, 0.015, n)
    price = 50000.0 * np.exp(np.cumsum(returns))

    high = price * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = price * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = price * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.uniform(10.0, 100.0, n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
            "symbol": "BTCUSDT",
        },
        index=dates,
    )

    fe = FeatureEngine()
    enriched = fe.compute(df)
    return enriched


# ── 1. Feature Selector Tests ─────────────────────────────────────────────────

def test_feature_selector_variance_and_collinearity():
    """Verify constant columns are dropped and correlated pairs are reduced."""
    n = 100
    df = pd.DataFrame({
        "constant_col": np.ones(n),
        "feat_a": np.linspace(0, 10, n),
        "feat_b": np.linspace(0, 10, n) + np.random.normal(0, 0.0001, n),  # Correlation ~1.0
        "feat_c": np.random.normal(0, 1, n),
        "target": np.random.choice([0, 1], n),
    })

    selector = FeatureSelector(correlation_threshold=0.95, variance_threshold=1e-5)
    X = df[["constant_col", "feat_a", "feat_b", "feat_c"]]
    y = df["target"]

    X_transformed = selector.fit_transform(X, y)

    assert "constant_col" not in X_transformed.columns
    # Exactly one of feat_a and feat_b should be kept
    assert ("feat_a" in X_transformed.columns) != ("feat_b" in X_transformed.columns)
    assert "feat_c" in X_transformed.columns

    # Test serialization
    data = selector.to_dict()
    loaded = FeatureSelector.from_dict(data)
    assert loaded.selected_features_ == selector.selected_features_


def test_feature_selector_top_k():
    """Verify top-k feature ranking selection."""
    np.random.seed(42)
    n = 150
    X = pd.DataFrame({f"f_{i}": np.random.normal(0, 1, n) for i in range(20)})
    y = pd.Series(np.random.choice([0, 1], n))

    selector = FeatureSelector(top_k=5)
    X_sel = selector.fit_transform(X, y)
    assert X_sel.shape[1] == 5
    assert len(selector.selected_features_) == 5


# ── 2. Purged & Embargoed Cross-Validation Tests ──────────────────────────────

def test_purged_kfold_no_overlap_and_purging():
    """Ensure training and test sets never overlap and purging removes horizon bars."""
    n = 100
    horizon = 5
    pkf = PurgedKFold(n_splits=5, horizon=horizon, embargo_pct=0.02)
    X = np.arange(n).reshape(-1, 1)

    splits = list(pkf.split(X))
    assert len(splits) == 5

    for fold, (train_idx, test_idx) in enumerate(splits):
        # 1. Zero overlap between train and test
        assert len(set(train_idx).intersection(set(test_idx))) == 0

        # 2. Check purging: samples in [test_start - horizon, test_start) must NOT be in train_idx
        test_start = test_idx[0]
        for p in range(max(0, test_start - horizon), test_start):
            assert p not in train_idx

        # 3. Check embargo: samples in [test_end, test_end + embargo] must NOT be in train_idx
        test_end = test_idx[-1] + 1
        embargo_bars = int(np.ceil(n * 0.02))
        for e in range(test_end, min(n, test_end + embargo_bars)):
            assert e not in train_idx


def test_purged_time_series_split():
    """Ensure chronological order and walk-forward expansion."""
    n = 100
    horizon = 4
    pts = PurgedTimeSeriesSplit(n_splits=3, horizon=horizon, min_train_pct=0.40)
    X = np.arange(n)

    splits = list(pts.split(X))
    assert len(splits) == 3

    prev_test_start = 0
    for train_idx, test_idx in splits:
        # All train samples strictly before test samples minus horizon
        assert np.max(train_idx) <= np.min(test_idx) - horizon
        assert np.min(test_idx) > prev_test_start
        prev_test_start = np.min(test_idx)


# ── 3. Model Zoo & Registry Tests ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "model_cls,model_type",
    [
        (LogisticRegressionModel, "logistic_regression"),
        (RandomForestModel, "random_forest"),
        (XGBoostModel, "xgboost"),
        (LightGBMModel, "lightgbm"),
    ],
)
def test_model_zoo_fit_predict_save_load(tmp_path: Path, model_cls, model_type):
    """Test all models for fit, prediction, feature importances, and serialization."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "mom_rsi": np.random.uniform(20, 80, n),
        "trend_macd": np.random.normal(0, 1, n),
        "vol_atr": np.random.uniform(10, 50, n),
    })
    y = pd.Series((X["mom_rsi"] > 50).astype(int))

    model: BaseMLModel = model_cls(params={"random_state": 42})
    model.fit(X, y)

    assert model.is_fitted
    assert len(model.feature_names) == 3

    # Predictions
    raw = model.predict_raw(X)
    assert len(raw) == n
    proba = model.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    # Single-row predict conforming to MLModelABC
    single_res = model.predict(X.iloc[0:1])
    assert "prediction" in single_res
    assert "probability" in single_res
    assert "confidence" in single_res

    # Feature importances
    importances = model.feature_importances_
    assert len(importances) == 3
    assert all(k in importances for k in ["mom_rsi", "trend_macd", "vol_atr"])

    # Serialization & reloading
    save_dir = tmp_path / f"saved_{model_type}"
    model.save(save_dir)

    loaded = ModelRegistry.load_model(save_dir)
    assert loaded.model_type == model_type
    assert loaded.is_fitted
    assert loaded.feature_names == model.feature_names

    # Assert identical predictions after load
    loaded_proba = loaded.predict_proba(X)
    np.testing.assert_allclose(proba, loaded_proba, atol=1e-5)


def test_model_registry_factory():
    """Verify registry listing and dynamic model creation."""
    models = ModelRegistry.list_models()
    assert "xgboost" in models
    assert "lightgbm" in models
    assert "random_forest" in models
    assert "logistic_regression" in models

    mod = ModelRegistry.create_model("xgboost", params={"max_depth": 3})
    assert isinstance(mod, XGBoostModel)
    assert mod.params["max_depth"] == 3


# ── 4. Metrics & Financial Evaluation Tests ───────────────────────────────────

def test_evaluate_classification_metrics():
    """Verify standard classification metrics calculations."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.6, 0.85, 0.1])

    m = evaluate_classification(y_true, y_pred, y_prob)
    assert m["accuracy"] == pytest.approx(5 / 6, rel=1e-2)
    assert 0.0 <= m["brier_score"] <= 1.0
    assert m["roc_auc"] > 0.5


def test_simulate_model_returns():
    """Verify financial trading simulation from predicted probabilities."""
    y_prob = np.array([0.8, 0.8, 0.2, 0.5, 0.5])
    returns = np.array([0.02, 0.01, -0.02, 0.01, -0.01])

    res = simulate_model_returns(y_prob, returns, long_threshold=0.6, short_threshold=0.4, fee_pct=0.0)
    # Bars 0, 1 long (+0.02, +0.01). Bar 2 short (-(-0.02) = +0.02). Bars 3, 4 flat.
    assert res["active_signal_bars"] == 3
    assert res["win_rate"] == 1.0
    assert res["cumulative_return"] > 0.04


# ── 5. MLSignalStrategy Tests ─────────────────────────────────────────────────

def test_ml_signal_strategy_generation(sample_ohlcv_features):
    """Verify MLSignalStrategy produces actionable long/short/no_trade signals."""
    df = sample_ohlcv_features
    # Fit a simple mock model
    features_only = df[[c for c in df.columns if c.startswith("mom_") or c.startswith("vol_atr_")]]
    clean_idx = features_only.dropna().index
    features_clean = features_only.loc[clean_idx]
    y = pd.Series(np.random.choice([0, 1], len(features_clean)), index=clean_idx)

    model = LogisticRegressionModel()
    model.fit(features_clean, y)

    strat = MLSignalStrategy(model=model, params={"long_threshold": 0.55, "short_threshold": 0.45})

    sig = strat.generate_signal(df.iloc[-50:], MarketRegime.RANGING)
    assert isinstance(sig.action, SignalAction)
    assert isinstance(sig.direction, Direction)
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.stop_loss > 0
    assert sig.take_profit > 0


def test_ml_strategy_backtest_integration(sample_ohlcv_features):
    """Verify MLSignalStrategy runs seamlessly in BacktestEngine."""
    df = sample_ohlcv_features
    features_only = df[[c for c in df.columns if c.startswith("mom_") or c.startswith("vol_atr_")]]
    clean_idx = features_only.dropna().index
    features_clean = features_only.loc[clean_idx]
    y = pd.Series(np.random.choice([0, 1], len(features_clean)), index=clean_idx)

    model = RandomForestModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 42})
    model.fit(features_clean, y)

    strat = MLSignalStrategy(model=model, params={"long_threshold": 0.52, "short_threshold": 0.48})
    engine = BacktestEngine(config=BacktestConfig(taker_fee=0.0006))
    result = engine.run(strat, df, initial_capital=10000.0)

    assert result.strategy_version_id == strat.version_id
    assert isinstance(result.total_return, float)
    assert isinstance(result.sharpe, float)


# ── 6. End-to-End ModelTrainer & DB Persistence Tests ─────────────────────────

@pytest.mark.asyncio
async def test_model_trainer_end_to_end_and_db(tmp_path: Path, sample_ohlcv_features):
    """Test full ModelTrainer workflow including Purged CV, evaluation, and DB persistence."""
    df = sample_ohlcv_features
    trainer = ModelTrainer(artifact_base_dir=tmp_path / "models")

    res = trainer.train(
        df=df,
        symbol="BTCUSDT",
        timeframe="1h",
        model_type="xgboost",
        target_type="direction",
        horizon=6,
        cv_folds=3,
        test_size=0.20,
    )

    assert res.model.is_fitted
    assert len(res.selected_features) > 0
    assert "cv_accuracy_mean" in res.cv_metrics
    assert "accuracy" in res.test_metrics
    assert Path(res.artifact_path).exists()
    assert (Path(res.artifact_path) / "training_summary.json").exists()

    # Verify DB persistence
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        await trainer.persist_to_db(session, res)

        repo = ModelRepository(session)
        registered = await repo.register_model("XGBOOST_BTCUSDT", "xgboost")
        latest = await repo.get_latest_version(registered.id)
        assert latest is not None
        assert latest.status == "CANDIDATE"
        assert latest.target == "direction"

    await engine.dispose()
