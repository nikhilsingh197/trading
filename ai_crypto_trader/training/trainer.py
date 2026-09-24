"""End-to-end ML training orchestrator with Purged Cross-Validation and artifact management."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.database.repositories.ai_repo import ModelRepository
from ai_crypto_trader.features.feature_engine import FeatureEngine
from ai_crypto_trader.features.feature_selector import FeatureSelector
from ai_crypto_trader.features.target_generator import TargetConfig, TargetGenerator
from ai_crypto_trader.models.base import BaseMLModel
from ai_crypto_trader.models.registry import ModelRegistry
from ai_crypto_trader.training.cross_validator import PurgedKFold
from ai_crypto_trader.training.metrics import evaluate_classification, simulate_model_returns

log = get_logger(__name__)


@dataclass
class TrainingResult:
    """Encapsulates the complete result of a model training experiment."""
    model: BaseMLModel
    model_version_id: str
    symbol: str
    timeframe: str
    target: str
    horizon: int
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    simulated_trading: dict[str, Any]
    feature_importances: dict[str, float]
    selected_features: list[str]
    artifact_path: str
    train_bars: int
    test_bars: int


class ModelTrainer:
    """Trains, cross-validates, evaluates, and persists machine learning models."""

    def __init__(
        self,
        artifact_base_dir: str | Path = "data/models",
        feature_engine: FeatureEngine | None = None,
        target_generator: TargetGenerator | None = None,
    ) -> None:
        self.artifact_base_dir = Path(artifact_base_dir)
        self.feature_engine = feature_engine or FeatureEngine()
        self.target_generator = target_generator or TargetGenerator()

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        target_type: str = "direction",
        horizon: int = 12,
        tp_pct: float = 2.0,
        sl_pct: float = 1.0,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Compute features, generate targets, quarantine leakages, and drop NaNs.

        Returns:
            (X, y, forward_returns)
        """
        # 1. Feature computation if needed
        enriched = df.copy()
        if not any(c.startswith("trend_") or c.startswith("mom_") for c in enriched.columns):
            enriched = self.feature_engine.compute(enriched)

        # 2. Target generation
        config = TargetConfig(horizon=horizon, tp_pct=tp_pct, sl_pct=sl_pct)
        enriched = self.target_generator.add_targets(enriched, config)

        # 3. Select target column
        target_mapping = {
            "direction": "target_direction",
            "tp_before_sl": "target_tp_before_sl",
            "significant_up": "target_significant_up",
            "significant_down": "target_significant_down",
        }
        target_col = target_mapping.get(target_type, target_type)
        if target_col not in enriched.columns:
            raise ValueError(f"Target column '{target_col}' not found in prepared data.")

        # 4. Quarantine all target columns out of feature matrix
        raw_cols = {
            "open", "high", "low", "close", "volume", "quote_volume",
            "trade_count", "quality_flag", "symbol", "timeframe", "source", "id",
        }
        target_cols = {c for c in enriched.columns if c.startswith("target_")}
        feature_cols = [
            c for c in enriched.columns
            if c not in raw_cols and c not in target_cols and pd.api.types.is_numeric_dtype(enriched[c])
        ]

        # 5. Extract X, y, forward_returns and drop NaNs
        clean_mask = enriched[feature_cols].notna().all(axis=1) & enriched[target_col].notna()
        clean_df = enriched[clean_mask]

        X = clean_df[feature_cols].copy()
        y = clean_df[target_col].astype(int).copy()
        forward_returns = clean_df["target_future_return"].fillna(0.0).copy()

        log.info(
            "dataset_prepared",
            total_bars=len(df),
            clean_bars=len(X),
            feature_count=len(feature_cols),
            target=target_col,
            positive_class_ratio=float(y.mean()),
        )
        return X, y, forward_returns

    def train(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        model_type: str = "xgboost",
        target_type: str = "direction",
        horizon: int = 12,
        cv_folds: int = 5,
        test_size: float = 0.20,
        hyperparams: dict[str, Any] | None = None,
        feature_selector: FeatureSelector | None = None,
        session: AsyncSession | None = None,
    ) -> TrainingResult:
        """Run full training, Purged CV, test evaluation, and artifact persistence."""
        X, y, forward_returns = self.prepare_dataset(
            df=df, target_type=target_type, horizon=horizon
        )

        n_samples = len(X)
        if n_samples < 50:
            raise ValueError(f"Insufficient samples for training ({n_samples} < 50).")

        # 1. Train / Test chronological split
        test_bars = int(n_samples * test_size)
        train_bars = n_samples - test_bars

        # Purge boundary: remove last `horizon` samples of training set
        purged_train_end = max(10, train_bars - horizon)
        train_idx = np.arange(0, purged_train_end)
        test_idx = np.arange(train_bars, n_samples)

        X_train_raw = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        ret_test = forward_returns.iloc[test_idx]

        # 2. Feature Selection (fitted strictly on train set)
        if feature_selector is None:
            feature_selector = FeatureSelector(
                correlation_threshold=0.92,
                variance_threshold=1e-6,
                top_k=min(30, len(X.columns)),
            )

        X_train = feature_selector.fit_transform(X_train_raw, y_train)
        X_test = feature_selector.transform(X_test_raw)
        selected_features = feature_selector.selected_features_

        # 3. Purged K-Fold Cross-Validation on training set
        pkf = PurgedKFold(n_splits=cv_folds, horizon=horizon, embargo_pct=0.01)
        fold_metrics: list[dict[str, float]] = []

        for fold, (cv_train_idx, cv_val_idx) in enumerate(pkf.split(X_train, y_train)):
            cv_model = ModelRegistry.create_model(
                model_type=model_type,
                name=f"{model_type}_fold_{fold}",
                params=hyperparams,
            )
            cv_model.fit(X_train.iloc[cv_train_idx], y_train.iloc[cv_train_idx])

            val_preds = cv_model.predict_raw(X_train.iloc[cv_val_idx])
            val_probs = cv_model.predict_proba(X_train.iloc[cv_val_idx])[:, 1]

            m = evaluate_classification(
                y_true=y_train.iloc[cv_val_idx],
                y_pred=val_preds,
                y_prob=val_probs,
            )
            fold_metrics.append(m)

        # Aggregate CV metrics
        cv_summary: dict[str, float] = {}
        if fold_metrics:
            keys = fold_metrics[0].keys()
            for k in keys:
                cv_summary[f"cv_{k}_mean"] = float(np.mean([m[k] for m in fold_metrics]))
                cv_summary[f"cv_{k}_std"] = float(np.std([m[k] for m in fold_metrics]))

        # 4. Final Model Training on complete train set
        final_model = ModelRegistry.create_model(
            model_type=model_type,
            name=f"{model_type}_{symbol}_{timeframe}",
            params=hyperparams,
        )
        final_model.fit(X_train, y_train)

        # 5. Evaluate on Holdout Test Set
        test_preds = final_model.predict_raw(X_test)
        test_probs = final_model.predict_proba(X_test)[:, 1]

        test_metrics = evaluate_classification(
            y_true=y_test,
            y_pred=test_preds,
            y_prob=test_probs,
        )

        simulated_trading = simulate_model_returns(
            y_prob=test_probs,
            forward_returns=ret_test,
            long_threshold=0.55,
            short_threshold=0.45,
        )

        final_model.metrics = {
            "cv_metrics": cv_summary,
            "test_metrics": test_metrics,
            "simulated_trading": simulated_trading,
        }

        # 6. Save Artifacts
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        artifact_dir = self.artifact_base_dir / f"{model_type}_{symbol}_{timestamp}"
        final_model.save(artifact_dir)

        # Save feature selector configuration
        with open(artifact_dir / "feature_selector.json", "w", encoding="utf-8") as f:
            json.dump(feature_selector.to_dict(), f, indent=2)

        # Save summary report
        summary = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_type": model_type,
            "model_version_id": final_model.model_version_id,
            "target": target_type,
            "horizon": horizon,
            "train_bars": len(X_train),
            "test_bars": len(X_test),
            "cv_metrics": cv_summary,
            "test_metrics": test_metrics,
            "simulated_trading": simulated_trading,
            "feature_importances": final_model.feature_importances_,
            "selected_features": selected_features,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(artifact_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return TrainingResult(
            model=final_model,
            model_version_id=final_model.model_version_id,
            symbol=symbol,
            timeframe=timeframe,
            target=target_type,
            horizon=horizon,
            cv_metrics=cv_summary,
            test_metrics=test_metrics,
            simulated_trading=simulated_trading,
            feature_importances=final_model.feature_importances_,
            selected_features=selected_features,
            artifact_path=str(artifact_dir),
            train_bars=len(X_train),
            test_bars=len(X_test),
        )

    async def persist_to_db(
        self,
        session: AsyncSession,
        result: TrainingResult,
    ) -> None:
        """Persist model and model version in database."""
        repo = ModelRepository(session)
        model = await repo.register_model(
            name=f"{result.model.model_type.upper()}_{result.symbol}",
            model_type=result.model.model_type,
            description=f"ML model for {result.symbol} {result.timeframe} forecasting {result.target}",
        )

        await repo.create_version(
            model_id=model.id,
            version=result.model_version_id[:8],
            artifact_path=result.artifact_path,
            params=result.model.params,
            feature_list=result.selected_features,
            target=result.target,
            val_metrics=result.cv_metrics,
            test_metrics=result.test_metrics,
            status="CANDIDATE",
        )
        await session.commit()
        log.info("model_persisted_to_db", model_id=str(model.id), version=result.model_version_id)
