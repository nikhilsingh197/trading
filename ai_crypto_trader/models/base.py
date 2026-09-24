"""Base class for all machine learning models in the platform."""
from __future__ import annotations

import abc
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ai_crypto_trader.core.interfaces import MLModelABC
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


class BaseMLModel(MLModelABC, abc.ABC):
    """Abstract base class for all tabular ML trading models."""

    def __init__(
        self,
        name: str = "BaseModel",
        model_type: str = "base",
        params: dict[str, Any] | None = None,
        model_version_id: str | None = None,
    ) -> None:
        self.name = name
        self.model_type = model_type
        self.params = params or {}
        self._model_version_id = model_version_id or str(uuid.uuid4())
        self.feature_names: list[str] = []
        self.is_fitted: bool = False
        self.metrics: dict[str, Any] = {}
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self._estimator: Any = None

    @property
    def model_version_id(self) -> str:
        return self._model_version_id

    @property
    def feature_importances_(self) -> dict[str, float]:
        """Dictionary of feature names mapped to relative importance scores."""
        return {}

    @abc.abstractmethod
    def _create_estimator(self) -> Any:
        """Instantiate underlying estimator with configured params."""
        ...

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> BaseMLModel:
        """Train the model on feature matrix X and target y."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            if not self.feature_names:
                self.feature_names = [f"f_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y)

        if self._estimator is None:
            self._estimator = self._create_estimator()

        self._estimator.fit(X_arr, y_arr)
        self.is_fitted = True

        log.debug(
            "model_fitted",
            model_type=self.model_type,
            n_samples=len(X_arr),
            n_features=len(self.feature_names),
        )
        return self

    def predict_raw(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate raw class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self._estimator.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate class probabilities. Returns array of shape (N, 2)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X_arr)
        # Fallback for models without predict_proba: simulate 0/1 proba
        raw = self.predict_raw(X_arr)
        return np.column_stack([1.0 - raw, raw])

    def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        """Satisfies MLModelABC interface for single-row or batch prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        # Ensure features match expected columns
        if self.feature_names and isinstance(features, pd.DataFrame):
            available = [c for c in self.feature_names if c in features.columns]
            if len(available) < len(self.feature_names):
                missing = set(self.feature_names) - set(available)
                raise ValueError(f"Features missing required columns: {missing}")
            features = features[self.feature_names]

        proba = self.predict_proba(features)
        raw = self.predict_raw(features)

        # If single row
        if len(features) == 1:
            p_up = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            pred_class = int(raw[0])
            confidence = abs(p_up - 0.5) * 2.0
            return {
                "prediction": pred_class,
                "probability": p_up,
                "confidence": confidence,
                "model_version_id": self.model_version_id,
                "model_type": self.model_type,
            }

        # Multi-row batch
        p_up = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        return {
            "prediction": raw.tolist(),
            "probability": p_up.tolist(),
            "confidence": (np.abs(p_up - 0.5) * 2.0).tolist(),
            "model_version_id": self.model_version_id,
            "model_type": self.model_type,
        }

    def save(self, path: Path | str) -> Path:
        """Save model bundle (weights, metadata, feature list) to a directory."""
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 1. Save estimator binary
        estimator_path = dir_path / "estimator.joblib"
        joblib.dump(self._estimator, estimator_path)

        # 2. Save metadata JSON
        meta = {
            "name": self.name,
            "model_type": self.model_type,
            "model_version_id": self._model_version_id,
            "params": self.params,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }
        with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        log.info("model_saved", path=str(dir_path), version_id=self._model_version_id)
        return dir_path

    @classmethod
    def load(cls, path: Path | str) -> BaseMLModel:
        """Load model from saved directory."""
        dir_path = Path(path)
        meta_path = dir_path / "metadata.json"
        estimator_path = dir_path / "estimator.joblib"

        if not meta_path.exists() or not estimator_path.exists():
            raise FileNotFoundError(f"Model artifacts not found in {dir_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls(
            name=meta.get("name", "LoadedModel"),
            params=meta.get("params", {}),
            model_version_id=meta.get("model_version_id"),
        )
        model.feature_names = meta.get("feature_names", [])
        model.is_fitted = meta.get("is_fitted", False)
        model.metrics = meta.get("metrics", {})
        model.created_at = meta.get("created_at", "")
        model._estimator = joblib.load(estimator_path)

        log.info("model_loaded", path=str(dir_path), version_id=model.model_version_id)
        return model
