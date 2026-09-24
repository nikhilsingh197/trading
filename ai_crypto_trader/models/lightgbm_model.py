"""LightGBM fast gradient boosting classifier for market signals."""
from __future__ import annotations

from typing import Any
import lightgbm as lgb

from ai_crypto_trader.models.base import BaseMLModel


class LightGBMModel(BaseMLModel):
    """LightGBM gradient boosting classifier."""

    def __init__(
        self,
        name: str = "LightGBM",
        params: dict[str, Any] | None = None,
        model_version_id: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            model_type="lightgbm",
            params=params or {},
            model_version_id=model_version_id,
        )

    def _create_estimator(self) -> lgb.LGBMClassifier:
        n_estimators = self.params.get("n_estimators", 100)
        num_leaves = self.params.get("num_leaves", 31)
        learning_rate = self.params.get("learning_rate", 0.05)
        min_child_samples = self.params.get("min_child_samples", 20)
        random_state = self.params.get("random_state", 42)

        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        )

    @property
    def feature_importances_(self) -> dict[str, float]:
        if not self.is_fitted or self._estimator is None or not self.feature_names:
            return {}
        if hasattr(self._estimator, "feature_importances_"):
            raw = self._estimator.feature_importances_
            total = raw.sum() + 1e-9
            normalized = raw / total
            items = sorted(
                zip(self.feature_names, normalized.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            return dict(items)
        return {}
