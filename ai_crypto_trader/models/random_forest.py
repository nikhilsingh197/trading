"""Random Forest classifier for cryptocurrency market direction prediction."""
from __future__ import annotations

from typing import Any
from sklearn.ensemble import RandomForestClassifier

from ai_crypto_trader.models.base import BaseMLModel


class RandomForestModel(BaseMLModel):
    """Ensemble Random Forest classifier with impurity-based feature importance."""

    def __init__(
        self,
        name: str = "RandomForest",
        params: dict[str, Any] | None = None,
        model_version_id: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            model_type="random_forest",
            params=params or {},
            model_version_id=model_version_id,
        )

    def _create_estimator(self) -> RandomForestClassifier:
        n_estimators = self.params.get("n_estimators", 100)
        max_depth = self.params.get("max_depth", 6)
        min_samples_leaf = self.params.get("min_samples_leaf", 10)
        class_weight = self.params.get("class_weight", "balanced")
        random_state = self.params.get("random_state", 42)

        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    @property
    def feature_importances_(self) -> dict[str, float]:
        if not self.is_fitted or self._estimator is None or not self.feature_names:
            return {}
        if hasattr(self._estimator, "feature_importances_"):
            importances = self._estimator.feature_importances_
            items = sorted(
                zip(self.feature_names, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            return dict(items)
        return {}
