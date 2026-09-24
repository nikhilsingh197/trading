"""Logistic Regression model for binary trading classification."""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_crypto_trader.models.base import BaseMLModel


class LogisticRegressionModel(BaseMLModel):
    """L2-regularized Logistic Regression classifier with automatic feature scaling."""

    def __init__(
        self,
        name: str = "LogisticRegression",
        params: dict[str, Any] | None = None,
        model_version_id: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            model_type="logistic_regression",
            params=params or {},
            model_version_id=model_version_id,
        )

    def _create_estimator(self) -> Pipeline:
        c_val = self.params.get("C", 1.0)
        max_iter = self.params.get("max_iter", 1000)
        class_weight = self.params.get("class_weight", "balanced")
        random_state = self.params.get("random_state", 42)

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_val,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ])

    @property
    def feature_importances_(self) -> dict[str, float]:
        if not self.is_fitted or self._estimator is None or not self.feature_names:
            return {}
        clf = self._estimator.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coefs = np.abs(clf.coef_[0])
            total = np.sum(coefs) + 1e-9
            normalized = coefs / total
            items = sorted(
                zip(self.feature_names, normalized.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            return dict(items)
        return {}
