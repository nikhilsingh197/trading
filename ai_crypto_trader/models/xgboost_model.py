"""XGBoost gradient boosting classifier for market signals."""
from __future__ import annotations

from typing import Any
import xgboost as xgb

from ai_crypto_trader.models.base import BaseMLModel


class XGBoostModel(BaseMLModel):
    """XGBoost gradient boosting decision tree classifier."""

    def __init__(
        self,
        name: str = "XGBoost",
        params: dict[str, Any] | None = None,
        model_version_id: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            model_type="xgboost",
            params=params or {},
            model_version_id=model_version_id,
        )

    def _create_estimator(self) -> xgb.XGBClassifier:
        n_estimators = self.params.get("n_estimators", 100)
        max_depth = self.params.get("max_depth", 4)
        learning_rate = self.params.get("learning_rate", 0.05)
        subsample = self.params.get("subsample", 0.8)
        colsample_bytree = self.params.get("colsample_bytree", 0.8)
        random_state = self.params.get("random_state", 42)

        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="logloss",
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
