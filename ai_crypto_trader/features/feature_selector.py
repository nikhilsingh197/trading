"""Feature selection and collinearity reduction for financial machine learning.

CRITICAL: FeatureSelector fits ONLY on the training split to avoid lookahead bias.
Any transformation on validation/test sets uses only parameters learned during fit.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif

from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


class FeatureSelector:
    """Selects predictive and non-redundant features for ML models.

    Steps during fit:
    1. Low-variance filter (drops constant or near-constant columns).
    2. Collinearity filter (drops one of pairs with correlation > threshold).
    3. Top-K ranking (ranks remaining features by tree importance or F-statistic).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.90,
        variance_threshold: float = 1e-6,
        top_k: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.top_k = top_k
        self.random_state = random_state

        self.selected_features_: list[str] = []
        self.feature_importances_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureSelector:
        """Fit selector on training features and optional target."""
        df = X.copy()
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df = df[numeric_cols]

        # 1. Variance filter
        variances = df.var(numeric_only=True)
        valid_cols = variances[variances > self.variance_threshold].index.tolist()
        df = df[valid_cols]

        if df.empty:
            raise ValueError("All features were eliminated by variance threshold.")

        # 2. Collinearity filter
        corr_matrix = df.corr().abs()
        to_drop: set[str] = set()

        columns = df.columns.tolist()
        for i in range(len(columns)):
            col_i = columns[i]
            if col_i in to_drop:
                continue
            for j in range(i + 1, len(columns)):
                col_j = columns[j]
                if col_j in to_drop:
                    continue
                if corr_matrix.loc[col_i, col_j] >= self.correlation_threshold:
                    # Drop the one with lower variance unless target correlation is known
                    var_i = variances[col_i]
                    var_j = variances[col_j]
                    if var_i >= var_j:
                        to_drop.add(col_j)
                    else:
                        to_drop.add(col_i)
                        break

        retained = [c for c in columns if c not in to_drop]
        df_retained = df[retained]

        # 3. Top-K selection (if specified and target provided)
        if self.top_k and len(retained) > self.top_k and y is not None:
            clean_idx = df_retained.dropna().index.intersection(y.dropna().index)
            X_clean = df_retained.loc[clean_idx]
            y_clean = y.loc[clean_idx]

            try:
                tree = ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                tree.fit(X_clean, y_clean)
                importances = dict(zip(retained, tree.feature_importances_))
            except Exception:
                # Fallback to ANOVA F-statistic
                scores, _ = f_classif(X_clean, y_clean)
                scores = np.nan_to_num(scores, nan=0.0)
                importances = dict(zip(retained, scores.tolist()))

            sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            self.selected_features_ = [f for f, _ in sorted_feats[: self.top_k]]
            self.feature_importances_ = dict(sorted_feats[: self.top_k])
        else:
            self.selected_features_ = retained
            self.feature_importances_ = {f: 1.0 for f in retained}

        log.info(
            "features_selected",
            input_features=len(X.columns),
            selected_features=len(self.selected_features_),
            dropped_collinear=len(to_drop),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature DataFrame to include only selected features."""
        if not self.selected_features_:
            raise ValueError("FeatureSelector must be fitted before transforming.")
        missing = [f for f in self.selected_features_ if f not in X.columns]
        if missing:
            raise KeyError(f"Missing required selected features: {missing}")
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration and selected features to dictionary."""
        return {
            "correlation_threshold": self.correlation_threshold,
            "variance_threshold": self.variance_threshold,
            "top_k": self.top_k,
            "selected_features": self.selected_features_,
            "feature_importances": self.feature_importances_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSelector:
        """Instantiate fitted selector from serialized dictionary."""
        selector = cls(
            correlation_threshold=data.get("correlation_threshold", 0.90),
            variance_threshold=data.get("variance_threshold", 1e-6),
            top_k=data.get("top_k"),
        )
        selector.selected_features_ = data.get("selected_features", [])
        selector.feature_importances_ = data.get("feature_importances", {})
        return selector
