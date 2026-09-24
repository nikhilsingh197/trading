"""Central registry for discovering and loading machine learning models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

from ai_crypto_trader.models.base import BaseMLModel
from ai_crypto_trader.models.logistic_regression import LogisticRegressionModel
from ai_crypto_trader.models.random_forest import RandomForestModel
from ai_crypto_trader.models.xgboost_model import XGBoostModel
from ai_crypto_trader.models.lightgbm_model import LightGBMModel


class ModelRegistry:
    """Registry pattern for model zoo instantiate and deserialization."""

    _models: dict[str, Type[BaseMLModel]] = {
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
    }

    @classmethod
    def register(cls, model_type: str, model_cls: Type[BaseMLModel]) -> None:
        cls._models[model_type.lower()] = model_cls

    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseMLModel]:
        key = model_type.lower()
        if key not in cls._models:
            raise KeyError(
                f"Model type '{model_type}' not found in registry. "
                f"Available: {list(cls._models.keys())}"
            )
        return cls._models[key]

    @classmethod
    def create_model(
        cls,
        model_type: str,
        name: str | None = None,
        params: dict[str, Any] | None = None,
        model_version_id: str | None = None,
    ) -> BaseMLModel:
        model_cls = cls.get_model_class(model_type)
        return model_cls(
            name=name or f"{model_type.capitalize()}Model",
            params=params or {},
            model_version_id=model_version_id,
        )

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._models.keys())

    @classmethod
    def load_model(cls, path: Path | str) -> BaseMLModel:
        """Inspects metadata.json in artifact directory and loads proper subclass."""
        dir_path = Path(path)
        meta_file = dir_path / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Missing metadata.json in {dir_path}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_type = meta.get("model_type", "base").lower()
        model_cls = cls.get_model_class(model_type)
        return model_cls.load(dir_path)
