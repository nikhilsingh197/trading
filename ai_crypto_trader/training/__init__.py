"""Model training package."""
from ai_crypto_trader.training.cross_validator import PurgedKFold, PurgedTimeSeriesSplit
from ai_crypto_trader.training.metrics import evaluate_classification, simulate_model_returns
from ai_crypto_trader.training.trainer import ModelTrainer, TrainingResult

__all__ = [
    "PurgedKFold",
    "PurgedTimeSeriesSplit",
    "evaluate_classification",
    "simulate_model_returns",
    "ModelTrainer",
    "TrainingResult",
]
