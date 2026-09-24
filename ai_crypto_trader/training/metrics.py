"""Evaluation metrics for financial machine learning models."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: True binary target labels (0 or 1).
        y_pred: Predicted class labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (1).

    Returns:
        Dictionary of computed metric names and values.
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
    }

    if y_prob is not None:
        ypr = np.asarray(y_prob)
        # Check if both classes are present in yt before computing ROC-AUC
        if len(np.unique(yt)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(yt, ypr))
            except Exception:
                metrics["roc_auc"] = 0.5
            try:
                metrics["log_loss"] = float(log_loss(yt, ypr, eps=1e-7))
            except Exception:
                metrics["log_loss"] = 1.0
            try:
                metrics["brier_score"] = float(brier_score_loss(yt, ypr))
            except Exception:
                metrics["brier_score"] = 0.25
        else:
            metrics["roc_auc"] = 0.5
            metrics["log_loss"] = 0.0
            metrics["brier_score"] = 0.0

    return metrics


def simulate_model_returns(
    y_prob: pd.Series | np.ndarray,
    forward_returns: pd.Series | np.ndarray,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    fee_pct: float = 0.0006,
) -> dict[str, Any]:
    """Simulate strategy returns based on model predicted probabilities.

    Position rule:
      Long (+1) if P(up) >= long_threshold
      Short (-1) if P(up) <= short_threshold
      Flat (0) otherwise

    Returns:
        Dictionary with simulated strategy statistics.
    """
    prob = np.asarray(y_prob)
    ret = np.asarray(forward_returns)

    signals = np.zeros_like(prob)
    signals[prob >= long_threshold] = 1.0
    signals[prob <= short_threshold] = -1.0

    # Number of trades
    trades = np.abs(np.diff(np.insert(signals, 0, 0))) > 0
    trade_count = int(np.sum(trades))

    # Gross return per bar
    strat_returns = signals * ret

    # Apply transaction fees when position changes
    fees = trades * fee_pct
    net_returns = strat_returns - fees

    cumulative_return = float(np.prod(1.0 + net_returns) - 1.0)
    market_return = float(np.prod(1.0 + ret) - 1.0)

    # Win rate on active signal bars
    active_mask = signals != 0
    if np.sum(active_mask) > 0:
        active_returns = strat_returns[active_mask]
        win_rate = float(np.mean(active_returns > 0))
        gains = active_returns[active_returns > 0]
        losses = -active_returns[active_returns < 0]
        sum_gains = float(np.sum(gains)) if len(gains) > 0 else 0.0
        sum_losses = float(np.sum(losses)) if len(losses) > 0 else 1e-9
        profit_factor = sum_gains / sum_losses
        expectancy = float(np.mean(active_returns))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        expectancy = 0.0

    return {
        "cumulative_return": cumulative_return,
        "market_return": market_return,
        "trade_count": trade_count,
        "active_signal_bars": int(np.sum(active_mask)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }
