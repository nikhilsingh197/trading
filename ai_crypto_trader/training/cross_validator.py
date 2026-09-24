"""Purged and Embargoed Time-Series Cross-Validation for Financial Machine Learning.

Reference:
    Marcos López de Prado (2018), Advances in Financial Machine Learning.
    - Purging: removes training samples whose target label overlaps with test events.
    - Embargo: removes training samples immediately following test events to eliminate autoregressive memory leakage.
"""
from __future__ import annotations

from typing import Iterator
import numpy as np
import pandas as pd

from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


class PurgedKFold:
    """K-Fold cross-validator with purging and embargoing for overlapping targets.

    Splits the dataset into K contiguous sequential blocks.
    In each fold, one block serves as the test set.
    The training set consists of all other blocks, with:
      1. Samples preceding the test block whose forward horizon overlaps the test block purged.
      2. Samples immediately succeeding the test block within the embargo window removed.
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 12,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Args:
            X: Training features.
            y: Target values (optional).
            groups: Group labels (optional).

        Yields:
            (train_indices, test_indices) arrays.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_bars = int(np.ceil(n_samples * self.embargo_pct))

        # Define K contiguous blocks
        fold_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)

        for fold in range(self.n_splits):
            test_start = fold_bounds[fold]
            test_end = fold_bounds[fold + 1]
            test_idx = indices[test_start:test_end]

            train_mask = np.ones(n_samples, dtype=bool)

            # 1. Mask out test indices
            train_mask[test_start:test_end] = False

            # 2. Purge: training samples before test_start whose horizon overlaps test_start
            purge_start = max(0, test_start - self.horizon)
            train_mask[purge_start:test_start] = False

            # 3. Embargo: training samples immediately after test_end
            embargo_end = min(n_samples, test_end + embargo_bars)
            train_mask[test_end:embargo_end] = False

            train_idx = indices[train_mask]

            if len(train_idx) == 0:
                log.warning("purged_kfold_empty_train_fold", fold=fold)
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits


class PurgedTimeSeriesSplit:
    """Expanding or rolling time-series cross-validator with purging and embargoing.

    Walks forward in time:
      Fold 0: Train [0 : t_split_0 - horizon], Test [t_split_0 : t_split_1]
      Fold 1: Train [0 : t_split_1 - horizon], Test [t_split_1 : t_split_2]
      ...
    """

    def __init__(
        self,
        n_splits: int = 4,
        horizon: int = 12,
        embargo_pct: float = 0.01,
        min_train_pct: float = 0.40,
    ) -> None:
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo_pct = embargo_pct
        self.min_train_pct = min_train_pct

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_bars = int(np.ceil(n_samples * self.embargo_pct))

        min_train_bars = int(n_samples * self.min_train_pct)
        remaining_bars = n_samples - min_train_bars
        test_size = remaining_bars // self.n_splits

        for i in range(self.n_splits):
            test_start = min_train_bars + i * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]

            # Train set is from 0 up to test_start - horizon (purged)
            train_end = max(0, test_start - self.horizon)
            train_idx = indices[:train_end]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx
