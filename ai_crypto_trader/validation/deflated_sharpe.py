"""Deflated Sharpe Ratio (DSR) and Multiple Testing Correction.

Based on Bailey and López de Prado (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality".

Adjusts the observed Sharpe ratio downwards based on:
1. The number of trials/configurations tested (N)
2. The variance of Sharpe ratios across all trials
3. The sample length of the backtest (T)
4. Non-normality (skewness and kurtosis of returns)
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def expected_max_sharpe(num_trials: int, variance_trials: float = 0.5) -> float:
    """Compute the expected maximum Sharpe ratio under the null hypothesis of no skill.

    Args:
        num_trials: Number of strategy variations or parameter sets tested (N).
        variance_trials: Variance of Sharpe ratios across the tested trials.

    Returns:
        Expected maximum Sharpe ratio from pure random chance.
    """
    if num_trials <= 1:
        return 0.0

    euler_mascheroni = 0.5772156649
    # Approximation of E[max(z_1, ..., z_N)] for standard normal variables
    z = (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / num_trials) + \
        euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (num_trials * math.e))

    return float(np.sqrt(variance_trials) * z)


def calculate_deflated_sharpe(
    observed_sharpe: float,
    returns: Sequence[float],
    num_trials: int = 10,
    variance_trials: float = 0.5,
) -> float:
    """Calculate the Deflated Sharpe Ratio (p-value / statistical confidence).

    Args:
        observed_sharpe: Non-annualized or annualized Sharpe of candidate strategy.
        returns: Time series of bar or trade percentage returns.
        num_trials: Number of historical strategy backtest variations tried.
        variance_trials: Variance of Sharpe ratios among tried strategies.

    Returns:
        DSR value between 0.0 and 1.0. A value >= 0.95 indicates statistical
        significance at the 5% level after correcting for multiple testing.
    """
    if len(returns) < 5 or observed_sharpe <= 0:
        return 0.0

    r = np.array(returns)
    n = len(r)

    # Skewness and Kurtosis of return distribution
    skew = float(stats.skew(r)) if len(r) > 2 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False)) if len(r) > 3 else 3.0  # Pearson kurtosis (normal = 3)

    sr_star = expected_max_sharpe(num_trials=num_trials, variance_trials=variance_trials)

    # Standard error of Sharpe ratio under non-normality
    # Var(SR) = (1 - skew * SR + (kurt - 1)/4 * SR^2) / (T - 1)
    denom_inner = 1.0 - (skew * observed_sharpe) + (((kurt - 1.0) / 4.0) * (observed_sharpe ** 2))
    denom_inner = max(1e-6, denom_inner)

    se_sr = np.sqrt(denom_inner / (n - 1.0))

    # Standardized test statistic
    z_stat = (observed_sharpe - sr_star) / se_sr

    # Cumulative probability from normal distribution
    dsr = float(stats.norm.cdf(z_stat))
    return max(0.0, min(1.0, dsr))
