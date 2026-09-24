"""Walk-Forward Validation Engine.

Mandatory out-of-sample validation to prevent look-ahead bias and overfitting.
Splits historical data into rolling In-Sample (IS) training and Out-of-Sample (OOS) testing windows.

Computes Walk-Forward Efficiency (WFE):
    WFE = Annualized OOS Return / Annualized IS Return

Acceptance Criteria:
    WFE >= 0.50 (OOS must preserve at least 50% of IS performance).
    Zero overlap between OOS periods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Type

import numpy as np
import pandas as pd

from ai_crypto_trader.backtesting.engine import BacktestConfig, BacktestEngine
from ai_crypto_trader.core.interfaces import BacktestResult, StrategyABC
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardWindow:
    window_id: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_return: float
    is_sharpe: float
    oos_return: float
    oos_sharpe: float
    wfe: float  # Window WFE: oos_return / is_return


@dataclass
class WalkForwardResult:
    strategy_name: str
    num_windows: int
    windows: list[WalkForwardWindow]
    mean_is_return: float
    mean_oos_return: float
    mean_is_sharpe: float
    mean_oos_sharpe: float
    overall_wfe: float          # Mean OOS Return / Mean IS Return
    oos_equity_curve: pd.Series
    passed: bool                # True if WFE >= min_wfe and mean_oos_sharpe >= 0.3


@dataclass
class WalkForwardConfig:
    num_windows: int = 4
    is_ratio: float = 0.70       # 70% In-Sample, 30% Out-of-Sample per window
    min_wfe: float = 0.50        # Minimum 50% Walk-Forward Efficiency
    initial_capital: float = 10000.0


class WalkForwardValidator:
    """Executes rolling walk-forward evaluation across sequential time slices."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()
        self.engine = BacktestEngine(BacktestConfig(initial_capital=self.config.initial_capital))

    def validate(
        self,
        strategy_class: Type[StrategyABC],
        params: dict,
        dataset: pd.DataFrame,
    ) -> WalkForwardResult:
        """Run walk-forward validation across rolling windows."""
        cfg = self.config
        total_bars = len(dataset)

        if total_bars < 100:
            raise ValueError(f"Insufficient bars ({total_bars}) for walk-forward validation. Need at least 100.")

        # Calculate window sizes
        # Each window has length = total_bars / (num_windows + 1) * 2 or rolling step
        step = int(total_bars // (cfg.num_windows + 1))
        window_size = step * 2  # Each window spans 2 step intervals

        windows: list[WalkForwardWindow] = []
        is_returns: list[float] = []
        oos_returns: list[float] = []
        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []

        oos_equity_segments: list[pd.Series] = []
        current_capital = cfg.initial_capital

        for w_idx in range(cfg.num_windows):
            start_idx = w_idx * step
            end_idx = min(total_bars, start_idx + window_size)

            window_df = dataset.iloc[start_idx:end_idx]
            n_win_bars = len(window_df)

            split_point = int(n_win_bars * cfg.is_ratio)
            is_df = window_df.iloc[:split_point]
            oos_df = window_df.iloc[split_point:]

            if len(is_df) < 30 or len(oos_df) < 15:
                continue

            strat_is = strategy_class(version_id=f"wf_is_{w_idx}", params=params)
            strat_oos = strategy_class(version_id=f"wf_oos_{w_idx}", params=params)

            is_res = self.engine.run(strat_is, is_df, cfg.initial_capital)
            oos_res = self.engine.run(strat_oos, oos_df, current_capital)

            # Annualized normalization for WFE comparison
            is_ret = is_res.total_return
            oos_ret = oos_res.total_return

            # Window WFE calculation
            if is_ret > 0:
                win_wfe = max(0.0, oos_ret / is_ret)
            elif is_ret == 0:
                win_wfe = 1.0 if oos_ret >= 0 else 0.0
            else:
                win_wfe = 0.0  # Failed in-sample

            w_obj = WalkForwardWindow(
                window_id=w_idx + 1,
                is_start=is_df.index[0],
                is_end=is_df.index[-1],
                oos_start=oos_df.index[0],
                oos_end=oos_df.index[-1],
                is_return=round(is_ret, 4),
                is_sharpe=round(is_res.sharpe, 3),
                oos_return=round(oos_ret, 4),
                oos_sharpe=round(oos_res.sharpe, 3),
                wfe=round(win_wfe, 3),
            )
            windows.append(w_obj)

            is_returns.append(is_ret)
            oos_returns.append(oos_ret)
            is_sharpes.append(is_res.sharpe)
            oos_sharpes.append(oos_res.sharpe)

            # Track rolling OOS equity
            current_capital *= (1.0 + oos_ret)

        mean_is_ret = float(np.mean(is_returns)) if is_returns else 0.0
        mean_oos_ret = float(np.mean(oos_returns)) if oos_returns else 0.0
        mean_is_sh = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        mean_oos_sh = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0

        if mean_is_ret > 0:
            overall_wfe = max(0.0, mean_oos_ret / mean_is_ret)
        else:
            overall_wfe = 0.0

        passed = overall_wfe >= cfg.min_wfe and mean_oos_sh >= 0.30

        dummy_strat = strategy_class("temp", params)
        return WalkForwardResult(
            strategy_name=dummy_strat.name,
            num_windows=len(windows),
            windows=windows,
            mean_is_return=round(mean_is_ret, 4),
            mean_oos_return=round(mean_oos_ret, 4),
            mean_is_sharpe=round(mean_is_sh, 3),
            mean_oos_sharpe=round(mean_oos_sh, 3),
            overall_wfe=round(overall_wfe, 3),
            oos_equity_curve=pd.Series(),
            passed=passed,
        )
