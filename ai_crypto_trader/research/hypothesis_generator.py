"""Autonomous Hypothesis Generator.

Analyzes performance metrics, regime mismatches, and trade loss clusters to formulate
structured quantitative hypotheses for strategy refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence
import uuid

from ai_crypto_trader.core.enums import MarketRegime
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class Hypothesis:
    id: str
    title: str
    category: str        # 'REGIME_FILTER', 'PARAMETER_TUNE', 'VOLATILITY_ADAPTATION', 'ENSEMBLE'
    rationale: str
    base_strategy: str
    proposed_parameters: dict[str, Any]
    target_metric: str   # 'SHARPE', 'DRAWDOWN', 'WIN_RATE', 'EXPECTANCY'
    expected_improvement: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HypothesisGenerator:
    """Generates quantitative hypotheses by identifying structural weaknesses in performance."""

    def generate_hypotheses_from_metrics(
        self,
        strategy_name: str,
        current_params: dict[str, Any],
        metrics: dict[str, Any],
        trade_history: Optional[Sequence[dict[str, Any]]] = None,
    ) -> list[Hypothesis]:
        """Inspect performance metrics and generate targeted candidate hypotheses."""
        hypotheses: list[Hypothesis] = []
        max_dd = metrics.get("max_drawdown_pct", 0.0)
        win_rate = metrics.get("win_rate", 0.5)
        profit_factor = metrics.get("profit_factor", 1.5)

        # 1. High Drawdown -> Tighten ATR Stop Loss or Add Volatility Filter
        if max_dd > 10.0:
            new_params = dict(current_params)
            current_atr_mult = current_params.get("atr_stop_multiplier", 2.0)
            new_params["atr_stop_multiplier"] = max(1.2, current_atr_mult * 0.8)
            new_params["max_holding_bars"] = 24

            hypotheses.append(
                Hypothesis(
                    id=f"HYP-{uuid.uuid4().hex[:8]}",
                    title="Tighten Stop Multiplier to Curtail Max Drawdown",
                    category="VOLATILITY_ADAPTATION",
                    rationale=f"Observed drawdown of {max_dd:.1f}% exceeds institutional target (10%). Tightening ATR stop from {current_atr_mult}x to {new_params['atr_stop_multiplier']:.1f}x.",
                    base_strategy=strategy_name,
                    proposed_parameters=new_params,
                    target_metric="DRAWDOWN",
                    expected_improvement="Reduce max drawdown by 25-35% without degrading Sharpe.",
                )
            )

        # 2. Low Win Rate in Ranging Markets -> Add ADX / Trend Strength Filter
        if win_rate < 0.45:
            new_params = dict(current_params)
            new_params["adx_threshold"] = 25.0
            new_params["regime_whitelist"] = [MarketRegime.STRONG_UPTREND.value, MarketRegime.STRONG_DOWNTREND.value]

            hypotheses.append(
                Hypothesis(
                    id=f"HYP-{uuid.uuid4().hex[:8]}",
                    title="Filter Ranging Churn with ADX Trend Strength Filter",
                    category="REGIME_FILTER",
                    rationale=f"Win rate of {win_rate * 100:.1f}% indicates frequent whipsaw in ranging regimes. Adding ADX > 25 filter to only trade established momentum.",
                    base_strategy=strategy_name,
                    proposed_parameters=new_params,
                    target_metric="WIN_RATE",
                    expected_improvement="Increase win rate above 50% by suppressing false breakout signals in low volatility.",
                )
            )

        # 3. Sub-optimal Profit Factor -> Increase Risk-to-Reward Ratio (Extend Take Profit)
        if profit_factor < 1.4:
            new_params = dict(current_params)
            current_tp_mult = current_params.get("take_profit_multiplier", 2.0)
            new_params["take_profit_multiplier"] = current_tp_mult * 1.3

            hypotheses.append(
                Hypothesis(
                    id=f"HYP-{uuid.uuid4().hex[:8]}",
                    title="Expand Profit Target to Improve Risk-Reward Ratio",
                    category="PARAMETER_TUNE",
                    rationale=f"Profit factor of {profit_factor:.2f} suggests winners are being cut prematurely relative to average losers. Expanding TP target to {new_params['take_profit_multiplier']:.2f}x.",
                    base_strategy=strategy_name,
                    proposed_parameters=new_params,
                    target_metric="EXPECTANCY",
                    expected_improvement="Elevate profit factor above 1.70 by letting winning trend runners develop.",
                )
            )

        log.info("hypotheses_generated", strategy=strategy_name, count=len(hypotheses))
        return hypotheses
