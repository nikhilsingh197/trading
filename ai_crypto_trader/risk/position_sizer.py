"""Position sizing algorithms.

All position sizes are computed HERE and nowhere else.
The AI/model layer MUST NOT compute position sizes independently.

Implements:
- Fixed fractional (fixed % of account per trade)
- ATR-based (risk = N * ATR, stop = entry +/- stop_distance)
- Kelly Criterion (capped at KELLY_FRACTION_CAP)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ai_crypto_trader.core.constants import KELLY_FRACTION_CAP
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class SizingResult:
    quantity: float          # Units to trade
    position_value: float    # Dollar value of position
    risk_amount: float       # Dollar amount at risk
    risk_pct: float          # % of equity at risk
    stop_distance: float     # Distance to stop loss
    method: str              # Sizing method used


class PositionSizer:
    """Computes position sizes enforcing hard risk limits."""

    def __init__(
        self,
        max_trade_risk_pct: float,
        max_position_size_pct: float,
    ) -> None:
        self._max_trade_risk_pct = max_trade_risk_pct / 100.0
        self._max_position_size_pct = max_position_size_pct / 100.0

    def fixed_fractional(
        self,
        equity: float,
        entry: float,
        stop_loss: float,
        risk_pct: Optional[float] = None,
    ) -> SizingResult:
        """Size position using fixed fractional (risk a fixed % per trade).

        Args:
            equity: Total account equity.
            entry: Entry price.
            stop_loss: Stop-loss price.
            risk_pct: Risk % of equity (defaults to max_trade_risk_pct).

        Returns:
            SizingResult with quantity and risk metrics.
        """
        effective_risk = min(
            (risk_pct / 100.0) if risk_pct else self._max_trade_risk_pct,
            self._max_trade_risk_pct,
        )

        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            log.warning("zero_stop_distance", entry=entry, stop_loss=stop_loss)
            return self._zero_result("fixed_fractional")

        risk_amount = equity * effective_risk
        raw_quantity = risk_amount / stop_distance

        # Cap at max position size
        max_quantity = (equity * self._max_position_size_pct) / entry
        quantity = min(raw_quantity, max_quantity)

        if quantity <= 0:
            return self._zero_result("fixed_fractional")

        return SizingResult(
            quantity=quantity,
            position_value=quantity * entry,
            risk_amount=quantity * stop_distance,
            risk_pct=quantity * stop_distance / equity * 100,
            stop_distance=stop_distance,
            method="fixed_fractional",
        )

    def kelly(
        self,
        equity: float,
        entry: float,
        stop_loss: float,
        win_rate: float,
        avg_win_loss_ratio: float,
    ) -> SizingResult:
        """Kelly Criterion, capped at KELLY_FRACTION_CAP.

        The Kelly fraction is further capped at max_trade_risk_pct.
        This is a conservative fractional Kelly implementation.

        Args:
            win_rate: Fraction of winning trades [0, 1].
            avg_win_loss_ratio: Average win / average loss.
        """
        if win_rate <= 0 or avg_win_loss_ratio <= 0:
            return self._zero_result("kelly")

        # Full Kelly: f* = W - (1-W)/R
        b = avg_win_loss_ratio
        p = win_rate
        q = 1.0 - p
        full_kelly = p - q / b

        if full_kelly <= 0:
            return self._zero_result("kelly")

        # Apply fraction cap
        fraction = min(full_kelly * KELLY_FRACTION_CAP, self._max_trade_risk_pct)

        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return self._zero_result("kelly")

        risk_amount = equity * fraction
        raw_quantity = risk_amount / stop_distance
        max_quantity = (equity * self._max_position_size_pct) / entry
        quantity = min(raw_quantity, max_quantity)

        if quantity <= 0:
            return self._zero_result("kelly")

        return SizingResult(
            quantity=quantity,
            position_value=quantity * entry,
            risk_amount=quantity * stop_distance,
            risk_pct=quantity * stop_distance / equity * 100,
            stop_distance=stop_distance,
            method=f"kelly({fraction:.3f})",
        )

    @staticmethod
    def _zero_result(method: str) -> SizingResult:
        return SizingResult(
            quantity=0.0,
            position_value=0.0,
            risk_amount=0.0,
            risk_pct=0.0,
            stop_distance=0.0,
            method=method,
        )
