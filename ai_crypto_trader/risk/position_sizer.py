"""Position sizing algorithms.

All position sizes are computed HERE and nowhere else.
The AI/model layer MUST NOT compute position sizes independently.

Implements:
- Fixed fractional (risk a fixed % of account equity per trade)
- Volatility Parity / ATR-based (normalizes risk across volatility regimes)
- Fractional Kelly Criterion (capped at KELLY_FRACTION_CAP and hard risk limits)
- Portfolio Heat tracking (total percentage of equity exposed to downside risk)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from ai_crypto_trader.core.constants import KELLY_FRACTION_CAP
from ai_crypto_trader.core.interfaces import PositionSnapshot
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class SizingResult:
    quantity: float          # Units to trade
    position_value: float    # Dollar value of position (quantity * entry)
    risk_amount: float       # Dollar amount at risk (quantity * stop_distance)
    risk_pct: float          # % of equity at risk
    stop_distance: float     # Distance to stop loss
    method: str              # Sizing method used


class PositionSizer:
    """Computes position sizes enforcing hard risk limits."""

    def __init__(
        self,
        max_trade_risk_pct: float,
        max_position_size_pct: float,
        max_leverage: float = 1.0,
    ) -> None:
        self._max_trade_risk_pct = max_trade_risk_pct / 100.0
        self._max_position_size_pct = max_position_size_pct / 100.0
        self._max_leverage = max(1.0, max_leverage)

    def fixed_fractional(
        self,
        equity: float,
        entry: float,
        stop_loss: float,
        risk_pct: Optional[float] = None,
        max_leverage: Optional[float] = None,
    ) -> SizingResult:
        """Size position using fixed fractional (risk a fixed % per trade).

        Args:
            equity: Total account equity.
            entry: Entry price.
            stop_loss: Stop-loss price.
            risk_pct: Risk % of equity (defaults to max_trade_risk_pct).
            max_leverage: Maximum leverage override (defaults to self._max_leverage).

        Returns:
            SizingResult with quantity and risk metrics.
        """
        if equity <= 0 or entry <= 0:
            return self._zero_result("fixed_fractional")

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

        # Cap at max position size % of equity
        max_qty_by_size = (equity * self._max_position_size_pct) / entry

        # Cap at max leverage
        lev = max_leverage or self._max_leverage
        max_qty_by_leverage = (equity * lev) / entry

        quantity = min(raw_quantity, max_qty_by_size, max_qty_by_leverage)

        if quantity <= 0:
            return self._zero_result("fixed_fractional")

        actual_risk_amount = quantity * stop_distance
        return SizingResult(
            quantity=quantity,
            position_value=quantity * entry,
            risk_amount=actual_risk_amount,
            risk_pct=(actual_risk_amount / equity) * 100.0,
            stop_distance=stop_distance,
            method="fixed_fractional",
        )

    def volatility_parity(
        self,
        equity: float,
        entry: float,
        atr: float,
        atr_multiplier: float = 2.0,
        risk_pct: Optional[float] = None,
        max_leverage: Optional[float] = None,
    ) -> SizingResult:
        """Size position inversely proportional to market volatility (ATR).

        This equalizes risk across high and low volatility assets.
        Stop distance is established at atr_multiplier * ATR.

        Args:
            equity: Total account equity.
            entry: Entry price.
            atr: Average True Range of the asset.
            atr_multiplier: Multiplier for stop distance (default 2.0).
            risk_pct: Target risk % of equity.
            max_leverage: Maximum allowed leverage.
        """
        if equity <= 0 or entry <= 0 or atr <= 0:
            return self._zero_result("volatility_parity")

        stop_distance = atr * max(0.5, atr_multiplier)
        effective_risk = min(
            (risk_pct / 100.0) if risk_pct else self._max_trade_risk_pct,
            self._max_trade_risk_pct,
        )

        target_risk_amount = equity * effective_risk
        raw_quantity = target_risk_amount / stop_distance

        # Cap at max position size % of equity
        max_qty_by_size = (equity * self._max_position_size_pct) / entry

        # Cap at max leverage
        lev = max_leverage or self._max_leverage
        max_qty_by_leverage = (equity * lev) / entry

        quantity = min(raw_quantity, max_qty_by_size, max_qty_by_leverage)

        if quantity <= 0:
            return self._zero_result("volatility_parity")

        actual_risk_amount = quantity * stop_distance
        return SizingResult(
            quantity=quantity,
            position_value=quantity * entry,
            risk_amount=actual_risk_amount,
            risk_pct=(actual_risk_amount / equity) * 100.0,
            stop_distance=stop_distance,
            method=f"volatility_parity(ATR={atr:.2f}, x{atr_multiplier})",
        )

    def kelly(
        self,
        equity: float,
        entry: float,
        stop_loss: float,
        win_rate: float,
        avg_win_loss_ratio: float,
        max_leverage: Optional[float] = None,
    ) -> SizingResult:
        """Kelly Criterion, capped at KELLY_FRACTION_CAP.

        The Kelly fraction is further capped at max_trade_risk_pct.
        This is a conservative fractional Kelly implementation.

        Args:
            equity: Account equity.
            entry: Entry price.
            stop_loss: Stop-loss price.
            win_rate: Fraction of winning trades [0, 1].
            avg_win_loss_ratio: Average win / average loss.
            max_leverage: Max allowed leverage.
        """
        if win_rate <= 0 or avg_win_loss_ratio <= 0 or equity <= 0 or entry <= 0:
            return self._zero_result("kelly")

        # Full Kelly: f* = W - (1-W)/R
        b = avg_win_loss_ratio
        p = win_rate
        q = 1.0 - p
        full_kelly = p - q / b

        if full_kelly <= 0:
            return self._zero_result("kelly")

        # Apply fraction cap and enforce hard risk limit
        fraction = min(full_kelly * KELLY_FRACTION_CAP, self._max_trade_risk_pct)

        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return self._zero_result("kelly")

        risk_amount = equity * fraction
        raw_quantity = risk_amount / stop_distance

        max_qty_by_size = (equity * self._max_position_size_pct) / entry
        lev = max_leverage or self._max_leverage
        max_qty_by_leverage = (equity * lev) / entry

        quantity = min(raw_quantity, max_qty_by_size, max_qty_by_leverage)

        if quantity <= 0:
            return self._zero_result("kelly")

        actual_risk_amount = quantity * stop_distance
        return SizingResult(
            quantity=quantity,
            position_value=quantity * entry,
            risk_amount=actual_risk_amount,
            risk_pct=(actual_risk_amount / equity) * 100.0,
            stop_distance=stop_distance,
            method=f"kelly(f*={full_kelly:.3f}, cap={fraction:.3f})",
        )

    def compute_portfolio_heat(
        self,
        open_positions: Sequence[PositionSnapshot],
        equity: float,
    ) -> float:
        """Compute aggregate portfolio heat (total % of equity at risk across all open positions).

        For each open position with a stop loss, heat = (quantity * abs(entry - stop_loss)) / equity * 100%.
        If no stop loss is defined, conservative estimate is position_value / equity * 100%.
        """
        if equity <= 0 or not open_positions:
            return 0.0

        total_risk_dollars = 0.0
        for pos in open_positions:
            if pos.stop_loss and pos.stop_loss > 0:
                stop_dist = abs(pos.entry_price - pos.stop_loss)
                total_risk_dollars += pos.quantity * stop_dist
            else:
                total_risk_dollars += pos.quantity * pos.entry_price * self._max_trade_risk_pct

        return (total_risk_dollars / equity) * 100.0

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
