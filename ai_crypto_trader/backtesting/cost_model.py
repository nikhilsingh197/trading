"""Execution cost model for realistic crypto backtesting.

Models:
1. Dynamic Slippage & Market Impact (order size relative to bar volume depth)
2. Half-spread execution costs
3. Taker & Maker transaction fees
4. Perpetual futures funding rate accrual (8-hour intervals or custom rates)
5. Volume participation limits (partial fills if order > max participation rate)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ai_crypto_trader.core.constants import (
    DEFAULT_MAKER_FEE,
    DEFAULT_SLIPPAGE,
    DEFAULT_TAKER_FEE,
)
from ai_crypto_trader.core.enums import Direction


@dataclass
class CostModelConfig:
    maker_fee: float = DEFAULT_MAKER_FEE          # 0.02%
    taker_fee: float = DEFAULT_TAKER_FEE          # 0.05%
    base_slippage: float = DEFAULT_SLIPPAGE      # 0.03% (3 bps)
    impact_multiplier: float = 0.10              # Market impact scaling factor
    impact_exponent: float = 0.5                 # Square-root law of market impact
    half_spread: float = 0.0001                  # 1 bps half-spread
    max_volume_participation: float = 0.05       # Max 5% of bar volume in one fill
    default_funding_rate_8h: float = 0.0001      # 0.01% per 8 hours (typical baseline)


@dataclass
class ExecutionFill:
    filled_quantity: float
    effective_price: float
    fee_paid: float
    slippage_paid: float
    is_partial: bool
    remaining_quantity: float


class CostModel:
    """Computes realistic execution fills and costs based on market liquidity."""

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()

    def calculate_fill(
        self,
        requested_quantity: float,
        base_price: float,
        direction: Direction,
        bar_volume: float,
        is_taker: bool = True,
    ) -> ExecutionFill:
        """Calculate execution fill price, slippage, and fee for an order.

        Uses the square-root market impact law:
        Slippage = base_slippage + half_spread + impact_multiplier * (qty / bar_vol) ^ exponent
        """
        cfg = self.config

        # 1. Volume participation limit
        max_fill_qty = bar_volume * cfg.max_volume_participation if bar_volume > 0 else requested_quantity
        fill_qty = min(requested_quantity, max_fill_qty)
        is_partial = fill_qty < requested_quantity
        remaining = requested_quantity - fill_qty

        # 2. Market impact slippage
        vol_ratio = fill_qty / (bar_volume + 1e-9)
        dynamic_slippage = cfg.base_slippage + cfg.half_spread + (cfg.impact_multiplier * (vol_ratio ** cfg.impact_exponent))

        if direction == Direction.LONG:
            effective_price = base_price * (1.0 + dynamic_slippage)
        else:
            effective_price = base_price * (1.0 - dynamic_slippage)

        slippage_cost = abs(effective_price - base_price) * fill_qty

        # 3. Transaction fee
        fee_rate = cfg.taker_fee if is_taker else cfg.maker_fee
        fee_paid = fill_qty * effective_price * fee_rate

        return ExecutionFill(
            filled_quantity=fill_qty,
            effective_price=effective_price,
            fee_paid=fee_paid,
            slippage_paid=slippage_cost,
            is_partial=is_partial,
            remaining_quantity=remaining,
        )

    def calculate_funding_payment(
        self,
        position_value: float,
        direction: Direction,
        funding_rate: Optional[float] = None,
    ) -> float:
        """Calculate funding payment for a perpetual futures position.

        Convention:
        - When funding_rate > 0: Longs pay shorts (long cash outflow, short cash inflow).
        - When funding_rate < 0: Shorts pay longs (short cash outflow, long cash inflow).

        Returns:
            Net cash flow for the position (positive = received, negative = paid).
        """
        rate = funding_rate if funding_rate is not None else self.config.default_funding_rate_8h

        if direction == Direction.LONG:
            # Longs pay when rate > 0
            return -position_value * rate
        elif direction == Direction.SHORT:
            # Shorts receive when rate > 0
            return position_value * rate
        return 0.0

    @staticmethod
    def is_funding_hour(ts: datetime) -> bool:
        """Perpetual funding occurs at 00:00, 08:00, and 16:00 UTC."""
        return ts.hour in (0, 8, 16) and ts.minute == 0
