"""Risk configuration dataclass.

Hard risk limits are deliberately separated from the general settings
so they can be reviewed independently. The AI/model layer must NEVER
import this module to modify values at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_crypto_trader.config.settings import Settings


@dataclass(frozen=True)  # frozen = immutable at runtime
class RiskConfig:
    """Immutable snapshot of hard risk limits extracted from Settings.

    Frozen dataclass prevents accidental mutation. The AI/model layer
    receives a RiskConfig instance and CANNOT modify it.
    """

    max_trade_risk_pct: float       # Max % of account equity per trade
    daily_loss_limit_pct: float     # Max % loss per day before halt
    max_drawdown_halt_pct: float    # Drawdown % that triggers kill switch
    max_drawdown_reduce_pct: float  # Drawdown % that halves position size
    max_open_positions: int
    max_leverage: float
    max_position_size_pct: float    # Max % of account in single position
    max_correlated_exposure_pct: float
    max_trades_per_day: int
    max_consecutive_losses: int

    @classmethod
    def from_settings(cls, settings: "Settings") -> "RiskConfig":
        return cls(
            max_trade_risk_pct=settings.risk_max_trade_risk_pct,
            daily_loss_limit_pct=settings.risk_daily_loss_limit_pct,
            max_drawdown_halt_pct=settings.risk_max_drawdown_halt_pct,
            max_drawdown_reduce_pct=settings.risk_max_drawdown_reduce_pct,
            max_open_positions=settings.risk_max_open_positions,
            max_leverage=settings.risk_max_leverage,
            max_position_size_pct=settings.risk_max_position_size_pct,
            max_correlated_exposure_pct=settings.risk_max_correlated_exposure_pct,
            max_trades_per_day=settings.risk_max_trades_per_day,
            max_consecutive_losses=settings.risk_max_consecutive_losses,
        )

    def __repr__(self) -> str:
        return (
            f"RiskConfig("
            f"max_trade_risk={self.max_trade_risk_pct}%, "
            f"daily_loss={self.daily_loss_limit_pct}%, "
            f"max_dd_halt={self.max_drawdown_halt_pct}%)"
        )
