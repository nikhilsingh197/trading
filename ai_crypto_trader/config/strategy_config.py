"""Strategy-level configuration and scorecard thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_crypto_trader.config.settings import Settings


@dataclass
class ScorecardThresholds:
    """Configurable thresholds for the strategy approval scorecard.

    Statistical rationale for defaults:
    - min_sharpe=0.5: Suggests the strategy earns more than 0.5 units of
      excess return per unit of risk, indicating the edge is non-trivial.
    - min_profit_factor=1.2: Gross profit must exceed gross loss by at
      least 20% to provide buffer against real-world degradation.
    - max_drawdown_pct=15.0: Drawdowns beyond 15% of equity make recovery
      mathematically difficult and psychologically unsustainable.
    - min_win_rate=0.40: Combined with min profit factor, ensures the
      average win is large enough to cover losses.
    - min_trades=50: Below 50 trades, confidence intervals on metrics
      are too wide for meaningful evaluation.
    - min_expectancy=0.0: Positive expectancy is a necessary (not
      sufficient) condition for a viable strategy.
    """
    min_sharpe: float = 0.5
    min_profit_factor: float = 1.2
    max_drawdown_pct: float = 15.0
    min_win_rate: float = 0.40
    min_trades: int = 50
    min_expectancy: float = 0.0
    min_sortino: float = 0.7
    max_consecutive_losses: int = 8

    @classmethod
    def from_settings(cls, settings: "Settings") -> "ScorecardThresholds":
        return cls(
            min_sharpe=settings.scorecard_min_sharpe,
            min_profit_factor=settings.scorecard_min_profit_factor,
            max_drawdown_pct=settings.scorecard_max_drawdown_pct,
            min_win_rate=settings.scorecard_min_win_rate,
            min_trades=settings.scorecard_min_trades,
            min_expectancy=settings.scorecard_min_expectancy,
        )


@dataclass
class PaperTradingConfig:
    """Paper trading advancement criteria."""
    min_period_days: int = 30
    min_trades: int = 100
    require_human_approval: bool = True

    @classmethod
    def from_settings(cls, settings: "Settings") -> "PaperTradingConfig":
        return cls(
            min_period_days=settings.paper_min_period_days,
            min_trades=settings.paper_min_trades,
        )
