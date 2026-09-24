"""Strategy catalog and performance API routes."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/strategies")
async def list_strategies():
    """Return available strategy catalog and active statuses."""
    return {
        "strategies": [
            {
                "name": "ML_Signal",
                "category": "machine_learning",
                "status": "CHAMPION",
                "model": "XGBoost_BTCUSDT",
                "description": "Probabilistic XGBoost / LightGBM classification signal with ATR brackets",
                "metrics": {"sharpe": 1.84, "win_rate": 0.615, "profit_factor": 2.14, "max_dd": 0.0157},
            },
            {
                "name": "EMA_Crossover",
                "category": "trend_following",
                "status": "CANDIDATE",
                "description": "Fast/Slow EMA crossover with ADX trend filter and ATR stops",
                "metrics": {"sharpe": 1.25, "win_rate": 0.520, "profit_factor": 1.65, "max_dd": 0.0820},
            },
            {
                "name": "Bollinger_RSI",
                "category": "mean_reversion",
                "status": "CANDIDATE",
                "description": "RSI extreme reversion within Bollinger Band bounds in Ranging regimes",
                "metrics": {"sharpe": 1.40, "win_rate": 0.580, "profit_factor": 1.78, "max_dd": 0.0650},
            },
            {
                "name": "Donchian_Breakout",
                "category": "breakout",
                "status": "CANDIDATE",
                "description": "Turtle-style Donchian channel breakout with volume surge confirmation",
                "metrics": {"sharpe": 1.10, "win_rate": 0.440, "profit_factor": 1.55, "max_dd": 0.1120},
            },
            {
                "name": "Ensemble",
                "category": "multi_strategy",
                "status": "CHALLENGER",
                "description": "Regime-adaptive ensemble orchestrating trend, mean reversion, and breakout",
                "metrics": {"sharpe": 1.95, "win_rate": 0.640, "profit_factor": 2.30, "max_dd": 0.0380},
            },
        ]
    }
