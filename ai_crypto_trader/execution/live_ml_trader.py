"""Live Paper Trading Engine powered by Champion XGBoost ML Model.

Fetches real candles from Binance up to the present moment, feeds them through
FeatureEngine and MLSignalStrategy (XGBoost), executes paper orders, and keeps running.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import ccxt.async_support as ccxt
import pandas as pd

from ai_crypto_trader.core.interfaces import Candle
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.models.xgboost_model import XGBoostModel
from ai_crypto_trader.paper_trading.paper_session import PaperSession
from ai_crypto_trader.strategies.ml_strategy import MLSignalStrategy

log = get_logger("live_ml_trader")

BINANCE_SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"


def _resolve_model_dir() -> Path:
    candidates = [
        Path("data/models/xgboost_BTCUSDT_20260924_073907"),
        Path("C:/Users/singh/Downloads/trading/data/models/xgboost_BTCUSDT_20260924_073907"),
    ]
    for c in candidates:
        if c.exists() and (c / "estimator.joblib").exists():
            return c

    models_dir = Path("data/models")
    if models_dir.exists():
        for d in sorted(models_dir.glob("xgboost*"), reverse=True):
            if (d / "estimator.joblib").exists():
                return d
        for d in sorted(models_dir.glob("lightgbm*"), reverse=True):
            if (d / "estimator.joblib").exists():
                return d
    return Path("data/models/xgboost_BTCUSDT_20260924_073907")


class LiveMLTrader:
    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.initial_capital = initial_capital
        model_dir = _resolve_model_dir()
        log.info("loading_champion_ml_model", path=str(model_dir))
        try:
            self.model = XGBoostModel.load(model_dir)
        except Exception as exc:
            log.warning("failed_loading_saved_model_using_fresh_instance", error=str(exc))
            self.model = XGBoostModel()
        
        # ML Strategy configured to trigger trades
        self.strategy = MLSignalStrategy(
            model=self.model,
            version_id="xgboost_champion_live",
            params={
                "long_threshold": 0.52,
                "short_threshold": 0.48,
                "min_confidence": 0.05,
                "atr_sl_multiplier": 1.5,
                "tp_ratio": 2.0,
            },
        )
        self.session = PaperSession(
            strategy=self.strategy,
            symbol="BTCUSDT",
            timeframe=TIMEFRAME,
            initial_capital=initial_capital,
            risk_per_trade_pct=0.015,
            max_position_pct=0.25,
        )
        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.last_processed_candle_time: datetime | None = None

    async def backfill_and_sync(self, limit: int = 500) -> int:
        """Fetch historical candles up to the current hour and process through ML model."""
        log.info("fetching_candles_from_binance", symbol=BINANCE_SYMBOL, timeframe=TIMEFRAME, limit=limit)
        ohlcv = await self.exchange.fetch_ohlcv(BINANCE_SYMBOL, TIMEFRAME, limit=limit)
        
        # We process all closed candles (exclude the last one if it is still open)
        closed_bars = ohlcv[:-1]
        log.info("received_closed_candles", count=len(closed_bars))

        processed = 0
        for row in closed_bars:
            ts_ms, o, h, l, c, v = row
            open_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            close_time = datetime.fromtimestamp((ts_ms + 3600 * 1000) / 1000, tz=timezone.utc)

            candle = Candle(
                symbol="BTCUSDT",
                timeframe=TIMEFRAME,
                open_time=open_time,
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
                close_time=close_time,
            )

            signal, orders = await self.session.step(candle)
            self.last_processed_candle_time = open_time
            processed += 1

            if orders:
                for o in orders:
                    log.info(
                        "ml_trade_executed",
                        open_time=open_time.isoformat(),
                        side=o.status.value,
                        qty=o.filled_qty,
                        fill_price=o.avg_fill_price,
                        equity=round(self.session.broker.portfolio.total_equity, 2),
                    )

        state = await self.session.broker.get_portfolio_state()
        trades = self.session.broker.portfolio.trades_history
        log.info(
            "backfill_sync_complete",
            total_candles=processed,
            total_trades=len(trades),
            current_equity=round(state.total_equity, 2),
            realized_pnl=round(state.realized_pnl, 2),
            unrealized_pnl=round(state.unrealized_pnl, 2),
            open_positions=len(state.open_positions),
        )
        return processed

    async def run_loop(self) -> None:
        """Continuously polls for new closed candles and feeds them to the ML model."""
        log.info("starting_live_ml_trading_loop")
        try:
            while True:
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(BINANCE_SYMBOL, TIMEFRAME, limit=3)
                    if len(ohlcv) >= 2:
                        last_closed = ohlcv[-2]
                        ts_ms, o, h, l, c, v = last_closed
                        open_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

                        if self.last_processed_candle_time is None or open_time > self.last_processed_candle_time:
                            close_time = datetime.fromtimestamp((ts_ms + 3600 * 1000) / 1000, tz=timezone.utc)
                            candle = Candle(
                                symbol="BTCUSDT",
                                timeframe=TIMEFRAME,
                                open_time=open_time,
                                open=float(o),
                                high=float(h),
                                low=float(l),
                                close=float(c),
                                volume=float(v),
                                close_time=close_time,
                            )
                            sig, orders = await self.session.step(candle)
                            self.last_processed_candle_time = open_time
                            state = await self.session.broker.get_portfolio_state()
                            log.info(
                                "new_candle_processed",
                                time=open_time.isoformat(),
                                close=c,
                                signal=sig.action.value if sig else "NONE",
                                orders=len(orders),
                                equity=round(state.total_equity, 2),
                            )

                    await asyncio.sleep(60)  # Check every 60s
                except Exception as e:
                    log.warning("poll_error_retrying", error=str(e))
                    await asyncio.sleep(30)
        finally:
            await self.exchange.close()


_global_live_trader: LiveMLTrader | None = None


def get_live_trader() -> LiveMLTrader:
    """Return the global singleton LiveMLTrader instance."""
    global _global_live_trader
    if _global_live_trader is None:
        _global_live_trader = LiveMLTrader()
    return _global_live_trader


async def main():
    trader = get_live_trader()
    await trader.backfill_and_sync(limit=500)
    await trader.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
