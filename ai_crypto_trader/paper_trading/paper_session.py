"""End-to-end paper trading session coordinating data ingestion, strategy signals, broker execution, and monitoring."""
from __future__ import annotations

from typing import Sequence
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ai_crypto_trader.core.enums import Direction, MarketRegime, Side, SignalAction
from ai_crypto_trader.core.interfaces import Candle, OrderRequest, OrderResult, Signal, StrategyABC
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.database.repositories.trading_repo import (
    OrderRepository,
    PortfolioSnapshotRepository,
    PositionRepository,
    TradeRepository,
)
from ai_crypto_trader.features.feature_engine import FeatureEngine
from ai_crypto_trader.paper_trading.paper_broker import PaperBroker
from ai_crypto_trader.paper_trading.paper_monitor import PaperMonitor, PaperTradingStats
from ai_crypto_trader.regimes.rule_based import RuleBasedRegimeDetector

log = get_logger(__name__)


class PaperSession:
    """Executes a paper trading session against real-time or replayed candle data."""

    def __init__(
        self,
        strategy: StrategyABC,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        initial_capital: float = 10_000.0,
        risk_per_trade_pct: float = 0.01,
        max_position_pct: float = 0.20,
        feature_engine: FeatureEngine | None = None,
        regime_detector: RuleBasedRegimeDetector | None = None,
        broker: PaperBroker | None = None,
        monitor: PaperMonitor | None = None,
    ) -> None:
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct

        self.feature_engine = feature_engine or FeatureEngine()
        self.regime_detector = regime_detector or RuleBasedRegimeDetector()
        self.broker = broker or PaperBroker(initial_capital=initial_capital)
        self.monitor = monitor or PaperMonitor()

        self._candle_buffer: list[dict] = []
        self._max_buffer_size = 500

    async def step(self, candle: Candle) -> tuple[Signal | None, list[OrderResult]]:
        """Process a single incoming candle, update stops, generate signals, and execute orders."""
        # 1. Update broker prices and process any active SL/TP triggers
        completed_trades = self.broker.on_candle(candle)

        # 2. Append candle to rolling buffer
        candle_dict = {
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": float(candle.volume),
            "symbol": candle.symbol,
            "open_time": candle.open_time,
        }
        self._candle_buffer.append(candle_dict)
        if len(self._candle_buffer) > self._max_buffer_size:
            self._candle_buffer.pop(0)

        # Wait until sufficient history is available to compute features (at least 30 bars)
        if len(self._candle_buffer) < 30:
            return None, []

        df_raw = pd.DataFrame(self._candle_buffer)
        df_raw["open_time"] = pd.to_datetime(df_raw["open_time"], utc=True)
        df_raw = df_raw.set_index("open_time")

        # 3. Compute Features and Regime
        try:
            features_df = self.feature_engine.compute(df_raw)
            regime = self.regime_detector.detect(features_df)
        except Exception as e:
            log.warning("paper_step_feature_error", error=str(e))
            return None, []

        # 4. Generate Strategy Signal
        signal = self.strategy.generate_signal(features_df, regime)

        # 5. Position Sizing and Order Routing
        portfolio_state = await self.broker.get_portfolio_state()
        equity = portfolio_state.total_equity
        entry_price = float(candle.close)
        orders_executed: list[OrderResult] = []

        existing_pos = self.broker.portfolio.get_position(self.symbol)

        if signal.action == SignalAction.ENTER_LONG and existing_pos is None:
            # Calculate position size based on risk per trade
            sl_distance = abs(entry_price - signal.stop_loss)
            risk_budget = equity * self.risk_per_trade_pct
            qty = risk_budget / max(1e-4, sl_distance)
            # Cap at max position percentage
            max_qty = (equity * self.max_position_pct) / entry_price
            qty = min(qty, max_qty)

            if qty * entry_price >= 10.0:  # Minimum notional threshold ($10)
                req = OrderRequest(
                    symbol=self.symbol,
                    side=Side.BUY,
                    order_type=signal.metadata.get("order_type", Side.BUY and "MARKET"),  # type: ignore
                    quantity=qty,
                    price=signal.take_profit,
                    stop_price=signal.stop_loss,
                    strategy_id=self.strategy.version_id,
                )
                res = await self.broker.submit_order(req)
                orders_executed.append(res)

        elif signal.action == SignalAction.ENTER_SHORT and existing_pos is None:
            sl_distance = abs(signal.stop_loss - entry_price)
            risk_budget = equity * self.risk_per_trade_pct
            qty = risk_budget / max(1e-4, sl_distance)
            max_qty = (equity * self.max_position_pct) / entry_price
            qty = min(qty, max_qty)

            if qty * entry_price >= 10.0:
                req = OrderRequest(
                    symbol=self.symbol,
                    side=Side.SELL,
                    order_type=signal.metadata.get("order_type", "MARKET"),  # type: ignore
                    quantity=qty,
                    price=signal.take_profit,
                    stop_price=signal.stop_loss,
                    strategy_id=self.strategy.version_id,
                )
                res = await self.broker.submit_order(req)
                orders_executed.append(res)

        elif signal.action == SignalAction.EXIT_LONG and existing_pos is not None and existing_pos.side == Direction.LONG:
            req = OrderRequest(
                symbol=self.symbol,
                side=Side.SELL,
                order_type=signal.metadata.get("order_type", "MARKET"),  # type: ignore
                quantity=existing_pos.quantity,
                strategy_id=self.strategy.version_id,
            )
            res = await self.broker.submit_order(req)
            orders_executed.append(res)

        elif signal.action == SignalAction.EXIT_SHORT and existing_pos is not None and existing_pos.side == Direction.SHORT:
            req = OrderRequest(
                symbol=self.symbol,
                side=Side.BUY,
                order_type=signal.metadata.get("order_type", "MARKET"),  # type: ignore
                quantity=existing_pos.quantity,
                strategy_id=self.strategy.version_id,
            )
            res = await self.broker.submit_order(req)
            orders_executed.append(res)

        # 6. Update Monitor
        self.monitor.update(portfolio_state, self.broker.portfolio.trades_history)

        return signal, orders_executed

    async def run_replay(self, candles: Sequence[Candle]) -> PaperTradingStats:
        """Fast-forward simulated live paper trading across historical candle stream."""
        for candle in candles:
            await self.step(candle)

        final_state = await self.broker.get_portfolio_state()
        return self.monitor.update(final_state, self.broker.portfolio.trades_history)

    async def persist_to_db(self, session: AsyncSession) -> None:
        """Persist session trades, orders, and portfolio snapshot to database."""
        state = await self.broker.get_portfolio_state()

        # 1. Snapshot
        snap_repo = PortfolioSnapshotRepository(session)
        await snap_repo.record_snapshot(
            total_equity=state.total_equity,
            available_cash=state.available_cash,
            unrealized_pnl=state.unrealized_pnl,
            realized_pnl=state.realized_pnl,
            drawdown_pct=state.drawdown_pct,
            peak_equity=state.peak_equity,
            environment="paper",
        )

        # 2. Trades
        trade_repo = TradeRepository(session)
        for t in self.broker.portfolio.trades_history:
            await trade_repo.record_trade(
                symbol=t.symbol,
                side=t.side.value,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                quantity=t.quantity,
                pnl=t.pnl,
                pnl_pct=t.pnl_pct,
                fees=t.fees,
                opened_at=t.opened_at,
                environment="paper",
                exit_reason=t.exit_reason,
                duration_seconds=t.duration_seconds,
            )

        await session.commit()
        log.info("paper_session_persisted_to_db", trades=len(self.broker.portfolio.trades_history))
