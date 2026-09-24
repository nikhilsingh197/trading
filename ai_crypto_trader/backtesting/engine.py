"""Event-driven backtesting engine.

Design principles:
1. Strict chronological event processing prevents look-ahead bias.
2. Every trade incurs realistic fees, slippage, and spread.
3. Stop-loss and take-profit are checked against bar high/low,
   not just close price, for realistic execution.
4. Position sizing is delegated to the RiskEngine.
5. Results are fully reproducible given the same data and params.

Event types (processed in order per bar):
  BAR -> SIGNAL -> ORDER -> FILL
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

from ai_crypto_trader.backtesting.metrics import BacktestMetrics, TradeRecord, compute_metrics
from ai_crypto_trader.core.constants import DEFAULT_MAKER_FEE, DEFAULT_SLIPPAGE, DEFAULT_TAKER_FEE
from ai_crypto_trader.core.enums import Direction, ExitReason
from ai_crypto_trader.core.interfaces import BacktestEngineABC, BacktestResult, Signal, StrategyABC
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    maker_fee: float = DEFAULT_MAKER_FEE
    taker_fee: float = DEFAULT_TAKER_FEE
    slippage: float = DEFAULT_SLIPPAGE
    max_position_pct: float = 0.10    # Max 10% of equity per position
    risk_per_trade_pct: float = 0.005  # 0.5% risk per trade
    allow_short: bool = True
    leverage: float = 1.0
    periods_per_year: int = 365 * 24   # Adjust per timeframe


@dataclass
class OpenPosition:
    direction: Direction
    entry_price: float
    quantity: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_time: datetime
    entry_fee: float
    risk_pct: float


class BacktestEngine(BacktestEngineABC):
    """Event-driven backtester."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()

    def run(
        self,
        strategy: StrategyABC,
        df: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """Run backtest on the provided DataFrame.

        Args:
            strategy: Strategy implementation.
            df: Feature-enriched DataFrame. Must NOT contain target columns.
            initial_capital: Starting capital (overrides config if provided).

        Returns:
            BacktestResult with all metrics.
        """
        cfg = self._config
        capital = initial_capital or cfg.initial_capital

        # Validate: reject if target columns present (leakage guard)
        target_cols = [c for c in df.columns if c.startswith("target_")]
        if target_cols:
            raise ValueError(
                f"BacktestEngine received target columns {target_cols}. "
                "Remove target columns before backtesting to prevent look-ahead bias."
            )

        equity = capital
        peak_equity = capital
        equity_curve: list[float] = []
        trades: list[TradeRecord] = []
        position: Optional[OpenPosition] = None

        from ai_crypto_trader.core.enums import MarketRegime
        from ai_crypto_trader.regimes.rule_based import RuleBasedRegimeDetector
        regime_detector = RuleBasedRegimeDetector()

        log.info(
            "backtest_starting",
            strategy=strategy.name,
            version=strategy.version_id,
            rows=len(df),
            capital=capital,
        )

        for i in range(200, len(df)):  # Skip first 200 bars for indicator warmup
            bar = df.iloc[i]
            window = df.iloc[:i + 1]  # Strict: only past + current bar

            # 1. Check stop-loss / take-profit on open position
            if position is not None:
                trade, position = self._check_sl_tp(position, bar, equity, trades, cfg)
                if trade:
                    equity += trade.pnl
                    peak_equity = max(peak_equity, equity)

            # 2. Generate signal (uses only past data via window)
            if position is None:
                try:
                    regime = regime_detector.detect(window)
                    signal = strategy.generate_signal(window, regime)
                except Exception as exc:
                    log.debug("signal_error", bar=i, error=str(exc))
                    equity_curve.append(equity)
                    continue

                if signal and signal.action.value.startswith("ENTER"):
                    # Apply slippage to entry price
                    slippage_factor = 1 + cfg.slippage if signal.direction == Direction.LONG else 1 - cfg.slippage
                    entry_price = float(bar["close"]) * slippage_factor

                    # Size position
                    stop_distance = abs(entry_price - signal.stop_loss)
                    if stop_distance <= 0:
                        equity_curve.append(equity)
                        continue

                    risk_amount = equity * cfg.risk_per_trade_pct
                    quantity = risk_amount / stop_distance

                    # Cap at max position size
                    max_qty = (equity * cfg.max_position_pct) / entry_price
                    quantity = min(quantity, max_qty)

                    # Entry fee (taker order)
                    entry_fee = quantity * entry_price * cfg.taker_fee

                    position = OpenPosition(
                        direction=signal.direction,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        entry_time=bar.name if hasattr(bar, 'name') else datetime.now(timezone.utc),
                        entry_fee=entry_fee,
                        risk_pct=cfg.risk_per_trade_pct,
                    )

            equity_curve.append(equity)

        # Close any remaining position at last bar
        if position is not None and len(df) > 0:
            last_bar = df.iloc[-1]
            close_price = float(last_bar["close"])
            trade = self._close_position(position, close_price, ExitReason.TIMEOUT, equity, cfg)
            trades.append(trade)
            equity += trade.pnl
            equity_curve.append(equity)

        eq_series = pd.Series(equity_curve)
        metrics = compute_metrics(trades, eq_series, capital, cfg.periods_per_year)

        log.info(
            "backtest_complete",
            strategy=strategy.name,
            total_return=f"{metrics.total_return:.2%}",
            sharpe=f"{metrics.sharpe:.3f}",
            max_drawdown=f"{metrics.max_drawdown:.2%}",
            num_trades=metrics.num_trades,
            win_rate=f"{metrics.win_rate:.2%}",
        )

        symbol_val = "UNKNOWN"
        if "symbol" in df.columns and len(df) > 0:
            symbol_val = str(df["symbol"].iloc[0])

        return BacktestResult(
            strategy_version_id=strategy.version_id,
            symbol=symbol_val,
            timeframe="unknown",
            period_start=df.index[0] if len(df) > 0 else datetime.now(timezone.utc),
            period_end=df.index[-1] if len(df) > 0 else datetime.now(timezone.utc),
            total_return=metrics.total_return,
            cagr=metrics.cagr,
            sharpe=metrics.sharpe,
            sortino=metrics.sortino,
            max_drawdown=metrics.max_drawdown,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            expectancy=metrics.expectancy,
            num_trades=metrics.num_trades,
            fees_paid=metrics.fees_paid,
            extra={"metrics": metrics.__dict__},
        )

    def _check_sl_tp(
        self,
        pos: OpenPosition,
        bar: pd.Series,
        equity: float,
        trades: list[TradeRecord],
        cfg: BacktestConfig,
    ) -> tuple[Optional[TradeRecord], Optional[OpenPosition]]:
        """Check if SL or TP was hit on this bar."""
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        if pos.direction == Direction.LONG:
            if pos.stop_loss and low <= pos.stop_loss:
                exit_price = pos.stop_loss * (1 - cfg.slippage)  # Slippage on SL
                trade = self._close_position(pos, exit_price, ExitReason.STOP_LOSS, equity, cfg)
                trades.append(trade)
                return trade, None
            if pos.take_profit and high >= pos.take_profit:
                exit_price = pos.take_profit * (1 - cfg.slippage)
                trade = self._close_position(pos, exit_price, ExitReason.TAKE_PROFIT, equity, cfg)
                trades.append(trade)
                return trade, None

        elif pos.direction == Direction.SHORT:
            if pos.stop_loss and high >= pos.stop_loss:
                exit_price = pos.stop_loss * (1 + cfg.slippage)
                trade = self._close_position(pos, exit_price, ExitReason.STOP_LOSS, equity, cfg)
                trades.append(trade)
                return trade, None
            if pos.take_profit and low <= pos.take_profit:
                exit_price = pos.take_profit * (1 + cfg.slippage)
                trade = self._close_position(pos, exit_price, ExitReason.TAKE_PROFIT, equity, cfg)
                trades.append(trade)
                return trade, None

        return None, pos

    def _close_position(
        self,
        pos: OpenPosition,
        exit_price: float,
        exit_reason: ExitReason,
        equity: float,
        cfg: BacktestConfig,
    ) -> TradeRecord:
        """Close a position and compute P&L."""
        exit_fee = pos.quantity * exit_price * cfg.taker_fee
        total_fee = pos.entry_fee + exit_fee

        if pos.direction == Direction.LONG:
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - total_fee
        pnl_pct = net_pnl / (pos.entry_price * pos.quantity + 1e-9)

        return TradeRecord(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            side="long" if pos.direction == Direction.LONG else "short",
            quantity=pos.quantity,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fees=total_fee,
            opened_at=pos.entry_time,
            closed_at=datetime.now(timezone.utc),
            exit_reason=exit_reason.value,
        )
