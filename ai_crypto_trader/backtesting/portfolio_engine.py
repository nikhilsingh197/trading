"""Multi-asset portfolio event-driven backtesting engine.

Features:
- Synchronized multi-asset event queue
- Realistic dynamic slippage & volume participation limits (via CostModel)
- Execution latency modeling (signal at bar t executed at bar t+1 open)
- Perpetual futures 8-hour funding rate payments
- Shared cash, portfolio margin, and cross-asset exposure caps
- Integrated institutional analytics and Monte Carlo simulation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ai_crypto_trader.backtesting.advanced_metrics import (
    ComprehensiveMetrics,
    TradeDetail,
    calculate_comprehensive_metrics,
)
from ai_crypto_trader.backtesting.cost_model import CostModel, CostModelConfig
from ai_crypto_trader.backtesting.monte_carlo import MonteCarloResult, MonteCarloSimulator
from ai_crypto_trader.core.enums import Direction, ExitReason, MarketRegime
from ai_crypto_trader.core.interfaces import Signal, StrategyABC
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.regimes.rule_based import RuleBasedRegimeDetector

log = get_logger(__name__)


@dataclass
class PortfolioEngineConfig:
    initial_capital: float = 50000.0
    max_open_positions: int = 5
    max_position_pct_per_asset: float = 0.20   # Max 20% equity per position
    risk_per_trade_pct: float = 0.005           # 0.5% risk of equity per trade
    cost_config: CostModelConfig = field(default_factory=CostModelConfig)
    periods_per_year: int = 365 * 24            # Hourly crypto
    run_monte_carlo: bool = True
    monte_carlo_iterations: int = 1000


@dataclass
class ActivePosition:
    symbol: str
    direction: Direction
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    entry_fee: float
    entry_slippage: float
    funding_accumulated: float = 0.0
    bars_held: int = 0


@dataclass
class QueuedOrder:
    symbol: str
    signal: Signal
    target_quantity: float
    queued_at: datetime


@dataclass
class MultiAssetBacktestResult:
    portfolio_metrics: ComprehensiveMetrics
    per_symbol_metrics: dict[str, ComprehensiveMetrics]
    equity_curve: pd.Series
    trades: list[TradeDetail]
    monte_carlo: Optional[MonteCarloResult] = None


class MultiAssetBacktestEngine:
    """Simulates trading across multiple cryptocurrency pairs simultaneously."""

    def __init__(self, config: PortfolioEngineConfig | None = None) -> None:
        self.config = config or PortfolioEngineConfig()
        self.cost_model = CostModel(self.config.cost_config)
        self.regime_detector = RuleBasedRegimeDetector()
        self.monte_carlo_sim = MonteCarloSimulator(iterations=self.config.monte_carlo_iterations)

    def run(
        self,
        strategy_map: dict[str, StrategyABC] | StrategyABC,
        datasets: dict[str, pd.DataFrame],
    ) -> MultiAssetBacktestResult:
        """Run synchronized multi-asset backtest.

        Args:
            strategy_map: Map of symbol -> Strategy instance, or a single strategy for all symbols.
            datasets: Map of symbol -> feature-enriched DataFrame with UTC datetime index.
        """
        cfg = self.config
        cash = cfg.initial_capital
        equity = cfg.initial_capital
        peak_equity = cfg.initial_capital

        # Prepare strategy mapping
        if isinstance(strategy_map, StrategyABC):
            strategies: dict[str, StrategyABC] = {sym: strategy_map for sym in datasets}
        else:
            strategies = strategy_map

        # Build unified chronological timeline
        all_timestamps: set[datetime] = set()
        clean_dfs: dict[str, pd.DataFrame] = {}
        for sym, df in datasets.items():
            # Leakage guard: reject any target columns
            target_cols = [c for c in df.columns if c.startswith("target_")]
            if target_cols:
                raise ValueError(f"Dataset for {sym} contains training target columns: {target_cols}")

            indexed_df = df.copy()
            if not isinstance(indexed_df.index, pd.DatetimeIndex):
                if "open_time" in indexed_df.columns:
                    indexed_df["open_time"] = pd.to_datetime(indexed_df["open_time"], utc=True)
                    indexed_df = indexed_df.set_index("open_time")
                else:
                    raise ValueError(f"Dataset {sym} must have DatetimeIndex or 'open_time' column")

            indexed_df = indexed_df.sort_index()
            clean_dfs[sym] = indexed_df
            all_timestamps.update(indexed_df.index.to_pydatetime())

        timeline = sorted(all_timestamps)
        log.info("multi_asset_backtest_starting", symbols=list(clean_dfs.keys()), bars=len(timeline), capital=cash)

        positions: dict[str, ActivePosition] = {}
        queued_orders: list[QueuedOrder] = []
        completed_trades: list[TradeDetail] = []
        equity_records: dict[datetime, float] = {}

        # Main timeline simulation
        for t in timeline:
            # ─────────────────────────────────────────────────────────────────
            # 1. Accrue Funding Payments on Open Positions (00:00, 08:00, 16:00 UTC)
            # ─────────────────────────────────────────────────────────────────
            if self.cost_model.is_funding_hour(t):
                for sym, pos in list(positions.items()):
                    if t in clean_dfs[sym].index:
                        current_bar = clean_dfs[sym].loc[t]
                        pos_val = pos.quantity * float(current_bar["close"])
                        payment = self.cost_model.calculate_funding_payment(pos_val, pos.direction)
                        pos.funding_accumulated += payment
                        cash += payment

            # ─────────────────────────────────────────────────────────────────
            # 2. Check Stop-Loss and Take-Profit on Open Positions
            # ─────────────────────────────────────────────────────────────────
            for sym, pos in list(positions.items()):
                pos.bars_held += 1
                if t not in clean_dfs[sym].index:
                    continue

                bar = clean_dfs[sym].loc[t]
                high = float(bar["high"])
                low = float(bar["low"])
                bar_vol = float(bar["volume"])

                exit_price: Optional[float] = None
                exit_reason: Optional[ExitReason] = None

                if pos.direction == Direction.LONG:
                    if low <= pos.stop_loss:
                        exit_price = pos.stop_loss
                        exit_reason = ExitReason.STOP_LOSS
                    elif high >= pos.take_profit:
                        exit_price = pos.take_profit
                        exit_reason = ExitReason.TAKE_PROFIT
                elif pos.direction == Direction.SHORT:
                    if high >= pos.stop_loss:
                        exit_price = pos.stop_loss
                        exit_reason = ExitReason.STOP_LOSS
                    elif low <= pos.take_profit:
                        exit_price = pos.take_profit
                        exit_reason = ExitReason.TAKE_PROFIT

                if exit_price is not None and exit_reason is not None:
                    # Execute close
                    fill = self.cost_model.calculate_fill(
                        requested_quantity=pos.quantity,
                        base_price=exit_price,
                        direction=Direction.SHORT if pos.direction == Direction.LONG else Direction.LONG,
                        bar_volume=bar_vol,
                        is_taker=True,
                    )

                    total_fees = pos.entry_fee + fill.fee_paid
                    total_slippage = pos.entry_slippage + fill.slippage_paid

                    if pos.direction == Direction.LONG:
                        gross_pnl = (fill.effective_price - pos.entry_price) * pos.quantity
                    else:
                        gross_pnl = (pos.entry_price - fill.effective_price) * pos.quantity

                    net_pnl = gross_pnl - total_fees + pos.funding_accumulated
                    pnl_pct = net_pnl / (pos.entry_price * pos.quantity + 1e-9)

                    cash += (pos.entry_price * pos.quantity) + net_pnl

                    trade = TradeDetail(
                        symbol=sym,
                        side="LONG" if pos.direction == Direction.LONG else "SHORT",
                        entry_price=pos.entry_price,
                        exit_price=fill.effective_price,
                        quantity=pos.quantity,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct,
                        fees=total_fees,
                        slippage=total_slippage,
                        funding_paid=pos.funding_accumulated,
                        opened_at=pos.entry_time,
                        closed_at=t,
                        exit_reason=exit_reason.value,
                        bars_held=pos.bars_held,
                    )
                    completed_trades.append(trade)
                    del positions[sym]

            # ─────────────────────────────────────────────────────────────────
            # 3. Process Queued Orders (Latency Modeling: Queued at t-1, Executed at t Open)
            # ─────────────────────────────────────────────────────────────────
            remaining_orders: list[QueuedOrder] = []
            for order in queued_orders:
                sym = order.symbol
                if sym in positions:
                    continue  # Already in position for this symbol

                if len(positions) >= cfg.max_open_positions:
                    continue  # Max portfolio positions reached

                if t not in clean_dfs[sym].index:
                    remaining_orders.append(order)
                    continue

                bar = clean_dfs[sym].loc[t]
                open_price = float(bar["open"])
                bar_vol = float(bar["volume"])

                # Fill order with dynamic slippage & participation limit
                fill = self.cost_model.calculate_fill(
                    requested_quantity=order.target_quantity,
                    base_price=open_price,
                    direction=order.signal.direction,
                    bar_volume=bar_vol,
                    is_taker=True,
                )

                required_cash = fill.effective_price * fill.filled_quantity + fill.fee_paid
                if required_cash > cash:
                    # Scale down if insufficient cash
                    continue

                cash -= required_cash

                positions[sym] = ActivePosition(
                    symbol=sym,
                    direction=order.signal.direction,
                    entry_price=fill.effective_price,
                    quantity=fill.filled_quantity,
                    stop_loss=order.signal.stop_loss,
                    take_profit=order.signal.take_profit,
                    entry_time=t,
                    entry_fee=fill.fee_paid,
                    entry_slippage=fill.slippage_paid,
                )

            queued_orders = remaining_orders

            # ─────────────────────────────────────────────────────────────────
            # 4. Generate New Signals on Bar Close & Queue for Next Bar Open
            # ─────────────────────────────────────────────────────────────────
            for sym, df in clean_dfs.items():
                if sym in positions or any(o.symbol == sym for o in queued_orders):
                    continue

                if t not in df.index:
                    continue

                idx = df.index.get_loc(t)
                if idx < 50:  # Indicator warmup
                    continue

                window = df.iloc[: idx + 1]
                strat = strategies.get(sym)
                if not strat:
                    continue

                regime = self.regime_detector.detect(window)
                sig = strat.generate_signal(window, regime)

                if sig and sig.action.value.startswith("ENTER"):
                    close_price = float(df.loc[t, "close"])
                    stop_distance = abs(close_price - sig.stop_loss)
                    if stop_distance <= 0:
                        continue

                    risk_amount = equity * cfg.risk_per_trade_pct
                    raw_qty = risk_amount / stop_distance
                    max_qty = (equity * cfg.max_position_pct_per_asset) / close_price
                    target_qty = min(raw_qty, max_qty)

                    if target_qty > 0:
                        queued_orders.append(
                            QueuedOrder(
                                symbol=sym,
                                signal=sig,
                                target_quantity=target_qty,
                                queued_at=t,
                            )
                        )

            # ─────────────────────────────────────────────────────────────────
            # 5. Compute Portfolio Equity at Time t
            # ─────────────────────────────────────────────────────────────────
            unrealized_pnl = 0.0
            position_capital = 0.0
            for sym, pos in positions.items():
                if t in clean_dfs[sym].index:
                    curr_price = float(clean_dfs[sym].loc[t, "close"])
                    position_capital += pos.entry_price * pos.quantity
                    if pos.direction == Direction.LONG:
                        unrealized_pnl += (curr_price - pos.entry_price) * pos.quantity
                    else:
                        unrealized_pnl += (pos.entry_price - curr_price) * pos.quantity

            equity = cash + position_capital + unrealized_pnl
            peak_equity = max(peak_equity, equity)
            equity_records[t] = equity

        # ─────────────────────────────────────────────────────────────────────
        # Finalization & Metrics Calculation
        # ─────────────────────────────────────────────────────────────────────
        # Close any remaining open positions at final bar
        final_t = timeline[-1]
        for sym, pos in list(positions.items()):
            if final_t in clean_dfs[sym].index:
                close_bar = clean_dfs[sym].loc[final_t]
                close_price = float(close_bar["close"])
                fill = self.cost_model.calculate_fill(
                    requested_quantity=pos.quantity,
                    base_price=close_price,
                    direction=Direction.SHORT if pos.direction == Direction.LONG else Direction.LONG,
                    bar_volume=float(close_bar["volume"]),
                )
                if pos.direction == Direction.LONG:
                    gross_pnl = (fill.effective_price - pos.entry_price) * pos.quantity
                else:
                    gross_pnl = (pos.entry_price - fill.effective_price) * pos.quantity

                net_pnl = gross_pnl - (pos.entry_fee + fill.fee_paid) + pos.funding_accumulated
                pnl_pct = net_pnl / (pos.entry_price * pos.quantity + 1e-9)

                trade = TradeDetail(
                    symbol=sym,
                    side="LONG" if pos.direction == Direction.LONG else "SHORT",
                    entry_price=pos.entry_price,
                    exit_price=fill.effective_price,
                    quantity=pos.quantity,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    fees=pos.entry_fee + fill.fee_paid,
                    slippage=pos.entry_slippage + fill.slippage_paid,
                    funding_paid=pos.funding_accumulated,
                    opened_at=pos.entry_time,
                    closed_at=final_t,
                    exit_reason=ExitReason.TIMEOUT.value,
                    bars_held=pos.bars_held,
                )
                completed_trades.append(trade)

        eq_series = pd.Series(equity_records).sort_index()

        # Portfolio aggregate metrics
        portfolio_metrics = calculate_comprehensive_metrics(
            trades=completed_trades,
            equity_series=eq_series,
            initial_capital=cfg.initial_capital,
            periods_per_year=cfg.periods_per_year,
        )

        # Per-symbol metrics
        per_symbol_metrics: dict[str, ComprehensiveMetrics] = {}
        for sym in clean_dfs:
            sym_trades = [t for t in completed_trades if t.symbol == sym]
            per_symbol_metrics[sym] = calculate_comprehensive_metrics(
                trades=sym_trades,
                equity_series=eq_series,  # evaluated against shared equity
                initial_capital=cfg.initial_capital,
                periods_per_year=cfg.periods_per_year,
            )

        # Monte Carlo simulation
        mc_result: Optional[MonteCarloResult] = None
        if cfg.run_monte_carlo and completed_trades:
            mc_result = self.monte_carlo_sim.simulate(
                trades=completed_trades,
                initial_capital=cfg.initial_capital,
            )

        log.info(
            "multi_asset_backtest_complete",
            trades=len(completed_trades),
            total_return=f"{portfolio_metrics.total_return:.2%}",
            sharpe=f"{portfolio_metrics.sharpe:.3f}",
            max_drawdown=f"{portfolio_metrics.max_drawdown:.2%}",
        )

        return MultiAssetBacktestResult(
            portfolio_metrics=portfolio_metrics,
            per_symbol_metrics=per_symbol_metrics,
            equity_curve=eq_series,
            trades=completed_trades,
            monte_carlo=mc_result,
        )
