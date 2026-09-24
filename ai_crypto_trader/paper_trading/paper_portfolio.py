"""Paper portfolio manager tracking cash, positions, margin, and equity curves."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ai_crypto_trader.core.enums import Direction, Side
from ai_crypto_trader.core.interfaces import PortfolioState, PositionSnapshot
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class PaperTradeRecord:
    """Completed trade record in the paper trading portfolio."""
    id: str
    symbol: str
    side: Direction
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    opened_at: datetime
    closed_at: datetime
    exit_reason: str
    strategy_id: str | None = None
    duration_seconds: int = 0


@dataclass
class PaperPosition:
    """Active open position in the paper trading portfolio."""
    symbol: str
    side: Direction
    entry_price: float
    quantity: float
    current_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_id: str | None = None

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.quantity

    @property
    def unrealized_pnl(self) -> float:
        if self.side == Direction.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis <= 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100.0

    def to_snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            symbol=self.symbol,
            side=self.side,
            entry_price=self.entry_price,
            quantity=self.quantity,
            current_price=self.current_price,
            unrealized_pnl=self.unrealized_pnl,
            unrealized_pnl_pct=self.unrealized_pnl_pct,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
        )


class PaperPortfolio:
    """Tracks account equity, cash balance, and open positions for paper trading."""

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.initial_capital = initial_capital
        self.available_cash = initial_capital
        self.realized_pnl = 0.0
        self.peak_equity = initial_capital
        self.positions: dict[str, PaperPosition] = {}
        self.trades_history: list[PaperTradeRecord] = []

    def update_price(self, symbol: str, current_price: float) -> None:
        """Update current market price of held symbol for mark-to-market valuations."""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
            self._update_peak_equity()

    def get_position(self, symbol: str) -> PaperPosition | None:
        return self.positions.get(symbol)

    def _update_peak_equity(self) -> None:
        eq = self.total_equity
        if eq > self.peak_equity:
            self.peak_equity = eq

    @property
    def unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def total_equity(self) -> float:
        # Cash + margin allocated to positions + open unrealized PnL
        positions_equity = sum(pos.cost_basis + pos.unrealized_pnl for pos in self.positions.values())
        return self.available_cash + positions_equity

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        dd = (self.peak_equity - self.total_equity) / self.peak_equity * 100.0
        return max(0.0, dd)

    def can_open_position(self, side: Side, fill_price: float, quantity: float, fee: float) -> bool:
        """Verify sufficient cash balance to fund the trade."""
        cost = fill_price * quantity + fee
        return self.available_cash >= cost

    def on_order_fill(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        fill_price: float,
        fee: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        exit_reason: str = "SIGNAL",
        strategy_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> PaperTradeRecord | None:
        """Handle execution of filled orders, updating positions and cash."""
        ts = timestamp or datetime.now(timezone.utc)
        trade_record: PaperTradeRecord | None = None

        if symbol not in self.positions:
            # OPEN NEW POSITION
            cost = fill_price * quantity + fee
            self.available_cash -= cost

            direction = Direction.LONG if side == Side.BUY else Direction.SHORT
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                side=direction,
                entry_price=fill_price,
                quantity=quantity,
                current_price=fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                opened_at=ts,
                strategy_id=strategy_id,
            )
            log.info("paper_position_opened", symbol=symbol, side=direction.value, qty=quantity, price=fill_price)
        else:
            # EXISTING POSITION
            pos = self.positions[symbol]

            # If same direction: add to position (dollar-cost average)
            if (pos.side == Direction.LONG and side == Side.BUY) or (pos.side == Direction.SHORT and side == Side.SELL):
                cost = fill_price * quantity + fee
                self.available_cash -= cost
                new_qty = pos.quantity + quantity
                pos.entry_price = (pos.cost_basis + fill_price * quantity) / new_qty
                pos.quantity = new_qty
                if stop_loss:
                    pos.stop_loss = stop_loss
                if take_profit:
                    pos.take_profit = take_profit
                log.info("paper_position_increased", symbol=symbol, new_qty=new_qty, avg_price=pos.entry_price)
            else:
                # Opposite direction: CLOSE OR REDUCE POSITION
                close_qty = min(quantity, pos.quantity)
                if pos.side == Direction.LONG:
                    pnl = (fill_price - pos.entry_price) * close_qty - fee
                else:
                    pnl = (pos.entry_price - fill_price) * close_qty - fee

                pnl_pct = (pnl / (pos.entry_price * close_qty)) * 100.0
                self.realized_pnl += pnl

                # Return capital + PnL back to available cash
                released_cost = pos.entry_price * close_qty
                self.available_cash += released_cost + pnl

                duration = int((ts - pos.opened_at).total_seconds())

                trade_record = PaperTradeRecord(
                    id=f"pt_{uuid.uuid4().hex[:12]}",
                    symbol=symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=fill_price,
                    quantity=close_qty,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    fees=fee,
                    opened_at=pos.opened_at,
                    closed_at=ts,
                    exit_reason=exit_reason,
                    strategy_id=strategy_id or pos.strategy_id,
                    duration_seconds=duration,
                )
                self.trades_history.append(trade_record)

                pos.quantity -= close_qty
                if pos.quantity <= 1e-8:
                    del self.positions[symbol]
                    log.info("paper_position_closed", symbol=symbol, pnl=pnl, pnl_pct=f"{pnl_pct:.2f}%")
                else:
                    log.info("paper_position_reduced", symbol=symbol, remaining_qty=pos.quantity, pnl=pnl)

        self._update_peak_equity()
        return trade_record

    def get_state(self) -> PortfolioState:
        """Returns frozen snapshot conforming to BrokerABC interface."""
        return PortfolioState(
            total_equity=self.total_equity,
            available_cash=self.available_cash,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            drawdown_pct=self.drawdown_pct,
            peak_equity=self.peak_equity,
            open_positions=[p.to_snapshot() for p in self.positions.values()],
        )
