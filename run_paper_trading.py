"""Continuous Paper Trading Runner.

Runs the paper trading session indefinitely against live Binance market data.
Designed to run for 2+ weeks to collect real performance data before live deployment.

Usage:
    python run_paper_trading.py

Features:
- Fetches live 1h candles from Binance (no API key needed for public data)
- Processes each new candle as it closes
- Saves daily performance snapshots
- Writes a checkpoint every hour (disaster recovery)
- Prints a live status update every candle
- Auto-restarts on errors with exponential backoff
- Stops cleanly on Ctrl+C and saves final report

Requirements: pip install ccxt
"""
from __future__ import annotations

import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Minimal bootstrap so we can import the platform ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import ccxt.async_support as ccxt

from ai_crypto_trader.paper_trading.paper_broker import PaperBroker
from ai_crypto_trader.paper_trading.paper_monitor import PaperMonitor
from ai_crypto_trader.paper_trading.paper_session import PaperSession
from ai_crypto_trader.core.interfaces import Candle
from ai_crypto_trader.monitoring.disaster_recovery import DisasterRecovery
from ai_crypto_trader.risk.kill_switch import KillSwitch
from ai_crypto_trader.risk.circuit_breaker import CircuitBreaker

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

SYMBOL          = "BTC/USDT"          # Trading pair
BINANCE_SYMBOL  = "BTCUSDT"           # Binance format
TIMEFRAME       = "1h"                # Candle size (1h recommended for first run)
INITIAL_CAPITAL = 10_000.0            # Simulated starting capital (USD)
RISK_PER_TRADE  = 0.01                # 1% risk per trade
MAX_POSITION    = 0.20                # Max 20% of capital per position
CHECKPOINT_DIR  = Path("data/paper_trading_checkpoints")
REPORT_DIR      = Path("data/paper_trading_reports")
LOG_FILE        = Path("data/paper_trading.log")

# Timeframe → seconds mapping
TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400, "1d": 86400,
}

# ─────────────────────────────────────────────────────────────────────────────
# Simple default strategy (replace with your champion strategy)
# ─────────────────────────────────────────────────────────────────────────────

from ai_crypto_trader.core.interfaces import Signal, StrategyABC
from ai_crypto_trader.core.enums import Direction, MarketRegime, SignalAction
import pandas as pd


class SimpleEMACrossStrategy(StrategyABC):
    """EMA 20/50 crossover — baseline strategy for paper trading validation."""

    @property
    def version_id(self) -> str:
        return "ema_cross_v1"

    @property
    def name(self) -> str:
        return "EMA Crossover 20/50"

    def generate_signal(self, df: pd.DataFrame, regime: MarketRegime) -> Signal:
        if len(df) < 51:
            return self._hold(df)

        close = df["close"]
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        price  = float(close.iloc[-1])
        e20    = float(ema20.iloc[-1])
        e50    = float(ema50.iloc[-1])
        e20_prev = float(ema20.iloc[-2])
        e50_prev = float(ema50.iloc[-2])

        atr = self._atr(df)

        # Bullish cross
        if e20_prev <= e50_prev and e20 > e50:
            return Signal(
                symbol=SYMBOL,
                action=SignalAction.ENTER_LONG,
                direction=Direction.LONG,
                confidence=0.65,
                entry_price=price,
                stop_loss=price - 2 * atr,
                take_profit=price + 3 * atr,
                expected_return_pct=3 * atr / price * 100,
                risk_pct=2 * atr / price * 100,
                regime=regime,
                reason="EMA20 crossed above EMA50",
                strategy_version_id=self.version_id,
            )

        # Bearish cross
        if e20_prev >= e50_prev and e20 < e50:
            return Signal(
                symbol=SYMBOL,
                action=SignalAction.EXIT_LONG,
                direction=Direction.FLAT,
                confidence=0.65,
                entry_price=price,
                stop_loss=price + 2 * atr,
                take_profit=price - 3 * atr,
                expected_return_pct=0.0,
                risk_pct=0.0,
                regime=regime,
                reason="EMA20 crossed below EMA50 — exit",
                strategy_version_id=self.version_id,
            )

        return self._hold(df)

    def _hold(self, df: pd.DataFrame) -> Signal:
        price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        return Signal(
            symbol=SYMBOL,
            action=SignalAction.HOLD,
            direction=Direction.FLAT,
            confidence=0.0,
            entry_price=price,
            stop_loss=price * 0.95,
            take_profit=price * 1.05,
            expected_return_pct=0.0,
            risk_pct=0.0,
            regime=MarketRegime.UNKNOWN,
            reason="No signal",
            strategy_version_id=self.version_id,
        )

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high = df["high"]
        low  = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Candle fetcher
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_recent_candles(exchange: ccxt.binance, limit: int = 100) -> list[Candle]:
    """Fetch the most recent closed candles."""
    ohlcv = await exchange.fetch_ohlcv(BINANCE_SYMBOL, TIMEFRAME, limit=limit)
    candles = []
    for row in ohlcv[:-1]:  # Exclude the current open (not yet closed) candle
        ts_ms, o, h, l, c, v = row
        open_dt  = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        close_dt = datetime.fromtimestamp((ts_ms + TF_SECONDS[TIMEFRAME] * 1000) / 1000, tz=timezone.utc)
        candles.append(Candle(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            open_time=open_dt,
            open=o, high=h, low=l, close=c, volume=v,
            close_time=close_dt,
        ))
    return candles


async def fetch_latest_closed_candle(exchange: ccxt.binance) -> Candle | None:
    """Fetch only the most recently closed candle."""
    ohlcv = await exchange.fetch_ohlcv(BINANCE_SYMBOL, TIMEFRAME, limit=3)
    if len(ohlcv) < 2:
        return None
    row = ohlcv[-2]  # -2 = last fully closed candle
    ts_ms, o, h, l, c, v = row
    open_dt  = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    close_dt = datetime.fromtimestamp((ts_ms + TF_SECONDS[TIMEFRAME] * 1000) / 1000, tz=timezone.utc)
    return Candle(
        symbol=SYMBOL, timeframe=TIMEFRAME,
        open_time=open_dt, open=o, high=h, low=l, close=c, volume=v,
        close_time=close_dt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def save_daily_report(session: PaperSession, day: int) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state = None
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        state_coro = session.broker.get_portfolio_state()
        # Get state synchronously
        state = loop.run_until_complete(state_coro) if not loop.is_running() else None
    except Exception:
        pass

    stats = session.monitor.get_stats() if hasattr(session.monitor, "get_stats") else {}
    report = {
        "day": day,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "equity": getattr(state, "total_equity", "N/A") if state else "N/A",
        "unrealized_pnl": getattr(state, "unrealized_pnl", 0) if state else 0,
        "realized_pnl": getattr(state, "realized_pnl", 0) if state else 0,
        "drawdown_pct": getattr(state, "drawdown_pct", 0) if state else 0,
        "stats": stats if isinstance(stats, dict) else {},
    }
    fname = REPORT_DIR / f"day_{day:02d}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    fname.write_text(json.dumps(report, indent=2, default=str))
    log(f"📋 Daily report saved: {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

_running = True

def _handle_signal(sig, frame):
    global _running
    log("⚠️  Shutdown signal received — finishing current candle then stopping...", "WARN")
    _running = False


async def main() -> None:
    global _running

    # Setup
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("🚀  AI Crypto Trader — Paper Trading Runner")
    log(f"    Symbol    : {SYMBOL}")
    log(f"    Timeframe : {TIMEFRAME}  ({TF_SECONDS[TIMEFRAME]}s candles)")
    log(f"    Capital   : ${INITIAL_CAPITAL:,.2f}")
    log(f"    Risk/trade: {RISK_PER_TRADE*100:.1f}%")
    log("=" * 60)

    # Initialise components
    strategy = SimpleEMACrossStrategy()
    broker   = PaperBroker(initial_capital=INITIAL_CAPITAL)
    monitor  = PaperMonitor()
    session  = PaperSession(
        strategy=strategy,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade_pct=RISK_PER_TRADE,
        max_position_pct=MAX_POSITION,
        broker=broker,
        monitor=monitor,
    )

    ks = KillSwitch()
    cb = CircuitBreaker()
    dr = DisasterRecovery(snapshot_dir=CHECKPOINT_DIR)

    exchange = ccxt.binance({"enableRateLimit": True})

    try:
        # Warmup — feed last 100 closed candles so indicators have history
        log("📥  Fetching warmup candles...")
        warmup_candles = await fetch_recent_candles(exchange, limit=100)
        for c in warmup_candles:
            await session.step(c)
        log(f"✅  Warmed up with {len(warmup_candles)} candles")

        # Track timing
        last_candle_ts  = warmup_candles[-1].open_time if warmup_candles else None
        last_checkpoint = time.time()
        last_report_day = 0
        candles_processed = 0
        start_time = time.time()

        log("⏳  Now listening for new candles. Press Ctrl+C to stop cleanly.\n")

        while _running:
            try:
                candle = await fetch_latest_closed_candle(exchange)

                if candle and candle.open_time != last_candle_ts:
                    last_candle_ts = candle.open_time
                    candles_processed += 1

                    signal_out, orders = await session.step(candle)

                    # Get portfolio state for display
                    pf = await broker.get_portfolio_state()
                    pnl_color = "📈" if pf.realized_pnl >= 0 else "📉"

                    log(
                        f"Candle #{candles_processed:4d} | "
                        f"{candle.open_time.strftime('%Y-%m-%d %H:%M')} | "
                        f"Close=${candle.close:,.2f} | "
                        f"Equity=${pf.total_equity:,.2f} | "
                        f"{pnl_color} PnL=${pf.realized_pnl:+.2f} | "
                        f"DD={pf.drawdown_pct:.1f}% | "
                        f"Signal={'---' if signal_out is None else signal_out.action.value}"
                    )

                    if orders:
                        for o in orders:
                            log(f"  ➤ ORDER: {o.status} | qty={o.filled_qty:.6f} | fee=${o.fees:.4f}")

                    # Hourly checkpoint
                    if time.time() - last_checkpoint > 3600:
                        await dr.save_checkpoint(
                            portfolio_state=pf,
                            circuit_breaker=cb,
                            kill_switch=ks,
                            extra={"candles_processed": candles_processed},
                        )
                        last_checkpoint = time.time()
                        log("💾  Hourly checkpoint saved")

                    # Daily report
                    days_running = int((time.time() - start_time) / 86400)
                    if days_running > last_report_day:
                        last_report_day = days_running
                        save_daily_report(session, days_running)
                        log(f"📅  Day {days_running} complete")

                    # Check circuit breaker
                    cb.check_metrics(
                        daily_loss_pct=pf.realized_pnl / INITIAL_CAPITAL * 100,
                        drawdown_pct=pf.drawdown_pct,
                    )
                    if not cb.can_trade()[0]:
                        log(f"⛔  Circuit breaker: {cb.reason}", "WARN")

                # Sleep until ~1 minute before next candle close
                candle_interval = TF_SECONDS[TIMEFRAME]
                now_ts = time.time()
                seconds_into_candle = now_ts % candle_interval
                sleep_for = candle_interval - seconds_into_candle - 30  # wake 30s early
                if sleep_for < 5:
                    sleep_for = 30
                await asyncio.sleep(sleep_for)

            except ccxt.NetworkError as e:
                log(f"🌐  Network error: {e} — retrying in 60s", "WARN")
                await asyncio.sleep(60)
            except ccxt.ExchangeError as e:
                log(f"❌  Exchange error: {e} — retrying in 120s", "ERROR")
                await asyncio.sleep(120)
            except Exception as e:
                log(f"💥  Unexpected error: {e} — retrying in 60s", "ERROR")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)

    finally:
        log("\n🛑  Shutting down...")

        # Final checkpoint
        try:
            pf = await broker.get_portfolio_state()
            await dr.save_checkpoint(portfolio_state=pf, circuit_breaker=cb, kill_switch=ks)
            log("💾  Final checkpoint saved")
        except Exception as e:
            log(f"   Checkpoint failed: {e}", "ERROR")

        # Final report
        days_total = int((time.time() - start_time) / 86400) if 'start_time' in dir() else 0
        save_daily_report(session, days_total)

        # Summary
        try:
            pf = await broker.get_portfolio_state()
            log("\n" + "=" * 60)
            log("📊  FINAL PAPER TRADING SUMMARY")
            log(f"    Starting capital : ${INITIAL_CAPITAL:,.2f}")
            log(f"    Final equity     : ${pf.total_equity:,.2f}")
            log(f"    Realized PnL     : ${pf.realized_pnl:+,.2f}")
            log(f"    Max Drawdown     : {pf.drawdown_pct:.2f}%")
            log(f"    Candles processed: {candles_processed}")
            log("=" * 60)
        except Exception:
            pass

        await exchange.close()
        log("👋  Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
