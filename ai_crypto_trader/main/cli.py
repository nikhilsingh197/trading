"""Command-line interface for the AI Crypto Trader platform.

Commands:
    act data backfill - Fetch historical OHLCV data with validation
    act data check    - Audit data integrity and detect gaps
    act data repair   - Interpolate short gaps and flag anomalies
    act data summary  - Display inventory of stored datasets
    act backtest      - Run backtest on historical data
    act risk          - Risk management and kill switch commands
    act info          - Show system configuration and status
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option("0.1.0", prog_name="act")
def main() -> None:
    """AI Crypto Trader — production-grade self-improving trading platform.

    WARNING: This system involves real financial risk.
    Paper trading is the default and safest mode.
    Explicit human approval is required for live trading.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# DATA COMMAND GROUP
# ─────────────────────────────────────────────────────────────────────────────

@main.group()
def data() -> None:
    """Market data pipeline commands (backfill, check, repair, summary)."""
    pass


@data.command("backfill")
@click.option("--symbol", default="BTCUSDT", help="Trading symbol (e.g. BTCUSDT, ETHUSDT)")
@click.option("--timeframe", default="1h", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
@click.option("--days", default=365, type=int, help="Days of history to fetch")
@click.option("--store-db/--no-db", default=False, help="Persist to database in addition to Parquet")
@click.option("--sandbox/--no-sandbox", default=True, help="Use exchange sandbox/testnet")
def data_backfill(symbol: str, timeframe: str, days: int, store_db: bool, sandbox: bool) -> None:
    """Fetch, validate, and store historical OHLCV data."""
    import ccxt.async_support as ccxt
    from ai_crypto_trader.config.settings import get_settings
    from ai_crypto_trader.core.logging import configure_logging, get_logger
    from ai_crypto_trader.ingestion.parquet_store import ParquetStore
    from ai_crypto_trader.ingestion.rest_backfill import RestBackfill

    settings = get_settings()
    configure_logging(settings.app_log_level)
    log = get_logger("cli.data.backfill")
    settings.ensure_directories()

    async def _run() -> None:
        exchange_cls = getattr(ccxt, settings.exchange_name)
        exchange = exchange_cls({
            "apiKey": settings.exchange_api_key,
            "secret": settings.exchange_api_secret,
            "sandbox": settings.exchange_sandbox or sandbox,
            "enableRateLimit": True,
        })

        backfiller = RestBackfill(exchange)
        start = datetime.now(timezone.utc) - timedelta(days=days)

        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        ccxt_symbol = f"{clean_symbol[:-4]}/{clean_symbol[-4:]}" if clean_symbol.endswith("USDT") else clean_symbol

        console.print(f"[cyan]Initiating backfill for [bold]{clean_symbol}[/bold] ({timeframe}) over past {days} days...[/cyan]")
        candles = await backfiller.fetch_candles(
            symbol=ccxt_symbol,
            timeframe=timeframe,
            start=start,
        )

        if not candles:
            console.print("[yellow]No candles retrieved from exchange.[/yellow]")
            await backfiller.close()
            return

        console.print(f"[green]Successfully fetched & validated [bold]{len(candles)}[/bold] candles.[/green]")

        # 1. Save to Parquet Store
        pstore = ParquetStore(settings.data_raw_dir)
        total_bars = pstore.save_candles(candles)
        console.print(f"[green]Parquet store updated: [bold]{total_bars}[/bold] total bars.[/green]")

        # 2. Optionally persist to DB
        if store_db:
            try:
                from ai_crypto_trader.database.connection import get_async_session
                from ai_crypto_trader.database.repositories.candle_repo import CandleRepository
                async with get_async_session() as session:
                    repo = CandleRepository(session)
                    count = await repo.upsert_many(candles)
                    console.print(f"[green]Database repository updated: [bold]{count}[/bold] records upserted.[/green]")
            except Exception as e:
                console.print(f"[yellow]Database persistence skipped or failed: {e}[/yellow]")

        await backfiller.close()

    asyncio.run(_run())


@data.command("check")
@click.option("--symbol", default="BTCUSDT", help="Trading symbol (e.g. BTCUSDT)")
@click.option("--timeframe", default="1h", help="Timeframe (e.g. 1h)")
def data_check(symbol: str, timeframe: str) -> None:
    """Audit data integrity, completeness, and detect sequence gaps."""
    from ai_crypto_trader.config.settings import get_settings
    from ai_crypto_trader.ingestion.data_integrity import DataIntegrityChecker
    from ai_crypto_trader.ingestion.parquet_store import ParquetStore

    settings = get_settings()
    pstore = ParquetStore(settings.data_raw_dir)

    clean_symbol = symbol.replace("/", "").replace("-", "").upper()
    if not pstore.exists(clean_symbol, timeframe):
        console.print(f"[red]Dataset for {clean_symbol}_{timeframe} not found. Run 'act data backfill' first.[/red]")
        sys.exit(1)

    candles = pstore.load_as_entities(clean_symbol, timeframe)
    checker = DataIntegrityChecker()
    report = checker.audit(candles, timeframe)

    table = Table(title=f"Data Quality Audit: {clean_symbol} ({timeframe})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Candles", str(report.total_candles))
    table.add_row("Expected Candles", str(report.expected_candles))
    table.add_row("Completeness", f"{report.completeness_pct:.2f}%")
    table.add_row("Health Score", f"{report.health_score:.1f} / 100")
    status_style = "green" if report.status in ("EXCELLENT", "GOOD") else "red"
    table.add_row("Status", f"[{status_style} bold]{report.status}[/{status_style} bold]")
    table.add_row("Sequence Gaps", str(report.gap_count))
    table.add_row("Total Missing Bars", str(report.total_missing_candles))
    table.add_row("Largest Gap", f"{report.largest_gap_candles} bars")
    table.add_row("Interpolated Bars", str(report.interpolated_count))
    table.add_row("Suspect Bars", str(report.suspect_count))
    table.add_row("Invalid OHLC Bars", str(report.invalid_ohlc_count))

    console.print(table)

    if report.gaps:
        console.print("\n[yellow bold]Identified Sequence Gaps:[/yellow bold]")
        gap_table = Table()
        gap_table.add_column("#", style="dim")
        gap_table.add_column("Gap Start", style="cyan")
        gap_table.add_column("Gap End", style="cyan")
        gap_table.add_column("Missing Bars", style="red")

        for idx, g in enumerate(report.gaps[:10], start=1):
            gap_table.add_row(
                str(idx),
                g.start_time.strftime("%Y-%m-%d %H:%M UTC"),
                g.end_time.strftime("%Y-%m-%d %H:%M UTC"),
                str(g.missing_count),
            )
        console.print(gap_table)
        if len(report.gaps) > 10:
            console.print(f"[dim]... and {len(report.gaps) - 10} more gaps.[/dim]")


@data.command("repair")
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe")
@click.option("--max-gap", default=5, type=int, help="Max gap size (bars) to linearly interpolate")
def data_repair(symbol: str, timeframe: str, max_gap: int) -> None:
    """Repair small sequence gaps via linear price interpolation."""
    from ai_crypto_trader.config.settings import get_settings
    from ai_crypto_trader.ingestion.data_cleaner import DataCleaner
    from ai_crypto_trader.ingestion.parquet_store import ParquetStore

    settings = get_settings()
    pstore = ParquetStore(settings.data_raw_dir)
    clean_symbol = symbol.replace("/", "").replace("-", "").upper()

    candles = pstore.load_as_entities(clean_symbol, timeframe)
    if not candles:
        console.print(f"[red]No data found for {clean_symbol}_{timeframe}[/red]")
        sys.exit(1)

    cleaner = DataCleaner()
    initial_count = len(candles)
    repaired = cleaner.interpolate_short_gaps(candles, timeframe, max_gap_bars=max_gap)

    added = len(repaired) - initial_count
    if added > 0:
        pstore.save_candles(repaired)
        console.print(f"[green]Successfully interpolated and added [bold]{added}[/bold] missing bars.[/green]")
    else:
        console.print("[cyan]No short gaps required interpolation.[/cyan]")


@data.command("summary")
def data_summary() -> None:
    """Display inventory of stored local market datasets."""
    from ai_crypto_trader.config.settings import get_settings
    from ai_crypto_trader.ingestion.parquet_store import ParquetStore

    settings = get_settings()
    pstore = ParquetStore(settings.data_raw_dir)
    summaries = pstore.get_summary()

    if not summaries:
        console.print(f"[yellow]No datasets found in {settings.data_raw_dir}.[/yellow]")
        return

    table = Table(title="Local Market Data Inventory")
    table.add_column("Symbol", style="cyan")
    table.add_column("Timeframe", style="magenta")
    table.add_column("Bars", justify="right", style="green")
    table.add_column("Earliest (UTC)", style="white")
    table.add_column("Latest (UTC)", style="white")
    table.add_column("Size (KB)", justify="right", style="yellow")

    for s in summaries:
        table.add_row(
            s["symbol"],
            s["timeframe"],
            f"{s['count']:,}",
            s["start"][:19] if s["start"] else "N/A",
            s["end"][:19] if s["end"] else "N/A",
            f"{s['size_kb']:,.1f}",
        )

    console.print(table)


# Backward-compatible alias
@main.command("backfill")
@click.option("--symbol", default="BTCUSDT", help="Trading symbol")
@click.option("--timeframe", default="1h", help="Timeframe")
@click.option("--days", default=365, type=int, help="Days of history to backfill")
@click.option("--sandbox/--no-sandbox", default=True, help="Use sandbox exchange")
@click.pass_context
def backfill_alias(ctx, symbol: str, timeframe: str, days: int, sandbox: bool) -> None:
    """Alias for 'act data backfill'."""
    ctx.invoke(data_backfill, symbol=symbol, timeframe=timeframe, days=days, store_db=False, sandbox=sandbox)


# ─────────────────────────────────────────────────────────────────────────────
# RISK COMMAND GROUP
# ─────────────────────────────────────────────────────────────────────────────

@main.group()
def risk() -> None:
    """Risk management commands."""
    pass


@risk.command("kill-switch")
@click.argument("action", type=click.Choice(["on", "off"]))
@click.option("--reason", default="manual CLI", help="Reason for activation")
def kill_switch_cmd(action: str, reason: str) -> None:
    """Activate or deactivate the global kill switch.

    This immediately stops all new position creation.
    Deactivation requires explicit human intent (this command).
    """
    async def _run() -> None:
        from ai_crypto_trader.config.settings import get_settings
        from ai_crypto_trader.risk.kill_switch import KillSwitch

        settings = get_settings()
        redis = None
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.redis_url, socket_connect_timeout=1)
            await r.ping()
            redis = r
        except Exception:
            redis = None

        ks = KillSwitch(redis_client=redis)

        if action == "on":
            await ks.activate(reason)
            console.print(f"[red bold]KILL SWITCH ACTIVATED[/red bold] Reason: {reason}")
            if not redis:
                console.print("[dim yellow](Note: Redis offline, set in-memory)[/dim yellow]")
        else:
            await ks.deactivate(actor="human:cli")
            console.print("[green]Kill switch deactivated[/green]")
            if not redis:
                console.print("[dim yellow](Note: Redis offline, cleared in-memory)[/dim yellow]")

        if redis:
            await redis.aclose()

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE COMMAND GROUP
# ─────────────────────────────────────────────────────────────────────────────

@main.group()
def db() -> None:
    """Database management and migration commands."""
    pass


@db.command("migrate")
def db_migrate() -> None:
    """Apply all pending Alembic database migrations."""
    from alembic.config import Config
    from alembic import command
    cfg = Config("alembic.ini")
    try:
        command.upgrade(cfg, "head")
        console.print("[green bold]Database migrations applied successfully (head).[/green bold]")
    except Exception as e:
        console.print(f"[red]Database migration error: {e}[/red]")
        sys.exit(1)


@db.command("rollback")
def db_rollback() -> None:
    """Roll back the last applied migration."""
    from alembic.config import Config
    from alembic import command
    cfg = Config("alembic.ini")
    try:
        command.downgrade(cfg, "-1")
        console.print("[yellow bold]Database downgraded by 1 revision.[/yellow bold]")
    except Exception as e:
        console.print(f"[red]Database rollback error: {e}[/red]")
        sys.exit(1)


@db.command("status")
def db_status() -> None:
    """Check current database migration version and connectivity."""
    from alembic.config import Config
    from alembic import command
    cfg = Config("alembic.ini")
    try:
        command.current(cfg, verbose=True)
    except Exception as e:
        console.print(f"[red]Could not determine database status: {e}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST COMMAND
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--symbol", default="BTCUSDT", help="Symbol to backtest")
@click.option("--timeframe", default="1h", help="Timeframe")
@click.option("--capital", default=10000.0, type=float, help="Initial capital")
@click.option("--strategy", default="EMA_Crossover", help="Strategy name")
def backtest(symbol: str, timeframe: str, capital: float, strategy: str) -> None:
    """Run a backtest on stored historical data."""
    import pandas as pd
    from ai_crypto_trader.config.settings import get_settings
    from ai_crypto_trader.core.logging import configure_logging
    from ai_crypto_trader.backtesting.engine import BacktestConfig, BacktestEngine
    from ai_crypto_trader.features.feature_engine import FeatureEngine
    from ai_crypto_trader.ingestion.parquet_store import ParquetStore
    from ai_crypto_trader.strategies.trend_following import EMACrossoverStrategy

    settings = get_settings()
    configure_logging(settings.app_log_level)

    pstore = ParquetStore(settings.data_raw_dir)
    clean_symbol = symbol.replace("/", "").replace("-", "").upper()
    df = pstore.load_candles(clean_symbol, timeframe)

    if df.empty:
        console.print(f"[red]Data not found for {clean_symbol}_{timeframe}. Run 'act data backfill' first.[/red]")
        sys.exit(1)

    df = df.set_index("open_time").sort_index()

    console.print(f"[cyan]Computing features ({len(df)} bars)...[/cyan]")
    engine = FeatureEngine()
    feature_df = engine.compute(df)

    strategy_map = {
        "EMA_Crossover": EMACrossoverStrategy,
    }
    if strategy not in strategy_map:
        console.print(f"[red]Unknown strategy: {strategy}. Available: {list(strategy_map.keys())}[/red]")
        sys.exit(1)

    strat = strategy_map[strategy](version_id="backtest_v1")
    bt_engine = BacktestEngine(BacktestConfig(initial_capital=capital))

    console.print(f"[cyan]Running backtest: {strategy}...[/cyan]")
    result = bt_engine.run(strat, feature_df, capital)

    table = Table(title=f"Backtest Results: {strategy} on {clean_symbol} {timeframe}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Return", f"{result.total_return:.2%}")
    table.add_row("CAGR", f"{result.cagr:.2%}")
    table.add_row("Sharpe Ratio", f"{result.sharpe:.3f}")
    table.add_row("Sortino Ratio", f"{result.sortino:.3f}")
    table.add_row("Max Drawdown", f"{result.max_drawdown:.2%}")
    table.add_row("Win Rate", f"{result.win_rate:.2%}")
    table.add_row("Profit Factor", f"{result.profit_factor:.3f}")
    table.add_row("Expectancy", f"{result.expectancy:.4f}")
    table.add_row("Total Trades", str(result.num_trades))
    table.add_row("Fees Paid", f"{result.fees_paid:.2f}")

    console.print(table)
    console.print()
    console.print(
        "[yellow]NOTE: Backtested performance is NOT indicative of future results.[/yellow]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# INFO COMMAND
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
def info() -> None:
    """Display system configuration and status."""
    from ai_crypto_trader.config.settings import get_settings

    try:
        settings = get_settings()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("[yellow]Copy .env.example to .env and fill in your values.[/yellow]")
        sys.exit(1)

    table = Table(title="AI Crypto Trader — System Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Environment", settings.app_env)
    table.add_row("Trading Mode", settings.trading_mode)
    table.add_row("Exchange", settings.exchange_name)
    table.add_row("Sandbox", str(settings.exchange_sandbox))
    table.add_row("Live Enabled", str(settings.live_trading_enabled))
    table.add_row("Max Trade Risk", f"{settings.risk_max_trade_risk_pct}%")
    table.add_row("Daily Loss Limit", f"{settings.risk_daily_loss_limit_pct}%")
    table.add_row("Max Drawdown Halt", f"{settings.risk_max_drawdown_halt_pct}%")
    table.add_row("Paper Min Days", str(settings.paper_min_period_days))
    table.add_row("Paper Min Trades", str(settings.paper_min_trades))

    console.print(table)
    console.print()
    console.print("[bold red]SAFETY REMINDER:[/bold red]")
    console.print("  • Paper trading is enabled by default")
    console.print("  • Live trading requires LIVE_TRADING_ENABLED=true AND explicit approval")
    console.print("  • Never share API keys or .env files")


if __name__ == "__main__":
    main()
