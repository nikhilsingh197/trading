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
@click.option("--symbols", default="BTCUSDT", help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT)")
@click.option("--timeframe", default="1h", help="Timeframe (e.g. 1h)")
@click.option("--capital", default=50000.0, type=float, help="Initial portfolio capital ($)")
@click.option("--strategy", default="EMA_Crossover", help="Strategy name")
@click.option("--monte-carlo/--no-monte-carlo", default=True, help="Run Monte Carlo bootstrap analysis")
def backtest(symbols: str, timeframe: str, capital: float, strategy: str, monte_carlo: bool) -> None:
    """Run event-driven backtest with realistic cost modeling and analytics."""
    import pandas as pd
    from ai_crypto_trader.config.settings import get_settings
    from ai_crypto_trader.core.logging import configure_logging
    from ai_crypto_trader.backtesting.portfolio_engine import (
        MultiAssetBacktestEngine,
        PortfolioEngineConfig,
    )
    from ai_crypto_trader.features.feature_engine import FeatureEngine
    from ai_crypto_trader.ingestion.parquet_store import ParquetStore
    from ai_crypto_trader.strategies import (
        BollingerRSIMeanReversion,
        DonchianBreakoutStrategy,
        EMACrossoverStrategy,
        MultiTimeframeTrendStrategy,
        StrategyEnsemble,
    )

    settings = get_settings()
    configure_logging(settings.app_log_level)

    strategy_map = {
        "EMA_Crossover": EMACrossoverStrategy,
        "Mean_Reversion": BollingerRSIMeanReversion,
        "Breakout": DonchianBreakoutStrategy,
        "MTF_Trend": MultiTimeframeTrendStrategy,
        "Ensemble": StrategyEnsemble,
    }
    if strategy not in strategy_map:
        console.print(f"[red]Unknown strategy: {strategy}. Available: {list(strategy_map.keys())}[/red]")
        sys.exit(1)

    symbol_list = [s.strip().replace("/", "").replace("-", "").upper() for s in symbols.split(",") if s.strip()]
    pstore = ParquetStore(settings.data_raw_dir)
    feature_engine = FeatureEngine()
    datasets: dict[str, pd.DataFrame] = {}

    for sym in symbol_list:
        raw_df = pstore.load_candles(sym, timeframe)
        if raw_df.empty:
            console.print(f"[red]No local data for {sym}_{timeframe}. Run 'act data backfill --symbol {sym}' first.[/red]")
            sys.exit(1)

        console.print(f"[cyan]Computing features for [bold]{sym}[/bold] ({len(raw_df)} bars)...[/cyan]")
        enriched_df = feature_engine.compute(raw_df)
        datasets[sym] = enriched_df

    strat_instance = strategy_map[strategy](version_id="backtest_v1")

    engine_cfg = PortfolioEngineConfig(
        initial_capital=capital,
        run_monte_carlo=monte_carlo,
        monte_carlo_iterations=1000,
    )
    portfolio_engine = MultiAssetBacktestEngine(config=engine_cfg)

    console.print(f"[cyan]Executing multi-asset event simulation ({', '.join(symbol_list)})...[/cyan]")
    result = portfolio_engine.run(strat_instance, datasets)
    pm = result.portfolio_metrics

    # 1. Portfolio Summary Table
    table = Table(title=f"Portfolio Performance Scorecard: {strategy} ({timeframe})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Initial Capital", f"${capital:,.2f}")
    final_eq = capital * (1.0 + pm.total_return)
    table.add_row("Final Equity", f"${final_eq:,.2f}")
    table.add_row("Total Return", f"[{'green' if pm.total_return >= 0 else 'red'}]{pm.total_return:.2%}[/]")
    table.add_row("CAGR", f"{pm.cagr:.2%}")
    table.add_row("Annualized Volatility", f"{pm.annualized_volatility:.2%}")
    table.add_row("Sharpe Ratio", f"{pm.sharpe:.3f}")
    table.add_row("Sortino Ratio", f"{pm.sortino:.3f}")
    table.add_row("Calmar Ratio", f"{pm.calmar:.3f}")
    table.add_row("Omega Ratio", f"{pm.omega_ratio:.3f}")
    table.add_row("Max Drawdown", f"[red]{pm.max_drawdown:.2%}[/red]")
    table.add_row("Max Drawdown Duration", f"{pm.max_drawdown_duration_bars} bars")
    table.add_row("Value at Risk (95%)", f"{pm.var_95:.2%}")
    table.add_row("Conditional VaR (95%)", f"{pm.cvar_95:.2%}")
    table.add_row("Win Rate", f"{pm.win_rate:.2%}")
    table.add_row("Profit Factor", f"{pm.profit_factor:.3f}")
    table.add_row("Payoff Ratio", f"{pm.payoff_ratio:.3f}")
    table.add_row("Total Trades", str(pm.num_trades))
    table.add_row("Total Fees Paid", f"${pm.fees_paid:,.2f}")
    table.add_row("Total Slippage Cost", f"${pm.slippage_paid:,.2f}")
    table.add_row("Net Funding Accrued", f"${pm.funding_paid:,.2f}")

    console.print(table)

    # 2. Per-Asset Breakdown Table
    if len(symbol_list) > 1:
        sym_table = Table(title="Per-Asset Performance Breakdown")
        sym_table.add_column("Symbol", style="cyan")
        sym_table.add_column("Trades", justify="right", style="white")
        sym_table.add_column("Win Rate", justify="right", style="green")
        sym_table.add_column("Profit Factor", justify="right", style="yellow")
        sym_table.add_column("Total Fees", justify="right", style="magenta")

        for sym, sm in result.per_symbol_metrics.items():
            sym_table.add_row(
                sym,
                str(sm.num_trades),
                f"{sm.win_rate:.2%}",
                f"{sm.profit_factor:.2f}" if sm.profit_factor != float("inf") else "N/A",
                f"${sm.fees_paid:,.2f}",
            )
        console.print(sym_table)

    # 3. Monte Carlo Robustness Table
    if result.monte_carlo and result.monte_carlo.trades_sampled > 0:
        mc = result.monte_carlo
        mc_table = Table(title=f"Monte Carlo Robustness Analysis ({mc.iterations} Iterations)")
        mc_table.add_column("Metric", style="cyan")
        mc_table.add_column("5th Percentile", style="red")
        mc_table.add_column("Median (50th)", style="yellow")
        mc_table.add_column("95th Percentile", style="green")

        mc_table.add_row(
            "Total Return",
            f"{mc.return_5th_pct:.2%}",
            f"{mc.median_return:.2%}",
            f"{mc.return_95th_pct:.2%}",
        )
        mc_table.add_row(
            "Max Drawdown (Worst)",
            f"[red]{mc.max_drawdown_95th_pct:.2%}[/red]",
            f"{mc.median_max_drawdown:.2%}",
            f"-",
        )
        mc_table.add_row(
            "Sharpe Ratio",
            f"{mc.sharpe_5th_pct:.2f}",
            f"{mc.median_sharpe:.2f}",
            "-",
        )
        mc_table.add_row("P(Drawdown > 10%)", f"{mc.prob_drawdown_exceeds_10pct:.1%}", "-", "-")
        mc_table.add_row("P(Drawdown > 15%)", f"{mc.prob_drawdown_exceeds_15pct:.1%}", "-", "-")

        console.print(mc_table)

    console.print()
    console.print(
        "[dim yellow]IMPORTANT: Backtested and simulated performance does not guarantee future results.[/dim yellow]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE COMMAND
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--symbol", default="BTCUSDT", help="Symbol to validate")
@click.option("--timeframe", default="1h", help="Timeframe")
@click.option("--strategy", default="EMA_Crossover", help="Strategy to evaluate")
@click.option("--windows", default=4, type=int, help="Number of rolling walk-forward windows")
def validate(symbol: str, timeframe: str, strategy: str, windows: int) -> None:
    """Run mandatory walk-forward validation and parameter stability checks."""
    import pandas as pd
    from ai_crypto_trader.config.settings import get_settings
    from ai_crypto_trader.core.logging import configure_logging
    from ai_crypto_trader.features.feature_engine import FeatureEngine
    from ai_crypto_trader.ingestion.parquet_store import ParquetStore
    from ai_crypto_trader.strategies import (
        BollingerRSIMeanReversion,
        DonchianBreakoutStrategy,
        EMACrossoverStrategy,
        MultiTimeframeTrendStrategy,
        StrategyEnsemble,
    )
    from ai_crypto_trader.validation import (
        ParameterStabilityTester,
        StrategyScorecard,
        WalkForwardConfig,
        WalkForwardValidator,
        calculate_deflated_sharpe,
    )

    settings = get_settings()
    configure_logging(settings.app_log_level)

    strategy_map = {
        "EMA_Crossover": EMACrossoverStrategy,
        "Mean_Reversion": BollingerRSIMeanReversion,
        "Breakout": DonchianBreakoutStrategy,
        "MTF_Trend": MultiTimeframeTrendStrategy,
        "Ensemble": StrategyEnsemble,
    }
    if strategy not in strategy_map:
        console.print(f"[red]Unknown strategy: {strategy}. Available: {list(strategy_map.keys())}[/red]")
        sys.exit(1)

    strat_cls = strategy_map[strategy]
    clean_symbol = symbol.replace("/", "").replace("-", "").upper()
    pstore = ParquetStore(settings.data_raw_dir)

    raw_df = pstore.load_candles(clean_symbol, timeframe)
    if raw_df.empty:
        console.print(f"[red]No local data for {clean_symbol}_{timeframe}. Run 'act data backfill' first.[/red]")
        sys.exit(1)

    console.print(f"[cyan]Computing features for [bold]{clean_symbol}[/bold] ({len(raw_df)} bars)...[/cyan]")
    fe = FeatureEngine()
    enriched_df = fe.compute(raw_df)

    strat_sample = strat_cls(version_id="temp")
    params = dict(strat_sample.params)

    # 1. Walk-Forward Validation
    console.print(f"[cyan]Running Walk-Forward Validation ({windows} rolling windows)...[/cyan]")
    wf_validator = WalkForwardValidator(WalkForwardConfig(num_windows=windows))
    wf_result = wf_validator.validate(strat_cls, params, enriched_df)

    wf_table = Table(title=f"Walk-Forward Windows: {strategy} on {clean_symbol} {timeframe}")
    wf_table.add_column("Window", style="dim")
    wf_table.add_column("IS Return", justify="right", style="cyan")
    wf_table.add_column("IS Sharpe", justify="right", style="cyan")
    wf_table.add_column("OOS Return", justify="right", style="magenta")
    wf_table.add_column("OOS Sharpe", justify="right", style="magenta")
    wf_table.add_column("WFE Ratio", justify="right", style="yellow")

    for w in wf_result.windows:
        wf_table.add_row(
            f"W{w.window_id}",
            f"{w.is_return:.2%}",
            f"{w.is_sharpe:.2f}",
            f"{w.oos_return:.2%}",
            f"{w.oos_sharpe:.2f}",
            f"{w.wfe:.1%}",
        )
    console.print(wf_table)
    console.print(f"[bold]Overall Walk-Forward Efficiency (WFE):[/bold] [{'green' if wf_result.overall_wfe >= 0.5 else 'red'}]{wf_result.overall_wfe:.1%}[/]")

    # 2. Parameter Stability Testing
    console.print(f"\n[cyan]Running Parameter Stability & Sensitivity Sweeps (+/-10%, +/-20%)...[/cyan]")
    stab_tester = ParameterStabilityTester()
    stab_report = stab_tester.evaluate(strat_cls, params, enriched_df)

    stab_table = Table(title="Parameter Sensitivity & Cliff Detection")
    stab_table.add_column("Parameter", style="cyan")
    stab_table.add_column("Perturbation", justify="right", style="white")
    stab_table.add_column("New Value", justify="right", style="white")
    stab_table.add_column("Sharpe", justify="right", style="yellow")
    stab_table.add_column("Drop %", justify="right", style="red")

    for r in stab_report.results[:8]:
        drop_color = "red" if r.degradation_pct > 30.0 else "green"
        stab_table.add_row(
            r.parameter_name,
            f"{r.perturbation_pct:+.0f}%",
            str(r.perturbed_value),
            f"{r.sharpe:.2f}",
            f"[{drop_color}]{r.degradation_pct:.1f}%[/{drop_color}]",
        )
    console.print(stab_table)
    if stab_report.has_cliff:
        console.print("[red bold]WARNING: Parameter Cliff Detected! (>30% drop on small perturbation).[/red bold]")
    else:
        console.print("[green]Parameter Surface is Smooth and Stable.[/green]")

    # 3. Strategy Scorecard Evaluation
    from ai_crypto_trader.backtesting.advanced_metrics import ComprehensiveMetrics
    from ai_crypto_trader.backtesting.engine import BacktestConfig, BacktestEngine
    bt_engine = BacktestEngine(BacktestConfig())
    base_res = bt_engine.run(strat_sample, enriched_df)

    comp_metrics = ComprehensiveMetrics(
        sharpe=base_res.sharpe,
        total_return=base_res.total_return,
        max_drawdown=base_res.max_drawdown,
        win_rate=base_res.win_rate,
        profit_factor=base_res.profit_factor,
        expectancy=base_res.expectancy,
        num_trades=base_res.num_trades,
    )

    trade_returns = [t.pnl_pct for t in base_res.trades] if hasattr(base_res, "trades") else []
    dsr = calculate_deflated_sharpe(base_res.sharpe, trade_returns, num_trials=10)

    scorecard = StrategyScorecard()
    sc_eval = scorecard.evaluate(
        strategy_name=strategy,
        metrics=comp_metrics,
        walk_forward=wf_result,
        stability=stab_report,
        dsr_p_value=dsr,
    )

    sc_table = Table(title="Strategy Qualification Scorecard")
    sc_table.add_column("Gate Criterion", style="cyan")
    sc_table.add_column("Required", style="white")
    sc_table.add_column("Actual", style="white")
    sc_table.add_column("Status", justify="center")

    for c in sc_eval.criteria:
        status_text = "[green bold]PASS[/green bold]" if c.passed else "[red bold]FAIL[/red bold]"
        sc_table.add_row(c.name, c.target, c.actual, status_text)

    console.print(sc_table)
    verdict_style = "green bold" if sc_eval.is_qualified else "red bold"
    console.print(f"\n[{verdict_style}]Final Verdict: {sc_eval.summary_verdict}[/{verdict_style}]")



# ─────────────────────────────────────────────────────────────────────────────
# ML TRAINING & MODEL MANAGEMENT COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--symbol", default="BTCUSDT", help="Trading pair symbol")
@click.option("--timeframe", default="1h", help="Candle timeframe")
@click.option(
    "--model",
    "model_type",
    default="xgboost",
    type=click.Choice(["xgboost", "lightgbm", "random_forest", "logistic_regression"]),
    help="Machine learning algorithm",
)
@click.option(
    "--target",
    "target_type",
    default="direction",
    type=click.Choice(["direction", "tp_before_sl", "significant_up"]),
    help="Target prediction variable",
)
@click.option("--horizon", default=12, type=int, help="Forward prediction horizon in bars")
@click.option("--cv-folds", default=5, type=int, help="Number of Purged K-Fold splits")
@click.option("--test-size", default=0.20, type=float, help="Fraction for holdout test split")
def train(
    symbol: str,
    timeframe: str,
    model_type: str,
    target_type: str,
    horizon: int,
    cv_folds: int,
    test_size: float,
) -> None:
    """Train and evaluate a machine learning trading model."""
    from ai_crypto_trader.ingestion.parquet_store import ParquetStore
    from ai_crypto_trader.training.trainer import ModelTrainer

    store = ParquetStore()
    df = store.load_candles(symbol, timeframe)
    if df is None or len(df) == 0:
        console.print(f"[red]No historical data found for {symbol} {timeframe}.[/red]")
        sys.exit(1)

    console.print(f"[bold cyan]Training {model_type.upper()} model for {symbol} ({len(df)} bars)...[/bold cyan]")
    trainer = ModelTrainer()

    try:
        result = trainer.train(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            target_type=target_type,
            horizon=horizon,
            cv_folds=cv_folds,
            test_size=test_size,
        )
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        sys.exit(1)

    # 1. Cross-Validation Results Table
    cv_table = Table(title=f"Purged K-Fold ({cv_folds} folds) Cross-Validation Performance")
    cv_table.add_column("Metric", style="cyan")
    cv_table.add_column("Mean +/- Std", style="white")

    for k in ["accuracy", "balanced_accuracy", "f1", "roc_auc", "precision", "recall", "brier_score"]:
        mean_val = result.cv_metrics.get(f"cv_{k}_mean", 0.0)
        std_val = result.cv_metrics.get(f"cv_{k}_std", 0.0)
        cv_table.add_row(k.replace("_", " ").title(), f"{mean_val:.4f} +/- {std_val:.4f}")

    console.print(cv_table)

    # 2. Holdout Test Evaluation Table
    test_table = Table(title="Out-of-Sample Holdout Test Evaluation")
    test_table.add_column("Metric", style="cyan")
    test_table.add_column("Score", style="white")

    for k, v in result.test_metrics.items():
        test_table.add_row(k.replace("_", " ").title(), f"{v:.4f}")

    console.print(test_table)

    # 3. Simulated Financial PnL Table
    sim_table = Table(title="Simulated Strategy Trading Returns (Holdout Test)")
    sim_table.add_column("Financial Metric", style="cyan")
    sim_table.add_column("Result", style="white")

    sim = result.simulated_trading
    sim_table.add_row("Simulated Strategy Return", f"{sim.get('cumulative_return', 0.0):.2%}")
    sim_table.add_row("Underlying Buy & Hold Return", f"{sim.get('market_return', 0.0):.2%}")
    sim_table.add_row("Signal Win Rate", f"{sim.get('win_rate', 0.0):.2%}")
    sim_table.add_row("Profit Factor", f"{sim.get('profit_factor', 0.0):.2f}")
    sim_table.add_row("Total Signals Generated", str(sim.get('trade_count', 0)))
    sim_table.add_row("Active Signal Bars", str(sim.get('active_signal_bars', 0)))

    console.print(sim_table)

    # 4. Top Feature Importances Table
    imp_table = Table(title="Top 10 Most Predictive Features")
    imp_table.add_column("Rank", justify="center")
    imp_table.add_column("Feature", style="cyan")
    imp_table.add_column("Importance", style="white")

    top_feats = list(result.feature_importances.items())[:10]
    for rank, (feat, score) in enumerate(top_feats, start=1):
        imp_table.add_row(str(rank), feat, f"{score:.4f}")

    console.print(imp_table)
    console.print(f"\n[green bold]Model artifact successfully persisted to:[/green bold] {result.artifact_path}")


@main.group()
def models() -> None:
    """Manage trained machine learning models."""
    pass


@models.command("list")
def list_models() -> None:
    """List all saved model artifacts."""
    from pathlib import Path
    models_dir = Path("data/models")
    if not models_dir.exists():
        console.print("[yellow]No models directory found.[/yellow]")
        return

    subdirs = [p for p in models_dir.iterdir() if p.is_dir() and (p / "training_summary.json").exists()]
    if not subdirs:
        console.print("[yellow]No trained models found.[/yellow]")
        return

    table = Table(title="Trained Machine Learning Models")
    table.add_column("Model Directory", style="cyan")
    table.add_column("Model Type", style="white")
    table.add_column("Symbol", style="white")
    table.add_column("Target", style="white")
    table.add_column("Test Acc", style="white")
    table.add_column("Created At", style="white")

    for d in sorted(subdirs, key=lambda p: p.stat().st_mtime, reverse=True):
        import json
        with open(d / "training_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        test_acc = summary.get("test_metrics", {}).get("accuracy", 0.0)
        table.add_row(
            d.name,
            summary.get("model_type", "unknown"),
            summary.get("symbol", "unknown"),
            summary.get("target", "unknown"),
            f"{test_acc:.2%}",
            summary.get("created_at", "")[:19].replace("T", " "),
        )

    console.print(table)



# ─────────────────────────────────────────────────────────────────────────────
# PAPER TRADING COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

@main.group()
def paper() -> None:
    """Run and monitor paper trading sessions."""
    pass


@paper.command("run")
@click.option("--symbol", default="BTCUSDT", help="Trading pair symbol")
@click.option("--timeframe", default="1h", help="Candle timeframe")
@click.option(
    "--strategy",
    "strategy_name",
    default="EMA_Crossover",
    type=click.Choice([
        "EMA_Crossover",
        "Bollinger_RSI",
        "Donchian_Breakout",
        "MultiTimeframe_Trend",
        "Ensemble",
        "ML_Signal",
    ]),
    help="Trading strategy",
)
@click.option("--capital", default=10_000.0, type=float, help="Initial paper balance")
@click.option("--risk-pct", default=1.0, type=float, help="Risk per trade as % of equity")
@click.option("--bars", default=0, type=int, help="Limit number of recent candles (0 for all)")
def run_paper(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    capital: float,
    risk_pct: float,
    bars: int,
) -> None:
    """Run an end-to-end paper trading session on market data."""
    import asyncio
    from ai_crypto_trader.core.interfaces import Candle
    from ai_crypto_trader.ingestion.parquet_store import ParquetStore
    from ai_crypto_trader.paper_trading.paper_session import PaperSession
    from ai_crypto_trader.strategies import (
        BollingerRSIMeanReversion,
        DonchianBreakoutStrategy,
        EMACrossoverStrategy,
        MLSignalStrategy,
        MultiTimeframeTrendStrategy,
        StrategyEnsemble,
    )

    store = ParquetStore()
    df = store.load_candles(symbol, timeframe)
    if df is None or len(df) == 0:
        console.print(f"[red]No historical data found for {symbol} {timeframe}.[/red]")
        sys.exit(1)

    if bars > 0 and len(df) > bars:
        df = df.iloc[-bars:].reset_index(drop=True)

    # Instantiate Strategy
    strat: Any
    if strategy_name == "EMA_Crossover":
        strat = EMACrossoverStrategy("paper_ema", {"fast_ema": 9, "slow_ema": 21})
    elif strategy_name == "Bollinger_RSI":
        strat = BollingerRSIMeanReversion("paper_mr", {"rsi_period": 14})
    elif strategy_name == "Donchian_Breakout":
        strat = DonchianBreakoutStrategy("paper_dc", {"dc_period": 20})
    elif strategy_name == "MultiTimeframe_Trend":
        strat = MultiTimeframeTrendStrategy("paper_mtf", {"fast_period": 9})
    elif strategy_name == "Ensemble":
        strat = StrategyEnsemble(
            version_id="paper_ens",
            strategies=[
                EMACrossoverStrategy("ens_ema", {}),
                BollingerRSIMeanReversion("ens_mr", {}),
                DonchianBreakoutStrategy("ens_dc", {}),
            ],
        )
    elif strategy_name == "ML_Signal":
        from pathlib import Path
        from ai_crypto_trader.models.registry import ModelRegistry
        models_dir = Path("data/models")
        subdirs = [p for p in models_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists()]
        if not subdirs:
            console.print("[red]No trained ML models found. Run 'act train' first.[/red]")
            sys.exit(1)
        latest_model_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
        model = ModelRegistry.load_model(latest_model_dir)
        strat = MLSignalStrategy(model=model, version_id="paper_ml", name="ML_Signal")
        console.print(f"[dim]Loaded model: {model.name} ({model.model_type}) from {latest_model_dir.name}[/dim]")

    # Convert DataFrame rows to Candle objects
    candles: list[Candle] = []
    for _, row in df.iterrows():
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                open_time=row["open_time"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                close_time=row.get("close_time", row["open_time"]),
            )
        )

    console.print(f"[bold cyan]Running Paper Trading Session ({len(candles)} candles)...[/bold cyan]")
    session = PaperSession(
        strategy=strat,
        symbol=symbol,
        timeframe=timeframe,
        initial_capital=capital,
        risk_per_trade_pct=risk_pct / 100.0,
    )

    async def _execute() -> Any:
        return await session.run_replay(candles)

    stats = asyncio.run(_execute())

    # 1. Performance Summary Table
    perf_table = Table(title=f"Paper Trading Performance - {symbol} ({strategy_name})")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="white")

    ret_color = "green" if stats.total_return_pct >= 0 else "red"
    perf_table.add_row("Initial Capital", f"${stats.initial_capital:,.2f}")
    perf_table.add_row("Ending Equity", f"${stats.current_equity:,.2f}")
    perf_table.add_row("Total Return", f"[{ret_color}]{stats.total_return_pct:+.2f}%[/{ret_color}]")
    perf_table.add_row("Peak Equity", f"${stats.peak_equity:,.2f}")
    perf_table.add_row("Current Drawdown", f"{stats.drawdown_pct:.2f}%")
    perf_table.add_row("Max Drawdown", f"{stats.max_drawdown_pct:.2f}%")
    perf_table.add_row("Total Trades", str(stats.total_trades))
    perf_table.add_row("Winning Trades", str(stats.winning_trades))
    perf_table.add_row("Losing Trades", str(stats.losing_trades))
    perf_table.add_row("Win Rate", f"{stats.win_rate:.2%}")
    perf_table.add_row("Profit Factor", f"{stats.profit_factor:.2f}")
    perf_table.add_row("Total Fees Paid", f"${stats.total_fees:.2f}")
    perf_table.add_row("System Status", f"[bold green]{stats.status}[/bold green]" if stats.status == "HEALTHY" else f"[bold red]{stats.status}[/bold red]")

    console.print(perf_table)

    # 2. Open Positions Table
    open_positions = session.broker.portfolio.positions
    if open_positions:
        pos_table = Table(title="Current Open Paper Positions")
        pos_table.add_column("Symbol", style="cyan")
        pos_table.add_column("Side", style="white")
        pos_table.add_column("Qty", style="white")
        pos_table.add_column("Entry Price", style="white")
        pos_table.add_column("Current Price", style="white")
        pos_table.add_column("Unrealized PnL", style="white")

        for p in open_positions.values():
            pnl_c = "green" if p.unrealized_pnl >= 0 else "red"
            pos_table.add_row(
                p.symbol,
                p.side.value,
                f"{p.quantity:.4f}",
                f"${p.entry_price:,.2f}",
                f"${p.current_price:,.2f}",
                f"[{pnl_c}]${p.unrealized_pnl:+,.2f} ({p.unrealized_pnl_pct:+.2f}%)[/{pnl_c}]",
            )
        console.print(pos_table)

    # 3. Recent Trades Table
    trades = session.broker.portfolio.trades_history
    if trades:
        trade_table = Table(title="Recent Paper Trades (Last 10)")
        trade_table.add_column("Closed At", style="cyan")
        trade_table.add_column("Side", style="white")
        trade_table.add_column("Entry", style="white")
        trade_table.add_column("Exit", style="white")
        trade_table.add_column("PnL ($)", style="white")
        trade_table.add_column("PnL (%)", style="white")
        trade_table.add_column("Reason", style="white")

        for t in trades[-10:]:
            tc = "green" if t.pnl >= 0 else "red"
            trade_table.add_row(
                t.closed_at.strftime("%Y-%m-%d %H:%M"),
                t.side.value,
                f"${t.entry_price:,.2f}",
                f"${t.exit_price:,.2f}",
                f"[{tc}]${t.pnl:+,.2f}[/{tc}]",
                f"[{tc}]{t.pnl_pct:+.2f}%[/{tc}]",
                t.exit_reason,
            )
        console.print(trade_table)

    # 4. Advancement Check
    eligible, reason = session.monitor.check_advancement(stats)
    verdict_style = "green bold" if eligible else "yellow"
    console.print(f"\n[{verdict_style}]Advancement Review: {reason}[/{verdict_style}]")


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
