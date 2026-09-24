# AI Crypto Trading Platform

> **⚠️ IMPORTANT DISCLAIMER**: This is research and educational software. Backtested or simulated performance is **NOT** indicative of future results. Cryptocurrency trading involves substantial risk of loss. **Never invest more than you can afford to lose.**

A modular, production-grade, self-improving AI cryptocurrency trading platform built with Python.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete system design.

**Technology Stack:**
- **Backend**: Python 3.11+, FastAPI, SQLAlchemy 2.0 async
- **Database**: PostgreSQL 15 + TimescaleDB
- **Cache/Queue**: Redis 7 + Celery
- **ML**: scikit-learn, XGBoost, LightGBM, PyTorch
- **Exchange**: CCXT Pro (unified REST + WebSocket)
- **Frontend**: React 18 + Vite + TailwindCSS
- **Deployment**: Docker + Docker Compose

---

## Quick Start (Milestone 1 — Paper Mode Only)

### Prerequisites
- Python 3.11+
- Docker Desktop

### 1. Clone and Install

```bash
git clone <repo>
cd trading

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — fill in your testnet API keys
# TRADING_MODE=paper is the default (safe)
# LIVE_TRADING_ENABLED=false by default
```

### 3. Start Infrastructure

```bash
# Start PostgreSQL + Redis
docker-compose up postgres redis -d

# Run database migrations
make migrate
```

### 4. Verify System Info

```bash
act info
```

### 5. Backfill Historical Data (Testnet)

```bash
act backfill --symbol BTCUSDT --timeframe 1h --days 365
```

### 6. Run a Backtest

```bash
act backtest --symbol BTCUSDT --timeframe 1h --capital 10000
```

### 7. Run Tests

```bash
make test-unit
```

---

## Development Milestones

| Milestone | Status | Description |
|---|---|---|
| 1 | ✅ **DONE** | Architecture + repository + configuration |
| 2 | ✅ **DONE** | Historical data pipeline (Parquet + DB repo, validation, gap repair, integrity audit) |
| 3 | ✅ **DONE** | Full database + Alembic migrations (20 tables, TimescaleDB hypertables, CRUD repos) |
| 4 | ✅ **DONE** | Advanced backtesting engine (Multi-asset portfolio, dynamic slippage, funding, Monte Carlo) |
| 5 | ✅ **DONE** | Additional rule-based strategies (Mean Reversion, Breakout, MTF Trend, Ensemble) |
| 6 | ✅ **DONE** | Walk-forward validation & parameter stability testing (rolling windows, cliff detection, DSR, scorecard) |
| 7 | ✅ **DONE** | ML research pipeline (feature selector, Purged K-Fold CV, model zoo: XGBoost/LightGBM/RF/Logistic, trainer, MLSignalStrategy) |
| 8 | 🔲 Next | Paper trading (live data) |
| 9 | 🔲 | Dashboard (React) |
| 10 | 🔲 | Full risk engine |
| 11 | 🔲 | Exchange integration (testnet) |
| 12 | 🔲 | Production monitoring |
| 13 | 🔲 | Controlled live deployment |
| 14 | 🔲 | Self-improvement engine |
| 15 | 🔲 | Champion/challenger deployment |

---

## Safety Rules

1. **Paper trading is the default** — live trading requires `LIVE_TRADING_ENABLED=true` AND explicit human approval
2. **Hard risk limits cannot be modified by AI** — only via config file + restart
3. **All strategies must pass validation** — backtest → walk-forward → paper trading
4. **Kill switch available** — `act risk kill-switch on --reason "emergency"`
5. **All decisions are logged** — complete audit trail in database
6. **No API secrets in code** — all credentials from environment variables only
7. **Version everything** — models, strategies, and configs are all versioned and immutable

---

## Module Structure

```
ai_crypto_trader/
├── config/          # Pydantic settings + risk config (human-controlled)
├── core/            # Enums, exceptions, interfaces, logging
├── database/        # SQLAlchemy models + connection management
├── ingestion/       # WebSocket collector + REST backfill + validator
├── indicators/      # Technical indicators (trend/momentum/volatility/volume)
├── features/        # Feature engine + target generator
├── regimes/         # Market regime detection
├── strategies/      # Strategy implementations
├── backtesting/     # Event-driven backtest engine + metrics
├── risk/            # Risk engine + kill switch + position sizer
├── portfolio/       # Portfolio management (Milestone 8+)
├── execution/       # Order management (Milestone 11+)
├── monitoring/      # Drift detection + performance monitoring (Milestone 12+)
├── research/        # AI research agent (Milestone 14+)
├── api/             # FastAPI backend
├── workers/         # Celery background tasks
└── main/            # CLI entry points
```

---

## Commands Reference

```bash
# System
act info                      # Show system configuration
act backfill --symbol BTCUSDT --timeframe 1h --days 365
act backtest --symbol BTCUSDT --timeframe 1h --capital 10000

# Risk management
act risk kill-switch on --reason "emergency"
act risk kill-switch off

# Development
make test-unit               # Run unit tests
make test                    # Run all tests with coverage
make lint                    # Run ruff linter
make format                  # Run black formatter
make migrate                 # Apply database migrations

# Docker
make docker-up               # Start all services
make docker-down             # Stop all services
make docker-logs             # Tail logs
```

---

## Performance Philosophy

The objective is **NOT** maximum backtest profit.

The objective is:
- **Robust** across market regimes
- **Out-of-sample validated** via walk-forward testing
- **Risk-adjusted** (Sharpe/Sortino over raw return)
- **Low overfitting** (parameter stability testing)
- **Realistic execution** (fees, slippage, latency modeled)
- **Controlled drawdown** (hard stop mechanisms)

A strategy with lower returns but significantly better robustness is preferred over a high-return overfit strategy.

---

## License

PROPRIETARY — Research and educational use only.
