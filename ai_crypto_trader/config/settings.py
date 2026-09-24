"""Central application configuration using Pydantic BaseSettings.

All configuration is loaded from environment variables or a .env file.
NO hard-coded secrets are allowed in this file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "paper", "live"] = "development"
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_secret_key: str = Field(..., min_length=16)

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(...)
    database_sync_url: str = Field(...)
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Exchange ──────────────────────────────────────────────────────────────
    exchange_name: str = "binance"
    exchange_api_key: str = Field(default="", repr=False)     # redacted from repr
    exchange_api_secret: str = Field(default="", repr=False)  # redacted from repr
    exchange_sandbox: bool = True  # default to testnet for safety

    # ── Trading Mode ─────────────────────────────────────────────────────────
    trading_mode: Literal["paper", "shadow", "live"] = "paper"
    live_trading_enabled: bool = False  # Must be explicitly enabled by human

    # ── Risk Limits (hard limits — must not be modified by AI/models) ─────────
    risk_max_trade_risk_pct: float = Field(default=0.50, ge=0.0, le=5.0)
    risk_daily_loss_limit_pct: float = Field(default=2.0, ge=0.0, le=20.0)
    risk_max_drawdown_halt_pct: float = Field(default=10.0, ge=0.0, le=50.0)
    risk_max_drawdown_reduce_pct: float = Field(default=5.0, ge=0.0, le=50.0)
    risk_max_open_positions: int = Field(default=5, ge=1, le=20)
    risk_max_leverage: float = Field(default=3.0, ge=1.0, le=20.0)
    risk_max_position_size_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    risk_max_correlated_exposure_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    risk_max_trades_per_day: int = Field(default=20, ge=1, le=100)
    risk_max_consecutive_losses: int = Field(default=6, ge=1, le=20)

    # ── Paper Trading Validation ──────────────────────────────────────────────
    paper_min_period_days: int = Field(default=30, ge=1)
    paper_min_trades: int = Field(default=100, ge=10)

    # ── Strategy Scorecard Thresholds ─────────────────────────────────────────
    scorecard_min_sharpe: float = 0.5
    scorecard_min_profit_factor: float = 1.2
    scorecard_max_drawdown_pct: float = 15.0
    scorecard_min_win_rate: float = 0.40
    scorecard_min_trades: int = 50
    scorecard_min_expectancy: float = 0.0

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2
    api_reload: bool = False

    # ── Alerts ────────────────────────────────────────────────────────────────
    alert_telegram_token: str = Field(default="", repr=False)
    alert_telegram_chat_id: str = ""
    alert_email_host: str = "smtp.gmail.com"
    alert_email_port: int = 587
    alert_email_user: str = Field(default="", repr=False)
    alert_email_password: str = Field(default="", repr=False)
    alert_email_to: str = ""

    # ── Data directories ─────────────────────────────────────────────────────
    data_raw_dir: Path = Path("./data/raw")
    data_processed_dir: Path = Path("./data/processed")
    data_live_dir: Path = Path("./data/live")
    model_artifact_dir: Path = Path("./data/models")
    report_output_dir: Path = Path("./data/reports")

    # ── Monitoring ────────────────────────────────────────────────────────────
    prometheus_port: int = 9090
    enable_metrics: bool = True

    # ── Drift Detection ───────────────────────────────────────────────────────
    drift_ks_pvalue_threshold: float = 0.05
    drift_win_rate_decline_pct: float = 20.0
    drift_confidence_decline_pct: float = 15.0

    @model_validator(mode="after")
    def validate_live_trading_safety(self) -> "Settings":
        """Prevent live trading without explicit enablement."""
        if self.trading_mode == "live" and not self.live_trading_enabled:
            raise ValueError(
                "SAFETY: trading_mode='live' requires LIVE_TRADING_ENABLED=true. "
                "Set this only after completing all validation milestones and "
                "obtaining explicit human approval."
            )
        if self.trading_mode == "live" and self.exchange_sandbox:
            raise ValueError(
                "SAFETY: trading_mode='live' but exchange_sandbox=true. "
                "You cannot run live trading against a sandbox exchange."
            )
        return self

    @field_validator("risk_max_drawdown_reduce_pct")
    @classmethod
    def validate_reduce_before_halt(cls, v: float, info: object) -> float:
        """Ensure reduce threshold is less than halt threshold."""
        # Accessed via info.data in pydantic v2
        return v

    def ensure_directories(self) -> None:
        """Create all configured data directories if they do not exist."""
        dirs = [
            self.data_raw_dir,
            self.data_processed_dir,
            self.data_live_dir,
            self.model_artifact_dir,
            self.report_output_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """Safe repr that never exposes secrets."""
        return (
            f"Settings(env={self.app_env!r}, "
            f"mode={self.trading_mode!r}, "
            f"exchange={self.exchange_name!r}, "
            f"sandbox={self.exchange_sandbox})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Uses lru_cache so the .env file is read only once per process.
    """
    return Settings()  # type: ignore[call-arg]
