"""Disaster Recovery & Emergency Orchestration.

Responsibilities:
- Emergency state snapshot: Serialize current positions, orders, equity to disk.
- Graceful shutdown sequence: Flush state → cancel orders → flatten positions → alert.
- Recovery procedures: Restore last known good state after a crash or restart.
- Checkpoint rotation: Keeps the last N snapshots to prevent disk bloat.

Design:
  The DisasterRecovery orchestrator is instantiated at startup and periodically
  calls save_checkpoint() to write an atomic JSON snapshot.  On unclean exit
  (SIGTERM / unhandled exception) the caller invokes emergency_shutdown().
  On restart, load_latest_checkpoint() restores the last saved state.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)

_SNAPSHOT_DIR_ENV = "DISASTER_RECOVERY_DIR"
_DEFAULT_SNAPSHOT_DIR = Path("data") / "recovery_snapshots"
_MAX_SNAPSHOTS = 10          # Keep last N checkpoint files
_CHECKPOINT_PREFIX = "snapshot_"


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecoverySnapshot:
    """Serialisable point-in-time state of the trading system."""
    snapshot_id: str
    captured_at: str                        # ISO-8601 UTC
    total_equity: float
    available_cash: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown_pct: float
    peak_equity: float
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    open_orders: list[dict[str, Any]] = field(default_factory=list)
    circuit_breaker_state: str = "UNKNOWN"
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    champion_strategy: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoverySnapshot":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "RecoverySnapshot":
        return cls.from_dict(json.loads(raw))


@dataclass
class ShutdownReport:
    """Summary emitted at the end of an emergency_shutdown() call."""
    initiated_at: str
    completed_at: str
    snapshot_saved: bool
    snapshot_path: Optional[str]
    orders_cancelled: int
    positions_flattened: int
    alerts_sent: int
    errors: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return len(self.errors) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class DisasterRecovery:
    """Orchestrates checkpoint persistence and graceful emergency shutdown.

    Intended usage
    --------------
    >>> dr = DisasterRecovery()
    >>> # Periodic checkpoint every N minutes:
    >>> await dr.save_checkpoint(portfolio_state, circuit_breaker, kill_switch)
    >>>
    >>> # On unclean exit:
    >>> report = await dr.emergency_shutdown(portfolio_state, circuit_breaker, kill_switch)
    >>>
    >>> # On restart:
    >>> snapshot = dr.load_latest_checkpoint()
    """

    def __init__(
        self,
        snapshot_dir: Optional[Path] = None,
        max_snapshots: int = _MAX_SNAPSHOTS,
        alert_manager=None,
        order_manager=None,
        execution_engine=None,
    ) -> None:
        env_override = os.environ.get(_SNAPSHOT_DIR_ENV)
        self.snapshot_dir = Path(env_override) if env_override else (snapshot_dir or _DEFAULT_SNAPSHOT_DIR)
        self.max_snapshots = max_snapshots
        self.alert_manager = alert_manager
        self.order_manager = order_manager
        self.execution_engine = execution_engine

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        log.info("disaster_recovery_initialised", snapshot_dir=str(self.snapshot_dir))

    # ── Checkpoint persistence ────────────────────────────────────────────────

    def build_snapshot(
        self,
        portfolio_state,
        circuit_breaker=None,
        kill_switch=None,
        open_orders: Optional[list] = None,
        champion_strategy: str = "",
        extra: Optional[dict] = None,
    ) -> RecoverySnapshot:
        """Construct a RecoverySnapshot from live system objects."""
        now_str = datetime.now(timezone.utc).isoformat()
        snap_id = f"{_CHECKPOINT_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        positions = []
        if portfolio_state is not None:
            for pos in (portfolio_state.open_positions or []):
                try:
                    positions.append({
                        "symbol": pos.symbol,
                        "side": str(pos.side),
                        "entry_price": pos.entry_price,
                        "quantity": pos.quantity,
                        "current_price": pos.current_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                        "stop_loss": pos.stop_loss,
                        "take_profit": pos.take_profit,
                    })
                except Exception as e:
                    log.warning("snapshot_position_serialisation_error", error=str(e))

        orders_raw = []
        if open_orders:
            for o in open_orders:
                try:
                    orders_raw.append(o if isinstance(o, dict) else vars(o))
                except Exception:
                    pass

        cb_state = "UNKNOWN"
        if circuit_breaker:
            try:
                cb_state = circuit_breaker.state.value
            except Exception:
                pass

        ks_active = False
        ks_reason = ""
        if kill_switch:
            ks_active = kill_switch.is_engaged
            ks_reason = getattr(kill_switch, "_reason", "")

        equity = getattr(portfolio_state, "total_equity", 0.0) if portfolio_state else 0.0
        cash = getattr(portfolio_state, "available_cash", 0.0) if portfolio_state else 0.0
        u_pnl = getattr(portfolio_state, "unrealized_pnl", 0.0) if portfolio_state else 0.0
        r_pnl = getattr(portfolio_state, "realized_pnl", 0.0) if portfolio_state else 0.0
        dd = getattr(portfolio_state, "drawdown_pct", 0.0) if portfolio_state else 0.0
        peak = getattr(portfolio_state, "peak_equity", equity) if portfolio_state else equity

        return RecoverySnapshot(
            snapshot_id=snap_id,
            captured_at=now_str,
            total_equity=equity,
            available_cash=cash,
            unrealized_pnl=u_pnl,
            realized_pnl=r_pnl,
            drawdown_pct=dd,
            peak_equity=peak,
            open_positions=positions,
            open_orders=orders_raw,
            circuit_breaker_state=cb_state,
            kill_switch_active=ks_active,
            kill_switch_reason=ks_reason,
            champion_strategy=champion_strategy,
            extra=extra or {},
        )

    def write_snapshot(self, snapshot: RecoverySnapshot) -> Path:
        """Write snapshot atomically (write temp → rename) and prune old files."""
        filename = f"{snapshot.snapshot_id}.json"
        target = self.snapshot_dir / filename
        tmp = target.with_suffix(".tmp")

        try:
            tmp.write_text(snapshot.to_json(), encoding="utf-8")
            shutil.move(str(tmp), str(target))
            log.info("checkpoint_saved", path=str(target), equity=snapshot.total_equity)
        except Exception as e:
            log.error("checkpoint_write_failed", error=str(e))
            raise

        self._prune_old_snapshots()
        return target

    def _prune_old_snapshots(self) -> None:
        snapshots = sorted(
            self.snapshot_dir.glob(f"{_CHECKPOINT_PREFIX}*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(snapshots) > self.max_snapshots:
            oldest = snapshots.pop(0)
            oldest.unlink(missing_ok=True)
            log.debug("old_snapshot_pruned", path=str(oldest))

    async def save_checkpoint(
        self,
        portfolio_state,
        circuit_breaker=None,
        kill_switch=None,
        open_orders: Optional[list] = None,
        champion_strategy: str = "",
        extra: Optional[dict] = None,
    ) -> Path:
        """Async wrapper: build + write snapshot."""
        snap = self.build_snapshot(
            portfolio_state=portfolio_state,
            circuit_breaker=circuit_breaker,
            kill_switch=kill_switch,
            open_orders=open_orders,
            champion_strategy=champion_strategy,
            extra=extra,
        )
        return self.write_snapshot(snap)

    # ── Recovery on restart ──────────────────────────────────────────────────

    def load_latest_checkpoint(self) -> Optional[RecoverySnapshot]:
        """Load the most recent snapshot from disk, or None if none exists."""
        snapshots = sorted(
            self.snapshot_dir.glob(f"{_CHECKPOINT_PREFIX}*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not snapshots:
            log.info("no_checkpoint_found", snapshot_dir=str(self.snapshot_dir))
            return None

        latest = snapshots[0]
        try:
            raw = latest.read_text(encoding="utf-8")
            snap = RecoverySnapshot.from_json(raw)
            log.info(
                "checkpoint_loaded",
                path=str(latest),
                snapshot_id=snap.snapshot_id,
                captured_at=snap.captured_at,
                equity=snap.total_equity,
            )
            return snap
        except Exception as e:
            log.error("checkpoint_load_failed", path=str(latest), error=str(e))
            return None

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return metadata for all available checkpoints, newest first."""
        snapshots = sorted(
            self.snapshot_dir.glob(f"{_CHECKPOINT_PREFIX}*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        result = []
        for p in snapshots:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                result.append({
                    "path": str(p),
                    "snapshot_id": data.get("snapshot_id"),
                    "captured_at": data.get("captured_at"),
                    "total_equity": data.get("total_equity"),
                    "open_positions": len(data.get("open_positions", [])),
                    "kill_switch_active": data.get("kill_switch_active", False),
                })
            except Exception:
                result.append({"path": str(p), "error": "parse_failed"})
        return result

    # ── Emergency shutdown ───────────────────────────────────────────────────

    async def emergency_shutdown(
        self,
        portfolio_state,
        circuit_breaker=None,
        kill_switch=None,
        reason: str = "Emergency shutdown initiated",
    ) -> ShutdownReport:
        """Graceful emergency shutdown sequence.

        Steps:
        1. Save state snapshot.
        2. Activate kill switch.
        3. Cancel all open orders (via order_manager).
        4. Flatten all open positions (via execution_engine).
        5. Send critical alert.
        """
        initiated = datetime.now(timezone.utc).isoformat()
        errors: list[str] = []
        snapshot_path: Optional[str] = None
        snapshot_saved = False
        orders_cancelled = 0
        positions_flattened = 0
        alerts_sent = 0

        log.critical("EMERGENCY_SHUTDOWN_INITIATED", reason=reason)

        # Step 2 — Activate kill switch FIRST (so snapshot captures it)
        if kill_switch and not kill_switch.is_engaged:
            try:
                await kill_switch.activate(reason)
            except Exception as e:
                errors.append(f"kill_switch_activate_failed: {e}")

        # Step 1 — Save snapshot (now includes kill switch engaged state)
        try:
            path = await self.save_checkpoint(
                portfolio_state=portfolio_state,
                circuit_breaker=circuit_breaker,
                kill_switch=kill_switch,
                extra={"shutdown_reason": reason},
            )
            snapshot_path = str(path)
            snapshot_saved = True
        except Exception as e:
            errors.append(f"snapshot_failed: {e}")

        # Step 3 — Cancel open orders
        if self.order_manager:
            try:
                open_ids = list(getattr(self.order_manager, "_open_orders", {}).keys())
                for oid in open_ids:
                    try:
                        await self.order_manager.cancel_order(oid)
                        orders_cancelled += 1
                    except Exception as ce:
                        errors.append(f"cancel_order {oid}: {ce}")
            except Exception as e:
                errors.append(f"order_cancel_sweep_failed: {e}")

        # Step 4 — Flatten positions
        if self.execution_engine:
            try:
                await self.execution_engine.emergency_flatten_all(reason)
                if portfolio_state:
                    positions_flattened = len(portfolio_state.open_positions or [])
            except Exception as e:
                errors.append(f"flatten_positions_failed: {e}")

        # Step 5 — Alert
        if self.alert_manager:
            try:
                from ai_crypto_trader.core.enums import AlertSeverity, AlertType
                await self.alert_manager.send_alert(
                    alert_type=AlertType.KILL_SWITCH_ACTIVATED,
                    severity=AlertSeverity.CRITICAL,
                    title="🚨 EMERGENCY SHUTDOWN",
                    message=(
                        f"Emergency shutdown sequence completed.\n"
                        f"Reason: {reason}\n"
                        f"Orders cancelled: {orders_cancelled}\n"
                        f"Positions flattened: {positions_flattened}\n"
                        f"Snapshot: {snapshot_path or 'FAILED'}\n"
                        f"Errors: {len(errors)}"
                    ),
                    force=True,
                )
                alerts_sent = 1
            except Exception as e:
                errors.append(f"alert_failed: {e}")

        completed = datetime.now(timezone.utc).isoformat()
        report = ShutdownReport(
            initiated_at=initiated,
            completed_at=completed,
            snapshot_saved=snapshot_saved,
            snapshot_path=snapshot_path,
            orders_cancelled=orders_cancelled,
            positions_flattened=positions_flattened,
            alerts_sent=alerts_sent,
            errors=errors,
        )

        log_fn = log.critical if not report.clean else log.info
        log_fn(
            "emergency_shutdown_complete",
            clean=report.clean,
            errors=errors,
            snapshot_path=snapshot_path,
        )

        return report
