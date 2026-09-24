"""Statistical Feature & Concept Drift Detector.

Monitors production data streams for distribution shifts and strategy degradation:
1. Covariate Feature Drift: Two-Sample Kolmogorov-Smirnov (KS) test & Population Stability Index (PSI).
2. Concept Drift: Rolling prediction confidence drops (> 15% decline) and rolling win-rate decay (> 20% decline).
3. Strategy Expectancy Drift: Demotion to paper mode when expectancy turns negative (2 sigma drop).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Optional, Sequence

import numpy as np
from scipy import stats

from ai_crypto_trader.alerts.alert_manager import AlertManager
from ai_crypto_trader.core.constants import DRIFT_CONFIDENCE_WINDOW, DRIFT_ROLLING_WINDOW
from ai_crypto_trader.core.enums import AlertSeverity, AlertType
from ai_crypto_trader.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class FeatureDriftResult:
    feature_name: str
    ks_statistic: float
    p_value: float
    psi: float
    is_drifted: bool
    status: str  # 'STABLE', 'WARNING', 'DRIFT_DETECTED'


@dataclass
class DriftReport:
    timestamp: datetime
    has_drift: bool
    recommended_action: str  # 'CONTINUE', 'WARN', 'REDUCE_SIZE', 'DEMOTE_TO_PAPER', 'TRIGGER_RETRAIN'
    feature_drifts: list[FeatureDriftResult] = field(default_factory=list)
    rolling_confidence: float = 0.0
    confidence_decline_pct: float = 0.0
    rolling_win_rate: float = 0.0
    win_rate_decline_pct: float = 0.0
    rolling_expectancy: float = 0.0
    details: str = ""


class DriftDetector:
    """Detects statistical covariate shift and concept drift in machine learning models and strategies."""

    def __init__(
        self,
        ks_pvalue_threshold: float = 0.05,
        confidence_decline_threshold_pct: float = 15.0,
        win_rate_decline_threshold_pct: float = 20.0,
        confidence_window_size: int = DRIFT_CONFIDENCE_WINDOW,
        trade_window_size: int = DRIFT_ROLLING_WINDOW,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.confidence_decline_threshold_pct = confidence_decline_threshold_pct
        self.win_rate_decline_threshold_pct = win_rate_decline_threshold_pct
        self.confidence_window_size = confidence_window_size
        self.trade_window_size = trade_window_size
        self.alert_manager = alert_manager

        # Baselines
        self._baseline_features: dict[str, np.ndarray] = {}
        self._baseline_confidence: float = 0.70
        self._baseline_win_rate: float = 0.55
        self._baseline_expectancy: float = 25.0

        # Production streaming buffers
        self._feature_streams: dict[str, deque[float]] = {}
        self._confidence_stream: deque[float] = deque(maxlen=confidence_window_size)
        self._trade_pnl_stream: deque[float] = deque(maxlen=trade_window_size)

    def set_baselines(
        self,
        baseline_features: dict[str, Sequence[float]],
        baseline_confidence: float = 0.70,
        baseline_win_rate: float = 0.55,
        baseline_expectancy: float = 25.0,
    ) -> None:
        """Establish in-sample baseline distributions for comparisons."""
        for feat, vals in baseline_features.items():
            self._baseline_features[feat] = np.array(vals, dtype=float)
            if feat not in self._feature_streams:
                self._feature_streams[feat] = deque(maxlen=len(vals) * 2)

        self._baseline_confidence = baseline_confidence
        self._baseline_win_rate = baseline_win_rate
        self._baseline_expectancy = baseline_expectancy

        log.info(
            "drift_baselines_set",
            features_count=len(baseline_features),
            baseline_conf=baseline_confidence,
            baseline_wr=baseline_win_rate,
        )

    def record_prediction(self, features: dict[str, float], confidence: float) -> None:
        """Stream an incoming prediction feature vector and model confidence score."""
        for feat, val in features.items():
            if feat in self._baseline_features:
                if feat not in self._feature_streams:
                    self._feature_streams[feat] = deque(maxlen=500)
                self._feature_streams[feat].append(float(val))

        self._confidence_stream.append(float(confidence))

    def record_trade_result(self, pnl: float) -> None:
        """Stream completed trade outcome for concept drift and expectancy tracking."""
        self._trade_pnl_stream.append(float(pnl))

    async def check_drift(self) -> DriftReport:
        """Perform comprehensive statistical drift evaluation across all features and metrics."""
        now = datetime.now(timezone.utc)
        feature_results: list[FeatureDriftResult] = []
        has_feature_drift = False

        # 1. Feature Drift (Two-Sample KS-Test & PSI)
        for feat_name, base_vals in self._baseline_features.items():
            prod_vals_deque = self._feature_streams.get(feat_name)
            if not prod_vals_deque or len(prod_vals_deque) < 30:
                continue

            prod_vals = np.array(prod_vals_deque, dtype=float)
            ks_res = stats.ks_2samp(base_vals, prod_vals)
            ks_stat = float(ks_res.statistic)
            p_val = float(ks_res.pvalue)

            psi = self._calculate_psi(base_vals, prod_vals)

            is_drifted = p_val < self.ks_pvalue_threshold
            if is_drifted:
                has_feature_drift = True
                status = "DRIFT_DETECTED"
            elif psi >= 0.20 or p_val < self.ks_pvalue_threshold * 2:
                status = "WARNING"
            else:
                status = "STABLE"

            feature_results.append(
                FeatureDriftResult(
                    feature_name=feat_name,
                    ks_statistic=round(ks_stat, 4),
                    p_value=round(p_val, 5),
                    psi=round(psi, 4),
                    is_drifted=is_drifted,
                    status=status,
                )
            )

        # 2. Concept Drift: Confidence Decay
        cur_conf = float(np.mean(self._confidence_stream)) if self._confidence_stream else self._baseline_confidence
        conf_drop_pct = 0.0
        if self._baseline_confidence > 0:
            conf_drop_pct = max(0.0, ((self._baseline_confidence - cur_conf) / self._baseline_confidence) * 100.0)

        confidence_drifted = conf_drop_pct >= self.confidence_decline_threshold_pct

        # 3. Concept Drift: Rolling Win-Rate Decay
        cur_wr = self._baseline_win_rate
        wr_drop_pct = 0.0
        if len(self._trade_pnl_stream) >= 10:
            wins = sum(1 for p in self._trade_pnl_stream if p > 0)
            cur_wr = wins / len(self._trade_pnl_stream)
            if self._baseline_win_rate > 0:
                wr_drop_pct = max(0.0, ((self._baseline_win_rate - cur_wr) / self._baseline_win_rate) * 100.0)

        wr_drifted = wr_drop_pct >= self.win_rate_decline_threshold_pct

        # 4. Strategy Expectancy Drift
        cur_expectancy = float(np.mean(self._trade_pnl_stream)) if len(self._trade_pnl_stream) >= 10 else self._baseline_expectancy
        expectancy_negative = (len(self._trade_pnl_stream) >= 15) and (cur_expectancy < 0.0)

        # Determine recommended action
        action = "CONTINUE"
        reasons = []

        if expectancy_negative:
            action = "DEMOTE_TO_PAPER"
            reasons.append(f"Expectancy turned negative (${cur_expectancy:.2f} vs baseline ${self._baseline_expectancy:.2f})")
        elif wr_drifted:
            action = "REDUCE_SIZE"
            reasons.append(f"Win rate dropped {wr_drop_pct:.1f}% (current {cur_wr * 100:.1f}%)")
        elif confidence_drifted:
            action = "REDUCE_SIZE"
            reasons.append(f"Prediction confidence declined {conf_drop_pct:.1f}% (current {cur_conf * 100:.1f}%)")
        elif has_feature_drift:
            action = "TRIGGER_RETRAIN"
            drifted_feats = [f.feature_name for f in feature_results if f.is_drifted]
            reasons.append(f"Covariate shift detected on features: {', '.join(drifted_feats)}")

        total_drift = has_feature_drift or confidence_drifted or wr_drifted or expectancy_negative
        details = " | ".join(reasons) if reasons else "Distributions within acceptable tolerance."

        report = DriftReport(
            timestamp=now,
            has_drift=total_drift,
            recommended_action=action,
            feature_drifts=feature_results,
            rolling_confidence=round(cur_conf, 4),
            confidence_decline_pct=round(conf_drop_pct, 2),
            rolling_win_rate=round(cur_wr, 4),
            win_rate_decline_pct=round(wr_drop_pct, 2),
            rolling_expectancy=round(cur_expectancy, 2),
            details=details,
        )

        if total_drift:
            log.warning("DRIFT_DETECTED", action=action, details=details)
            if self.alert_manager:
                severity = AlertSeverity.CRITICAL if action == "DEMOTE_TO_PAPER" else AlertSeverity.WARNING
                await self.alert_manager.send_alert(
                    alert_type=AlertType.DRIFT_DETECTED,
                    severity=severity,
                    title=f"Drift Detected: {action}",
                    message=f"Statistical drift alarm engaged.\nAction: {action}\nDetails: {details}",
                    metadata={"action": action, "confidence_drop": conf_drop_pct, "win_rate_drop": wr_drop_pct},
                )

        return report

    @staticmethod
    def _calculate_psi(baseline: np.ndarray, production: np.ndarray, num_bins: int = 10) -> float:
        """Compute Population Stability Index between baseline and production distributions."""
        if len(baseline) == 0 or len(production) == 0:
            return 0.0

        # Create quantiles from baseline
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(baseline, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        base_counts, _ = np.histogram(baseline, bins=bin_edges)
        prod_counts, _ = np.histogram(production, bins=bin_edges)

        # Normalize to fractions with epsilon smoothing
        eps = 1e-4
        base_pct = (base_counts + eps) / (len(baseline) + eps * num_bins)
        prod_pct = (prod_counts + eps) / (len(production) + eps * num_bins)

        # PSI formula: sum((prod - base) * ln(prod / base))
        psi_val = np.sum((prod_pct - base_pct) * np.log(prod_pct / base_pct))
        return float(max(0.0, psi_val))
