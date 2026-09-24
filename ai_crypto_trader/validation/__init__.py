"""Validation and Overfitting Prevention package.

Provides Walk-Forward validation, Parameter Stability sensitivity sweeps,
Deflated Sharpe Ratio calculations, and automated Scorecard qualification gates.
"""
from ai_crypto_trader.validation.approval_gate import (
    ApprovalRequest,
    HumanApprovalGate,
)
from ai_crypto_trader.validation.deployment_checklist import (
    CheckResult,
    CheckStatus,
    DeploymentChecklist,
    DeploymentReadiness,
    LivePromotionThresholds,
)
from ai_crypto_trader.validation.deflated_sharpe import (
    calculate_deflated_sharpe,
    expected_max_sharpe,
)
from ai_crypto_trader.validation.parameter_stability import (
    ParameterStabilityTester,
    PerturbationResult,
    StabilityReport,
)
from ai_crypto_trader.validation.scorecard import (
    ScorecardCriterion,
    ScorecardEvaluation,
    StrategyScorecard,
)
from ai_crypto_trader.validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardValidator,
    WalkForwardWindow,
)

__all__ = [
    "ApprovalRequest",
    "CheckResult",
    "CheckStatus",
    "DeploymentChecklist",
    "DeploymentReadiness",
    "HumanApprovalGate",
    "LivePromotionThresholds",
    "WalkForwardValidator",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardWindow",
    "ParameterStabilityTester",
    "StabilityReport",
    "PerturbationResult",
    "StrategyScorecard",
    "ScorecardEvaluation",
    "ScorecardCriterion",
    "calculate_deflated_sharpe",
    "expected_max_sharpe",
]
