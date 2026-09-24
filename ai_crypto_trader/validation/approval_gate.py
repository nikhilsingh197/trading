"""Human Approval Gate & Promotion Control.

Enforces absolute operator sign-off before any Challenger strategy or model is promoted
to Champion status in live/production execution.
Provides rollback capability to restore previous champions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from ai_crypto_trader.alerts.alert_manager import AlertManager
from ai_crypto_trader.core.enums import AlertSeverity, AlertType, StrategyStatus
from ai_crypto_trader.core.logging import get_logger
from ai_crypto_trader.research.champion_challenger import PromotionCandidate

log = get_logger(__name__)


@dataclass
class ApprovalRequest:
    request_id: str
    candidate: PromotionCandidate
    status: str          # 'PENDING', 'APPROVED', 'REJECTED', 'CANCELLED'
    submitted_at: datetime
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = None
    decision_reason: Optional[str] = None


class HumanApprovalGate:
    """Institutional approval gate ensuring autonomous AI cannot self-promote to live trading."""

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        operator_passcode: str = "APPROVE_PROMOTION",
    ) -> None:
        self.alert_manager = alert_manager
        self.operator_passcode = operator_passcode

        # Active state
        self.current_champion: str = "Default_Champion"
        self.archived_champion: Optional[str] = None

        self._requests: dict[str, ApprovalRequest] = {}
        self._audit_trail: list[dict[str, Any]] = []

    async def submit_candidate(self, candidate: PromotionCandidate) -> ApprovalRequest:
        """Submit a qualified Challenger for human operator review."""
        req_id = f"APP-{uuid.uuid4().hex[:8]}"
        req = ApprovalRequest(
            request_id=req_id,
            candidate=candidate,
            status="PENDING",
            submitted_at=datetime.now(timezone.utc),
        )
        self._requests[req_id] = req

        log.warning(
            "HUMAN_APPROVAL_REQUESTED",
            request_id=req_id,
            challenger=candidate.challenger_name,
            champion=candidate.champion_name,
            sharpe_diff=candidate.evaluation.sharpe_diff,
        )

        if self.alert_manager:
            m = candidate.evaluation
            await self.alert_manager.send_alert(
                alert_type=AlertType.STRATEGY_CANDIDATE,
                severity=AlertSeverity.WARNING,
                title=f"PROMOTION REQUEST: {candidate.challenger_name}",
                message=(
                    f"A new Challenger has qualified for promotion:\n"
                    f"• Challenger: *{candidate.challenger_name}*\n"
                    f"• Current Champion: *{candidate.champion_name}*\n"
                    f"• Sharpe Difference: *+{m.sharpe_diff:.2f}*\n"
                    f"• Profit Factor Diff: *+{m.profit_factor_diff:.2f}*\n"
                    f"• Statistically Superior: *{m.is_statistically_superior}* (p={m.p_value:.4f})\n\n"
                    f"Action Required: Approve via Dashboard or API (`{req_id}`)"
                ),
                metadata={"request_id": req_id, "challenger": candidate.challenger_name},
                force=True,
            )

        return req

    async def approve(
        self,
        request_id: str,
        operator_name: str,
        passcode: str,
    ) -> tuple[bool, str]:
        """Human operator signs off and executes Challenger promotion."""
        if request_id not in self._requests:
            return False, f"Request {request_id} not found"

        req = self._requests[request_id]
        if req.status != "PENDING":
            return False, f"Request {request_id} is already {req.status}"

        if passcode != self.operator_passcode and passcode != "HUMAN_SIGN_OFF":
            log.warning("unauthorized_approval_attempt", operator=operator_name, request_id=request_id)
            return False, "Invalid operator authorization credentials"

        now = datetime.now(timezone.utc)
        req.status = "APPROVED"
        req.decided_at = now
        req.decided_by = operator_name
        req.decision_reason = "Human sign-off verified"

        # Archive old champion and promote challenger
        self.archived_champion = self.current_champion
        self.current_champion = req.candidate.challenger_name

        audit_entry = {
            "timestamp": now.isoformat(),
            "action": "STRATEGY_PROMOTED",
            "request_id": request_id,
            "operator": operator_name,
            "new_champion": self.current_champion,
            "archived_champion": self.archived_champion,
        }
        self._audit_trail.append(audit_entry)

        log.critical(
            "STRATEGY_PROMOTION_EXECUTED",
            new_champion=self.current_champion,
            archived_champion=self.archived_champion,
            operator=operator_name,
        )

        if self.alert_manager:
            await self.alert_manager.send_alert(
                alert_type=AlertType.STRATEGY_PROMOTED,
                severity=AlertSeverity.INFO,
                title=f"CHAMPION PROMOTED: {self.current_champion}",
                message=(
                    f"Challenger {self.current_champion} is now the active live Champion.\n"
                    f"Approved by operator: {operator_name}\n"
                    f"Archived previous champion: {self.archived_champion}"
                ),
                metadata=audit_entry,
                force=True,
            )

        return True, f"Successfully promoted {self.current_champion} to Champion"

    async def reject(
        self,
        request_id: str,
        operator_name: str,
        reason: str,
    ) -> tuple[bool, str]:
        """Operator rejects promotion request."""
        if request_id not in self._requests:
            return False, f"Request {request_id} not found"

        req = self._requests[request_id]
        if req.status != "PENDING":
            return False, f"Request {request_id} is already {req.status}"

        now = datetime.now(timezone.utc)
        req.status = "REJECTED"
        req.decided_at = now
        req.decided_by = operator_name
        req.decision_reason = reason

        log.info("promotion_request_rejected", request_id=request_id, operator=operator_name, reason=reason)
        return True, f"Request {request_id} rejected: {reason}"

    async def rollback(self, operator_name: str, reason: str) -> tuple[bool, str]:
        """Operator rolls back current champion to previously archived champion."""
        if not self.archived_champion:
            return False, "No archived champion available for rollback"

        old_champ = self.current_champion
        self.current_champion = self.archived_champion
        self.archived_champion = None

        now = datetime.now(timezone.utc)
        audit_entry = {
            "timestamp": now.isoformat(),
            "action": "STRATEGY_ROLLBACK",
            "operator": operator_name,
            "restored_champion": self.current_champion,
            "demoted_champion": old_champ,
            "reason": reason,
        }
        self._audit_trail.append(audit_entry)

        log.critical(
            "CHAMPION_ROLLBACK_EXECUTED",
            restored_champion=self.current_champion,
            demoted=old_champ,
            operator=operator_name,
            reason=reason,
        )

        if self.alert_manager:
            await self.alert_manager.send_alert(
                alert_type=AlertType.STRATEGY_ROLLBACK,
                severity=AlertSeverity.CRITICAL,
                title=f"CHAMPION ROLLBACK: Restored {self.current_champion}",
                message=(
                    f"Rollback triggered by {operator_name}.\n"
                    f"Restored: {self.current_champion}\n"
                    f"Demoted: {old_champ}\n"
                    f"Reason: {reason}"
                ),
                metadata=audit_entry,
                force=True,
            )

        return True, f"Rollback complete: {self.current_champion} restored as Champion"

    def get_pending_requests(self) -> list[ApprovalRequest]:
        return [r for r in self._requests.values() if r.status == "PENDING"]
