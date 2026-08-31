"""Phase 9 candidate 编排：validate、dedupe、policy 与审批状态。"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime, timezone

from app.ingestion.contracts import (
    ApprovalStatus,
    IngestionApproval,
    IngestionCandidate,
    IngestionCandidateStatus,
    IngestionCandidateSummary,
    IngestionPrincipalType,
    IngestionRequester,
    KnowledgeDocument,
    KnowledgeSourceType,
    PolicyDecisionType,
)
from app.ingestion.identity import build_candidate_id
from app.ingestion.policy import (
    INGESTION_MAX_CONTENT_CHARS,
    evaluate_ingestion_candidate,
)
from app.ingestion.repository import (
    CandidateRepository,
    InMemoryCandidateRepository,
)


class CandidateNotFoundError(LookupError):
    """请求的 candidate 不存在。"""


class InvalidCandidateTransitionError(ValueError):
    """拒绝 terminal candidate 的反转或其他非法状态迁移。"""


_POLICY_REQUESTER = IngestionRequester(
    principal_id="ingestion-policy-v1",
    principal_type=IngestionPrincipalType.SYSTEM,
    source="policy",
)


class IngestionService:
    """只管理候选 domain state；不认识 RAG、Chroma、MCP 或 HTTP。"""

    def __init__(
        self,
        repository: CandidateRepository | None = None,
        *,
        max_content_chars: int = INGESTION_MAX_CONTENT_CHARS,
        supported_source_types: Collection[KnowledgeSourceType] | None = None,
    ) -> None:
        if (
            isinstance(max_content_chars, bool)
            or not isinstance(max_content_chars, int)
            or max_content_chars < 1
        ):
            raise ValueError("max_content_chars 必须是正整数")
        self._repository = (
            repository
            if repository is not None
            else InMemoryCandidateRepository()
        )
        self._max_content_chars = max_content_chars
        self._supported_source_types = (
            None
            if supported_source_types is None
            else frozenset(supported_source_types)
        )

    @property
    def repository(self) -> CandidateRepository:
        """暴露抽象 repository，便于组合根和测试读取 domain state。"""
        return self._repository

    def propose(
        self,
        *,
        document: KnowledgeDocument,
        requester: IngestionRequester,
        reason: str | None = None,
        requested_at: datetime | None = None,
    ) -> IngestionCandidate:
        """幂等创建候选；相同来源与内容返回首次保存的候选。"""
        candidate_id = build_candidate_id(
            document.source_id,
            document.content_hash,
        )
        existing = self._repository.get(candidate_id)
        if existing is not None:
            return existing

        timestamp = requested_at or datetime.now(timezone.utc)
        policy = evaluate_ingestion_candidate(
            document,
            max_content_chars=self._max_content_chars,
            supported_source_types=self._supported_source_types,
        )
        status, approval = self._initial_state(policy.decision, timestamp, policy.reasons)
        candidate = IngestionCandidate(
            candidate_id=candidate_id,
            document=document,
            status=status,
            requested_at=timestamp,
            requested_by=requester,
            reason=reason,
            policy_decision=policy.decision,
            policy_reasons=policy.reasons,
            approval=approval,
        )
        return self._repository.save(candidate)

    def approve(
        self,
        candidate_id: str,
        *,
        decided_by: IngestionRequester,
        reason: str | None = None,
        decided_at: datetime | None = None,
    ) -> IngestionCandidate:
        """把 pending candidate 变为 approved，但不调用索引。"""
        candidate = self._require_pending(candidate_id)
        approval = IngestionApproval(
            status=ApprovalStatus.APPROVED,
            decided_at=decided_at or datetime.now(timezone.utc),
            decided_by=decided_by,
            reason=reason,
        )
        return self._replace_state(
            candidate,
            status=IngestionCandidateStatus.APPROVED,
            approval=approval,
        )

    def reject(
        self,
        candidate_id: str,
        *,
        decided_by: IngestionRequester,
        reason: str,
        decided_at: datetime | None = None,
    ) -> IngestionCandidate:
        """把 pending candidate 变为 rejected，但不调用索引或网络。"""
        candidate = self._require_pending(candidate_id)
        approval = IngestionApproval(
            status=ApprovalStatus.REJECTED,
            decided_at=decided_at or datetime.now(timezone.utc),
            decided_by=decided_by,
            reason=reason,
        )
        return self._replace_state(
            candidate,
            status=IngestionCandidateStatus.REJECTED,
            approval=approval,
        )

    @staticmethod
    def summarize(candidate: IngestionCandidate) -> IngestionCandidateSummary:
        """构造不含正文与 metadata 的安全展示摘要。"""
        document = candidate.document
        return IngestionCandidateSummary(
            candidate_id=candidate.candidate_id,
            title=document.title,
            source_type=document.source_type,
            source_uri=document.source_uri,
            content_hash=document.content_hash,
            content_length=len(document.content),
            trust_level=document.trust_level,
            policy_decision=candidate.policy_decision,
            policy_reasons=candidate.policy_reasons,
            status=candidate.status,
            approval_status=candidate.approval.status,
        )

    @staticmethod
    def _initial_state(decision, timestamp, reasons):
        if decision is PolicyDecisionType.ALLOW:
            return (
                IngestionCandidateStatus.APPROVED,
                IngestionApproval(status=ApprovalStatus.NOT_REQUIRED),
            )
        if decision is PolicyDecisionType.REQUIRE_APPROVAL:
            return (
                IngestionCandidateStatus.PENDING,
                IngestionApproval(status=ApprovalStatus.PENDING),
            )
        return (
            IngestionCandidateStatus.REJECTED,
            IngestionApproval(
                status=ApprovalStatus.REJECTED,
                decided_at=timestamp,
                decided_by=_POLICY_REQUESTER,
                reason=",".join(reason.value for reason in reasons),
            ),
        )

    def _require_pending(self, candidate_id: str) -> IngestionCandidate:
        candidate = self._repository.get(candidate_id)
        if candidate is None:
            raise CandidateNotFoundError(candidate_id)
        if candidate.status is not IngestionCandidateStatus.PENDING:
            raise InvalidCandidateTransitionError(
                f"candidate {candidate_id} 当前为 {candidate.status.value}，"
                "仅 pending 状态允许 approve/reject"
            )
        return candidate

    def _replace_state(
        self,
        candidate: IngestionCandidate,
        *,
        status: IngestionCandidateStatus,
        approval: IngestionApproval,
    ) -> IngestionCandidate:
        payload = candidate.model_dump()
        payload["status"] = status
        payload["approval"] = approval
        updated = IngestionCandidate.model_validate(payload)
        return self._repository.replace(updated)


__all__ = [
    "CandidateNotFoundError",
    "IngestionService",
    "InvalidCandidateTransitionError",
]
