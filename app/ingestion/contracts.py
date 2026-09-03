"""永久知识进入流程的版本化 Pydantic domain contracts。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictStr,
    field_validator,
    model_validator,
)

from app.common.redaction import redact_sensitive_text
from app.ingestion.identity import (
    build_candidate_id,
    build_content_hash,
    build_document_uid,
    build_source_id,
    canonicalize_source_uri,
    is_stable_id,
)
from app.ingestion.normalization import (
    CONTENT_NORMALIZATION,
    prepare_content,
    prepare_metadata,
)


KNOWLEDGE_DOCUMENT_SCHEMA_VERSION = 1
INGESTION_CANDIDATE_SCHEMA_VERSION = 1
_REQUESTER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$")
_REQUEST_SOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class KnowledgeSourceType(str, Enum):
    """Phase 9 支持的受控知识来源类型。"""

    LOCAL_FILE = "local_file"
    UPLOAD = "upload"
    GITHUB = "github"
    WEB = "web"
    FIRECRAWL = "firecrawl"


class TrustLevel(str, Enum):
    """由可信调用边界指定、正文无权改变的来源信任级别。"""

    TRUSTED_INTERNAL = "trusted_internal"
    TRUSTED_EXTERNAL = "trusted_external"
    UNTRUSTED_EXTERNAL = "untrusted_external"


class IngestionCandidateStatus(str, Enum):
    """候选在进入索引之前的 domain 状态。"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ApprovalStatus(str, Enum):
    """知识摄取审批状态；与 MCP tool execution approval 无关。"""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PolicyDecisionType(str, Enum):
    """无副作用策略评估的结果类型。"""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class PolicyReason(str, Enum):
    """固定、安全、可审计的策略原因码。"""

    TRUSTED_LOCAL = "trusted_local"
    TRUSTED_EXTERNAL = "trusted_external"
    UNTRUSTED_EXTERNAL = "untrusted_external"
    SOURCE_REQUIRES_APPROVAL = "source_requires_approval"
    EMPTY_CONTENT = "empty_content"
    CONTENT_TOO_LARGE = "content_too_large"
    UNSUPPORTED_SOURCE = "unsupported_source"
    INVALID_TRUST_ASSIGNMENT = "invalid_trust_assignment"


class IngestionPrincipalType(str, Enum):
    """提出或决定摄取请求的主体类别。"""

    USER = "user"
    SERVICE = "service"
    SYSTEM = "system"


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KnowledgeDocument(_FrozenContract):
    """进入 Policy 之前的完整、安全、可追溯知识文档。"""

    schema_version: int = Field(
        default=KNOWLEDGE_DOCUMENT_SCHEMA_VERSION,
        frozen=True,
    )
    source_id: StrictStr
    document_uid: StrictStr
    source_type: KnowledgeSourceType
    source_uri: StrictStr = Field(max_length=2048)
    title: StrictStr = Field(min_length=1, max_length=512)
    content: StrictStr = Field(repr=False)
    content_hash: StrictStr
    content_normalization: StrictStr = Field(
        default=CONTENT_NORMALIZATION,
        frozen=True,
    )
    content_redacted: bool = False
    version: StrictStr | None = Field(default=None, max_length=256)
    fetched_at: AwareDatetime
    mime_type: StrictStr | None = Field(default=None, max_length=255)
    language: StrictStr | None = Field(default=None, max_length=64)
    metadata: dict[str, JsonValue] = Field(default_factory=dict, repr=False)
    metadata_redacted: bool = False
    trust_level: TrustLevel

    @model_validator(mode="before")
    @classmethod
    def _prepare_untrusted_text_and_metadata(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        prepared = dict(data)
        if "content" in prepared:
            content, redacted = prepare_content(prepared["content"])
            prepared["content"] = content
            prepared["content_redacted"] = bool(
                prepared.get("content_redacted", False)
            ) or redacted
        metadata, redacted = prepare_metadata(prepared.get("metadata", {}))
        prepared["metadata"] = metadata
        prepared["metadata_redacted"] = bool(
            prepared.get("metadata_redacted", False)
        ) or redacted
        return prepared

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != KNOWLEDGE_DOCUMENT_SCHEMA_VERSION:
            raise ValueError(
                "不支持的 KnowledgeDocument schema_version: "
                f"{value!r}"
            )
        return value

    @field_validator("title")
    @classmethod
    def _sanitize_title(cls, value: str) -> str:
        title = redact_sensitive_text(value.strip())
        if not title:
            raise ValueError("title 去除首尾空白后不能为空")
        return title

    @field_validator("version", "mime_type", "language")
    @classmethod
    def _normalize_optional_provenance(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if redact_sensitive_text(normalized) != normalized:
            raise ValueError("provenance 字段不能包含凭据")
        return normalized

    @model_validator(mode="after")
    def _validate_identity_and_provenance(self) -> KnowledgeDocument:
        canonical_uri = canonicalize_source_uri(self.source_type, self.source_uri)
        if self.source_uri != canonical_uri:
            raise ValueError("source_uri 必须使用 canonical 格式")
        if not is_stable_id(self.source_id, "src_"):
            raise ValueError("source_id 格式无效")
        if not is_stable_id(self.document_uid, "doc_"):
            raise ValueError("document_uid 格式无效")
        expected_source_id = build_source_id(self.source_type, self.source_uri)
        if self.source_id != expected_source_id:
            raise ValueError("source_id 与 source_type/source_uri 不一致")
        if self.document_uid != build_document_uid(self.source_id):
            raise ValueError("document_uid 与 source_id 不一致")
        if self.content_hash != build_content_hash(self.content):
            raise ValueError("content_hash 与规范化 content 不一致")
        if self.content_normalization != CONTENT_NORMALIZATION:
            raise ValueError("不支持的 content normalization 版本")
        return self


class IngestionRequester(_FrozenContract):
    """禁止以任意自由文本冒充主体的结构化请求者。"""

    principal_id: StrictStr
    principal_type: IngestionPrincipalType
    source: StrictStr

    @field_validator("principal_id")
    @classmethod
    def _validate_principal_id(cls, value: str) -> str:
        if not _REQUESTER_ID_PATTERN.fullmatch(value):
            raise ValueError("principal_id 必须是长度不超过 128 的安全结构化标识")
        if redact_sensitive_text(value) != value:
            raise ValueError("principal_id 不能包含凭据")
        return value

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        if not _REQUEST_SOURCE_PATTERN.fullmatch(value):
            raise ValueError("requester source 必须是安全的小写结构化标识")
        return value


class IngestionApproval(_FrozenContract):
    """只描述永久知识审批，不触发任何索引或外部调用。"""

    status: ApprovalStatus
    decided_at: AwareDatetime | None = None
    decided_by: IngestionRequester | None = None
    reason: StrictStr | None = Field(default=None, max_length=1000, repr=False)

    @field_validator("reason")
    @classmethod
    def _sanitize_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        reason = redact_sensitive_text(value.strip())
        return reason or None

    @model_validator(mode="after")
    def _validate_decision_fields(self) -> IngestionApproval:
        if self.status in {ApprovalStatus.PENDING, ApprovalStatus.NOT_REQUIRED}:
            if self.decided_at is not None or self.decided_by is not None:
                raise ValueError("未决定的 approval 不能包含决定人或决定时间")
            if self.reason is not None:
                raise ValueError("未决定的 approval 不能包含决定原因")
        else:
            if self.decided_at is None or self.decided_by is None:
                raise ValueError("已决定的 approval 必须包含决定人和决定时间")
            if self.status is ApprovalStatus.REJECTED and not self.reason:
                raise ValueError("rejected approval 必须包含原因")
        return self


class IngestionPolicyDecision(_FrozenContract):
    """确定性 ingestion policy 的纯值输出。"""

    decision: PolicyDecisionType
    reasons: tuple[PolicyReason, ...] = Field(min_length=1)


class IngestionCandidate(_FrozenContract):
    """已评估但尚未进入任何持久索引的摄取候选。"""

    schema_version: int = Field(
        default=INGESTION_CANDIDATE_SCHEMA_VERSION,
        frozen=True,
    )
    candidate_id: StrictStr
    document: KnowledgeDocument
    status: IngestionCandidateStatus
    requested_at: AwareDatetime
    requested_by: IngestionRequester
    reason: StrictStr | None = Field(default=None, max_length=1000, repr=False)
    policy_decision: PolicyDecisionType
    policy_reasons: tuple[PolicyReason, ...] = Field(min_length=1)
    approval: IngestionApproval

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != INGESTION_CANDIDATE_SCHEMA_VERSION:
            raise ValueError(
                "不支持的 IngestionCandidate schema_version: "
                f"{value!r}"
            )
        return value

    @field_validator("reason")
    @classmethod
    def _sanitize_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        reason = redact_sensitive_text(value.strip())
        return reason or None

    @model_validator(mode="after")
    def _validate_identity_and_state(self) -> IngestionCandidate:
        if not is_stable_id(self.candidate_id, "cand_"):
            raise ValueError("candidate_id 格式无效")
        expected_id = build_candidate_id(
            self.document.source_id,
            self.document.content_hash,
        )
        if self.candidate_id != expected_id:
            raise ValueError("candidate_id 与 source_id/content_hash 不一致")

        approval_status = self.approval.status
        if self.status is IngestionCandidateStatus.PENDING:
            if approval_status is not ApprovalStatus.PENDING:
                raise ValueError("pending candidate 必须具有 pending approval")
        elif self.status is IngestionCandidateStatus.APPROVED:
            if approval_status not in {
                ApprovalStatus.APPROVED,
                ApprovalStatus.NOT_REQUIRED,
            }:
                raise ValueError("approved candidate 必须已获批或无需审批")
        elif self.status is IngestionCandidateStatus.REJECTED:
            if approval_status is not ApprovalStatus.REJECTED:
                raise ValueError("rejected candidate 必须具有 rejected approval")

        if self.policy_decision is PolicyDecisionType.DENY:
            if self.status is not IngestionCandidateStatus.REJECTED:
                raise ValueError("DENY policy 只能产生 rejected candidate")
        elif self.policy_decision is PolicyDecisionType.ALLOW:
            if (
                self.status is not IngestionCandidateStatus.APPROVED
                or approval_status is not ApprovalStatus.NOT_REQUIRED
            ):
                raise ValueError("ALLOW policy 必须产生无需审批的 approved candidate")
        return self


class IngestionCandidateSummary(_FrozenContract):
    """适合日志/列表展示且不包含正文和 metadata 的安全摘要。"""

    candidate_id: StrictStr
    title: StrictStr
    source_type: KnowledgeSourceType
    source_uri: StrictStr
    content_hash: StrictStr
    content_length: int = Field(ge=0)
    trust_level: TrustLevel
    policy_decision: PolicyDecisionType
    policy_reasons: tuple[PolicyReason, ...]
    status: IngestionCandidateStatus
    approval_status: ApprovalStatus
    requested_at: AwareDatetime


class KnowledgeSourceAdapter(Protocol):
    """把已取得的 payload 纯转换为文档；不得在此协议中发起网络请求。"""

    def to_document(
        self,
        payload: Mapping[str, JsonValue],
        *,
        fetched_at: datetime | None = None,
    ) -> KnowledgeDocument:
        """转换已取得的 source payload。"""


__all__ = [
    "ApprovalStatus",
    "INGESTION_CANDIDATE_SCHEMA_VERSION",
    "IngestionApproval",
    "IngestionCandidate",
    "IngestionCandidateStatus",
    "IngestionCandidateSummary",
    "IngestionPolicyDecision",
    "IngestionPrincipalType",
    "IngestionRequester",
    "KNOWLEDGE_DOCUMENT_SCHEMA_VERSION",
    "KnowledgeDocument",
    "KnowledgeSourceAdapter",
    "KnowledgeSourceType",
    "PolicyDecisionType",
    "PolicyReason",
    "TrustLevel",
]
