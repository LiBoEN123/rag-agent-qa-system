"""独立于 RAG、Agent 与 MCP 的 Knowledge ingestion domain。"""

from app.ingestion.contracts import (
    ApprovalStatus,
    IngestionApproval,
    IngestionCandidate,
    IngestionCandidateStatus,
    IngestionCandidateSummary,
    IngestionPolicyDecision,
    IngestionPrincipalType,
    IngestionRequester,
    KnowledgeDocument,
    KnowledgeSourceAdapter,
    KnowledgeSourceType,
    PolicyDecisionType,
    PolicyReason,
    TrustLevel,
)
from app.ingestion.factory import build_knowledge_document
from app.ingestion.policy import (
    INGESTION_MAX_CONTENT_CHARS,
    evaluate_ingestion_candidate,
)
from app.ingestion.repository import (
    CandidateRepository,
    InMemoryCandidateRepository,
)
from app.ingestion.service import (
    CandidateNotFoundError,
    IngestionService,
    InvalidCandidateTransitionError,
)


__all__ = [
    "ApprovalStatus",
    "CandidateNotFoundError",
    "CandidateRepository",
    "INGESTION_MAX_CONTENT_CHARS",
    "InMemoryCandidateRepository",
    "IngestionApproval",
    "IngestionCandidate",
    "IngestionCandidateStatus",
    "IngestionCandidateSummary",
    "IngestionPolicyDecision",
    "IngestionPrincipalType",
    "IngestionRequester",
    "IngestionService",
    "InvalidCandidateTransitionError",
    "KnowledgeDocument",
    "KnowledgeSourceAdapter",
    "KnowledgeSourceType",
    "PolicyDecisionType",
    "PolicyReason",
    "TrustLevel",
    "build_knowledge_document",
    "evaluate_ingestion_candidate",
]
