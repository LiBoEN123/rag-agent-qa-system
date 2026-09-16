"""Phase 9 policy、idempotency、lifecycle 与零副作用 contracts。"""

from __future__ import annotations

import importlib
import socket
from datetime import datetime, timedelta, timezone

import pytest

from app.ingestion import (
    ApprovalStatus,
    CandidateNotFoundError,
    InMemoryCandidateRepository,
    IngestionCandidateStatus,
    IngestionPrincipalType,
    IngestionRequester,
    IngestionService,
    InvalidCandidateTransitionError,
    KnowledgeSourceType,
    PolicyDecisionType,
    PolicyReason,
    TrustLevel,
    build_knowledge_document,
)


FIXED_TIME = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


def _requester(principal_id="phase9-requester"):
    return IngestionRequester(
        principal_id=principal_id,
        principal_type=IngestionPrincipalType.USER,
        source="pytest",
    )


def _document(
    *,
    source_type=KnowledgeSourceType.WEB,
    source_uri="https://example.com/page",
    content="knowledge",
    trust_level=None,
):
    return build_knowledge_document(
        source_type=source_type,
        source_uri=source_uri,
        title="Knowledge",
        content=content,
        fetched_at=FIXED_TIME,
        trust_level=trust_level,
    )


def test_trusted_local_is_automatically_approved_without_indexing(tmp_path):
    document = _document(
        source_type=KnowledgeSourceType.LOCAL_FILE,
        source_uri=str(tmp_path / "guide.md"),
    )
    candidate = IngestionService().propose(
        document=document,
        requester=_requester(),
    )

    assert document.trust_level is TrustLevel.TRUSTED_INTERNAL
    assert candidate.policy_decision is PolicyDecisionType.ALLOW
    assert candidate.policy_reasons == (PolicyReason.TRUSTED_LOCAL,)
    assert candidate.status is IngestionCandidateStatus.APPROVED
    assert candidate.approval.status is ApprovalStatus.NOT_REQUIRED


@pytest.mark.parametrize(
    ("source_type", "source_uri"),
    [
        (KnowledgeSourceType.UPLOAD, "upload://tenant/upload-1"),
        (KnowledgeSourceType.GITHUB, "github://owner/repo/docs/guide.md"),
        (KnowledgeSourceType.WEB, "https://example.com/web"),
        (KnowledgeSourceType.FIRECRAWL, "https://example.com/crawl"),
    ],
)
def test_external_sources_default_to_pending_approval(source_type, source_uri):
    candidate = IngestionService().propose(
        document=_document(source_type=source_type, source_uri=source_uri),
        requester=_requester(),
    )

    assert candidate.policy_decision is PolicyDecisionType.REQUIRE_APPROVAL
    assert candidate.status is IngestionCandidateStatus.PENDING
    assert candidate.approval.status is ApprovalStatus.PENDING
    assert PolicyReason.UNTRUSTED_EXTERNAL in candidate.policy_reasons


def test_caller_marked_trusted_external_github_can_be_allowed():
    candidate = IngestionService().propose(
        document=_document(
            source_type=KnowledgeSourceType.GITHUB,
            source_uri="github://owner/repo/docs/guide.md",
            trust_level=TrustLevel.TRUSTED_EXTERNAL,
        ),
        requester=_requester(),
    )

    assert candidate.policy_decision is PolicyDecisionType.ALLOW
    assert candidate.status is IngestionCandidateStatus.APPROVED
    assert candidate.approval.status is ApprovalStatus.NOT_REQUIRED


def test_empty_and_oversized_content_are_denied_without_truncation():
    empty = IngestionService().propose(
        document=_document(content=" \r\n\t"),
        requester=_requester(),
    )
    oversized_document = _document(content="x" * 11)
    oversized = IngestionService(max_content_chars=10).propose(
        document=oversized_document,
        requester=_requester(),
    )

    assert empty.status is IngestionCandidateStatus.REJECTED
    assert empty.policy_reasons == (PolicyReason.EMPTY_CONTENT,)
    assert oversized.status is IngestionCandidateStatus.REJECTED
    assert oversized.policy_reasons == (PolicyReason.CONTENT_TOO_LARGE,)
    assert oversized.document.content == "x" * 11


def test_deployment_can_fail_closed_for_a_recognized_but_unsupported_source():
    candidate = IngestionService(
        supported_source_types={KnowledgeSourceType.LOCAL_FILE}
    ).propose(
        document=_document(
            source_type=KnowledgeSourceType.UPLOAD,
            source_uri="upload://tenant/upload-1",
        ),
        requester=_requester(),
    )

    assert candidate.policy_decision is PolicyDecisionType.DENY
    assert candidate.policy_reasons == (PolicyReason.UNSUPPORTED_SOURCE,)
    assert candidate.status is IngestionCandidateStatus.REJECTED


def test_same_candidate_propose_is_idempotent_and_keeps_first_request():
    repository = InMemoryCandidateRepository()
    service = IngestionService(repository)
    document = _document()
    first = service.propose(
        document=document,
        requester=_requester("first-user"),
        requested_at=FIXED_TIME,
    )
    second = service.propose(
        document=document,
        requester=_requester("second-user"),
        requested_at=FIXED_TIME + timedelta(days=1),
    )

    assert first is second
    assert first.candidate_id == second.candidate_id
    assert second.requested_by.principal_id == "first-user"
    assert second.requested_at == FIXED_TIME
    assert repository.count() == 1


def test_changed_content_creates_a_second_logical_candidate():
    repository = InMemoryCandidateRepository()
    service = IngestionService(repository)
    first = service.propose(
        document=_document(content="version one"),
        requester=_requester(),
    )
    second = service.propose(
        document=_document(content="version two"),
        requester=_requester(),
    )

    assert first.candidate_id != second.candidate_id
    assert repository.count() == 2


def test_approve_pending_candidate_and_reject_terminal_reversal():
    service = IngestionService()
    pending = service.propose(
        document=_document(),
        requester=_requester(),
    )
    approved = service.approve(
        pending.candidate_id,
        decided_by=_requester("reviewer"),
        reason="provenance verified",
        decided_at=FIXED_TIME,
    )

    assert approved.status is IngestionCandidateStatus.APPROVED
    assert approved.approval.status is ApprovalStatus.APPROVED
    assert approved.document == pending.document
    with pytest.raises(InvalidCandidateTransitionError, match="仅 pending"):
        service.reject(
            approved.candidate_id,
            decided_by=_requester("reviewer"),
            reason="reverse",
        )


def test_reject_pending_candidate_and_reject_terminal_reversal():
    service = IngestionService()
    pending = service.propose(
        document=_document(source_uri="https://example.com/reject"),
        requester=_requester(),
    )
    rejected = service.reject(
        pending.candidate_id,
        decided_by=_requester("reviewer"),
        reason="provenance not acceptable",
        decided_at=FIXED_TIME,
    )

    assert rejected.status is IngestionCandidateStatus.REJECTED
    assert rejected.approval.status is ApprovalStatus.REJECTED
    with pytest.raises(InvalidCandidateTransitionError, match="仅 pending"):
        service.approve(
            rejected.candidate_id,
            decided_by=_requester("reviewer"),
        )


def test_candidate_lifecycle_has_zero_rag_chroma_mcp_web_or_network_calls(
    monkeypatch,
):
    rag_indexing = importlib.import_module("app.rag.indexing")
    rag_pipeline = importlib.import_module("app.rag.pipeline")
    mcp_client = importlib.import_module("app.tools.mcp.client")
    web_fetch_module = importlib.import_module("app.tools.web_fetch")
    calls = {
        "chroma_writes": 0,
        "rag_rebuilds": 0,
        "mcp_calls": 0,
        "web_calls": 0,
        "network_calls": 0,
    }

    def forbidden(name):
        def fail(*_args, **_kwargs):
            calls[name] += 1
            pytest.fail(f"Phase 9 不得产生 {name}")

        return fail

    monkeypatch.setattr(
        rag_indexing,
        "sync_documents",
        forbidden("chroma_writes"),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "rebuild_and_persist_index",
        forbidden("rag_rebuilds"),
    )
    monkeypatch.setattr(
        mcp_client,
        "initialize_mcp_runtime",
        forbidden("mcp_calls"),
    )
    monkeypatch.setattr(
        web_fetch_module,
        "web_fetch",
        forbidden("web_calls"),
    )
    monkeypatch.setattr(socket.socket, "connect", forbidden("network_calls"))

    service = IngestionService()
    pending = service.propose(
        document=_document(),
        requester=_requester(),
    )
    approved = service.approve(
        pending.candidate_id,
        decided_by=_requester("reviewer"),
    )
    other = service.propose(
        document=_document(source_uri="https://example.com/other"),
        requester=_requester(),
    )
    service.reject(
        other.candidate_id,
        decided_by=_requester("reviewer"),
        reason="not approved",
    )

    assert approved.status is IngestionCandidateStatus.APPROVED
    service.delete_candidate(approved.candidate_id)
    service.delete_candidate(other.candidate_id)
    assert service.repository.count() == 0
    assert calls == {
        "chroma_writes": 0,
        "rag_rebuilds": 0,
        "mcp_calls": 0,
        "web_calls": 0,
        "network_calls": 0,
    }


def test_delete_candidate_missing_and_concurrent_delete(monkeypatch):
    service = IngestionService()
    with pytest.raises(CandidateNotFoundError):
        service.delete_candidate("missing")
    candidate = service.propose(document=_document(), requester=_requester())
    monkeypatch.setattr(service.repository, "delete", lambda _id: False)
    with pytest.raises(CandidateNotFoundError):
        service.delete_candidate(candidate.candidate_id)
