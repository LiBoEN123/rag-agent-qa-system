"""External approved candidate 的真实 LangChain incremental indexing 集成。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.ingestion import (
    IngestionCandidateStatus,
    IngestionPrincipalType,
    IngestionRequester,
    IngestionService,
)
from app.rag import indexing
from app.rag.chroma import load_index_from_chroma
from app.rag.store import load_index_cache
from app.sources import (
    ExternalKnowledgeSourceService,
    FIRECRAWL_RECORD_NAMESPACE,
    FirecrawlPagePayload,
    FirecrawlPageSourceAdapter,
    GITHUB_RECORD_NAMESPACE,
    GitHubFilePayload,
    GitHubFileSourceAdapter,
)


FIXED_TIME = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)


class CountingEmbeddings(Embeddings):
    """不访问网络的确定性 embedding，并记录 document 调用。"""

    def __init__(self) -> None:
        self.document_texts: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float(len(text) + 1), float(sum(map(ord, text)) % 997 + 1)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_texts.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def reset(self) -> None:
        self.document_texts.clear()


def _requester():
    return IngestionRequester(
        principal_id="external-index-operator",
        principal_type=IngestionPrincipalType.SYSTEM,
        source="source-test",
    )


def _github_document(content: str):
    return GitHubFileSourceAdapter().to_document(
        GitHubFilePayload(
            owner="owner",
            repository="repo",
            path="README.md",
            content=content,
            resolved_sha="a" * 40,
            mime_type="text/markdown",
        ),
        fetched_at=FIXED_TIME,
    )


def _firecrawl_document(content: str):
    return FirecrawlPageSourceAdapter().to_document(
        FirecrawlPagePayload(
            requested_url="https://example.com/article",
            final_url="https://example.com/article",
            title="Article",
            content=content,
            mime_type="text/html",
            status_code=200,
        ),
        fetched_at=FIXED_TIME,
    )


def _approve(service: IngestionService, document):
    candidate = service.propose(document=document, requester=_requester())
    assert candidate.status is IngestionCandidateStatus.PENDING
    return service.approve(candidate.candidate_id, decided_by=_requester())


def test_real_external_incremental_indexing_skips_updates_and_isolates_sources(tmp_path):
    settings = get_settings()
    embeddings = CountingEmbeddings()
    local_dir = tmp_path / "local-corpus"
    local_dir.mkdir()
    (local_dir / "local.txt").write_text("local durable content", encoding="utf-8")
    local_documents = indexing.load_local_langchain_documents(str(local_dir))
    local_result = indexing.sync_documents(local_documents, embeddings, settings=settings)
    assert local_result == {
        "num_added": 1,
        "num_updated": 0,
        "num_skipped": 0,
        "num_deleted": 0,
    }

    domain = IngestionService()
    runtime_sources: list[set[str]] = []

    def reload_runtime():
        result = load_index_cache(model=embeddings, rerank_model=object())
        runtime_sources.append(
            {
                chunk["metadata"]["source_type"]
                for chunk in result["chunks"]
            }
        )
        return bool(result["ok"]), str(result["message"])

    source_service = ExternalKnowledgeSourceService(
        requester=_requester(),
        ingestion_service=domain,
        embeddings=embeddings,
        settings=settings,
        runtime_reload=reload_runtime,
    )

    github_v1 = _approve(domain, _github_document("github content v1"))
    vector_count_before_proposal = len(local_documents)
    chunks_before, _ = load_index_from_chroma(settings)
    assert len(chunks_before) == vector_count_before_proposal

    embeddings.reset()
    github_first = source_service.index_approved(github_v1.candidate_id)
    assert github_first == {
        "num_added": 1,
        "num_updated": 0,
        "num_skipped": 0,
        "num_deleted": 0,
    }
    assert embeddings.document_texts == ["github content v1"]

    embeddings.reset()
    github_same = source_service.index_approved(github_v1.candidate_id)
    assert github_same == {
        "num_added": 0,
        "num_updated": 0,
        "num_skipped": 1,
        "num_deleted": 0,
    }
    assert embeddings.document_texts == []

    firecrawl_v1 = _approve(domain, _firecrawl_document("firecrawl content v1"))
    embeddings.reset()
    firecrawl_first = source_service.index_approved(firecrawl_v1.candidate_id)
    assert firecrawl_first["num_added"] == 1
    assert embeddings.document_texts == ["firecrawl content v1"]

    embeddings.reset()
    firecrawl_same = source_service.index_approved(firecrawl_v1.candidate_id)
    assert firecrawl_same["num_skipped"] == 1
    assert embeddings.document_texts == []

    github_v2 = _approve(domain, _github_document("github content v2"))
    assert github_v2.document.source_id == github_v1.document.source_id
    assert github_v2.candidate_id != github_v1.candidate_id
    embeddings.reset()
    github_modified = source_service.index_approved(github_v2.candidate_id)
    assert github_modified["num_added"] == 1
    assert github_modified["num_deleted"] == 1
    assert embeddings.document_texts == ["github content v2"]

    firecrawl_v2 = _approve(domain, _firecrawl_document("firecrawl content v2"))
    embeddings.reset()
    firecrawl_modified = source_service.index_approved(firecrawl_v2.candidate_id)
    assert firecrawl_modified["num_added"] == 1
    assert firecrawl_modified["num_deleted"] == 1
    assert embeddings.document_texts == ["firecrawl content v2"]

    chunks, _ = load_index_from_chroma(settings)
    by_source = {
        chunk["metadata"]["source_type"]: chunk["text"] for chunk in chunks
    }
    assert by_source == {
        "local_file": "local durable content",
        "github": "github content v2",
        "firecrawl": "firecrawl content v2",
    }
    assert runtime_sources[-1] == {"local_file", "github", "firecrawl"}
    assert indexing.create_record_manager(
        settings,
        namespace=GITHUB_RECORD_NAMESPACE,
    ).list_keys()
    assert indexing.create_record_manager(
        settings,
        namespace=FIRECRAWL_RECORD_NAMESPACE,
    ).list_keys()

    embeddings.reset()
    local_again = indexing.sync_documents(local_documents, embeddings, settings=settings)
    assert local_again["num_skipped"] == 1
    assert local_again["num_deleted"] == 0
    assert embeddings.document_texts == []
    after_local_full, _ = load_index_from_chroma(settings)
    assert {chunk["metadata"]["source_type"] for chunk in after_local_full} == {
        "local_file",
        "github",
        "firecrawl",
    }


def test_pending_external_candidate_cannot_index(monkeypatch):
    domain = IngestionService()
    pending = domain.propose(document=_github_document("pending"), requester=_requester())
    service = ExternalKnowledgeSourceService(
        requester=_requester(),
        ingestion_service=domain,
        embeddings=CountingEmbeddings(),
        runtime_reload=lambda: (True, "unused"),
    )
    calls = 0

    def forbidden_sync(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr("app.sources.service.sync_documents", forbidden_sync)
    with pytest.raises(ValueError, match="approved"):
        service.index_approved(pending.candidate_id)
    assert calls == 0
