"""External approved candidate 的真实 LangChain incremental indexing 集成。"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr
from sqlalchemy import delete, update

from app.config import get_settings
from app.api.application import create_app
from app.api.ingestion import IngestionOperationsDependencies
from app.ingestion import (
    IngestionCandidateStatus,
    IngestionPrincipalType,
    IngestionRequester,
    IngestionService,
    KnowledgeSourceType,
    build_knowledge_document,
)
from app.rag import indexing, pipeline, store as rag_store
from app.rag.chroma import load_index_from_chroma
from app.rag.retrieval import tokenize
from app.rag.store import load_index_cache
from app.sources import (
    ExternalKnowledgeSourceService,
    FIRECRAWL_RECORD_NAMESPACE,
    FirecrawlPagePayload,
    FirecrawlPageSourceAdapter,
    GITHUB_RECORD_NAMESPACE,
    GitHubFilePayload,
    GitHubFileSourceAdapter,
    SourceIngestionError,
)
from app.storage.database import _IngestionCandidateRecord, get_engine
from app.storage.ingestion import (
    SQLiteCandidateRepository,
    SQLiteIndexRunRepository,
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


@pytest.fixture
def persisted_runtime(tmp_path, monkeypatch):
    settings = get_settings()
    embeddings = CountingEmbeddings()
    for name in (
        "chunks", "texts", "docs_registry", "doc_embeddings", "bm25", "model",
        "rerank_model", "RAG_READY", "RAG_INIT_MESSAGE", "INDEX_SOURCE",
    ):
        monkeypatch.setattr(pipeline, name, getattr(pipeline, name))
    monkeypatch.setattr(pipeline, "model", embeddings)
    monkeypatch.setattr(pipeline, "rerank_model", object())
    monkeypatch.setattr(pipeline, "_load_embedding_model", lambda: embeddings)
    monkeypatch.setattr(rag_store, "_load_rerank_model", lambda: object())
    local_dir = tmp_path / "local-corpus"
    local_dir.mkdir()
    (local_dir / "local.txt").write_text("local durable content", encoding="utf-8")
    ok, message = pipeline.rebuild_and_persist_index(str(local_dir))
    assert ok, message
    domain = IngestionService(SQLiteCandidateRepository(settings.sqlite_db_path))
    source_service = ExternalKnowledgeSourceService(
        requester=_requester(), ingestion_service=domain,
        embeddings=embeddings, settings=settings,
    )
    return SimpleNamespace(
        settings=settings, embeddings=embeddings, local_dir=local_dir,
        domain=domain, source_service=source_service,
        runs=SQLiteIndexRunRepository(settings.sqlite_db_path),
    )


def _index_with_run(runtime, candidate):
    run = runtime.runs.create_running(candidate.candidate_id)
    try:
        result = runtime.source_service.index_approved(candidate.candidate_id)
    except SourceIngestionError as exc:
        runtime.runs.fail(run.id, error_code=exc.code)
        raise
    runtime.runs.succeed(
        run.id, **{name.removeprefix("num_"): count for name, count in result.items()},
    )
    return result


def test_force_rebuild_preserves_external_records_runtime_and_success_history(persisted_runtime):
    runtime = persisted_runtime
    keyword = "EXTERNAL_REBUILD_KEYWORD_9271"
    github = _approve(runtime.domain, _github_document(f"github {keyword}"))
    firecrawl = _approve(runtime.domain, _firecrawl_document("firecrawl preserved"))
    for candidate in (github, firecrawl):
        _index_with_run(runtime, candidate)
    approved_only = _approve(runtime.domain, _github_document("APPROVED_ONLY_9271"))
    assert approved_only.status is IngestionCandidateStatus.APPROVED
    history = runtime.runs.list()
    # 模拟遗留一致性问题：external keys 存在，但实际 Chroma records 缺失。
    vector_store = indexing.create_vector_store(None)
    vector_store.delete(vector_store.get(where={"source_type": "github"})["ids"])
    runtime.embeddings.reset()

    ok, message = pipeline.rebuild_and_persist_index(
        str(runtime.local_dir), force_rebuild=True,
    )

    assert ok, message
    persisted = indexing.create_vector_store(None).get(include=["documents", "metadatas"])
    assert set(persisted["documents"]) == {
        "local durable content", f"github {keyword}", "firecrawl preserved",
    }
    assert set(runtime.embeddings.document_texts) == set(persisted["documents"])
    assert set(pipeline.texts) == set(persisted["documents"])
    assert pipeline.RAG_READY is True
    assert len(pipeline.docs_registry) == 3
    assert pipeline.doc_embeddings.shape == (3, 2)
    assert pipeline.bm25.corpus_size == 3
    github_position = pipeline.texts.index(f"github {keyword}")
    assert all(
        pipeline.bm25.doc_freqs[github_position].get(token, 0) > 0
        for token in tokenize(keyword)
    )
    assert runtime.runs.list() == history
    for source_type, namespace in (
        ("local_file", indexing.LOCAL_RECORD_NAMESPACE),
        ("github", GITHUB_RECORD_NAMESPACE),
        ("firecrawl", FIRECRAWL_RECORD_NAMESPACE),
    ):
        keys = set(indexing.create_record_manager(namespace=namespace).list_keys())
        actual_ids = {
            record_id for record_id, metadata in zip(persisted["ids"], persisted["metadatas"])
            if metadata["source_type"] == source_type
        }
        assert keys == actual_ids
        assert keys
    runtime.embeddings.reset()
    for candidate in (github, firecrawl):
        assert runtime.source_service.index_approved(candidate.candidate_id)["num_skipped"] == 1
    assert runtime.embeddings.document_texts == []


@pytest.mark.parametrize("document_factory", [_github_document, _firecrawl_document])
@pytest.mark.parametrize("newer_status", ["success", "failed", "running"])
def test_force_rebuild_restores_latest_successful_source_version(
    persisted_runtime, monkeypatch, document_factory, newer_status,
):
    runtime = persisted_runtime
    v1 = _approve(runtime.domain, document_factory("successful old version"))
    _index_with_run(runtime, v1)
    v2 = _approve(runtime.domain, document_factory("new version"))
    assert v1.document.source_id == v2.document.source_id
    if newer_status == "success":
        _index_with_run(runtime, v2)
    elif newer_status == "failed":
        # 索引写入已发生、runtime reload 失败，SUCCESS 历史仍应决定恢复版本。
        monkeypatch.setattr(runtime.source_service, "_runtime_reload", lambda: (False, "failed"))
        with pytest.raises(SourceIngestionError, match="reload"):
            _index_with_run(runtime, v2)
    else:
        runtime.runs.create_running(v2.candidate_id)
    history = runtime.runs.list()

    ok, message = pipeline.rebuild_and_persist_index(
        str(runtime.local_dir), force_rebuild=True,
    )

    assert ok, message
    expected = "new version" if newer_status == "success" else "successful old version"
    persisted, _ = load_index_from_chroma()
    assert {chunk["text"] for chunk in persisted} == {"local durable content", expected}
    assert set(pipeline.texts) == {"local durable content", expected}
    assert runtime.runs.list() == history


@pytest.mark.parametrize("local_change", ["modified", "empty", "empty_force"])
def test_local_sync_retains_external_and_builds_complete_runtime(persisted_runtime, local_change):
    runtime = persisted_runtime
    for factory in (_github_document, _firecrawl_document):
        _index_with_run(runtime, _approve(runtime.domain, factory("external durable content")))
    external_keys = {
        namespace: indexing.create_record_manager(namespace=namespace).list_keys()
        for namespace in indexing.EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE.values()
    }
    path = runtime.local_dir / "local.txt"
    if local_change == "modified":
        path.write_text("local changed content", encoding="utf-8")
    else:
        path.unlink()
    runtime.embeddings.reset()

    ok, message = pipeline.rebuild_and_persist_index(
        str(runtime.local_dir), force_rebuild=local_change == "empty_force",
    )

    assert ok, message
    assert pipeline.RAG_READY is True
    sources = {chunk["metadata"]["source_type"] for chunk in pipeline.chunks}
    assert {"github", "firecrawl"} <= sources
    assert pipeline.bm25.corpus_size == len(pipeline.chunks)
    assert pipeline.texts.count("external durable content") == 2
    if local_change == "modified":
        assert runtime.embeddings.document_texts == ["local changed content"]
    else:
        assert sources == {"github", "firecrawl"}
        assert indexing.create_record_manager().list_keys() == []
        assert runtime.embeddings.document_texts == (
            ["external durable content"] * 2 if local_change == "empty_force" else []
        )
    for namespace, keys in external_keys.items():
        assert indexing.create_record_manager(namespace=namespace).list_keys() == keys


@pytest.mark.parametrize("damage", ["missing", "invalid_json", "unsupported", "pending"])
def test_global_rebuild_validates_successful_candidates_before_reset(
    persisted_runtime, damage,
):
    runtime = persisted_runtime
    if damage == "unsupported":
        document = build_knowledge_document(
            source_type=KnowledgeSourceType.WEB, source_uri="https://example.com/unsupported",
            title="unsupported", content="persisted unsupported source",
        )
    else:
        document = _github_document("persisted external source")
    candidate = runtime.domain.propose(document=document, requester=_requester())
    if damage != "pending":
        candidate = runtime.domain.approve(candidate.candidate_id, decided_by=_requester())
    runtime.runs.succeed(runtime.runs.create_running(candidate.candidate_id).id, added=1)
    with get_engine(runtime.settings.sqlite_db_path).begin() as connection:
        if damage == "missing":
            connection.execute(delete(_IngestionCandidateRecord))
        elif damage == "invalid_json":
            connection.execute(update(_IngestionCandidateRecord).values(candidate_json="{}"))
    old_chunks = pipeline.chunks
    old_manifest = (runtime.settings.index_store_dir / "index_manifest.json").read_bytes()
    old_ids = indexing.create_vector_store(None).get(include=[])["ids"]

    ok, _message = pipeline.rebuild_and_persist_index(
        str(runtime.local_dir), force_rebuild=True,
    )

    assert ok is False
    assert pipeline.chunks is old_chunks
    assert pipeline.RAG_READY is True
    assert indexing.create_vector_store(None).get(include=[])["ids"] == old_ids
    assert (runtime.settings.index_store_dir / "index_manifest.json").read_bytes() == old_manifest


def test_upload_approve_index_retrieve_and_force_rebuild(persisted_runtime, monkeypatch):
    runtime = persisted_runtime
    keyword = "UPLOAD_V1_UNIQUE_987654"
    # 只替代模型推理；保留真实 SQLite、LangChain index、Chroma、reload 和检索。
    reranker = SimpleNamespace(predict=lambda pairs: [float(keyword in text) for _, text in pairs])
    monkeypatch.setattr(rag_store, "_load_rerank_model", lambda: reranker)
    settings = runtime.settings.model_copy(update={
        "ingestion_admin_api_enabled": True,
        "ingestion_admin_api_key": SecretStr("upload-test-admin"),
    })
    deps = IngestionOperationsDependencies(
        ingestion_service=runtime.domain, source_service=runtime.source_service,
        index_runs=runtime.runs, requester=_requester(),
        admin_api_key=settings.ingestion_admin_api_key,
    )
    client = TestClient(create_app(settings=settings, ingestion_dependencies=deps))
    headers = {"X-Ingestion-Admin-Key": "upload-test-admin"}
    response = client.post(
        "/admin/ingestion/sources/upload", headers=headers,
        files={"file": ("unique.txt", keyword.encode())},
    )
    assert response.status_code == 201, response.text
    candidate_id = response.json()["candidate_id"]
    path = f"/admin/ingestion/candidates/{candidate_id}"
    assert response.json()["status"] == "pending"
    assert client.post(f"{path}/index", headers=headers).status_code == 409
    assert pipeline.texts == ["local durable content"]
    assert keyword not in indexing.create_vector_store(None).get()["documents"]
    assert client.post(f"{path}/approve", headers=headers, json={}).status_code == 200
    assert runtime.runs.list() == []
    assert keyword not in indexing.create_vector_store(None).get()["documents"]

    indexed = client.post(f"{path}/index", headers=headers)
    assert indexed.status_code == 200, indexed.text
    assert indexed.json()["status"] == "success"
    assert indexed.json()["added"] == 1
    runtime.embeddings.reset()
    assert client.post(f"{path}/index", headers=headers).json()["skipped"] == 1
    assert runtime.embeddings.document_texts == []
    duplicate = client.post(
        "/admin/ingestion/sources/upload", headers=headers,
        files={"file": ("again.txt", keyword.encode())},
    )
    assert duplicate.json()["candidate_id"] == candidate_id
    assert duplicate.json()["status"] == "approved"
    assert runtime.domain.repository.count() == 1

    for after_rebuild in (False, True):
        if after_rebuild:
            # 从持久 candidate 全文恢复，不依赖临时文件或现有向量记录。
            history = runtime.runs.list()
            indexing.create_vector_store(None).delete_collection()
            ok, message = pipeline.rebuild_and_persist_index(
                str(runtime.local_dir), force_rebuild=True,
            )
            assert ok, message
            assert runtime.runs.list() == history
        assert pipeline.bm25.corpus_size == len(pipeline.chunks) == 2
        upload_position = pipeline.texts.index(keyword)
        assert all(pipeline.bm25.doc_freqs[upload_position].get(token, 0) for token in tokenize(keyword))
        # 禁止 Chroma 查询失败被本地向量 fallback 掩盖。
        def no_fallback(*args, **kwargs):
            pytest.fail("Chroma retrieval unexpectedly fell back to local vectors")
        monkeypatch.setattr("app.rag.retrieval._retrieve_with_local_embeddings", no_fallback)
        hits = pipeline.search_relevant_chunks(keyword, top_k=2)
        hit = next(item for item in hits if keyword in item["text"])
        assert hit["vector_score"] > 0.99
        assert hit["metadata"]["source_type"] == "upload"
        assert hit["source_path"].startswith("upload://local/")
        assert pipeline.rerank(hits, keyword)[0]["text"] == keyword
        keys = indexing.create_record_manager(namespace=indexing.UPLOAD_RECORD_NAMESPACE).list_keys()
        records = indexing.create_vector_store(None).get(where={"source_type": "upload"})
        assert set(keys) == set(records["ids"])
        assert len(keys) == 1
        assert [item.candidate_id for item in SQLiteCandidateRepository(
            runtime.settings.sqlite_db_path,
        ).load_latest_successfully_indexed()] == [candidate_id]
