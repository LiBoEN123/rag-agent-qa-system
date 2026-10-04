"""External approved candidate 的真实 LangChain incremental indexing 集成。"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
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
from app.indexing import langchain as indexing
from app.rag import runtime as pipeline
from app.indexing import service as indexing_service, manifest as rag_store
from app.indexing.service import IndexingService, IndexingError
from app.api.application import refresh_query_runtime
from app.rag.chroma import load_index_from_chroma
from app.rag.retrieval import tokenize
from app.rag.runtime import _load_index_bundle as load_index_cache
from app.sources import (
    ExternalKnowledgeSourceService,
    FirecrawlPagePayload,
    FirecrawlPageSourceAdapter,
    GitHubFilePayload,
    GitHubFileSourceAdapter,
)
from app.storage.ingestion import _IngestionCandidateRecord
from app.indexing.langchain import FIRECRAWL_RECORD_NAMESPACE, GITHUB_RECORD_NAMESPACE
from app.storage.database import get_engine
from app.storage.ingestion import (
    IndexRunOperation,
    IndexRunStatus,
    SQLiteCandidateRepository,
    SQLiteIndexRunRepository,
)
from app.sources.upload import load_upload_document
from tests.support.structured_document import admission_structure, build_table_docx, imut_admission_structure


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


def _sync_and_load(data_dir, *, force_rebuild=False):
    settings = get_settings().model_copy(update={"rag_file_path": Path(data_dir)})
    try:
        IndexingService(settings=settings, embeddings=pipeline.model).sync_local_corpus(
            force_rebuild=force_rebuild,
        )
    except Exception as exc:
        return False, str(exc)
    return pipeline.load_index_cache(allow_empty=True)


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

    source_service = IndexingService(
        candidates=domain.repository,
        embeddings=embeddings,
        settings=settings,
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
    assert runtime_sources == []
    assert reload_runtime()[0]
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
    service = IndexingService(
        candidates=domain.repository,
        embeddings=CountingEmbeddings(),
    )
    calls = 0

    def forbidden_sync(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr("app.indexing.service.sync_documents", forbidden_sync)
    with pytest.raises(ValueError, match="approved"):
        service.index_approved(pending.candidate_id)
    assert calls == 0


def test_writer_embedding_failure_returns_safe_error_without_creating_index(monkeypatch):
    domain = IngestionService()
    candidate = _approve(domain, _github_document("approved content"))
    service = IndexingService(candidates=domain.repository)

    def failed_model(_settings):
        raise RuntimeError("private model path")

    monkeypatch.setattr(indexing_service, "load_embedding_model", failed_model)
    with pytest.raises(IndexingError) as error:
        service.index_approved(candidate.candidate_id)
    assert error.value.code == "SOURCE_EMBEDDINGS_UNAVAILABLE"
    assert "private" not in str(error.value)
    assert not get_settings().chroma_persist_dir.exists()


@pytest.fixture
def persisted_runtime(tmp_path, monkeypatch):
    settings = get_settings()
    embeddings = CountingEmbeddings()
    for name in (
        "chunks", "docs_registry", "doc_embeddings", "bm25", "model",
        "rerank_model", "RAG_READY", "RAG_INIT_MESSAGE", "INDEX_SOURCE",
    ):
        monkeypatch.setattr(pipeline, name, getattr(pipeline, name))
    monkeypatch.setattr(pipeline, "model", embeddings)
    monkeypatch.setattr(pipeline, "rerank_model", object())
    monkeypatch.setattr(indexing_service, "load_embedding_model", lambda _settings: embeddings)
    monkeypatch.setattr(pipeline, "_load_rerank_model", lambda: object())
    local_dir = tmp_path / "local-corpus"
    local_dir.mkdir()
    (local_dir / "local.txt").write_text("local durable content", encoding="utf-8")
    ok, message = _sync_and_load(str(local_dir))
    assert ok, message
    domain = IngestionService(SQLiteCandidateRepository(settings.sqlite_db_path))
    source_service = ExternalKnowledgeSourceService(
        requester=_requester(), ingestion_service=domain, settings=settings,
    )
    return SimpleNamespace(
        settings=settings, embeddings=embeddings, local_dir=local_dir,
        domain=domain, source_service=source_service,
        indexer=IndexingService(settings=settings, candidates=domain.repository, embeddings=embeddings),
        runs=SQLiteIndexRunRepository(settings.sqlite_db_path),
    )


def _index_with_run(runtime, candidate):
    run = runtime.runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id)
    try:
        result = runtime.indexer.index_approved(candidate.candidate_id)
    except IndexingError as exc:
        runtime.runs.fail(run.id, error_code=exc.code)
        raise
    runtime.runs.succeed(
        run.id, **{name.removeprefix("num_"): count for name, count in result.items()},
    )
    refresh_query_runtime()
    return result


def _remove_client(runtime):
    settings = runtime.settings.model_copy(update={
        "ingestion_admin_api_enabled": True,
        "ingestion_admin_api_key": SecretStr("remove-test-admin"),
    })
    deps = IngestionOperationsDependencies(
        ingestion_service=runtime.domain, source_service=runtime.source_service,
        indexing_service=runtime.indexer, on_index_changed=refresh_query_runtime,
        index_runs=runtime.runs, requester=_requester(),
        admin_api_key=settings.ingestion_admin_api_key,
    )
    return TestClient(
        create_app(settings=settings, ingestion_dependencies=deps),
        headers={"X-Ingestion-Admin-Key": "remove-test-admin"},
    )


def _upload_document(content):
    return load_upload_document(BytesIO(content.encode()), filename="remove.txt", max_bytes=10000)


@pytest.mark.parametrize("factory", [_github_document, _firecrawl_document, _upload_document])
def test_remove_clears_source_runtime_and_history_survives_rebuild_and_reindex(persisted_runtime, factory):
    runtime = persisted_runtime
    candidate = _approve(runtime.domain, factory("REMOVE_UNIQUE_KEYWORD_9271"))
    _index_with_run(runtime, candidate)
    # 同 namespace 的另一 source 及 Local 均须保留。
    other = _approve(runtime.domain, build_knowledge_document(
        source_type=candidate.document.source_type,
        source_uri=candidate.document.source_uri + "-other",
        title="other", content="other external retained", fetched_at=FIXED_TIME,
    ))
    _index_with_run(runtime, other)
    manager = indexing.create_record_manager(namespace=indexing.EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE[candidate.document.source_type])
    runtime.embeddings.reset()
    client = _remove_client(runtime)
    path = f"/admin/ingestion/candidates/{candidate.candidate_id}"
    removed = client.post(f"{path}/remove")
    assert removed.status_code == 200, removed.text
    assert removed.json()["operation"] == "remove"
    assert removed.json()["status"] == "success"
    assert removed.json()["deleted"] == 1
    assert runtime.embeddings.document_texts == []
    assert manager.list_keys(group_ids=[candidate.document.source_id]) == []
    assert indexing.create_vector_store(None).get(where={"source_id": candidate.document.source_id})["ids"] == []
    assert {chunk["text"] for chunk in pipeline.chunks} == {"local durable content", "other external retained"}
    assert pipeline.bm25.corpus_size == len(pipeline.chunks) == 2
    assert len(pipeline.doc_embeddings) == len(pipeline.docs_registry) == 2
    assert all("REMOVE_UNIQUE_KEYWORD_9271" not in hit["text"] for hit in pipeline.search_relevant_chunks("REMOVE_UNIQUE_KEYWORD_9271", top_k=10))
    assert runtime.domain.repository.get(candidate.candidate_id) == candidate
    assert client.post(f"{path}/remove").json()["deleted"] == 0
    for _ in range(2):
        ok, message = _sync_and_load(str(runtime.local_dir), force_rebuild=True)
        assert ok, message
        assert {chunk["text"] for chunk in pipeline.chunks} == {"local durable content", "other external retained"}
        assert manager.list_keys(group_ids=[candidate.document.source_id]) == []
    # 已清除 RecordManager keys，同一 approved candidate 再入库必须 added，而非 skipped。
    indexed = client.post(f"{path}/index")
    assert indexed.status_code == 200, indexed.text
    assert indexed.json()["operation"] == "index"
    assert indexed.json()["added"] == 1
    assert indexed.json()["skipped"] == 0
    ok, message = _sync_and_load(str(runtime.local_dir), force_rebuild=True)
    assert ok, message
    assert "REMOVE_UNIQUE_KEYWORD_9271" in [chunk["text"] for chunk in pipeline.chunks]


@pytest.mark.parametrize("factory", [_github_document, _firecrawl_document])
def test_remove_from_old_candidate_removes_current_source_version(persisted_runtime, factory):
    runtime = persisted_runtime
    v1 = _approve(runtime.domain, factory("old version"))
    v2 = _approve(runtime.domain, factory("\n".join(f"new version section {i}" for i in range(200))))
    _index_with_run(runtime, v1)
    _index_with_run(runtime, v2)
    result = _remove_client(runtime).post(f"/admin/ingestion/candidates/{v1.candidate_id}/remove")
    assert result.status_code == 200, result.text
    assert result.json()["deleted"] > 1
    assert runtime.domain.repository.load_latest_successfully_indexed() == []
    ok, message = _sync_and_load(str(runtime.local_dir), force_rebuild=True)
    assert ok, message
    assert [chunk["text"] for chunk in pipeline.chunks] == ["local durable content"]


def test_force_rebuild_after_deleting_all_removed_candidate_versions(persisted_runtime):
    runtime = persisted_runtime
    versions = [_approve(runtime.domain, _github_document(content)) for content in ("deleted v1", "deleted v2")]
    for candidate in versions:
        _index_with_run(runtime, candidate)
    client = _remove_client(runtime)
    assert client.post(f"/admin/ingestion/candidates/{versions[0].candidate_id}/remove").status_code == 200
    history = runtime.runs.list()
    for candidate in versions:
        assert client.delete(f"/admin/ingestion/candidates/{candidate.candidate_id}").status_code == 204
    for _ in range(2):
        ok, message = _sync_and_load(str(runtime.local_dir), force_rebuild=True)
        assert ok, message
        assert [chunk["text"] for chunk in pipeline.chunks] == ["local durable content"]
        assert indexing.create_vector_store(None).get(where={"source_id": versions[0].document.source_id})["ids"] == []
    assert runtime.runs.list() == history


@pytest.mark.parametrize("missing", ["chroma", "record_manager"])
def test_remove_repairs_partial_persistent_state(persisted_runtime, missing):
    runtime = persisted_runtime
    candidate = _approve(runtime.domain, _github_document("partial state"))
    _index_with_run(runtime, candidate)
    manager = indexing.create_record_manager(namespace=GITHUB_RECORD_NAMESPACE)
    vector_store = indexing.create_vector_store(None)
    if missing == "chroma":
        vector_store.delete(where={"source_id": candidate.document.source_id})
    else:
        manager.delete_keys(manager.list_keys())
    response = _remove_client(runtime).post(f"/admin/ingestion/candidates/{candidate.candidate_id}/remove")
    assert response.status_code == 200, response.text
    assert response.json()["deleted"] == (0 if missing == "chroma" else 1)
    assert manager.list_keys() == []
    assert vector_store.get(where={"source_id": candidate.document.source_id})["ids"] == []
    assert [chunk["text"] for chunk in pipeline.chunks] == ["local durable content"]


@pytest.mark.parametrize("failure", ["chroma", "record_manager"])
def test_failed_remove_retains_authoritative_index_for_rebuild(persisted_runtime, monkeypatch, failure):
    runtime = persisted_runtime
    candidate = _approve(runtime.domain, _github_document("restore after failed remove"))
    _index_with_run(runtime, candidate)
    def fail(*args, **kwargs):
        raise RuntimeError("sensitive backend failure")
    with monkeypatch.context() as patch:
        if failure == "chroma":
            patch.setattr("langchain_chroma.Chroma.delete", fail)
        elif failure == "record_manager":
            patch.setattr("langchain_classic.indexes.SQLRecordManager.delete_keys", fail)
        response = _remove_client(runtime).post(f"/admin/ingestion/candidates/{candidate.candidate_id}/remove")
    assert response.status_code == 500
    assert "sensitive" not in response.text
    run = runtime.runs.latest(candidate.candidate_id)
    assert (run.operation, run.status) == (IndexRunOperation.REMOVE, IndexRunStatus.FAILED)
    assert "sensitive" not in run.model_dump_json()
    assert runtime.domain.repository.load_latest_successfully_indexed() == [candidate]
    ok, message = _sync_and_load(str(runtime.local_dir), force_rebuild=True)
    assert ok, message
    assert {chunk["text"] for chunk in pipeline.chunks} == {"local durable content", "restore after failed remove"}


@pytest.mark.parametrize("raise_error", [False, True])
def test_refresh_failure_keeps_successful_remove_authoritative(persisted_runtime, monkeypatch, raise_error):
    runtime = persisted_runtime
    candidate = _approve(runtime.domain, _github_document("removed permanently"))
    _index_with_run(runtime, candidate)

    def failed_refresh(**kwargs):
        if raise_error:
            raise RuntimeError("private refresh failure")
        return False, "private refresh failure"

    with monkeypatch.context() as patch:
        patch.setattr("app.api.application.load_index_cache", failed_refresh)
        response = _remove_client(runtime).post(f"/admin/ingestion/candidates/{candidate.candidate_id}/remove")
    assert response.status_code == 200
    assert runtime.runs.latest(candidate.candidate_id).status is IndexRunStatus.SUCCESS
    assert runtime.domain.repository.load_latest_successfully_indexed() == []
    ok, message = _sync_and_load(str(runtime.local_dir), force_rebuild=True)
    assert ok, message
    assert [chunk["text"] for chunk in pipeline.chunks] == ["local durable content"]


def test_remove_last_source_publishes_empty_runtime(persisted_runtime):
    runtime = persisted_runtime
    candidate = _approve(runtime.domain, _github_document("last external"))
    _index_with_run(runtime, candidate)
    (runtime.local_dir / "local.txt").unlink()
    assert _sync_and_load(str(runtime.local_dir))[0]
    response = _remove_client(runtime).post(f"/admin/ingestion/candidates/{candidate.candidate_id}/remove")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "success"
    assert pipeline.RAG_READY is False
    assert pipeline.chunks == pipeline.docs_registry == []
    assert pipeline.doc_embeddings is pipeline.bm25 is None
    assert pipeline.search_relevant_chunks("last external") == []
    assert _sync_and_load(str(runtime.local_dir), force_rebuild=True)[0]
    assert pipeline.chunks == []


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

    ok, message = _sync_and_load(
        str(runtime.local_dir), force_rebuild=True,
    )

    assert ok, message
    persisted = indexing.create_vector_store(None).get(include=["documents", "metadatas"])
    assert set(persisted["documents"]) == {
        "local durable content", f"github {keyword}", "firecrawl preserved",
    }
    assert set(runtime.embeddings.document_texts) == set(persisted["documents"])
    assert {chunk["text"] for chunk in pipeline.chunks} == set(persisted["documents"])
    assert pipeline.RAG_READY is True
    assert len(pipeline.docs_registry) == 3
    assert pipeline.doc_embeddings.shape == (3, 2)
    assert pipeline.bm25.corpus_size == 3
    github_position = [chunk["text"] for chunk in pipeline.chunks].index(f"github {keyword}")
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
        assert runtime.indexer.index_approved(candidate.candidate_id)["num_skipped"] == 1
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
        # 部分写入失败也不能替代上一次 SUCCESS 的 authoritative version。
        runtime.indexer.index_approved(v2.candidate_id)
        run = runtime.runs.create_running(v2.candidate_id, source_id=v2.document.source_id)
        runtime.runs.fail(run.id, error_code="INDEX_OPERATION_FAILED")
    else:
        runtime.runs.create_running(v2.candidate_id, source_id=v2.document.source_id)
    history = runtime.runs.list()

    ok, message = _sync_and_load(
        str(runtime.local_dir), force_rebuild=True,
    )

    assert ok, message
    expected = "new version" if newer_status == "success" else "successful old version"
    persisted, _ = load_index_from_chroma()
    assert {chunk["text"] for chunk in persisted} == {"local durable content", expected}
    assert {chunk["text"] for chunk in pipeline.chunks} == {"local durable content", expected}
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

    ok, message = _sync_and_load(
        str(runtime.local_dir), force_rebuild=local_change == "empty_force",
    )

    assert ok, message
    assert pipeline.RAG_READY is True
    sources = {chunk["metadata"]["source_type"] for chunk in pipeline.chunks}
    assert {"github", "firecrawl"} <= sources
    assert pipeline.bm25.corpus_size == len(pipeline.chunks)
    assert [chunk["text"] for chunk in pipeline.chunks].count("external durable content") == 2
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
    runtime.runs.succeed(runtime.runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id, added=1)
    with get_engine(runtime.settings.sqlite_db_path).begin() as connection:
        if damage == "missing":
            connection.execute(delete(_IngestionCandidateRecord))
        elif damage == "invalid_json":
            connection.execute(update(_IngestionCandidateRecord).values(candidate_json="{}"))
    old_chunks = pipeline.chunks
    old_manifest = (runtime.settings.index_store_dir / "index_manifest.json").read_bytes()
    old_ids = indexing.create_vector_store(None).get(include=[])["ids"]

    ok, _message = _sync_and_load(
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
    monkeypatch.setattr(pipeline, "_load_rerank_model", lambda: reranker)
    settings = runtime.settings.model_copy(update={
        "ingestion_admin_api_enabled": True,
        "ingestion_admin_api_key": SecretStr("upload-test-admin"),
    })
    deps = IngestionOperationsDependencies(
        ingestion_service=runtime.domain, source_service=runtime.source_service,
        indexing_service=runtime.indexer, on_index_changed=refresh_query_runtime,
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
    assert [chunk["text"] for chunk in pipeline.chunks] == ["local durable content"]
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
            ok, message = _sync_and_load(
                str(runtime.local_dir), force_rebuild=True,
            )
            assert ok, message
            assert runtime.runs.list() == history
        assert pipeline.bm25.corpus_size == len(pipeline.chunks) == 2
        upload_position = [chunk["text"] for chunk in pipeline.chunks].index(keyword)
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


@pytest.mark.parametrize("old_version", [2, 3, 4])
def test_version_migration_rechunks_sqlite_structure_and_preserves_legacy(persisted_runtime, monkeypatch, old_version):
    runtime = persisted_runtime
    document = load_upload_document(
        BytesIO(build_table_docx(merged=True)), filename="table.docx", max_bytes=2_000_000,
    )
    candidate = _approve(runtime.domain, document)
    # 模拟旧版已成功索引的正文 chunk 与 manifest，SQLite 仍保存完整 structure。
    old_chunk = Document(page_content=document.content, metadata={
        "source_id": document.source_id, "document_uid": document.document_uid,
        "source_type": "upload", "source_uri": document.source_uri,
        "source_path": document.source_uri, "file_name": "table.docx", "file_type": "docx", "position": 1,
    })
    result = indexing.sync_documents(
        [old_chunk], runtime.embeddings, namespace=indexing.UPLOAD_RECORD_NAMESPACE, cleanup="incremental",
    )
    runtime.runs.succeed(runtime.runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id, added=result["num_added"])
    legacy = _approve(runtime.domain, build_knowledge_document(
        source_type="upload", source_uri="upload://local/legacy-docx", title="legacy.docx",
        content="legacy plain evidence", metadata={"file_type": "docx"},
    ))
    _index_with_run(runtime, legacy)
    local_table = runtime.local_dir / "local.docx"
    local_table.write_bytes(build_table_docx())
    old_manifest = {**rag_store.build_current_manifest(str(runtime.local_dir)), "pipeline_version": old_version}
    rag_store.save_manifest(old_manifest)
    assert rag_store.needs_rebuild(str(runtime.local_dir)) == (True, "pipeline_version 变化，需要 force rebuild")
    old_upload_ids = indexing.create_vector_store(None).get(where={"source_id": document.source_id})["ids"]
    restored_inputs = []
    parsed_local_paths = []
    convert = indexing_service.candidate_to_langchain_documents
    parse_local = indexing.parse_document

    def capture(candidate, **kwargs):
        restored_inputs.append(candidate)
        return convert(candidate, **kwargs)

    def capture_local(path):
        parsed_local_paths.append(path)
        return parse_local(path)

    monkeypatch.setattr(indexing_service, "candidate_to_langchain_documents", capture)
    monkeypatch.setattr(indexing, "parse_document", capture_local)
    # 没有原文件、原 parser 或 source fetch，force rebuild 只能消费 candidate_json。
    def no_parse(*args, **kwargs):
        pytest.fail("force rebuild attempted to parse an uploaded document again")

    monkeypatch.setattr("app.sources.upload.parse_document", no_parse)
    history = runtime.runs.list()
    # 使用真实启动判断和现有 global force rebuild，无 migration script 或新索引。
    ok, message = _sync_and_load(str(runtime.local_dir))
    assert ok, message
    assert rag_store.load_manifest()["pipeline_version"] == 6
    assert parsed_local_paths == [local_table]
    assert rag_store.needs_rebuild(str(runtime.local_dir))[0] is False
    restored = next(item for item in restored_inputs if item.candidate_id == candidate.candidate_id)
    assert restored == candidate
    assert restored.document.structured_content == document.structured_content
    assert any(cell.row_span == 2 for cell in restored.document.structured_content.blocks[1].cells)
    assert runtime.runs.list() == history
    persisted, _ = load_index_from_chroma()
    upload_chunks = [chunk for chunk in persisted if chunk["metadata"]["source_id"] == document.source_id]
    assert [chunk["metadata"]["element_type"] for chunk in upload_chunks] == ["text", "table", "text"]
    table_chunk = upload_chunks[1]
    assert "：年满18岁" in table_chunk["text"]
    assert "材料：身份证\n证明。" in table_chunk["text"]
    assert (table_chunk["metadata"]["row_start"], table_chunk["metadata"]["row_end"]) == (1, 2)
    assert [chunk["metadata"]["position"] for chunk in upload_chunks] == [1, 2, 3]
    assert any(chunk["metadata"]["source_type"] == "local_file" and
               chunk["metadata"].get("element_type") == "table" for chunk in persisted)
    [legacy_chunk] = [chunk for chunk in persisted if chunk["metadata"]["source_id"] == legacy.document.source_id]
    assert legacy_chunk["text"] == "legacy plain evidence"
    assert "element_type" not in legacy_chunk["metadata"]
    records = indexing.create_vector_store(None).get(where={"source_type": "upload"})
    assert set(old_upload_ids).isdisjoint(records["ids"])
    assert set(records["ids"]) == set(indexing.create_record_manager(namespace=indexing.UPLOAD_RECORD_NAMESPACE).list_keys())
    assert pipeline.bm25.corpus_size == len(persisted)
    runtime.embeddings.reset()
    assert runtime.indexer.index_approved(candidate.candidate_id)["num_skipped"] == len(upload_chunks)
    assert runtime.embeddings.document_texts == []


@pytest.mark.parametrize("structure_factory, old_version", [
    (admission_structure, 3), (imut_admission_structure, 4), (imut_admission_structure, 5),
])
@pytest.mark.parametrize("source_type, source_uri, file_type", [
    ("upload", "upload://local/admission", "docx"),
    ("firecrawl", "https://example.com/admission", "html"),
])
def test_section_pipeline_migration_rechunks_persisted_candidates_without_acquisition(
    persisted_runtime, monkeypatch, source_type, source_uri, file_type, structure_factory, old_version,
):
    runtime = persisted_runtime
    document = build_knowledge_document(
        source_type=source_type, source_uri=source_uri, title="录取分数统计", content="",
        structured_content=structure_factory(), metadata={"file_type": file_type},
    )
    candidate = _approve(runtime.domain, document)
    namespace = indexing.EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE[document.source_type]
    # 持久化原始结构与成功历史；旧 chunk 无 section，重建必须替换其表示。
    old_chunk = Document(page_content=document.content, metadata={
        "source_id": document.source_id, "document_uid": document.document_uid,
        "source_type": source_type, "source_uri": source_uri, "position": 1,
    })
    expected = indexing.candidate_to_langchain_documents(candidate)
    old_chunks = [Document(
        page_content=chunk.page_content.removeprefix(f"[标题路径] {document.title}\n\n"),
        metadata={key: value for key, value in chunk.metadata.items() if key != "title_path"},
    ) for chunk in expected] if old_version == 5 else [old_chunk]
    result = indexing.sync_documents(old_chunks, runtime.embeddings, namespace=namespace)
    runtime.runs.succeed(runtime.runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id, added=result["num_added"])
    old_ids = indexing.create_vector_store(None).get(where={"source_id": document.source_id})["ids"]
    manifest = rag_store.build_current_manifest(str(runtime.local_dir))
    rag_store.save_manifest({**manifest, "pipeline_version": old_version})
    history = runtime.runs.list()

    def forbidden(*args, **kwargs):
        pytest.fail("section rebuild must use SQLite structure, without upload, fetch or parser")

    monkeypatch.setattr(runtime.source_service, "_fetch", forbidden)
    monkeypatch.setattr("app.sources.service.invoke_mcp_tool", forbidden)
    monkeypatch.setattr("app.sources.upload.parse_document", forbidden)
    monkeypatch.setattr("app.sources.firecrawl.parse_html_document", forbidden)
    monkeypatch.setattr("app.ingestion.parsing._document_converter", forbidden)
    assert rag_store.needs_rebuild(str(runtime.local_dir)) == (True, "pipeline_version 变化，需要 force rebuild")
    ok, message = _sync_and_load(str(runtime.local_dir))
    assert ok, message
    assert rag_store.load_manifest() == manifest
    assert manifest["pipeline_version"] == 6
    assert rag_store.needs_rebuild(str(runtime.local_dir))[0] is False
    restored = SQLiteCandidateRepository(runtime.settings.sqlite_db_path).get(candidate.candidate_id)
    assert restored == candidate
    assert runtime.runs.list() == history
    actual = [chunk for chunk in pipeline.chunks if chunk["metadata"]["source_id"] == document.source_id]
    assert [chunk["text"] for chunk in actual] == [chunk.page_content for chunk in expected]
    assert {chunk["metadata"]["section_index"] for chunk in actual} == {1, 2}
    assert all(chunk["metadata"]["title_path"] == document.title for chunk in actual)
    assert all("上下文：内蒙古工业大学2026年分省录取分数统计" in chunk["text"] for chunk in actual)
    records = indexing.create_vector_store(None).get(where={"source_id": document.source_id})
    ordered = sorted(zip(records["documents"], records["metadatas"]), key=lambda item: item[1]["position"])
    assert ordered == [(chunk.page_content, chunk.metadata) for chunk in expected]
    assert set(old_ids).isdisjoint(records["ids"])
    assert set(records["ids"]) == set(indexing.create_record_manager(namespace=namespace).list_keys())


def test_firecrawl_html_sqlite_index_retrieval_and_rebuild_keep_legacy(persisted_runtime, monkeypatch):
    import asyncio
    import json
    import sqlite3

    from app.ingestion.document_blocks import canonical_document_text
    from app.tools.context import ToolInvocationContext
    from tests.support.structured_document import REGISTRATION_HTML

    runtime = persisted_runtime
    html = REGISTRATION_HTML.replace(
        "<body>", "<body><h1>办理事项</h1><h2>申请材料</h2><h3>证件清单</h3>",
    )
    title_path = "办理事项 > 申请材料 > 证件清单"

    async def fetch(name, args, context):
        assert name == "firecrawl_firecrawl_scrape"
        assert args["formats"] == ["markdown", "html"]
        return {"content_blocks": [{"type": "text", "text": json.dumps({
            "markdown": "fallback", "html": html, "success": True,
            "metadata": {"title": "办理事项", "contentType": "text/html", "url": args["url"], "statusCode": 200},
        })}]}

    monkeypatch.setattr(runtime.source_service, "_fetch", fetch)
    pending = asyncio.run(runtime.source_service.propose_firecrawl_page(
        url="https://example.com/registration", context=ToolInvocationContext(source="test"),
    ))
    candidate = runtime.domain.approve(pending.candidate_id, decided_by=_requester())
    assert runtime.runs.list() == []
    assert all("企业注册" not in chunk["text"] for chunk in pipeline.chunks)
    chunks = indexing.candidate_to_langchain_documents(candidate)
    [table] = [chunk for chunk in chunks if chunk.metadata.get("element_type") == "table"]
    expected = "事项：企业注册；条件：年满18岁；材料：身份证、申请表。"
    assert expected in table.page_content
    assert table.metadata["title_path"] == title_path
    assert table.page_content.startswith(f"[标题路径] {title_path}\n\n[表格]")
    assert _index_with_run(runtime, candidate)["num_added"] == 3
    legacy = _approve(runtime.domain, _firecrawl_document("legacy markdown"))
    assert legacy.document.structured_content is None
    _index_with_run(runtime, legacy)

    with sqlite3.connect(runtime.settings.sqlite_db_path) as connection:
        payloads = [row[0] for row in connection.execute("SELECT candidate_json FROM ingestion_candidates")]
    assert all("<table>" not in payload for payload in payloads)
    repository = SQLiteCandidateRepository(runtime.settings.sqlite_db_path)
    restored = repository.get(candidate.candidate_id)
    assert restored == candidate
    assert restored.document.structure_hash == candidate.document.structure_hash
    assert restored.document.content == canonical_document_text(restored.document.structured_content)

    def forbidden(*args, **kwargs):
        pytest.fail("persisted Firecrawl rebuild must not fetch or parse")

    monkeypatch.setattr(runtime.source_service, "_fetch", forbidden)
    monkeypatch.setattr("app.sources.service.invoke_mcp_tool", forbidden)
    monkeypatch.setattr("app.sources.firecrawl.parse_html_document", forbidden)
    monkeypatch.setattr("app.ingestion.parsing.parse_html_document", forbidden)
    monkeypatch.setattr("app.ingestion.parsing._document_converter", forbidden)
    monkeypatch.setattr("app.rag.retrieval._retrieve_with_local_embeddings", forbidden)
    history = runtime.runs.list()
    ok, message = _sync_and_load(str(runtime.local_dir), force_rebuild=True)
    assert ok, message
    assert rag_store.load_manifest()["pipeline_version"] == 6
    assert runtime.runs.list() == history
    assert repository.get(candidate.candidate_id) == restored
    [legacy_chunk] = [chunk for chunk in pipeline.chunks if chunk["metadata"]["source_id"] == legacy.document.source_id]
    assert legacy_chunk["text"] == "legacy markdown"
    assert "element_type" not in legacy_chunk["metadata"]
    table_position = next(i for i, chunk in enumerate(pipeline.chunks)
                          if chunk["metadata"].get("element_type") == "table")
    assert all(pipeline.bm25.doc_freqs[table_position].get(token, 0) for token in tokenize("身份证"))
    assert all(pipeline.bm25.doc_freqs[table_position].get(token, 0) for token in tokenize("申请材料"))
    hits = pipeline.search_relevant_chunks("企业注册需要哪些材料", top_k=5)
    assert any(expected in hit["text"] and hit["metadata"].get("element_type") == "table" for hit in hits)
    records = indexing.create_vector_store(None).get(where={"source_type": "firecrawl"})
    assert len(records["ids"]) == 4
    assert all(metadata.get("title_path") == title_path for metadata in records["metadatas"]
               if metadata["source_id"] == candidate.document.source_id)
    assert set(records["ids"]) == set(indexing.create_record_manager(namespace=FIRECRAWL_RECORD_NAMESPACE).list_keys())
