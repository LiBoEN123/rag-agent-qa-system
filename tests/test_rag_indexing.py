"""LangChain-native Local incremental indexing 的框架与业务边界测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.api.lifecycle import StartupDependencies, initialize_rag
from app.config import get_settings
from app.ingestion import (
    IngestionCandidateStatus,
    IngestionPrincipalType,
    IngestionRequester,
    IngestionService,
    KnowledgeSourceType,
    build_knowledge_document,
)
from app.rag import indexing, pipeline
from app.rag.chroma import load_index_from_chroma
from app.rag.ingest import clean_documents, load_files, split_chunks


class CountingEmbeddings(Embeddings):
    """不访问模型或网络，并记录实际 document embedding 文本。"""

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


def _requester(principal_id: str = "index-test") -> IngestionRequester:
    return IngestionRequester(
        principal_id=principal_id,
        principal_type=IngestionPrincipalType.SYSTEM,
        source="test",
    )


def _document(tmp_path, *, source_type=KnowledgeSourceType.LOCAL_FILE):
    source_uri = (
        str((tmp_path / "guide.txt").resolve())
        if source_type is KnowledgeSourceType.LOCAL_FILE
        else "https://example.com/guide"
    )
    return build_knowledge_document(
        source_type=source_type,
        source_uri=source_uri,
        title="guide.txt",
        content="第一段。\n\n第二段。",
        metadata={
            "file_name": "guide.txt",
            "file_type": "txt",
            "source_path": source_uri,
        },
    )


def test_framework_record_manager_schema_and_methods_work():
    settings = get_settings()
    manager = indexing.create_record_manager(settings)
    indexing.create_record_manager(settings)

    assert manager.exists(["key-a"]) == [False]
    manager.update(["key-a"], group_ids=["source-a"])
    assert manager.exists(["key-a"]) == [True]
    assert manager.list_keys(group_ids=["source-a"]) == ["key-a"]
    manager.delete_keys(["key-a"])
    assert manager.list_keys() == []


def test_candidate_adapter_enforces_approved_only_and_preserves_phase9_ids(tmp_path):
    local_service = IngestionService()
    approved = local_service.propose(
        document=_document(tmp_path),
        requester=_requester(),
    )
    converted = indexing.candidate_to_langchain_documents(
        approved,
        chunk_size=20,
        chunk_overlap=0,
    )

    assert approved.status is IngestionCandidateStatus.APPROVED
    assert converted
    assert {item.metadata["source_id"] for item in converted} == {
        approved.document.source_id
    }
    assert {item.metadata["document_uid"] for item in converted} == {
        approved.document.document_uid
    }
    assert all("candidate" not in item.metadata for item in converted)
    assert all("approval" not in item.metadata for item in converted)

    external_service = IngestionService()
    pending = external_service.propose(
        document=_document(tmp_path, source_type=KnowledgeSourceType.WEB),
        requester=_requester(),
    )
    with pytest.raises(ValueError, match="approved"):
        indexing.candidate_to_langchain_documents(pending)

    rejected = external_service.reject(
        pending.candidate_id,
        decided_by=_requester("reviewer"),
        reason="reject",
    )
    with pytest.raises(ValueError, match="approved"):
        indexing.candidate_to_langchain_documents(rejected)

    superseded = rejected.model_copy(
        update={"status": IngestionCandidateStatus.SUPERSEDED}
    )
    with pytest.raises(ValueError, match="approved"):
        indexing.candidate_to_langchain_documents(superseded)


def test_local_adapter_reuses_existing_clean_and_split_behavior(tmp_path):
    file_path = tmp_path / "guide.md"
    file_path.write_text("# 标题\n\n第一段。\n\n第二段。", encoding="utf-8")
    raw_docs, _registry = load_files(str(tmp_path))
    existing_chunks, _counts = split_chunks(
        clean_documents(raw_docs),
        chunk_size=30,
        chunk_overlap=0,
    )

    converted = indexing.load_local_langchain_documents(
        str(tmp_path),
        chunk_size=30,
        chunk_overlap=0,
    )

    assert [item.page_content for item in converted] == [
        item["text"] for item in existing_chunks
    ]
    assert [item.metadata["position"] for item in converted] == [
        item["metadata"]["position"] for item in existing_chunks
    ]


def test_real_langchain_incremental_add_skip_modify_add_delete(tmp_path):
    settings = get_settings()
    first_path = tmp_path / "b.txt"
    second_path = tmp_path / "c.txt"
    first_path.write_text("bravo original", encoding="utf-8")
    second_path.write_text("charlie original", encoding="utf-8")
    embeddings = CountingEmbeddings()

    documents = indexing.load_local_langchain_documents(str(tmp_path))
    initial = indexing.sync_documents(documents, embeddings, settings=settings)
    assert initial == {
        "num_added": 2,
        "num_updated": 0,
        "num_skipped": 0,
        "num_deleted": 0,
    }
    assert len(embeddings.document_texts) == 2

    embeddings.reset()
    unchanged = indexing.sync_documents(documents, embeddings, settings=settings)
    assert unchanged == {
        "num_added": 0,
        "num_updated": 0,
        "num_skipped": 2,
        "num_deleted": 0,
    }
    assert embeddings.document_texts == []

    first_path.write_text("bravo modified", encoding="utf-8")
    embeddings.reset()
    modified = indexing.sync_documents(
        indexing.load_local_langchain_documents(str(tmp_path)),
        embeddings,
        settings=settings,
    )
    assert modified["num_added"] == 1
    assert modified["num_deleted"] == 1
    assert modified["num_skipped"] == 1
    assert embeddings.document_texts == ["bravo modified"]

    added_path = tmp_path / "a.txt"
    added_path.write_text("alpha new", encoding="utf-8")
    embeddings.reset()
    added = indexing.sync_documents(
        indexing.load_local_langchain_documents(str(tmp_path)),
        embeddings,
        settings=settings,
    )
    assert added["num_added"] == 1
    assert added["num_skipped"] == 2
    assert embeddings.document_texts == ["alpha new"]

    second_path.unlink()
    embeddings.reset()
    deleted = indexing.sync_documents(
        indexing.load_local_langchain_documents(str(tmp_path)),
        embeddings,
        settings=settings,
    )
    assert deleted["num_deleted"] == 1
    assert deleted["num_skipped"] == 2
    assert embeddings.document_texts == []

    manager = indexing.create_record_manager(settings)
    assert len(manager.list_keys()) == 2
    chunks, matrix = load_index_from_chroma(settings)
    assert [chunk["metadata"]["chunk_id"] for chunk in chunks] == [1, 2]
    assert [chunk["metadata"]["doc_id"] for chunk in chunks] == ["doc_1", "doc_2"]
    assert matrix.shape == (2, 2)


def test_force_rebuild_preserves_legacy_collection(tmp_path):
    settings = get_settings()
    embeddings = CountingEmbeddings()
    legacy = Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=embeddings,
    )
    legacy.add_texts(["legacy"], ids=["legacy-record"])
    (tmp_path / "current.txt").write_text("current", encoding="utf-8")
    result = indexing.sync_documents(
        indexing.load_local_langchain_documents(str(tmp_path)),
        embeddings,
        settings=settings,
        force_rebuild=True,
    )

    assert result["num_added"] == 1
    assert len(legacy.get(include=[])["ids"]) == 1
    current = indexing.create_vector_store(
        None,
        settings,
        create_collection_if_not_exists=False,
    )
    assert len(current.get(include=[])["ids"]) == 1


def test_startup_always_syncs_and_propagates_force_rebuild():
    calls = []
    settings = SimpleNamespace(
        rag_file_path="/tmp/kb",
        chunk_size=100,
        chunk_overlap=10,
        auto_rebuild_on_startup=False,
    )

    def sync(**kwargs):
        calls.append(kwargs)
        return True, "ok"

    initialize_rag(
        StartupDependencies(
            settings=settings,
            needs_rebuild=lambda _data_dir: (False, "unchanged"),
            index_cache_exists=lambda: True,
            load_index_cache=lambda: pytest.fail("startup 必须走 index sync"),
            rebuild_and_persist_index=sync,
        )
    )

    assert calls == [
        {
            "data_dir": "/tmp/kb",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "force_rebuild": False,
        }
    ]


def test_sync_failure_does_not_publish_new_runtime(monkeypatch):
    sentinel_chunks = [{"text": "old", "metadata": {"chunk_id": 1}}]
    monkeypatch.setattr(pipeline, "chunks", sentinel_chunks)
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "model", CountingEmbeddings())
    monkeypatch.setattr(
        pipeline,
        "load_local_langchain_documents",
        lambda *_args, **_kwargs: [SimpleNamespace()],
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("index failed")

    monkeypatch.setattr(pipeline, "sync_documents", fail)
    ok, message = pipeline.rebuild_and_persist_index("/tmp/kb")

    assert ok is False
    assert "index failed" in message
    assert pipeline.chunks is sentinel_chunks
    assert pipeline.RAG_READY is True


def test_loaded_v2_embeddings_are_finite_and_aligned(tmp_path):
    settings = get_settings()
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    documents = indexing.load_local_langchain_documents(str(tmp_path))
    indexing.sync_documents(documents, CountingEmbeddings(), settings=settings)

    chunks, embeddings = load_index_from_chroma(settings)

    assert len(chunks) == len(documents)
    assert embeddings.shape[0] == len(chunks)
    assert np.isfinite(embeddings).all()
