"""LangChain-native Local incremental indexing 的框架与业务边界测试。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.ingestion import (
    IngestionCandidateStatus,
    IngestionPrincipalType,
    IngestionRequester,
    IngestionService,
    KnowledgeSourceType,
    build_knowledge_document,
)
from app.indexing import langchain as indexing, manifest as store
from app.rag import runtime as pipeline
from app.indexing.service import IndexingService
from app.indexing import service as writer
from app.rag.chroma import load_index_from_chroma
from app.sources.upload import load_upload_document
from tests.support.structured_document import build_table_docx, registration_structure


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


def test_local_adapter_preserves_cleaning_and_markdown_chunk_contract(tmp_path):
    file_path = tmp_path / "guide.md"
    file_path.write_text(
        "# 一级\r\n\r\n开头\t 内容。\r\n\r\n"
        "## 二级\r\n\r\n第二段。\r\n\r\n"
        "### 三级\r\n\r\n第三段内容。<!-- hidden -->\r\n\r\n尾段。",
        encoding="utf-8",
    )

    converted = indexing.load_local_langchain_documents(
        str(tmp_path),
        chunk_size=30,
        chunk_overlap=0,
    )

    assert [item.page_content for item in converted] == [
        "[标题路径] 一级\n\n开头 内容。",
        "[标题路径] 一级 > 二级\n\n第二段。",
        "[标题路径] 一级 > 二级 > 三级\n\n第三段内容。  \n尾段。",
    ]
    assert [
        {
            key: value
            for key, value in item.metadata.items()
            if key in {"position", "title_path", "h1", "h2", "h3"}
        }
        for item in converted
    ] == [
        {"position": 1, "title_path": "一级", "h1": "一级"},
        {
            "position": 2,
            "title_path": "一级 > 二级",
            "h1": "一级",
            "h2": "二级",
        },
        {
            "position": 3,
            "title_path": "一级 > 二级 > 三级",
            "h1": "一级",
            "h2": "二级",
            "h3": "三级",
        },
    ]
    assert all("doc_id" not in item.metadata for item in converted)
    assert all("chunk_id" not in item.metadata for item in converted)


def test_local_plain_text_cleaning_and_metadata_surface(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "  alpha\t beta\r\n\r\n\r\n gamma  ",
        encoding="utf-8",
    )

    [document] = indexing.load_local_langchain_documents(
        str(tmp_path),
        chunk_size=100,
        chunk_overlap=0,
    )

    assert document.page_content == "alpha beta\n\n gamma"
    assert set(document.metadata) == {
        "source_id",
        "document_uid",
        "source_type",
        "source_uri",
        "file_name",
        "file_type",
        "source_path",
        "position",
    }
    assert document.metadata["file_name"] == "notes.txt"
    assert document.metadata["file_type"] == "txt"
    assert document.metadata["source_path"] == str(file_path.resolve())
    assert document.metadata["position"] == 1


@pytest.mark.parametrize("file_type", ["pdf", "docx"])
def test_legacy_pdf_and_docx_without_structure_keep_plain_text_path(tmp_path, file_type):
    source_path = (tmp_path / f"manual.{file_type}").resolve()
    document = build_knowledge_document(
        source_type=KnowledgeSourceType.LOCAL_FILE,
        source_uri=str(source_path),
        title=source_path.name,
        content="第一段内容。\n\n第二段内容。",
        metadata={
            "file_name": source_path.name,
            "file_type": file_type,
            "source_path": str(source_path),
        },
    )
    candidate = IngestionService().propose(
        document=document,
        requester=_requester(),
    )

    converted = indexing.candidate_to_langchain_documents(
        candidate,
        chunk_size=20,
        chunk_overlap=0,
    )

    assert [item.page_content for item in converted] == [
        "第一段内容。\n\n第二段内容。"
    ]
    assert {item.metadata["file_type"] for item in converted} == {file_type}
    assert candidate.document.structured_content is None
    assert all("element_type" not in item.metadata for item in converted)


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
    settings = get_settings().model_copy(update={"rag_file_path": tmp_path})
    embeddings = CountingEmbeddings()
    legacy = Chroma(collection_name=settings.chroma_collection_name,
                    persist_directory=str(settings.chroma_persist_dir), embedding_function=embeddings)
    legacy.add_texts(["legacy"], ids=["legacy-record"])
    (tmp_path / "current.txt").write_text("current", encoding="utf-8")
    result = IndexingService(settings=settings, embeddings=embeddings).sync_local_corpus(force_rebuild=True)
    assert result["num_added"] == 1
    assert legacy.get(include=[])["ids"] == ["legacy-record"]
    assert indexing.create_vector_store(None, settings).get()["documents"] == ["current"]


@pytest.mark.parametrize("force_rebuild", [False, True])
def test_local_sync_clears_only_owned_namespace_unless_force(tmp_path, force_rebuild):
    settings = get_settings().model_copy(update={"rag_file_path": tmp_path})
    service = IndexingService(settings=settings, embeddings=CountingEmbeddings())
    path = tmp_path / "removed.txt"
    path.write_text("old local content", encoding="utf-8")
    service.sync_local_corpus()
    for namespace in indexing.EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE.values():
        indexing.create_record_manager(namespace=namespace).update(["stale-external-key"])
    path.unlink()
    service.sync_local_corpus(force_rebuild=force_rebuild)
    assert indexing.create_vector_store(None).get(include=[])["ids"] == []
    assert indexing.create_record_manager().list_keys() == []
    for namespace in indexing.EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE.values():
        assert indexing.create_record_manager(namespace=namespace).list_keys() == (
            [] if force_rebuild else ["stale-external-key"]
        )





def test_empty_reload_does_not_publish_on_collection_read_failure(monkeypatch):
    old_chunks = [{"text": "retained runtime"}]
    monkeypatch.setattr(pipeline, "chunks", old_chunks)
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "_load_index_bundle", lambda **kwargs: {
        "ok": False, "chunks": [], "message": "read failed",
    })
    def fail(*args, **kwargs):
        raise RuntimeError("read failed")
    monkeypatch.setattr(pipeline, "create_vector_store", fail)
    assert pipeline.load_index_cache(allow_empty=True)[0] is False
    assert pipeline.chunks is old_chunks
    assert pipeline.RAG_READY is True





def test_sync_failure_keeps_manifest_and_query_runtime(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"rag_file_path": tmp_path})
    service = IndexingService(settings=settings, embeddings=CountingEmbeddings())
    (tmp_path / "one.txt").write_text("original", encoding="utf-8")
    service.sync_local_corpus()
    manifest = (settings.index_store_dir / "index_manifest.json").read_bytes()
    sentinel_chunks = [{"text": "old", "metadata": {"chunk_id": 1}}]
    monkeypatch.setattr(pipeline, "chunks", sentinel_chunks)
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    def fail(*_args, **_kwargs):
        raise RuntimeError("index failed")
    monkeypatch.setattr(writer, "sync_documents", fail)
    with pytest.raises(RuntimeError, match="index failed"):
        service.sync_local_corpus()
    assert (settings.index_store_dir / "index_manifest.json").read_bytes() == manifest
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


@pytest.mark.parametrize("file_type", ["pdf", "docx"])
def test_local_and_upload_keep_same_structure_and_table_aware_chunks(tmp_path, monkeypatch, file_type):
    payload = build_table_docx(merged=True) if file_type == "docx" else (
        Path(__file__).parent / "fixtures" / "structured-table.pdf"
    ).read_bytes()
    path = tmp_path / f"table.{file_type}"
    path.write_bytes(payload)
    uploaded = load_upload_document(BytesIO(payload), filename=path.name, max_bytes=2_000_000)
    admitted = []
    propose = IngestionService.propose

    def capture(self, **kwargs):
        candidate = propose(self, **kwargs)
        admitted.append(candidate)
        return candidate

    monkeypatch.setattr(IngestionService, "propose", capture)
    documents = indexing.load_local_langchain_documents(str(path), chunk_size=1000, chunk_overlap=0)
    [local] = admitted
    assert local.document.structured_content == uploaded.structured_content
    assert local.document.structure_hash == uploaded.structure_hash
    assert local.document.content == uploaded.content
    upload_service = IngestionService()
    uploaded_candidate = propose(upload_service, document=uploaded, requester=_requester())
    uploaded_candidate = upload_service.approve(uploaded_candidate.candidate_id, decided_by=_requester())
    upload_chunks = indexing.candidate_to_langchain_documents(uploaded_candidate, chunk_size=1000, chunk_overlap=0)
    assert [chunk.page_content for chunk in documents] == [chunk.page_content for chunk in upload_chunks]
    assert [chunk.metadata["element_type"] for chunk in documents] == ["text", "table", "text"]
    assert documents[0].page_content == (
        "段落 A" if file_type == "docx" else "Paragraph A: registration requirements."
    )
    assert documents[1].metadata["table_id"] == "b1"
    assert ("：年满18岁" if file_type == "docx" else "：Age 18+") in documents[1].page_content
    assert all("structured_content" not in document.metadata for document in documents)


@pytest.mark.parametrize("old_version", [2, 3, 4])
def test_pipeline_version_change_requires_force_rebuild_without_renaming_index(tmp_path, monkeypatch, old_version):
    current = store.build_current_manifest(str(tmp_path))
    assert current["pipeline_version"] == 5
    assert current["chroma_collection"] == get_settings().chroma_index_collection_name
    assert current["record_manager_namespace"] == indexing.LOCAL_RECORD_NAMESPACE
    monkeypatch.setattr(store, "load_manifest", lambda *_args: {**current, "pipeline_version": old_version})
    assert store.needs_rebuild(str(tmp_path)) == (True, "pipeline_version 变化，需要 force rebuild")
    indexing.create_record_manager()
    monkeypatch.setattr(store, "create_vector_store", lambda *args, **kwargs: object())
    monkeypatch.setattr(store, "load_manifest", lambda *_args: current)
    assert store.needs_rebuild(str(tmp_path))[0] is False


@pytest.mark.parametrize("cleanup", ["full", "incremental"])
def test_structured_chunks_real_index_add_skip_update_delete(tmp_path, cleanup):
    def documents(material):
        structure = registration_structure()
        table = structure.blocks[0]
        cells = tuple(cell.model_copy(update={"text": material}) if (
            cell.row_index == 1 and cell.column_index == 2
        ) else cell for cell in table.cells)
        document = build_knowledge_document(
            source_type="local_file", source_uri=str(tmp_path / "table.pdf"), title="table.pdf",
            content="", metadata={"file_type": "pdf"},
            structured_content=structure.model_copy(update={"blocks": (table.model_copy(update={"cells": cells}),)}),
        )
        candidate = IngestionService().propose(document=document, requester=_requester())
        return indexing.candidate_to_langchain_documents(candidate, chunk_size=85, chunk_overlap=0)

    embeddings = CountingEmbeddings()
    first = documents("身份证、申请表")
    assert len(first) == 2
    assert indexing.sync_documents(first, embeddings, cleanup=cleanup) == {
        "num_added": 2, "num_updated": 0, "num_skipped": 0, "num_deleted": 0,
    }
    embeddings.reset()
    assert indexing.sync_documents(documents("身份证、申请表"), embeddings, cleanup=cleanup) == {
        "num_added": 0, "num_updated": 0, "num_skipped": 2, "num_deleted": 0,
    }
    assert embeddings.document_texts == []
    changed = documents("身份证、申请表、补充证明")
    # LangChain 内容寻址的 update 是新增新 chunk + 清理旧 chunk。
    assert indexing.sync_documents(changed, embeddings, cleanup=cleanup) == {
        "num_added": 1, "num_updated": 0, "num_skipped": 1, "num_deleted": 1,
    }
    assert embeddings.document_texts == [changed[0].page_content]
    records = indexing.create_vector_store(None).get()
    assert set(records["documents"]) == {chunk.page_content for chunk in changed}
    assert set(records["ids"]) == set(indexing.create_record_manager().list_keys())
    embeddings.reset()
    assert indexing.sync_documents([], embeddings, cleanup="full")["num_deleted"] == 2
    assert indexing.create_vector_store(None).get()["ids"] == []
    assert indexing.create_record_manager().list_keys() == []
    assert embeddings.document_texts == []
