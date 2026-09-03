"""LangChain-native 的 Local Knowledge 增量索引薄适配。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index

from app.config import AppSettings, get_settings
from app.ingestion import (
    IngestionCandidate,
    IngestionCandidateStatus,
    IngestionPrincipalType,
    IngestionRequester,
    IngestionService,
    KnowledgeSourceType,
    build_knowledge_document,
)
from app.rag.ingest import clean_documents, load_files, split_doc


LOCAL_RECORD_NAMESPACE = "rag-agent/local/v1"
RECORD_MANAGER_FILE_NAME = "record_manager.sqlite3"
_LOCAL_REQUESTER = IngestionRequester(
    principal_id="local-index-sync-v1",
    principal_type=IngestionPrincipalType.SYSTEM,
    source="rag-indexing",
)
_MIME_TYPES = {
    "txt": "text/plain",
    "md": "text/markdown",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def candidate_to_langchain_documents(
    candidate: IngestionCandidate,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """仅把 approved candidate 用现有 splitter 转换为 LangChain chunks。"""
    if candidate.status is not IngestionCandidateStatus.APPROVED:
        raise ValueError("只有 approved ingestion candidate 可以进入永久索引")

    document = candidate.document
    source_metadata = document.metadata
    file_type = str(source_metadata.get("file_type") or "txt").lower()
    file_name = str(source_metadata.get("file_name") or document.title)
    source_path = str(source_metadata.get("source_path") or document.source_uri)
    legacy_doc = {
        "doc_id": document.document_uid,
        "file_name": file_name,
        "source_path": source_path,
        "file_type": file_type,
        "text": document.content,
    }
    chunks = split_doc(
        legacy_doc,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    langchain_documents: list[Document] = []
    for chunk in chunks:
        metadata = {
            key: value
            for key, value in (chunk.get("metadata") or {}).items()
            if key not in {"doc_id", "chunk_id"} and value is not None
        }
        metadata.update(
            {
                "source_id": document.source_id,
                "document_uid": document.document_uid,
                "source_type": document.source_type.value,
                "source_uri": document.source_uri,
                "file_name": file_name,
                "file_type": file_type,
                "source_path": source_path,
            }
        )
        langchain_documents.append(
            Document(
                page_content=str(chunk.get("text") or ""),
                metadata=metadata,
            )
        )
    return langchain_documents


def load_local_langchain_documents(
    data_dir: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """复用现有 Local loader/cleaner，并经过 Phase 9 admission contract。"""
    raw_docs, _docs_registry = load_files(data_dir)
    service = IngestionService()
    langchain_documents: list[Document] = []

    for raw_doc in clean_documents(raw_docs):
        source_path = Path(str(raw_doc["source_path"])).resolve()
        file_type = str(raw_doc["file_type"])
        knowledge_document = build_knowledge_document(
            source_type=KnowledgeSourceType.LOCAL_FILE,
            source_uri=str(source_path),
            title=str(raw_doc["file_name"]),
            content=str(raw_doc["text"]),
            fetched_at=datetime.fromtimestamp(
                source_path.stat().st_mtime,
                tz=timezone.utc,
            ),
            mime_type=_MIME_TYPES.get(file_type),
            metadata={
                "file_name": str(raw_doc["file_name"]),
                "file_type": file_type,
                "source_path": str(source_path),
            },
        )
        candidate = service.propose(
            document=knowledge_document,
            requester=_LOCAL_REQUESTER,
        )
        langchain_documents.extend(
            candidate_to_langchain_documents(
                candidate,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return langchain_documents


def record_manager_path(settings: AppSettings | None = None) -> Path:
    """返回官方 SQLRecordManager 的 SQLite 文件位置。"""
    resolved = settings or get_settings()
    return resolved.index_store_dir / RECORD_MANAGER_FILE_NAME


def create_record_manager(
    settings: AppSettings | None = None,
    *,
    namespace: str = LOCAL_RECORD_NAMESPACE,
) -> SQLRecordManager:
    """创建并幂等初始化官方 SQLRecordManager。"""
    db_path = record_manager_path(settings)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    manager = SQLRecordManager(
        namespace=namespace,
        db_url=f"sqlite:///{db_path.resolve()}",
    )
    manager.create_schema()
    return manager


def create_vector_store(
    embeddings: Embeddings | None,
    settings: AppSettings | None = None,
    *,
    create_collection_if_not_exists: bool = True,
) -> Chroma:
    """创建由 langchain-chroma 管理读写的 v2 VectorStore。"""
    resolved = settings or get_settings()
    resolved.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=resolved.chroma_index_collection_name,
        persist_directory=str(resolved.chroma_persist_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        create_collection_if_not_exists=create_collection_if_not_exists,
    )


def sync_documents(
    documents: list[Document],
    embeddings: Embeddings,
    *,
    settings: AppSettings | None = None,
    namespace: str = LOCAL_RECORD_NAMESPACE,
    cleanup: Literal["full", "incremental"] = "full",
    force_rebuild: bool = False,
) -> dict[str, int]:
    """用 LangChain Indexing API 同步指定 namespace 的受控文档。"""
    resolved = settings or get_settings()
    if force_rebuild and cleanup != "full":
        raise ValueError("force_rebuild 只允许用于 full authoritative sync")
    manager = create_record_manager(resolved, namespace=namespace)
    vector_store = create_vector_store(embeddings, resolved)

    if force_rebuild:
        vector_store.delete_collection()
        keys = manager.list_keys()
        if keys:
            manager.delete_keys(keys)
        vector_store = create_vector_store(embeddings, resolved)

    result: dict[str, Any] = index(
        documents,
        manager,
        vector_store,
        cleanup=cleanup,
        source_id_key="source_id",
        key_encoder="sha256",
    )
    return {key: int(value) for key, value in result.items()}


__all__ = [
    "LOCAL_RECORD_NAMESPACE",
    "RECORD_MANAGER_FILE_NAME",
    "candidate_to_langchain_documents",
    "create_record_manager",
    "create_vector_store",
    "load_local_langchain_documents",
    "record_manager_path",
    "sync_documents",
]
