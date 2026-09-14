"""LangChain-native 的受控文档转换与单 namespace 索引薄适配。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

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
from app.rag.ingest import (
    clean_markdown_text,
    clean_text,
    collect_supported_files,
    read_file_text,
)


LOCAL_RECORD_NAMESPACE = "rag-agent/local/v1"
GITHUB_RECORD_NAMESPACE = "rag-agent/github/v1"
FIRECRAWL_RECORD_NAMESPACE = "rag-agent/firecrawl/v1"
EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE = {
    KnowledgeSourceType.GITHUB: GITHUB_RECORD_NAMESPACE,
    KnowledgeSourceType.FIRECRAWL: FIRECRAWL_RECORD_NAMESPACE,
}
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
_PLAIN_TEXT_FILE_TYPES = {"txt", "pdf", "docx"}
_TEXT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ""]
_MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def candidate_to_langchain_documents(
    candidate: IngestionCandidate,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """仅把 approved candidate 直接切分为 LangChain chunks。"""
    if candidate.status is not IngestionCandidateStatus.APPROVED:
        raise ValueError("只有 approved ingestion candidate 可以进入永久索引")

    document = candidate.document
    source_metadata = document.metadata
    file_type = str(source_metadata.get("file_type") or "txt").lower()
    file_name = str(source_metadata.get("file_name") or document.title)
    source_path = str(source_metadata.get("source_path") or document.source_uri)
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size if chunk_size is None else chunk_size,
        chunk_overlap=(
            settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        ),
        separators=_TEXT_SEPARATORS,
    )

    if file_type == "md":
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MARKDOWN_HEADERS,
        ).split_text(document.content)
        chunks = splitter.split_documents(sections)
    elif file_type in _PLAIN_TEXT_FILE_TYPES:
        chunks = splitter.create_documents([document.content])
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")

    provenance = {
        "source_id": document.source_id,
        "document_uid": document.document_uid,
        "source_type": document.source_type.value,
        "source_uri": document.source_uri,
        "file_name": file_name,
        "file_type": file_type,
        "source_path": source_path,
    }
    result: list[Document] = []
    for position, chunk in enumerate(chunks, start=1):
        page_content = chunk.page_content
        metadata = {**provenance, "position": position}
        if file_type == "md":
            title_path = " > ".join(
                chunk.metadata[key]
                for key in ("h1", "h2", "h3")
                if key in chunk.metadata
            )
            if title_path:
                page_content = f"[标题路径] {title_path}\n\n{page_content}"
            metadata.update({"title_path": title_path, **chunk.metadata})
        result.append(Document(page_content=page_content, metadata=metadata))
    return result


def load_local_langchain_documents(
    data_dir: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """读取并清洗 Local 文件，再经过 Phase 9 admission contract。"""
    service = IngestionService()
    langchain_documents: list[Document] = []

    for file_path in collect_supported_files(data_dir):
        source_path = file_path.resolve()
        file_type = file_path.suffix.lower().lstrip(".")
        content = read_file_text(str(file_path))
        content = (
            clean_markdown_text(content)
            if file_type == "md"
            else clean_text(content)
        )
        knowledge_document = build_knowledge_document(
            source_type=KnowledgeSourceType.LOCAL_FILE,
            source_uri=str(source_path),
            title=file_path.name,
            content=content,
            fetched_at=datetime.fromtimestamp(
                source_path.stat().st_mtime,
                tz=timezone.utc,
            ),
            mime_type=_MIME_TYPES.get(file_type),
            metadata={
                "file_name": file_path.name,
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
) -> dict[str, int]:
    """用 LangChain Indexing API 同步指定 namespace 的受控文档。"""
    resolved = settings or get_settings()
    manager = create_record_manager(resolved, namespace=namespace)
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
    "EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE",
    "FIRECRAWL_RECORD_NAMESPACE",
    "GITHUB_RECORD_NAMESPACE",
    "LOCAL_RECORD_NAMESPACE",
    "RECORD_MANAGER_FILE_NAME",
    "candidate_to_langchain_documents",
    "create_record_manager",
    "create_vector_store",
    "load_local_langchain_documents",
    "record_manager_path",
    "sync_documents",
]
