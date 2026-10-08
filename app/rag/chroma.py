"""通过 langchain-chroma 读取和查询本地向量索引。"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import AppSettings
from app.storage.vector import create_vector_store


class ChromaUnavailableError(RuntimeError):
    """ChromaDB 不可用时抛出的异常。"""


def chroma_collection_has_chunks() -> bool:
    """判断配置的 ChromaDB collection 是否存在且包含 chunk。"""
    try:
        store = create_vector_store(None, create_collection_if_not_exists=False)
        return bool(store.get(limit=1, include=[]).get("ids"))
    except Exception:
        return False


def load_index_from_chroma(
    settings: AppSettings | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """从 v2 Chroma 恢复 chunks，并重建 runtime compatibility IDs。"""
    store = create_vector_store(
        None,
        settings,
        create_collection_if_not_exists=False,
    )
    data = store.get(include=["documents", "metadatas", "embeddings"])

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    raw_embeddings = data.get("embeddings")

    if not documents:
        raise ValueError("ChromaDB 索引缓存中没有 documents")
    if len(metadatas) != len(documents):
        raise ValueError("ChromaDB 索引缓存中的 metadata 数量与 document 数量不一致")
    if raw_embeddings is None:
        raise ValueError("ChromaDB 索引缓存中缺少 embeddings")

    try:
        embedding_matrix = np.asarray(raw_embeddings, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("ChromaDB 索引缓存中的 embeddings 不是有效数值矩阵") from exc

    if embedding_matrix.ndim != 2 or embedding_matrix.shape[1] == 0:
        raise ValueError("ChromaDB 索引缓存中的 embeddings 必须是非空二维矩阵")
    if embedding_matrix.shape[0] != len(documents):
        raise ValueError("ChromaDB 索引缓存中的 embedding 数量与 document 数量不一致")
    if not np.isfinite(embedding_matrix).all():
        raise ValueError("ChromaDB 索引缓存中的 embeddings 包含非有限数值")

    records = [
        {
            "chunk": {"text": document or "", "metadata": metadata or {}},
            "embedding": embedding_matrix[index],
        }
        for index, (document, metadata) in enumerate(zip(documents, metadatas))
    ]

    def sort_key(record: dict[str, Any]):
        """优先按稳定 provenance 排序，兼容旧测试数据的 chunk_id。"""
        metadata = record["chunk"].get("metadata") or {}
        source_path = str(metadata.get("source_path") or "")
        try:
            position = int(metadata.get("position", 0))
        except (TypeError, ValueError):
            position = 0
        if source_path:
            return (0, source_path, position)
        try:
            chunk_id = int(metadata.get("chunk_id", 0))
        except (TypeError, ValueError):
            chunk_id = 0
        return (1, "", chunk_id)

    records.sort(key=sort_key)
    doc_ids: dict[str, str] = {}
    for runtime_chunk_id, record in enumerate(records, start=1):
        metadata = record["chunk"].setdefault("metadata", {})
        source_path = str(metadata.get("source_path") or "")
        if not source_path:
            continue
        if source_path not in doc_ids:
            doc_ids[source_path] = f"doc_{len(doc_ids) + 1}"
        metadata["doc_id"] = doc_ids[source_path]
        metadata["chunk_id"] = runtime_chunk_id
    chunks = [record["chunk"] for record in records]
    aligned_embeddings = np.stack(
        [record["embedding"] for record in records]
    ).astype(np.float32, copy=False)
    return chunks, aligned_embeddings


def query_chroma(query_embedding: Any, n_results: int) -> list[dict[str, Any]]:
    """用 langchain-chroma 和既有 cosine distance 语义执行向量检索。"""
    store = create_vector_store(None, create_collection_if_not_exists=False)
    matches = store.similarity_search_by_vector_with_relevance_scores(
        [float(value) for value in query_embedding],
        k=max(1, int(n_results)),
    )
    return [
        {
            "record_id": document.id,
            "text": document.page_content,
            "metadata": document.metadata,
            "distance": float(distance),
            "vector_score": 1.0 - float(distance),
        }
        for document, distance in matches
    ]
