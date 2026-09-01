"""ChromaDB 存储适配模块。

封装 collection 初始化、chunk 向量持久化、查询和缓存恢复等底层操作。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.config import AppSettings, get_settings
from app.common.logging import get_logger

logger = get_logger("chat_service")


class ChromaUnavailableError(RuntimeError):
    """ChromaDB 不可用时抛出的异常。"""


def _import_chromadb():
    """延迟导入 ChromaDB，避免测试阶段过早依赖它。"""
    try:
        import chromadb
    except ImportError as exc:
        raise ChromaUnavailableError(
            "未安装 chromadb，请先运行 `pip install -r requirements.txt`。"
        ) from exc

    return chromadb


def _persist_dir(settings: AppSettings | None = None) -> Path:
    """读取 ChromaDB 本地持久化目录配置。"""
    return (settings or get_settings()).chroma_persist_dir


def _collection_name(settings: AppSettings | None = None) -> str:
    """读取 LangChain-native v2 ChromaDB collection 名称。"""
    return (settings or get_settings()).chroma_index_collection_name


def get_chroma_client(settings: AppSettings | None = None):
    """创建本地持久化 ChromaDB 客户端。"""
    chromadb = _import_chromadb()
    persist_dir = _persist_dir(settings)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_chroma_collection(
    create: bool = True,
    settings: AppSettings | None = None,
):
    """获取存放 chunk 的 collection，必要时按 cosine 距离创建。"""
    client = get_chroma_client(settings)
    name = _collection_name(settings)

    if create:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    return client.get_collection(name=name)


def reset_chroma_collection():
    """拒绝旧 direct-reset path；force rebuild 由 langchain-chroma 执行。"""
    raise RuntimeError(
        "direct Chroma reset 已停用，请使用 LangChain indexing force_rebuild"
    )


def chroma_collection_has_chunks() -> bool:
    """判断配置的 ChromaDB collection 是否存在且包含 chunk。"""
    try:
        collection = get_chroma_collection(create=False)
        return collection.count() > 0
    except Exception:
        return False


def _as_embedding_list(embedding: Any) -> list[float]:
    """将不同来源的 embedding 转换为 ChromaDB 可写入的 float 列表。"""
    if hasattr(embedding, "tolist"):
        return [float(x) for x in embedding.tolist()]
    return [float(x) for x in embedding]


def persist_chunks_to_chroma(
    chunks: list[dict[str, Any]],
    embeddings: Any,
    reset: bool = True,
    batch_size: int = 500,
) -> int:
    """拒绝旧 direct-upsert path；保留符号仅用于明确迁移错误。"""
    del chunks, embeddings, reset, batch_size
    raise RuntimeError(
        "direct Chroma upsert 已停用，请使用 LangChain indexing sync_documents"
    )


def load_index_from_chroma(
    settings: AppSettings | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """从 v2 Chroma 恢复 chunks，并重建 runtime compatibility IDs。"""
    collection = get_chroma_collection(create=False, settings=settings)
    data = collection.get(include=["documents", "metadatas", "embeddings"])

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    raw_embeddings = data.get("embeddings")

    if not documents:
        raise ValueError("ChromaDB 索引缓存中没有 documents")
    if len(metadatas) != len(documents):
        raise ValueError(
            "ChromaDB 索引缓存中的 metadata 数量与 document 数量不一致"
        )
    if raw_embeddings is None:
        raise ValueError("ChromaDB 索引缓存中缺少 embeddings")

    try:
        embedding_matrix = np.asarray(raw_embeddings, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("ChromaDB 索引缓存中的 embeddings 不是有效数值矩阵") from exc

    if embedding_matrix.ndim != 2 or embedding_matrix.shape[1] == 0:
        raise ValueError("ChromaDB 索引缓存中的 embeddings 必须是非空二维矩阵")
    if embedding_matrix.shape[0] != len(documents):
        raise ValueError(
            "ChromaDB 索引缓存中的 embedding 数量与 document 数量不一致"
        )
    if not np.isfinite(embedding_matrix).all():
        raise ValueError("ChromaDB 索引缓存中的 embeddings 包含非有限数值")

    records = [
        {
            "chunk": {
                "text": document or "",
                "metadata": metadata or {},
            },
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
    """使用预先计算好的 query embedding 在 ChromaDB 中执行向量检索。"""
    collection = get_chroma_collection(create=False)
    count = collection.count()
    if count <= 0:
        return []

    n_results = max(1, min(int(n_results), count))
    result = collection.query(
        query_embeddings=[_as_embedding_list(query_embedding)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits = []
    for record_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        distance_value = float(distance)
        hits.append({
            "record_id": record_id,
            "text": document or "",
            "metadata": metadata or {},
            "distance": distance_value,
            "vector_score": 1.0 - distance_value,
        })

    return hits
