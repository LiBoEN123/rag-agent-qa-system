from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.config import Settings
from app.untils import get_logger

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

def _persist_dir() -> Path:
    return Path(getattr(Settings, "CHROMA_PERSIST_DIR"))


def _collection_name() -> str:
    return str(getattr(Settings, "CHROMA_COLLECTION_NAME", "mini_rag_chunks"))


def get_chroma_client():
    """创建本地持久化 ChromaDB 客户端。"""
    chromadb = _import_chromadb()
    persist_dir = _persist_dir()
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_chroma_collection(create: bool = True):
    """获取存放 chunk 的 collection，必要时按 cosine 距离创建。"""
    client = get_chroma_client()
    name = _collection_name()

    if create:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    return client.get_collection(name=name)


def reset_chroma_collection():
    """删除并重建 chunk collection，用于完整重建索引。"""
    client = get_chroma_client()
    name = _collection_name()

    try:
        client.delete_collection(name=name)
    except Exception:
        # collection 不存在时 ChromaDB 会抛异常，这里可以安全忽略。
        pass

    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def chroma_collection_has_chunks() -> bool:
    """判断配置的 ChromaDB collection 是否存在且包含 chunk。"""
    try:
        collection = get_chroma_collection(create=False)
        return collection.count() > 0
    except Exception:
        return False


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, str | int | float | bool]:
    """将 chunk 元数据转换为 ChromaDB 可接受的标量类型。"""
    clean: dict[str, str | int | float | bool] = {}

    for key, value in (metadata or {}).items():
        if value is None:
            continue

        if isinstance(value, bool):
            clean[str(key)] = value
        elif isinstance(value, (str, int, float)):
            clean[str(key)] = value
        else:
            clean[str(key)] = str(value)

    return clean


def make_chroma_id(chunk: dict[str, Any]) -> str:
    """根据来源路径、位置和正文内容生成稳定的 ChromaDB 记录 ID。"""
    metadata = chunk.get("metadata") or {}
    raw = "\n".join([
        str(metadata.get("source_path", "")),
        str(metadata.get("position", "")),
        str(chunk.get("text", "")),
    ])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"chunk-{digest}"


def _as_embedding_list(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        return [float(x) for x in embedding.tolist()]
    return [float(x) for x in embedding]


def persist_chunks_to_chroma(
    chunks: list[dict[str, Any]],
    embeddings: Any,
    reset: bool = True,
    batch_size: int = 500,
) -> int:
    """将 chunk 文本、元数据和预先计算好的 embedding 写入 ChromaDB。"""
    if embeddings is None:
        raise ValueError("写入 ChromaDB 时必须提供 doc_embeddings")

    collection = reset_chroma_collection() if reset else get_chroma_collection(create=True)
    saved = 0

    batch_ids: list[str] = []
    batch_documents: list[str] = []
    batch_metadatas: list[dict[str, str | int | float | bool]] = []
    batch_embeddings: list[list[float]] = []

    def flush_batch() -> None:
        nonlocal saved, batch_ids, batch_documents, batch_metadatas, batch_embeddings
        if not batch_ids:
            return

        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings,
        )
        saved += len(batch_ids)
        batch_ids = []
        batch_documents = []
        batch_metadatas = []
        batch_embeddings = []

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "") or ""
        metadata = sanitize_metadata(chunk.get("metadata") or {})
        record_id = make_chroma_id(chunk)
        metadata["record_id"] = record_id
        metadata["text_length"] = len(text)

        batch_ids.append(record_id)
        batch_documents.append(text)
        batch_metadatas.append(metadata)
        batch_embeddings.append(_as_embedding_list(embeddings[idx]))

        if len(batch_ids) >= batch_size:
            flush_batch()

    flush_batch()
    return saved


def load_chunks_from_chroma() -> list[dict[str, Any]]:
    """从 ChromaDB 加载所有 chunk，并恢复为内存中的 chunk 结构。"""
    collection = get_chroma_collection(create=False)
    data = collection.get(include=["documents", "metadatas"])

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    chunks = [
        {
            "text": document or "",
            "metadata": metadata or {},
        }
        for document, metadata in zip(documents, metadatas)
    ]

    def sort_key(chunk: dict[str, Any]):
        metadata = chunk.get("metadata") or {}
        try:
            return int(metadata.get("chunk_id", 0))
        except (TypeError, ValueError):
            return 0

    chunks.sort(key=sort_key)
    return chunks


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