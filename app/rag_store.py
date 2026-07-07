import json

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.chroma_store import (
    chroma_collection_has_chunks,
    load_chunks_from_chroma,
    persist_chunks_to_chroma,
)
from app.config import Settings
from app.rag_ingest import collect_supported_files
from app.rag_retrieval import tokenize
from app.untils import get_logger


devcie = 'cuda' if torch.cuda.is_available() else "cpu"
logger = get_logger('chat_service')

INDEX_STORE_DIR = Settings.INDEX_STORE_DIR

MANIFEST_PATH = INDEX_STORE_DIR / "index_manifest.json"


def _model_cache_folder() -> str:
    return str(Settings.ensure_model_cache_dir())


def _load_embedding_model():
    cache_folder = _model_cache_folder()
    logger.info(
        f"[models] loading embedding model={Settings.EMBEDDING_MODEL_NAME} "
        f"device={devcie} cache_dir={cache_folder}"
    )
    return SentenceTransformer(
        Settings.EMBEDDING_MODEL_NAME,
        device=devcie,
        cache_folder=cache_folder,
        local_files_only=True
    )


def _load_rerank_model():
    cache_folder = _model_cache_folder()
    logger.info(
        f"[models] loading rerank model={Settings.RERANK_MODEL_NAME} "
        f"device={devcie} cache_dir={cache_folder}"
    )
    return CrossEncoder(
        Settings.RERANK_MODEL_NAME,
        device=devcie,
        cache_folder=cache_folder,
        local_files_only=True
    )


def build_index(all_chunks: list[dict], model=None, rerank_model=None):
    """基于 chunk 文本构建 BM25、向量嵌入和重排模型。"""
    if not all_chunks:
        return {
            "texts": [],
            "bm25": None,
            "doc_embeddings": None,
            "model": model,
            "rerank_model": rerank_model
        }

    local_texts = [chunk["text"] for chunk in all_chunks]
    tokenized_texts = [tokenize(text) for text in local_texts]

    if model is None:
        model = _load_embedding_model()

    if rerank_model is None:
        rerank_model = _load_rerank_model()

    local_doc_embeddings = model.encode(
        local_texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    local_bm25 = BM25Okapi(tokenized_texts)

    return {
        "texts": local_texts,
        "bm25": local_bm25,
        "doc_embeddings": local_doc_embeddings,
        "model": model,
        "rerank_model": rerank_model
    }


def build_file_snapshot(data_dir: str):
    """记录知识库文件的路径、大小和修改时间快照。"""
    files = collect_supported_files(data_dir)
    snapshot = []

    for file_path in files:
        stat = file_path.stat()
        snapshot.append({
            "path": str(file_path.resolve()),
            "size": stat.st_size,
            "time": stat.st_mtime
        })

    snapshot.sort(key=lambda x: x["path"])
    return snapshot


def build_current_manifest(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    """生成当前索引构建配置和文件快照的 manifest。"""
    return {
        "pipeline_version": 1,
        "embedding_model": Settings.EMBEDDING_MODEL_NAME,
        "rerank_model": Settings.RERANK_MODEL_NAME,
        "build_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "files": build_file_snapshot(data_dir),
    }


def load_manifest():
    """从磁盘读取历史索引 manifest。"""
    if not MANIFEST_PATH.exists():
        return None

    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_manifest(manifest: dict):
    """将当前 manifest 写入磁盘。"""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def needs_rebuild(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
) -> tuple[bool, str]:
    """对比历史 manifest 与当前状态判断是否需要重建索引。"""
    old_manifest = load_manifest()
    current_manifest = build_current_manifest(data_dir, chunk_size, chunk_overlap)

    if old_manifest is None:
        return True, "没有找到历史 manifest，需要首次构建"

    if old_manifest.get("pipeline_version") != current_manifest.get("pipeline_version"):
        return True, "pipeline_version 变化，需要重建"

    if old_manifest.get("embedding_model") != current_manifest.get("embedding_model"):
        return True, "embedding 模型变化，需要重建"

    if old_manifest.get("rerank_model") != current_manifest.get("rerank_model"):
        return True, "rerank 模型变化，需要重建"

    if old_manifest.get("build_config") != current_manifest.get("build_config"):
        return True, "chunk 参数变化，需要重建"

    if old_manifest.get("files") != current_manifest.get("files"):
        return True, "data 目录文件变化，需要重建"

    if not chroma_collection_has_chunks():
        return True, "ChromaDB collection 中未找到 chunk，需要重建"

    return False, "目录和参数都没有变化，无需重建"


def index_cache_exists() -> bool:
    """Return whether the ChromaDB index cache is available."""
    return chroma_collection_has_chunks()


def get_index_cache_source() -> str:
    """返回当前可用的持久化索引来源。"""
    if index_cache_exists():
        return "chroma"
    return "none"


def save_index_cache(
    chunks: list[dict],
    doc_embeddings
):
    """Persist chunk text, metadata, and embeddings to ChromaDB."""
    try:
        saved_count = persist_chunks_to_chroma(
            chunks=chunks,
            embeddings=doc_embeddings,
            reset=True,
        )
        logger.info(f"[index] saved chunks to ChromaDB count={saved_count}")
    except Exception:
        logger.exception("[index] failed to save chunks to ChromaDB")
        raise


def _load_models(model=None, rerank_model=None):
    """在索引缓存可用时加载 embedding 和 rerank 模型。"""
    if model is None:
        model = _load_embedding_model()

    if rerank_model is None:
        rerank_model = _load_rerank_model()

    return model, rerank_model


def _build_bm25(texts: list[str]):
    """基于 ChromaDB 中恢复出的 chunk 文本重建 BM25 索引。"""
    tokenized_chunks = [tokenize(text) for text in texts]
    return BM25Okapi(tokenized_chunks) if tokenized_chunks else None


def build_docs_registry_from_chunks(chunks: list[dict]) -> list[dict]:
    """Rebuild the document registry from ChromaDB chunk metadata."""
    docs_by_source: dict[str, dict] = {}
    docs_registry: list[dict] = []

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        source_path = str(metadata.get("source_path") or "")
        if not source_path:
            continue

        if source_path not in docs_by_source:
            doc = {
                "doc_id": metadata.get("doc_id") or f"doc_{len(docs_registry) + 1}",
                "file_name": metadata.get("file_name") or "",
                "source_path": source_path,
                "file_type": metadata.get("file_type") or "",
                "chunk_count": 0,
            }
            docs_by_source[source_path] = doc
            docs_registry.append(doc)

        docs_by_source[source_path]["chunk_count"] += 1

    return docs_registry


def load_index_cache(model=None, rerank_model=None):
    """从 ChromaDB 恢复索引数据、BM25 和模型实例。"""
    if not index_cache_exists():
        return {
            "ok": False,
            "message": "未找到 ChromaDB 索引缓存",
            "source": "none",
            "chunks": [],
            "texts": [],
            "docs_registry": [],
            "doc_embeddings": None,
            "bm25": None,
            "model": model,
            "rerank_model": rerank_model
        }

    try:
        chunks = load_chunks_from_chroma()
        texts = [chunk["text"] for chunk in chunks]
        docs_registry = build_docs_registry_from_chunks(chunks)
        bm25 = _build_bm25(texts)
        model, rerank_model = _load_models(model=model, rerank_model=rerank_model)

        return {
            "ok": True,
            "message": "已从 ChromaDB 加载索引",
            "source": "chroma",
            "chunks": chunks,
            "texts": texts,
            "docs_registry": docs_registry,
            "doc_embeddings": None,
            "bm25": bm25,
            "model": model,
            "rerank_model": rerank_model
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"加载 ChromaDB 索引缓存失败：{e}",
            "source": "none",
            "chunks": [],
            "texts": [],
            "docs_registry": [],
            "doc_embeddings": None,
            "bm25": None,
            "model": model,
            "rerank_model": rerank_model
        }
