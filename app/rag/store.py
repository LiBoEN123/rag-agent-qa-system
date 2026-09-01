"""RAG 索引存储与模型加载模块。

管理模型缓存、BM25 构建、索引清单、ChromaDB 缓存保存和恢复。
"""

import json

import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.rag.chroma import (
    chroma_collection_has_chunks,
    load_index_from_chroma,
    persist_chunks_to_chroma,
)
from app.config import ensure_model_cache_dir, get_settings
from app.rag.ingest import collect_supported_files
from app.rag.indexing import LOCAL_RECORD_NAMESPACE, record_manager_path
from app.rag.retrieval import tokenize
from app.common.logging import get_logger


devcie = 'cuda' if torch.cuda.is_available() else "cpu"
logger = get_logger('chat_service')

def _manifest_path():
    """返回当前配置对应的索引 manifest 路径。"""
    return get_settings().index_store_dir / "index_manifest.json"


def _model_cache_folder() -> str:
    """返回模型加载使用的本地缓存目录。"""
    settings = get_settings()
    return str(ensure_model_cache_dir(settings))


def _load_embedding_model():
    """通过官方 LangChain integration 从本地缓存加载 embedding。"""
    settings = get_settings()
    cache_folder = _model_cache_folder()
    logger.info(
        f"[models] loading embedding model={settings.embedding_model_name} "
        f"device={devcie} cache_dir={cache_folder}"
    )
    return HuggingFaceEmbeddings(
        model=settings.embedding_model_name,
        cache_folder=cache_folder,
        model_kwargs={
            "device": devcie,
            "local_files_only": True,
        },
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_rerank_model():
    """从本地缓存加载交叉编码器重排模型。"""
    settings = get_settings()
    cache_folder = _model_cache_folder()
    logger.info(
        f"[models] loading rerank model={settings.rerank_model_name} "
        f"device={devcie} cache_dir={cache_folder}"
    )
    return CrossEncoder(
        settings.rerank_model_name,
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

    if hasattr(model, "embed_documents"):
        local_doc_embeddings = np.asarray(
            model.embed_documents(local_texts),
            dtype=np.float32,
        )
    else:
        local_doc_embeddings = model.encode(
            local_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
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
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    """生成当前索引构建配置和文件快照的 manifest。"""
    settings = get_settings()
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    chunk_overlap = (
        settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    return {
        "pipeline_version": 2,
        "embedding_model": settings.embedding_model_name,
        "rerank_model": settings.rerank_model_name,
        "record_manager_namespace": LOCAL_RECORD_NAMESPACE,
        "chroma_collection": settings.chroma_index_collection_name,
        "build_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "files": build_file_snapshot(data_dir),
    }


def load_manifest():
    """从磁盘读取历史索引 manifest。"""
    path = _manifest_path()
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_manifest(manifest: dict):
    """将当前 manifest 写入磁盘。"""
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def needs_rebuild(
    data_dir: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[bool, str]:
    """判断是否需要清空 v2 索引；普通文件变化只做增量 sync。"""
    settings = get_settings()
    old_manifest = load_manifest()
    current_manifest = build_current_manifest(data_dir, chunk_size, chunk_overlap)

    if old_manifest is None:
        return True, "没有找到 v2 manifest，需要首次 bootstrap"

    if old_manifest.get("pipeline_version") != current_manifest.get("pipeline_version"):
        return True, "pipeline_version 变化，需要 force rebuild"

    if old_manifest.get("embedding_model") != current_manifest.get("embedding_model"):
        return True, "embedding 模型变化，需要 force rebuild"

    if (
        old_manifest.get("record_manager_namespace")
        != current_manifest.get("record_manager_namespace")
        or old_manifest.get("chroma_collection")
        != current_manifest.get("chroma_collection")
    ):
        return True, "index namespace 或 collection 变化，需要 force rebuild"

    if old_manifest.get("build_config") != current_manifest.get("build_config"):
        return True, "chunk 参数变化，需要 force rebuild"

    if not record_manager_path(settings).is_file():
        return True, "SQLRecordManager 不存在，需要 force rebuild"

    if not chroma_collection_has_chunks():
        return True, "v2 Chroma collection 中未找到 chunk，需要 force rebuild"

    if old_manifest.get("files") != current_manifest.get("files"):
        return False, "Local corpus 变化，将执行 LangChain 增量同步"

    if old_manifest.get("rerank_model") != current_manifest.get("rerank_model"):
        return False, "仅 reranker 变化，不重新 embedding"

    return False, "配置与 corpus 未变化，将验证 unchanged skip"


def index_cache_exists() -> bool:
    """判断 ChromaDB 中是否存在可用的索引缓存。"""
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
    """将 chunk、元数据和向量持久化到 ChromaDB。"""
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
    """根据 ChromaDB chunk 元数据重建文档注册表。"""
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
        chunks, doc_embeddings = load_index_from_chroma()
        if doc_embeddings is None:
            raise ValueError("ChromaDB 索引缓存中缺少 embeddings")
        if getattr(doc_embeddings, "ndim", None) != 2:
            raise ValueError("ChromaDB 索引缓存中的 embeddings 必须是二维矩阵")
        if len(doc_embeddings) != len(chunks):
            raise ValueError(
                "ChromaDB 索引缓存中的 embedding 数量与 chunk 数量不一致"
            )

        texts = [chunk["text"] for chunk in chunks]
        docs_registry = build_docs_registry_from_chunks(chunks)
        bm25 = _build_bm25(texts)
        model, rerank_model = _load_models(model=model, rerank_model=rerank_model)
        logger.info(
            "[index] loaded cache from ChromaDB "
            f"chunk_count={len(chunks)} embedding_count={len(doc_embeddings)} "
            f"embedding_shape={doc_embeddings.shape}"
        )

        return {
            "ok": True,
            "message": "已从 ChromaDB 加载索引",
            "source": "chroma",
            "chunks": chunks,
            "texts": texts,
            "docs_registry": docs_registry,
            "doc_embeddings": doc_embeddings,
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
