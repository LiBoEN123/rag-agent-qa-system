"""只读 RAG runtime：加载持久索引、管理内存状态并执行查询。"""

from typing import Any, Dict, List

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.common.logging import get_logger
from app.config import ensure_model_cache_dir, get_settings
from app.rag.chroma import chroma_collection_has_chunks, load_index_from_chroma
from app.storage.vector import create_vector_store, load_embedding_model
from app.rag.retrieval import tokenize
from app.rag.debug import (
    get_chunk_full_text as get_chunk_full_text_impl,
    preview_chunks as preview_chunks_impl,
    preview_docs as preview_docs_impl,
)
from app.rag.policy import should_refuse
from app.rag.retrieval import (
    retrieve as retrieve_impl,
    rerank as rerank_impl,
    search_relevant_chunks as search_relevant_chunks_impl,
)
from app.common.numeric import safe_float

logger = get_logger("chat_service")


def _load_rerank_model():
    """从本地缓存加载交叉编码器重排模型。"""
    settings = get_settings()
    cache_folder = str(ensure_model_cache_dir(settings))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"[models] loading rerank model={settings.rerank_model_name} "
        f"device={device} cache_dir={cache_folder}"
    )
    return CrossEncoder(
        settings.rerank_model_name,
        device=device,
        cache_folder=cache_folder,
        local_files_only=True
    )


def _load_models(model=None, rerank_model=None):
    """在索引缓存可用时加载 embedding 和 rerank 模型。"""
    if model is None:
        model = load_embedding_model()

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


def _load_index_bundle(model=None, rerank_model=None):
    """从 ChromaDB 恢复索引数据、BM25 和模型实例。"""
    if not chroma_collection_has_chunks():
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


# 进程内索引状态。
chunks: list[dict] = []
texts: list[str] = []
docs_registry: list[dict] = []

bm25 = None
model = None
rerank_model = None
doc_embeddings = None

RAG_READY = False
RAG_INIT_MESSAGE = "RAG 尚未初始化"
INDEX_SOURCE = "none"
NO_DOCUMENTS_MESSAGE = "当前知识库为空，请先显式同步知识索引后再提问"


# =========================
# 状态函数
# =========================
def get_rag_status():
    """返回当前内存索引是否可用，以及文档和 chunk 数量。"""
    return {
        "ready": RAG_READY,
        "message": RAG_INIT_MESSAGE,
        "doc_count": len(docs_registry),
        "chunk_count": len(chunks),
        "index_source": INDEX_SOURCE
    }


# =========================
# 初始化 / 构建
# =========================
def _publish_runtime_bundle(bundle: dict) -> None:
    """只在索引读取和 runtime 验证成功后替换进程状态。"""
    global chunks, texts, docs_registry, bm25, model, rerank_model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE, INDEX_SOURCE

    chunks = bundle["chunks"]
    texts = bundle["texts"]
    docs_registry = bundle["docs_registry"]
    doc_embeddings = bundle["doc_embeddings"]
    bm25 = bundle["bm25"]
    model = bundle["model"]
    rerank_model = bundle["rerank_model"]
    RAG_READY = bool(chunks)
    RAG_INIT_MESSAGE = "OK" if chunks else NO_DOCUMENTS_MESSAGE
    INDEX_SOURCE = bundle.get("source", "none") if chunks else "none"


def load_index_cache(*, allow_empty: bool = False) -> tuple[bool, str]:
    """恢复全局 RAG 状态；来源操作可在严格确认空库后发布空 runtime。"""
    global chunks, texts, docs_registry, bm25, model, rerank_model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE, INDEX_SOURCE

    result = _load_index_bundle(model=model)

    if allow_empty and (not result["ok"] or not result["chunks"]):
        try:
            store = create_vector_store(None, create_collection_if_not_exists=False)
            if store.get(limit=1, include=[])["ids"]:
                return False, result["message"]
        except Exception:
            return False, "无法验证 ChromaDB 空索引"
        result.update(ok=True, message=NO_DOCUMENTS_MESSAGE)
        _publish_runtime_bundle(result)
        return True, result["message"]

    if not result["ok"]:
        RAG_READY = False
        RAG_INIT_MESSAGE = result["message"]
        INDEX_SOURCE = result.get("source", "none")
        return False, RAG_INIT_MESSAGE

    if not result["chunks"]:
        RAG_READY = False
        RAG_INIT_MESSAGE = "索引缓存为空"
        INDEX_SOURCE = result.get("source", "none")
        return False, RAG_INIT_MESSAGE

    _publish_runtime_bundle(result)
    return True, result["message"]


# =========================
# 查询阶段：检索包装
# =========================
def retrieve(question: str, top_k: int = 3):
    """使用当前全局索引执行一次基础召回。"""
    return retrieve_impl(
        question=question,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=top_k
    )

def rerank(candidates: list[dict], question: str=""):
    """使用当前全局重排模型对候选 chunk 重新排序。"""
    return rerank_impl(
        rerank_model=rerank_model,
        candidates=candidates,
        question=question,
    )


def search_relevant_chunks(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """使用当前全局索引检索与问题相关的 chunk。"""
    return search_relevant_chunks_impl(
        question=question,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=top_k
    )


# =========================
# 调试接口包装
# =========================
def run_diagnostic_search(question: str, top_k: int) -> dict[str, Any]:
    """按当前索引执行单查询检索、重排和证据门控，不进入问答工作流。"""
    rag_status = get_rag_status()
    if not rag_status["ready"]:
        return {
            "status": "empty_index",
            "message": rag_status["message"],
            "reason": "知识库未准备完成",
            "chunks": [],
        }

    retrieved = search_relevant_chunks(question, top_k)
    if not retrieved:
        return {
            "status": "no_evidence",
            "message": "success",
            "reason": "未检索到结果",
            "chunks": [],
        }

    final_chunks = rerank(retrieved, question)[:top_k]
    if not final_chunks:
        return {
            "status": "no_evidence",
            "message": "success",
            "reason": "重排后没有可用结果",
            "chunks": [],
        }

    if should_refuse(final_chunks):
        max_vector_score = max(
            safe_float(item.get("vector_score")) for item in final_chunks
        )
        return {
            "status": "no_evidence",
            "message": "success",
            "reason": f"证据分数低于阈值：max_vector={max_vector_score:.4f}",
            "chunks": [],
        }

    return {
        "status": "ok",
        "message": "success",
        "reason": "检索成功",
        "chunks": final_chunks,
    }


def preview_docs():
    """返回当前知识库文档注册表的调试预览。"""
    return preview_docs_impl(docs_registry)


def preview_chunks(max_len:int, limit: int = 70, file_name: str | None = None):
    """返回当前知识库 chunk 的调试预览。"""
    return preview_chunks_impl(chunks, max_len=max_len, limit=limit, file_name=file_name)


def get_chunk_full_text(chunk_id: int | str):
    """根据 chunk_id 返回对应 chunk 的完整文本。"""
    return get_chunk_full_text_impl(chunks, chunk_id)
