"""RAG 公共门面。

暴露索引、检索和 LangGraph 问答能力，并动态转发运行时索引状态。
"""

from typing import Any

from app.config import Settings as _Settings
from app.rag import pipeline as _pipeline


__all__ = [
    "INDEX_SOURCE",
    "NO_DOCUMENTS_MESSAGE",
    "RAG_INIT_MESSAGE",
    "RAG_READY",
    "bm25",
    "chunks",
    "doc_embeddings",
    "docs_registry",
    "get_chunk_full_text",
    "get_rag_status",
    "index_cache_exists",
    "load_index_cache",
    "model",
    "needs_rebuild",
    "preview_chunks",
    "preview_docs",
    "rebuild_and_persist_index",
    "rebuild_index",
    "rerank",
    "rerank_model",
    "retrieve",
    "run_rag_subgraph",
    "save_index_cache",
    "search_relevant_chunks",
    "texts",
]


NO_DOCUMENTS_MESSAGE = _pipeline.NO_DOCUMENTS_MESSAGE

get_rag_status = _pipeline.get_rag_status
rebuild_index = _pipeline.rebuild_index
rebuild_and_persist_index = _pipeline.rebuild_and_persist_index
needs_rebuild = _pipeline.needs_rebuild
index_cache_exists = _pipeline.index_cache_exists
save_index_cache = _pipeline.save_index_cache
load_index_cache = _pipeline.load_index_cache
retrieve = _pipeline.retrieve
rerank = _pipeline.rerank
search_relevant_chunks = _pipeline.search_relevant_chunks
preview_docs = _pipeline.preview_docs
preview_chunks = _pipeline.preview_chunks
get_chunk_full_text = _pipeline.get_chunk_full_text


def __getattr__(name: str):
    """动态读取重建索引时可能被重新绑定的运行时状态。"""
    # Index state is rebound during rebuild/load, so resolve it dynamically.
    if name in {
        "chunks",
        "texts",
        "docs_registry",
        "bm25",
        "model",
        "rerank_model",
        "doc_embeddings",
        "RAG_READY",
        "RAG_INIT_MESSAGE",
        "INDEX_SOURCE",
    }:
        return getattr(_pipeline, name)
    raise AttributeError(name)


def run_rag_subgraph(
    question: str,
    top_k: int = _Settings.TOP_K,
    decompose_top_k: int = _Settings.DECOMPOSE_TOP_K,
    retrieved_chunks: list[dict[str, Any]] | None = None,
    trace_id: str = "",
) -> dict[str, Any]:
    """延迟调用 LangGraph RAG 子图，避免 RAG 门面导入时编译或执行图。"""
    from app.rag.graph import run_rag_subgraph as graph_runner

    return graph_runner(
        question=question,
        top_k=top_k,
        decompose_top_k=decompose_top_k,
        retrieved_chunks=retrieved_chunks,
        trace_id=trace_id,
    )
