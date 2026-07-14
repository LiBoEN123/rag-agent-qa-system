"""RAG 公共门面。

统一暴露索引和问答能力，并动态转发可能在运行期间重建的索引状态。
"""

from typing import Any

from app.rag import pipeline as _pipeline


__all__ = [
    "INDEX_SOURCE",
    "NO_DOCUMENTS_MESSAGE",
    "RAG_INIT_MESSAGE",
    "RAG_READY",
    "STOP_WORD",
    "Settings",
    "ask",
    "bm25",
    "build_evidence_gate_step",
    "build_rag_result",
    "call_llm",
    "chunks",
    "decompose_query_with_llm",
    "doc_embeddings",
    "docs_registry",
    "format_reference",
    "get_chunk_full_text",
    "get_rag_status",
    "index_cache_exists",
    "load_index_cache",
    "model",
    "needs_rebuild",
    "parse_llm_output",
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
    "should_refuse",
    "texts",
    "verify_answer_citations",
]


NO_DOCUMENTS_MESSAGE = _pipeline.NO_DOCUMENTS_MESSAGE
STOP_WORD = _pipeline.STOP_WORD
Settings = _pipeline.Settings

build_evidence_gate_step = _pipeline.build_evidence_gate_step
build_rag_result = _pipeline.build_rag_result
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

# 这些别名有意设计成可以修改，测试代码或调用者可以替换门面层上的函数。
decompose_query_with_llm = _pipeline.decompose_query_with_llm
call_llm = _pipeline.call_llm
should_refuse = _pipeline.should_refuse
parse_llm_output = _pipeline.parse_llm_output
format_reference = _pipeline.format_reference
verify_answer_citations = _pipeline.verify_answer_citations


def ask(*args, **kwargs):
    """执行 RAG 问答流程，并应用门面层注入的可替换依赖。"""
    _pipeline.decompose_query_with_llm = decompose_query_with_llm
    _pipeline.call_llm = call_llm
    _pipeline.rerank = rerank
    _pipeline.should_refuse = should_refuse
    _pipeline.parse_llm_output = parse_llm_output
    _pipeline.format_reference = format_reference
    _pipeline.verify_answer_citations = verify_answer_citations
    return _pipeline.ask(*args, **kwargs)


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
    top_k: int = Settings.TOP_K,
    decompose_top_k: int = Settings.DECOMPOSE_TOP_K,
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
