"""RAG 公共门面。

统一暴露索引和问答能力，并动态转发可能在运行期间重建的索引状态。
"""

from app.rag import pipeline as _pipeline


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

# These aliases are intentionally mutable: tests and callers patch the facade.
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
