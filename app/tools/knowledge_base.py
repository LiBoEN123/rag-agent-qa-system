"""知识库工具模块。

向 Agent 暴露文档列表、chunk 原文和相关证据检索等基础能力。
"""

from app.tools.context import ToolInvocationContext
from app.tools.result import tool_error, tool_success


def list_documents() -> dict:
    """列出知识库中的文档并包装为标准工具结果。"""
    tool_name = "list_documents"
    tool_input = {}
    try:
        from app.rag.pipeline import preview_docs

        documents = preview_docs()
        return tool_success(
            tool_name,
            tool_input,
            {"count": len(documents), "documents": documents},
        )
    except Exception as exc:
        return tool_error(tool_name, tool_input, str(exc))


def get_chunk_text(chunk_id: int | str) -> dict:
    """按 chunk 编号读取完整文本并包装为标准工具结果。"""
    tool_name = "get_chunk_text"
    tool_input = {"chunk_id": chunk_id}
    try:
        from app.rag.pipeline import get_chunk_full_text

        chunk = get_chunk_full_text(chunk_id)
        if not chunk.get("found"):
            return tool_error(
                tool_name,
                tool_input,
                f"未找到 chunk_id={chunk_id} 的 chunk，可能是知识库未初始化或 chunk_id 不存在",
            )
        return tool_success(
            tool_name,
            tool_input,
            {"chunk_id": chunk_id, "chunk": chunk},
        )
    except Exception as exc:
        return tool_error(tool_name, tool_input, str(exc))


def _get_chunk_value(chunk: dict, key: str, default=None):
    """优先从 chunk 顶层读取字段，其次读取 metadata。"""
    if not isinstance(chunk, dict):
        return default
    value = chunk.get(key)
    if value is not None:
        return value
    metadata = chunk.get("metadata") or {}
    return metadata.get(key, default) if isinstance(metadata, dict) else default


def _round_chunk_score(chunk: dict, key: str) -> float:
    """安全读取并保留四位小数的 chunk 分数。"""
    try:
        return round(float(_get_chunk_value(chunk, key, 0.0) or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


def search_knowledge_base(
    args: dict,
    context: ToolInvocationContext | None = None,
) -> dict:
    """检索本地证据并在工具内部执行独立 RAG Graph。"""
    tool_name = "search_knowledge_base"
    try:
        from app.config import get_settings
        from app.rag.pipeline import search_relevant_chunks
        from app.rag.graph import run_rag_subgraph

        query = args["query"]
        top_k = args["top_k"]
        tool_input = {"query": query, "top_k": top_k}

        prefetched_chunks = search_relevant_chunks(query, top_k=top_k)
        effective_question = (
            (context.effective_question if context is not None else "").strip()
            or query
        )
        rag_top_k = (
            context.top_k
            if context is not None and context.top_k > 0
            else top_k
        )
        decompose_top_k = (
            context.decompose_top_k
            if context is not None and context.decompose_top_k > 0
            else get_settings().decompose_top_k
        )
        rag_result = run_rag_subgraph(
            question=effective_question,
            top_k=rag_top_k,
            decompose_top_k=decompose_top_k,
            retrieved_chunks=prefetched_chunks,
            trace_id=context.trace_id if context is not None else "",
        )
        final_chunks = rag_result.get("final_chunks") or []
        return tool_success(
            tool_name,
            tool_input,
            {
                "count": len(final_chunks),
                "chunks": final_chunks,
                "top_chunk_ids": [
                    _get_chunk_value(chunk, "chunk_id")
                    for chunk in final_chunks
                ],
                "top_scores": [
                    _round_chunk_score(chunk, "hybrid_score")
                    for chunk in final_chunks
                ],
                "rag_result": rag_result,
            },
        )
    except Exception as exc:
        return tool_error(
            tool_name,
            args if isinstance(args, dict) else {},
            str(exc),
        )
