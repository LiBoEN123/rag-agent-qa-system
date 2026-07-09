"""Tools that expose knowledge-base documents and chunks."""

from app.tools.result import tool_error, tool_success


def list_documents() -> dict:
    tool_name = "list_documents"
    tool_input = {}
    try:
        from app.rag import preview_docs

        documents = preview_docs()
        return tool_success(
            tool_name,
            tool_input,
            {"count": len(documents), "documents": documents},
        )
    except Exception as exc:
        return tool_error(tool_name, tool_input, str(exc))


def get_chunk_text(chunk_id: int | str) -> dict:
    tool_name = "get_chunk_text"
    tool_input = {"chunk_id": chunk_id}
    try:
        from app.rag import get_chunk_full_text

        if not chunk_id or not str(chunk_id).strip():
            return tool_error(tool_name, tool_input, "chunk_id 不能为空")
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
    if not isinstance(chunk, dict):
        return default
    value = chunk.get(key)
    if value is not None:
        return value
    metadata = chunk.get("metadata") or {}
    return metadata.get(key, default) if isinstance(metadata, dict) else default


def _round_chunk_score(chunk: dict, key: str) -> float:
    try:
        return round(float(_get_chunk_value(chunk, key, 0.0) or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


def search_knowledge_base(args: dict) -> dict:
    tool_name = "search_knowledge_base"
    try:
        from app.config import Settings
        from app.rag import search_relevant_chunks

        args = args or {}
        query = str(args.get("query") or "").strip()
        raw_top_k = args.get("top_k", Settings.TOP_K)
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            return tool_error(
                tool_name,
                {"query": query, "top_k": raw_top_k},
                "top_k 必须是整数",
            )
        top_k = max(1, min(top_k, 20))
        tool_input = {"query": query, "top_k": top_k}
        if not query:
            return tool_error(tool_name, tool_input, "query 不能为空")

        chunks = search_relevant_chunks(query, top_k=top_k)
        return tool_success(
            tool_name,
            tool_input,
            {
                "count": len(chunks),
                "chunks": chunks,
                "top_chunk_ids": [
                    _get_chunk_value(chunk, "chunk_id") for chunk in chunks
                ],
                "top_scores": [
                    _round_chunk_score(chunk, "hybrid_score") for chunk in chunks
                ],
            },
        )
    except Exception as exc:
        return tool_error(
            tool_name,
            args if isinstance(args, dict) else {},
            str(exc),
        )
