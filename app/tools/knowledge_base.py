"""知识库工具模块。

向 Agent 暴露文档列表、chunk 原文和相关证据检索等基础能力。
"""

from urllib.parse import quote

from langchain_core.documents import Document

from app.tools.context import ToolInvocationContext
from app.tools.evidence import build_evidence_document
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.result import tool_error, tool_success


def get_chunk_context(
    args: dict,
    context: ToolInvocationContext | None = None,
) -> dict:
    """按已有 source_id + position 扩展上下文，不检索或修改 runtime chunks。"""
    tool_name = "get_chunk_context"
    try:
        from app.rag import runtime
        from app.rag.debug import get_chunk_full_text

        # anchor 与邻居使用同一 runtime 快照；复用原文工具底层的唯一查找逻辑。
        chunks = runtime.chunks
        chunk_id = args["chunk_id"]
        anchor = get_chunk_full_text(chunks, chunk_id)
        if not anchor.get("found"):
            raise ToolExecutionError(
                f"未找到 chunk_id={chunk_id}。chunk_id 必须来自知识库 evidence 的 "
                "metadata.chunk_id，不能使用 Document.id / evidence id。",
                code=ToolErrorCode.INVALID_ARGUMENTS, retryable=False,
            )
        metadata = anchor["metadata"]
        source_id = metadata.get("source_id")
        position = metadata.get("position")
        if (
            not isinstance(source_id, str) or not source_id.strip()
            or type(position) is not int or position < 1
        ):
            raise ToolExecutionError(
                f"chunk_id={chunk_id} 缺少有效的 source_id 或正整数 position，"
                "无法确定同文档上下文。",
                code=ToolErrorCode.INVALID_ARGUMENTS, retryable=False,
            )

        before, after = args["before"], args["after"]
        start, end = max(1, position - before), position + after
        neighbors = []
        for chunk in chunks:
            neighbor_metadata = chunk.get("metadata") or {}
            neighbor_position = neighbor_metadata.get("position")
            if (
                neighbor_metadata.get("source_id") == source_id
                and type(neighbor_position) is int
                and start <= neighbor_position <= end
            ):
                neighbors.append(chunk)
        neighbors.sort(key=lambda chunk: chunk["metadata"]["position"])
        evidence: dict[str, Document] = {}
        for chunk in neighbors:
            document = chunk_to_evidence_document(chunk)
            evidence.setdefault(document.id, document)
        return tool_success(tool_name, args, {
            "anchor_chunk_id": chunk_id, "source_id": source_id,
            "anchor_position": position, "before": before, "after": after,
            "count": len(evidence), "evidence": list(evidence.values()),
        })
    except ToolExecutionError:
        raise
    except Exception as exc:
        return tool_error(tool_name, args, str(exc))


def chunk_to_evidence_document(chunk: dict) -> Document:
    """将检索或原文读取的 chunk 投影为同一 Document，不改变正文和分数。"""
    metadata = dict(chunk.get("metadata") or {})
    source_uri = metadata.get("source_uri")
    # 原索引 source_type 表示入库来源；transport source_type 表示工具证据种类。
    if metadata.get("source_type"):
        metadata["knowledge_source_type"] = metadata["source_type"]
    for key in (
        "source_id", "source_uri", "source_path", "document_uid", "file_type",
        "page_number", "title_path", "file_name", "chunk_id", "position",
        "bm25_score", "vector_score", "hybrid_score", "rerank_score",
    ):
        if chunk.get(key) is not None:
            metadata[key] = chunk[key]
    for key in (
        "file_name", "chunk_id", "position", "bm25_score", "vector_score",
        "hybrid_score", "rerank_score",
    ):
        metadata.setdefault(key, None)
    source_id = metadata.get("source_id")
    position = metadata.get("position")
    chunk_id = metadata.get("chunk_id")
    source_uri = source_uri or metadata.get("source_uri") or metadata.get("source_path")
    if source_id and position is not None:
        identity = ("source_position", source_id, str(position))
    elif source_uri and position is not None:
        identity = ("uri_position", source_uri, str(position))
    elif chunk_id is not None and str(chunk_id).strip():
        identity = ("chunk", source_id or source_uri or "", str(chunk_id))
    else:
        raise ValueError("chunk 缺少稳定来源和 locator")
    # 旧 chunk 可能仅有 chunk_id；URI 明确标记为 KB locator，不伪造文件路径。
    source_uri = source_uri or (
        f"knowledge_base://source/{quote(str(source_id), safe='')}" if source_id
        else f"knowledge_base://chunk/{quote(str(chunk_id), safe='')}"
    )
    return build_evidence_document(
        page_content=chunk["text"], source_type="knowledge_base",
        source_uri=source_uri, identity=identity, metadata=metadata,
    )


def search_knowledge_base(
    args: dict,
    context: ToolInvocationContext | None = None,
) -> dict:
    """按本轮工具参数混合召回、重排，只返回候选证据。"""
    tool_name = "search_knowledge_base"
    try:
        from app.rag.runtime import get_rag_status, rerank, search_relevant_chunks

        if not get_rag_status()["ready"]:
            raise ToolExecutionError(
                "知识库尚未准备完成",
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                retryable=False,
            )

        query = args["query"]
        top_k = args["top_k"]
        tool_input = {"query": query, "top_k": top_k}

        candidates = search_relevant_chunks(query, top_k=top_k)
        final_chunks = rerank(candidates, query)[:top_k]
        return tool_success(
            tool_name,
            tool_input,
            {
                "query": query,
                "count": len(final_chunks),
                "evidence": [chunk_to_evidence_document(chunk) for chunk in final_chunks],
            },
        )
    except ToolExecutionError:
        raise
    except Exception as exc:
        return tool_error(
            tool_name,
            args if isinstance(args, dict) else {},
            str(exc),
        )
