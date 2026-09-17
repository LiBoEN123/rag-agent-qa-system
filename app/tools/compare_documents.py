"""在两份指定知识库文档中检索证据，由 Agent 完成比较。"""

from typing import Any

from langchain_core.documents import Document

from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.knowledge_base import chunk_to_evidence_document
from app.tools.result import tool_error, tool_success


def _normalize_doc_label(value: Any) -> str:
    """标准化文档名，便于文件名精确或模糊匹配。"""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = text.replace("\\", "/").split("/")[-1]
    text = text.strip(" 《》<>\"'“”‘’")
    return re.sub(r"\s+", "", text)


def _strip_known_suffix(value: str) -> str:
    """去掉常见文档扩展名，兼容用户只输入标题的情况。"""
    for suffix in [".md", ".txt", ".pdf", ".docx", ".doc"]:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _get_compare_chunk_file_name(chunk: dict) -> str:
    """从 chunk 顶层或 metadata 中提取用于文档匹配的文件名。"""
    metadata = chunk.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return str(
        chunk.get("file_name")
        or metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("source_path")
        or ""
    )


def _doc_name_matches(file_name: str, doc_name: str, *, exact: bool) -> bool:
    """判断 chunk 文件名是否匹配用户指定的文档名或关键词。"""
    file_label = _normalize_doc_label(file_name)
    doc_label = _normalize_doc_label(doc_name)
    if not file_label or not doc_label:
        return False

    file_no_suffix = _strip_known_suffix(file_label)
    doc_no_suffix = _strip_known_suffix(doc_label)
    if exact:
        return (
            doc_label in {file_label, file_no_suffix}
            or doc_no_suffix in {file_label, file_no_suffix}
        )

    return (
        doc_label in file_label
        or file_label in doc_label
        or doc_no_suffix in file_no_suffix
        or file_no_suffix in doc_no_suffix
    )


def _filter_chunks_by_document(chunks: list[dict], doc_name: str) -> list[dict]:
    """按文档名筛选 chunk，优先精确匹配，失败后再做包含匹配。"""
    candidates = [chunk for chunk in chunks if isinstance(chunk, dict)]
    exact_matches = [
        chunk
        for chunk in candidates
        if _doc_name_matches(_get_compare_chunk_file_name(chunk), doc_name, exact=True)
    ]
    if exact_matches:
        return exact_matches

    return [
        chunk
        for chunk in candidates
        if _doc_name_matches(_get_compare_chunk_file_name(chunk), doc_name, exact=False)
    ]


def _select_compare_evidence(
    chunks: list[dict],
    doc_name: str,
    query: str,
    top_k: int,
    side: str,
) -> list[Document]:
    """复制已过滤的候选，复用重排与 KB identity，仅为返回视图添加分组。"""
    from app.rag.pipeline import rerank

    candidates = [{**chunk, "metadata": dict(chunk.get("metadata") or {})} for chunk in chunks]
    evidence = []
    for chunk in rerank(candidates, query)[:top_k]:
        document = chunk_to_evidence_document(chunk)
        evidence.append(document.model_copy(update={"metadata": {
            **document.metadata,
            "comparison_side": side,
            "comparison_document": doc_name,
        }}))
    return evidence


def compare_documents_tool(args: dict) -> dict:
    """分别返回两份指定文档的 Top-K Document evidence，不生成比较结论。"""
    tool_name = "compare_documents"
    try:
        query = args["query"]
        doc_a = args["doc_a"]
        doc_b = args["doc_b"]
        top_k = args["top_k"]
        tool_input = {"query": query, "doc_a": doc_a, "doc_b": doc_b, "top_k": top_k}
        if _strip_known_suffix(_normalize_doc_label(doc_a)) == _strip_known_suffix(
            _normalize_doc_label(doc_b)
        ):
            return tool_error(tool_name, tool_input, "doc_a 和 doc_b 不能相同")

        from app.rag import pipeline

        if not pipeline.get_rag_status()["ready"]:
            raise ToolExecutionError(
                "知识库尚未准备完成",
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                retryable=False,
            )
        current_chunks = pipeline.chunks
        doc_a_chunks = _filter_chunks_by_document(current_chunks, doc_a)
        doc_b_chunks = _filter_chunks_by_document(current_chunks, doc_b)
        if not doc_a_chunks:
            return tool_error(tool_name, tool_input, "doc_a 未找到匹配文档 chunk")
        if not doc_b_chunks:
            return tool_error(tool_name, tool_input, "doc_b 未找到匹配文档 chunk")

        evidence_a = _select_compare_evidence(doc_a_chunks, doc_a, query, top_k, "doc_a")
        evidence_b = _select_compare_evidence(doc_b_chunks, doc_b, query, top_k, "doc_b")
        return tool_success(
            tool_name,
            tool_input,
            {
                "query": query,
                "doc_a": doc_a,
                "doc_b": doc_b,
                "doc_a_count": len(evidence_a),
                "doc_b_count": len(evidence_b),
                "count": len(evidence_a) + len(evidence_b),
                "evidence": evidence_a + evidence_b,
            },
        )
    except ToolExecutionError:
        raise
    except Exception as exc:
        return tool_error(tool_name, args if isinstance(args, dict) else {}, str(exc))
