"""从现有 KB metadata 发现文档，并在精确来源范围内检索正文。"""

import unicodedata
from pathlib import PurePath

from app.rag.retrieval import tokenize
from app.tools.context import ToolInvocationContext
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.knowledge_base import chunk_to_evidence_document
from app.tools.result import tool_error, tool_success
from app.tools.schemas import DocumentSearchItem


def _normalize(value: str | None) -> str:
    """保留路径和词边界，只统一 Unicode、大小写和空白。"""
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def _documents_from_chunks(chunks: list[dict]) -> list[DocumentSearchItem]:
    """按已有 source_id 聚合；不把缺失的稳定 ID 替换为运行时 doc_id。"""
    by_source: dict[str, dict] = {}
    fields = (
        "document_uid", "file_name", "source_uri", "source_path", "file_type", "source_type",
    )
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        source_id = metadata.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            continue
        item = by_source.setdefault(source_id, {
            "source_id": source_id, "file_name": "", "chunk_count": 0,
        })
        item["chunk_count"] += 1
        for field in fields:
            if not item.get(field) and metadata.get(field):
                item[field] = metadata[field]
    return [DocumentSearchItem.model_validate(item) for item in by_source.values()]


def _match_score(document: DocumentSearchItem, query: str, tokens: set[str]) -> float:
    """按文件名精确、文件名包含、locator 包含、关键词交集分层评分。"""
    file_name = _normalize(document.file_name)
    locators = [_normalize(value) for value in (
        document.source_path, document.source_uri, document.source_id, document.document_uid,
    )]
    if query == file_name:
        boost = 400
    elif query == PurePath(file_name).stem:
        boost = 350
    elif query in file_name:
        boost = 250
    elif any(query in value for value in locators):
        boost = 150
    else:
        boost = 0
    # overlap 有界，不能盖过更高优先级的匹配；复用现有 jieba helper。
    keywords = {token for value in [file_name, *locators] for token in tokenize(value)
                if token.isalnum()}
    overlap = len(tokens & keywords) / len(tokens) if tokens else 0.0
    return boost + 50.0 * overlap


def _resolve_document(documents: list[DocumentSearchItem], locator: str) -> DocumentSearchItem:
    """按 ID、完整文件名、无扩展名依次精确匹配，任一级歧义立即失败。"""
    normalized = _normalize(locator)
    match_groups = (
        [item for item in documents if item.source_id == locator],
        [item for item in documents if item.document_uid == locator],
        [item for item in documents if _normalize(item.file_name) == normalized],
        [item for item in documents if PurePath(_normalize(item.file_name)).stem == normalized],
    )
    for matches in match_groups:
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ToolExecutionError(
                "ambiguous document：匹配到多份文档，请先使用 find_documents 获取唯一 source_id。",
                code=ToolErrorCode.INVALID_ARGUMENTS, retryable=False,
            )
    raise ToolExecutionError(
        "未找到精确匹配的文档，请先使用 find_documents 获取 source_id / document_uid。",
        code=ToolErrorCode.INVALID_ARGUMENTS, retryable=False,
    )


def _ready_chunks() -> list[dict]:
    from app.rag import pipeline

    if not pipeline.get_rag_status()["ready"]:
        raise ToolExecutionError(
            "知识库尚未准备完成", code=ToolErrorCode.DEPENDENCY_UNAVAILABLE, retryable=False,
        )
    return pipeline.chunks


def find_documents(args: dict, context: ToolInvocationContext | None = None) -> dict:
    """仅对运行时文档元数据排序，不读取正文、不调用模型或创建索引。"""
    try:
        query = _normalize(args.get("query"))
        tokens = {token for token in tokenize(query) if token.isalnum()}
        ranked = []
        for document in _documents_from_chunks(_ready_chunks()):
            score = _match_score(document, query, tokens) if query else 0.0
            if not query or score > 0:
                ranked.append(document.model_copy(update={"score": score}))
        ranked.sort(key=lambda item: (-item.score, _normalize(item.file_name), item.source_id))
        documents = ranked[:args["limit"]]
        return tool_success("find_documents", args, {
            "query": args.get("query"), "count": len(documents), "total": len(ranked),
            "documents": [item.model_dump(mode="json") for item in documents],
        })
    except ToolExecutionError:
        raise
    except Exception as exc:
        return tool_error("find_documents", args, str(exc))


def search_document(args: dict, context: ToolInvocationContext | None = None) -> dict:
    """先固定 runtime chunk 快照并按来源筛选，再复用现有重排与 evidence identity。"""
    try:
        from app.rag.pipeline import rerank

        chunks = _ready_chunks()
        document = _resolve_document(_documents_from_chunks(chunks), args["document"])
        # 重排前复制候选，避免给共享 runtime 写入分数。
        candidates = [
            {**chunk, "metadata": dict(chunk["metadata"])}
            for chunk in chunks
            if (chunk.get("metadata") or {}).get("source_id") == document.source_id
        ]
        final_chunks = rerank(candidates, args.get("query"))[:args["top_k"]]
        return tool_success("search_document", args, {
            "query": args["query"], "document": args["document"],
            "source_id": document.source_id, "count": len(final_chunks),
            "evidence": [chunk_to_evidence_document(chunk) for chunk in final_chunks],
        })
    except ToolExecutionError:
        raise
    except Exception as exc:
        return tool_error("search_document", args, str(exc))
