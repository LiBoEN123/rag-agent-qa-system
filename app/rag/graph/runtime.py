"""LangGraph RAG 子图同步运行入口。"""

from typing import Any

from pydantic import ValidationError

from app.config import get_settings
from app.rag.graph.builder import get_compiled_rag_graph
from app.rag.results import build_default_timings
from app.rag.schemas import RAGResult


class RAGGraphError(RuntimeError):
    """RAG 子图未能生成合法业务结果时抛出的统一编排错误。"""


def _validate_rag_result(result: dict[str, Any]) -> RAGResult | None:
    """校验结果且不把 Pydantic 错误或原始证据泄漏到编排边界。"""
    try:
        return RAGResult.model_validate(result)
    except ValidationError:
        return None


def run_rag_subgraph(
    question: str,
    top_k: int | None = None,
    decompose_top_k: int | None = None,
    retrieved_chunks: list[dict[str, Any]] | None = None,
    trace_id: str = "",
) -> dict[str, Any]:
    """同步执行 RAG 子图，并返回统一的 RAG 结果字典。"""
    settings = get_settings()
    graph_input = {
        "question": question,
        "top_k": settings.top_k if top_k is None else top_k,
        "decompose_top_k": (
            settings.decompose_top_k
            if decompose_top_k is None
            else decompose_top_k
        ),
        "retrieved_chunks": retrieved_chunks,
        "trace_id": trace_id,
        "timings": build_default_timings(),
        "decompose_llm_used": False,
        "answer_llm_used": False,
    }
    output = get_compiled_rag_graph().invoke(graph_input)
    result = output.get("result") if isinstance(output, dict) else None
    if not isinstance(result, dict):
        raise RAGGraphError("RAG subgraph did not produce a valid result")

    validated = _validate_rag_result(result)
    if validated is None:
        raise RAGGraphError("RAG subgraph produced an invalid result")
    return validated.model_dump(mode="json", exclude_none=True)


__all__ = ["RAGGraphError", "run_rag_subgraph"]
