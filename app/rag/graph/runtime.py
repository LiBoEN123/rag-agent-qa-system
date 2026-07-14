"""LangGraph RAG 子图同步运行入口。"""

from typing import Any

from app.config import Settings
from app.rag.graph.builder import get_compiled_rag_graph
from app.rag.results import build_default_timings


REQUIRED_RAG_RESULT_FIELDS = {
    "status",
    "answer",
    "references",
    "final_chunks",
    "retrieval_confidence",
    "answer_confidence",
    "need_clarification",
    "clarification_question",
    "reason",
    "timings",
    "decompose_llm_used",
    "answer_llm_used",
    "llm_call_count",
}


class RAGGraphError(RuntimeError):
    """RAG 子图未能生成合法业务结果时抛出的统一编排错误。"""


def run_rag_subgraph(
    question: str,
    top_k: int = Settings.TOP_K,
    decompose_top_k: int = Settings.DECOMPOSE_TOP_K,
    retrieved_chunks: list[dict[str, Any]] | None = None,
    trace_id: str = "",
) -> dict[str, Any]:
    """同步执行 RAG 子图，只返回与旧 ``rag.ask()`` 一致的结果字典。"""
    graph_input = {
        "question": question,
        "top_k": top_k,
        "decompose_top_k": decompose_top_k,
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

    missing_fields = REQUIRED_RAG_RESULT_FIELDS.difference(result)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise RAGGraphError(
            f"RAG subgraph result is missing required fields: {missing}"
        )
    return result


__all__ = ["RAGGraphError", "run_rag_subgraph"]
