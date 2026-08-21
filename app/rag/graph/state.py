"""LangGraph RAG 子图状态定义。"""

from typing import Any, TypedDict


class RAGGraphState(TypedDict, total=False):
    """在 RAG 子图节点间传递轻量、可序列化的问答状态。"""

    question: str
    top_k: int
    decompose_top_k: int
    retrieved_chunks: list[dict[str, Any]] | None
    trace_id: str

    timings: dict[str, int | bool | str]
    decompose_result: dict[str, Any]
    decompose_llm_used: bool
    answer_llm_used: bool

    final_chunks: list[dict[str, Any]]
    retrieval_confidence: float
    evidence_gate: dict[str, Any]

    context: str
    answer_output: dict[str, Any]
    answer_output_invalid: bool
    parsed_output: dict[str, Any]

    final_references: list[dict[str, Any]]
    citation_verification: dict[str, Any]

    terminal_reason: str
    result: dict[str, Any]


__all__ = ["RAGGraphState"]
