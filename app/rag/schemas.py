"""RAG 顶层结果契约。"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class RAGResult(BaseModel):
    """一次 RAG 子图执行的唯一顶层结果契约。"""

    model_config = ConfigDict(extra="forbid")

    status: str
    answer: str
    references: list[dict[str, Any]]
    final_chunks: list[dict[str, Any]]
    retrieval_confidence: float
    answer_confidence: float
    need_clarification: bool
    clarification_question: str
    reason: str
    timings: dict[str, Any]
    decompose_llm_used: bool
    answer_llm_used: bool
    llm_call_count: int
    evidence_gate: dict[str, Any] | None = None
    citation_verification: dict[str, Any] | None = None


__all__ = ["RAGResult"]
