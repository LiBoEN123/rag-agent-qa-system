"""RAG 结果构造模块。

统一问答返回字段，并生成记录候选证据强度的门控追踪信息。
"""

from typing import Any

from app.rag.retrieval import safe_float
from app.rag.schemas import RAGResult


def build_default_timings() -> dict[str, int | bool | str]:
    """构造 RAG 子图使用的默认阶段耗时结构。"""
    return {
        "decompose_cost_ms": 0,
        "decompose_llm_used": False,
        "decompose_rule_reason": "",
        "decompose_skipped_reason": "",
        "rerank_cost_ms": 0,
        "retrieve_with_sub_queries_cost_ms": 0,
        "context_build_cost_ms": 0,
        "llm_cost_ms": 0,
        "parse_cost_ms": 0,
    }


def build_evidence_gate_step(
    final_chunks: list[dict[str, Any]],
    refused: bool,
    reason: str,
) -> dict:
    """汇总候选证据分数并构造证据门控追踪步骤。"""
    candidates = [
        item for item in (final_chunks or []) if isinstance(item, dict)
    ]
    return {
        "step": "evidence_gate",
        "refused": bool(refused),
        "max_vector_score": round(
            max(
                (safe_float(item.get("vector_score")) for item in candidates),
                default=0.0,
            ),
            4,
        ),
        "max_rerank_score": round(
            max(
                (safe_float(item.get("rerank_score")) for item in candidates),
                default=0.0,
            ),
            4,
        ),
        "max_hybrid_score": round(
            max(
                (safe_float(item.get("hybrid_score")) for item in candidates),
                default=0.0,
            ),
            4,
        ),
        "reason": reason,
    }


def build_rag_result(
    status: str,
    answer: str = "",
    references: list[dict[str, Any]] | None = None,
    final_chunks: list[dict[str, Any]] | None = None,
    retrieval_confidence: float = 0.0,
    answer_confidence: float = 0.0,
    need_clarification: bool = False,
    clarification_question: str = "",
    reason: str = "",
    timings: dict[str, Any] | None = None,
    decompose_llm_used: bool = False,
    answer_llm_used: bool = False,
    evidence_gate: dict[str, Any] | None = None,
    citation_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造字段完整且状态统一的 RAG 问答结果。"""
    result = RAGResult(
        status=status,
        answer=answer,
        references=references or [],
        final_chunks=final_chunks or [],
        retrieval_confidence=retrieval_confidence,
        answer_confidence=answer_confidence,
        need_clarification=need_clarification,
        clarification_question=clarification_question,
        reason=reason,
        timings=timings or {},
        decompose_llm_used=bool(decompose_llm_used),
        answer_llm_used=bool(answer_llm_used),
        llm_call_count=(
            int(bool(decompose_llm_used)) + int(bool(answer_llm_used))
        ),
        evidence_gate=evidence_gate,
        citation_verification=citation_verification,
    )
    return result.model_dump(mode="json", exclude_none=True)
