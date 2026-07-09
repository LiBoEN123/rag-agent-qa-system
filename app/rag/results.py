"""RAG 结果构造模块。

统一问答返回字段，并生成记录候选证据强度的门控追踪信息。
"""

from typing import Any

from app.rag.retrieval import safe_float


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
    references: list | None = None,
    final_chunks: list | None = None,
    retrieval_confidence: float = 0.0,
    answer_confidence: float = 0.0,
    need_clarification: bool = False,
    clarification_question: str = "",
    reason: str = "",
    timings: dict | None = None,
    decompose_llm_used: bool = False,
    answer_llm_used: bool = False,
    **extra,
) -> dict:
    """构造字段完整且状态统一的 RAG 问答结果。"""
    result = {
        "status": status,
        "answer": answer,
        "references": references or [],
        "final_chunks": final_chunks or [],
        "retrieval_confidence": retrieval_confidence,
        "answer_confidence": answer_confidence,
        "need_clarification": need_clarification,
        "clarification_question": clarification_question,
        "reason": reason,
        "timings": timings or {},
        "decompose_llm_used": bool(decompose_llm_used),
        "answer_llm_used": bool(answer_llm_used),
        "llm_call_count": int(bool(decompose_llm_used)) + int(bool(answer_llm_used)),
    }
    result.update(extra)
    return result
