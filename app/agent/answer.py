"""Agent 答案合成与 RAG 结果映射。"""

from typing import Any

from app.agent.state import AgentState


ERROR_STATUSES = {"tool_error", "error", "model_error"}
INTERNAL_ERROR_TEXT = "Agent 内部执行异常，请稍后重试"

DIRECT_ANSWER_TEXT = (
    "我是一个 RAG 知识库问答助手。"
    "我可以基于已导入的文档回答问题并提供引用，"
    "也可以列出知识库文档、查看指定 chunk、对比文档内容，以及进行简单四则运算。"
)


def map_rag_status(status: str) -> str:
    """将现有 RAG 状态映射为 Agent 的统一状态。"""
    normalized = str(status or "").strip()
    mapping = {
        "OK": "ok",
        "NEED_CLARIFICATION": "clarify",
        "NO_RELEVANT_INFO": "no_evidence",
        "MODEL_ERROR": "model_error",
        "MODEL_BAD_OUTPUT": "model_error",
        "ok": "ok",
        "clarify": "clarify",
        "no_evidence": "no_evidence",
        "model_error": "model_error",
    }
    return mapping.get(normalized, "error")


def apply_rag_result_to_state(
    state: AgentState,
    rag_result: dict[str, Any],
) -> AgentState:
    """将 RAG 子图的统一结果映射回现有 AgentState。"""
    state.status = map_rag_status(rag_result.get("status", ""))
    state.final_answer = rag_result.get("answer", "") or ""
    state.references = rag_result.get("references", []) or []
    state.reason = rag_result.get("reason", "") or ""
    state.decompose_llm_used = bool(
        rag_result.get("decompose_llm_used", False)
    )
    state.answer_llm_used = bool(rag_result.get("answer_llm_used", False))
    rag_llm_call_count = int(rag_result.get("llm_call_count", 0) or 0)
    state.llm_call_count += rag_llm_call_count

    timings = rag_result.get("timings")
    state.tool_trace.append(
        {
            "step": "compose_final_answer",
            "mode": "rag",
            "status": state.status,
            "answer_len": len(state.final_answer or ""),
            "reference_count": len(state.references or []),
            "reason": state.reason,
            "retrieval_confidence": rag_result.get(
                "retrieval_confidence",
                0.0,
            ),
            "answer_confidence": rag_result.get("answer_confidence", 0.0),
            "timings": dict(timings) if isinstance(timings, dict) else {},
            "rag_llm_call_count": rag_llm_call_count,
        }
    )
    return state


def compose_final_answer(state: AgentState) -> AgentState:
    """完成确定性短路、框架回答或稳定错误结果。"""
    if state.status == "clarify":
        state.tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "clarify",
                "status": state.status,
                "answer_len": len(state.final_answer),
                "reason": state.reason,
            }
        )
        return state

    if state.status == "direct_answer":
        if not state.final_answer:
            state.final_answer = DIRECT_ANSWER_TEXT
        state.tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "direct",
                "status": state.status,
                "answer_len": len(state.final_answer),
                "reason": state.reason,
            }
        )
        return state

    if state.status in ERROR_STATUSES:
        if state.status == "error":
            state.final_answer = INTERNAL_ERROR_TEXT
        elif state.status == "model_error" and not state.final_answer:
            state.final_answer = "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
        elif not state.final_answer:
            state.final_answer = (
                f"工具执行失败，无法生成可靠回答。原因：{state.reason}"
                if state.reason
                else "工具执行失败，无法生成可靠回答。"
            )
        state.tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "error",
                "status": state.status,
                "answer_len": len(state.final_answer),
                "reason": state.reason,
            }
        )
        return state

    if state.status == "ok" and state.final_answer:
        state.tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "agent_loop",
                "status": state.status,
                "answer_len": len(state.final_answer),
                "reason": state.reason,
            }
        )
        return state

    previous_status = state.status
    state.status = "error"
    state.reason = f"Unexpected Agent state before compose: {previous_status}"
    state.final_answer = INTERNAL_ERROR_TEXT
    state.tool_trace.append(
        {
            "step": "compose_final_answer",
            "mode": "error",
            "status": state.status,
            "reason": state.reason,
        }
    )
    return state


__all__ = [
    "apply_rag_result_to_state",
    "compose_final_answer",
    "map_rag_status",
]
