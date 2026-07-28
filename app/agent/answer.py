"""Agent 答案合成模块。

映射 RAG 子图结果，或根据普通工具观察结果生成最终回答状态。
"""

from typing import Any

from app.agent.state import AgentObservation, AgentState


ERROR_STATUSES = {"tool_error", "error", "model_error"}

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


def find_last_successful_observation(state: AgentState) -> AgentObservation | None:
    """从状态中查找最后一条成功的工具观察结果。"""
    for observation in reversed(state.observations):
        if observation.success:
            return observation
    return None


def _format_tool_answer(tool_name: str, output: dict) -> str:
    """调用现有工具答案格式化函数，局部导入用于避免循环依赖。"""
    from app.agent.formatting import format_tool_answer

    return format_tool_answer(tool_name, output)


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


def compose_tool_answer(state: AgentState, observation: AgentObservation) -> AgentState:
    """将成功的普通工具观察结果格式化为最终答案并写回状态。"""
    try:
        answer = _format_tool_answer(observation.tool_name, observation.output)
    except Exception as exc:
        state.status = "error"
        state.reason = f"Failed to format tool answer: {exc}"
        state.final_answer = "工具结果格式化失败，无法生成可靠回答。"
        state.tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "tool",
                "status": state.status,
                "reason": state.reason,
            }
        )
        return state

    state.final_answer = answer
    state.references = []
    state.status = "ok"
    state.reason = "Tool result formatted as final answer"
    state.tool_trace.append(
        {
            "step": "compose_final_answer",
            "mode": "tool",
            "tool_name": observation.tool_name,
            "status": state.status,
            "answer_len": len(state.final_answer or ""),
            "reason": state.reason,
        }
    )
    return state


def compose_final_answer(state: AgentState) -> AgentState:
    """根据已完成的工具观察结果生成 Agent 最终回答。"""
    if state.status == "clarify":
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
        if not state.final_answer:
            if state.reason:
                state.final_answer = f"工具执行失败，无法生成可靠回答。原因：{state.reason}"
            else:
                state.final_answer = "工具执行失败，无法生成可靠回答。"
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

    if not state.observations:
        state.status = "error"
        state.reason = "No observations available for final answer"
        state.final_answer = "当前没有可用于生成回答的工具结果。"
        state.tool_trace.append(
            {
                "step": "compose_final_answer",
                "status": "error",
                "reason": state.reason,
            }
        )
        return state

    successful_rag_observation = any(
        observation.tool_name == "search_knowledge_base"
        and observation.success
        for observation in state.observations
    )
    if successful_rag_observation:
        raise RuntimeError(
            "Successful knowledge-base observations must be handled "
            "by the LangGraph RAG subgraph."
        )

    observation = find_last_successful_observation(state)
    if observation is None:
        state.status = "tool_error"
        state.reason = state.reason or "No successful tool observation"
        state.final_answer = "工具没有返回可用结果，无法生成可靠回答。"
        state.tool_trace.append(
            {
                "step": "compose_final_answer",
                "status": state.status,
                "reason": state.reason,
            }
        )
        return state

    return compose_tool_answer(state, observation)


__all__ = [
    "apply_rag_result_to_state",
    "compose_final_answer",
    "compose_tool_answer",
    "find_last_successful_observation",
    "map_rag_status",
]
