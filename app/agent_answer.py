from typing import Any

from app.agent_state import AgentObservation, AgentState


ERROR_STATUSES = {"tool_error", "error", "model_error"}


def map_rag_status(status: str) -> str:
    """将现有 RAG 状态映射为 Agent v2 的统一状态。"""
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


def find_first_observation(state: AgentState, tool_name: str) -> AgentObservation | None:
    """从状态中查找指定工具产生的第一条观察结果。"""
    for observation in state.observations:
        if observation.tool_name == tool_name:
            return observation
    return None


def find_last_successful_observation(state: AgentState) -> AgentObservation | None:
    """从状态中查找最后一条成功的工具观察结果。"""
    for observation in reversed(state.observations):
        if observation.success:
            return observation
    return None


def _call_ask(
    *,
    question: str,
    top_k: int,
    decompose_top_k: int,
    retrieved_chunks: list,
    trace_id: str,
) -> dict[str, Any]:
    """调用现有 RAG ask 流程，局部导入用于避免循环依赖。"""
    from app.rag import ask

    return ask(
        question=question,
        top_k=top_k,
        decompose_top_k=decompose_top_k,
        retrieved_chunks=retrieved_chunks,
        trace_id=trace_id,
    )


def _format_tool_answer(tool_name: str, output: dict) -> str:
    """调用现有工具答案格式化函数，局部导入用于避免循环依赖。"""
    from app.agent import format_tool_answer

    return format_tool_answer(tool_name, output)


def compose_rag_answer(state: AgentState, observation: AgentObservation) -> AgentState:
    """基于知识库检索观察结果调用 RAG 流程并写回最终答案。"""
    retrieved_chunks = observation.output.get("chunks", []) if isinstance(observation.output, dict) else []
    question = state.rewritten_question or state.original_question

    try:
        rag_result = _call_ask(
            question=question,
            top_k=state.top_k,
            decompose_top_k=state.decompose_top_k,
            retrieved_chunks=retrieved_chunks,
            trace_id=state.trace_id,
        )
    except Exception as exc:
        state.status = "model_error"
        state.reason = f"RAG answer generation failed: {exc}"
        state.final_answer = "模型生成回答失败，无法生成可靠回答。"
        state.tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "rag",
                "status": state.status,
                "reason": state.reason,
            }
        )
        return state

    state.status = map_rag_status(rag_result.get("status", ""))
    state.final_answer = rag_result.get("answer", "") or ""
    state.references = rag_result.get("references", []) or []
    state.reason = rag_result.get("reason", "") or ""
    state.decompose_llm_used = bool(rag_result.get("decompose_llm_used", False))
    state.answer_llm_used = bool(rag_result.get("answer_llm_used", False))
    state.llm_call_count += int(rag_result.get("llm_call_count", 0) or 0)

    state.tool_trace.append(
        {
            "step": "compose_final_answer",
            "mode": "rag",
            "status": state.status,
            "answer_len": len(state.final_answer or ""),
            "reference_count": len(state.references or []),
            "reason": state.reason,
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
    """根据已完成的工具观察结果生成 Agent v2 最终回答。"""
    if state.status in {"clarify", "direct_answer"}:
        return state

    if state.status in ERROR_STATUSES:
        if not state.final_answer:
            if state.reason:
                state.final_answer = f"工具执行失败，无法生成可靠回答。原因：{state.reason}"
            else:
                state.final_answer = "工具执行失败，无法生成可靠回答。"
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

    rag_observation = find_first_observation(state, "search_knowledge_base")
    if rag_observation is not None and rag_observation.success:
        return compose_rag_answer(state, rag_observation)

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
    "compose_final_answer",
    "compose_rag_answer",
    "compose_tool_answer",
    "find_first_observation",
    "find_last_successful_observation",
    "map_rag_status",
]
