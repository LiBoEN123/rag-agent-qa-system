"""LangGraph 父图业务节点与框架 Agent 薄适配。"""

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.answer import apply_rag_result_to_state, compose_final_answer
from app.agent.framework_agent import (
    apply_framework_events,
    framework_metrics,
)
from app.agent.guardrails import AgentStopReason
from app.agent.router import route_agent_action
from app.agent.runtime import (
    append_llm_usage_trace,
    build_decide_trace,
    build_v2_result,
    maybe_save_agent_trace,
    normalize_llm_call_count,
)
from app.agent.state import AgentState, get_effective_question
from app.config import Settings
from app.graph.state import AgentGraphState
from app.rag.graph import run_rag_subgraph


def _safe_decision(decision: object) -> dict[str, Any]:
    """将 route 节点写入的内部路由决策安全规整为字典。"""
    return decision if isinstance(decision, dict) else {}


def _require_agent_state(state: AgentGraphState) -> AgentState:
    """读取已初始化的业务状态，缺失时立即失败。"""
    agent_state = state.get("agent_state")
    if not isinstance(agent_state, AgentState):
        raise RuntimeError("LangGraph agent_state is not initialized")
    return agent_state


def _resolve_state_questions(state: AgentGraphState) -> tuple[str, str]:
    """解析并清理父图中的原始问题与有效问题。"""
    return (
        (state.get("original_question") or "").strip(),
        (state.get("effective_question") or "").strip(),
    )


def route_node(state: AgentGraphState) -> dict[str, Any]:
    """执行并计时唯一的确定性 Router，把结果写回父图。"""
    _, effective_question = _resolve_state_questions(state)
    started = time.perf_counter()
    decision = route_agent_action(effective_question)
    return {
        "decision": decision,
        "route_cost_ms": max(
            0,
            int((time.perf_counter() - started) * 1000),
        ),
    }


def initialize_agent_state_node(state: AgentGraphState) -> dict[str, Any]:
    """创建 API 兼容业务状态和框架消息入口。"""
    original_question, effective_question = _resolve_state_questions(state)
    decision = _safe_decision(state.get("decision"))
    agent_state = AgentState(
        trace_id=state.get("trace_id", ""),
        session_id=state.get("session_id"),
        original_question=original_question,
        rewritten_question=(
            effective_question if effective_question != original_question else ""
        ),
        top_k=state.get("top_k", Settings.TOP_K),
        decompose_top_k=state.get("decompose_top_k", Settings.DECOMPOSE_TOP_K),
    )
    agent_state.tool_trace.append(
        build_decide_trace(
            decision,
            route_cost_ms=state.get("route_cost_ms", 0),
        )
    )
    action = str(decision.get("action") or "").strip()
    if action == "clarify":
        clarification = (
            str(decision.get("clarification_question") or "").strip()
            or str(decision.get("reason") or "").strip()
            or "问题需要进一步澄清"
        )
        agent_state.status = "clarify"
        agent_state.reason = clarification
        agent_state.final_answer = clarification
        agent_state.tool_trace.append(
            {
                "step": "clarify",
                "step_id": 1,
                "status": "success",
                "reason": clarification,
            }
        )
    elif action == "direct_answer":
        agent_state.status = "direct_answer"
        agent_state.reason = str(
            decision.get("reason") or "该问题可以直接回答"
        )
        agent_state.tool_trace.append(
            {
                "step": "direct_answer",
                "step_id": 1,
                "status": "success",
                "reason": agent_state.reason,
            }
        )
    elif action != "agent":
        safe_action = action[:100] or "<missing>"
        agent_state.status = "error"
        agent_state.reason = f"Unsupported route action: {safe_action}"
    return {
        "original_question": original_question,
        "effective_question": effective_question,
        "messages": [HumanMessage(content=effective_question)],
        "governance_events": [],
        "framework_events_applied": 0,
        "iteration_count": 0,
        "tool_call_count": 0,
        "tool_error_count": 0,
        "agent_elapsed_ms": 0,
        "stop_reason": "",
        "pending_rag_result": {},
        "agent_state": agent_state,
    }


def _normalize_ai_content(content: object) -> str:
    """把模型支持的文本内容形式归一化为字符串。"""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content or "").strip()
    parts = [
        item if isinstance(item, str) else item.get("text", "")
        for item in content
        if isinstance(item, (str, dict))
    ]
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def _last_final_ai(state: AgentGraphState) -> AIMessage | None:
    """查找最近一条不再请求工具的最终模型消息。"""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message
    return None


def _apply_stop(agent_state: AgentState, reason: str) -> None:
    """把框架停止原因映射到现有 API 兼容状态。"""
    tool_errors = {
        AgentStopReason.FATAL_TOOL_ERROR.value,
        AgentStopReason.APPROVAL_REQUIRED.value,
        AgentStopReason.MISSING_IDEMPOTENCY_KEY.value,
        AgentStopReason.RUNTIME_UNAVAILABLE.value,
    }
    if reason in tool_errors:
        agent_state.status = "tool_error"
        agent_state.final_answer = "工具调用被治理策略或运行时终止，无法继续本次请求。"
    else:
        agent_state.status = "model_error"
        agent_state.final_answer = "本次请求达到 Agent 执行上限，已安全停止。请缩小问题范围后重试。"
    if reason == "agent_model_call_failed":
        agent_state.final_answer = "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
    agent_state.reason = reason


def adapt_framework_agent_node(state: AgentGraphState) -> dict[str, Any]:
    """把 create_agent 事件映射回现有 AgentState/API 兼容字段。"""
    agent_state = _require_agent_state(state)
    applied = apply_framework_events(state, agent_state)
    events = list(state.get("governance_events", []))
    metrics = framework_metrics(events)
    successful_models = sum(
        item.get("kind") == "model" and item.get("ok") for item in events
    )
    agent_state.answer_llm_used = successful_models > 0
    agent_state.llm_call_count = successful_models
    updates: dict[str, Any] = {
        "agent_state": agent_state,
        "framework_events_applied": applied,
        **metrics,
    }
    if state.get("pending_rag_result"):
        return updates
    reason = str(state.get("stop_reason") or "")
    if reason:
        _apply_stop(agent_state, reason)
        return updates
    message = _last_final_ai(state)
    answer = _normalize_ai_content(message.content if message else "")
    agent_state.final_answer = answer or "模型未生成可用的最终回答。"
    agent_state.references = []
    agent_state.status = "ok" if answer else "model_error"
    agent_state.reason = (
        "Agent model produced final answer"
        if answer
        else "Agent model returned empty final answer"
    )
    updates["stop_reason"] = AgentStopReason.COMPLETED.value
    return updates


def rag_subgraph_node(state: AgentGraphState) -> dict[str, Any]:
    """消费框架路径的知识库 observation，复用原 RAG 子图。"""
    agent_state = _require_agent_state(state)
    applied = apply_framework_events(state, agent_state)
    observation = next(
        (
            item
            for item in reversed(agent_state.observations)
            if item.tool_name == "search_knowledge_base" and item.success
        ),
        None,
    )
    if observation is None or not isinstance(observation.output, dict):
        raise RuntimeError("RAG path requires a successful knowledge base observation")
    chunks = observation.output.get("chunks")
    if not isinstance(chunks, list):
        raise RuntimeError("Knowledge base observation chunks must be a list")
    rag_result = run_rag_subgraph(
        question=get_effective_question(agent_state),
        top_k=agent_state.top_k,
        decompose_top_k=agent_state.decompose_top_k,
        retrieved_chunks=chunks,
        trace_id=agent_state.trace_id,
    )
    return {
        "agent_state": apply_rag_result_to_state(agent_state, rag_result),
        "framework_events_applied": applied,
    }


def compose_node(state: AgentGraphState) -> dict[str, AgentState]:
    """调用现有 Answer Composer。"""
    return {"agent_state": compose_final_answer(_require_agent_state(state))}


def finalize_node(state: AgentGraphState) -> dict[str, Any]:
    """复用 API 映射、统计与 trace 持久化。"""
    agent_state = _require_agent_state(state)
    decision = _safe_decision(state.get("decision"))
    normalize_llm_call_count(agent_state)
    append_llm_usage_trace(agent_state)
    result = build_v2_result(agent_state, decision)
    maybe_save_agent_trace(
        trace_id=agent_state.trace_id,
        session_id=agent_state.session_id,
        question=agent_state.original_question,
        rewritten_question=agent_state.rewritten_question,
        final_answer=result.get("answer", ""),
        status=result.get("status", ""),
        reason=result.get("reason", ""),
        tool_trace=result.get("tool_trace", []),
    )
    return {"agent_state": agent_state, "result": result}


__all__ = [
    "adapt_framework_agent_node",
    "compose_node",
    "finalize_node",
    "initialize_agent_state_node",
    "rag_subgraph_node",
    "route_node",
]
