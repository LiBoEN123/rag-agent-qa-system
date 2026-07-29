"""LangGraph 父图业务节点与框架 Agent 薄适配。"""

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.answer import (
    build_compose_answer_updates,
    build_rag_answer_updates,
)
from app.agent.framework_agent import (
    framework_metrics,
    project_framework_events,
)
from app.agent.guardrails import AgentStopReason
from app.agent.router import route_agent_action
from app.agent.results import build_agent_result
from app.agent.runtime import (
    build_decide_trace,
    maybe_save_agent_trace,
    resolve_llm_call_count,
    upsert_llm_usage_trace,
    validate_runtime_counts,
)
from app.config import get_settings
from app.graph.state import AgentGraphState
from app.rag.graph import run_rag_subgraph


def _safe_decision(decision: object) -> dict[str, Any]:
    """将 route 节点写入的内部路由决策安全规整为字典。"""
    return decision if isinstance(decision, dict) else {}


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


def initialize_node(state: AgentGraphState) -> dict[str, Any]:
    """初始化唯一的平铺父图状态和框架消息入口。"""
    settings = get_settings()
    original_question, effective_question = _resolve_state_questions(state)
    decision = _safe_decision(state.get("decision"))
    top_k, decompose_top_k = validate_runtime_counts(
        top_k=state.get("top_k", settings.top_k),
        decompose_top_k=state.get(
            "decompose_top_k", settings.decompose_top_k
        ),
    )
    tool_trace = [
        build_decide_trace(
            decision,
            route_cost_ms=state.get("route_cost_ms", 0),
        )
    ]
    action = str(decision.get("action") or "").strip()
    status = "running"
    reason = ""
    final_answer = ""
    if action == "clarify":
        clarification = (
            str(decision.get("clarification_question") or "").strip()
            or str(decision.get("reason") or "").strip()
            or "问题需要进一步澄清"
        )
        status = "clarify"
        reason = clarification
        final_answer = clarification
        tool_trace.append(
            {
                "step": "clarify",
                "step_id": 1,
                "status": "success",
                "reason": clarification,
            }
        )
    elif action == "direct_answer":
        status = "direct_answer"
        reason = str(decision.get("reason") or "该问题可以直接回答")
        tool_trace.append(
            {
                "step": "direct_answer",
                "step_id": 1,
                "status": "success",
                "reason": reason,
            }
        )
    elif action != "agent":
        safe_action = action[:100] or "<missing>"
        status = "error"
        reason = f"Unsupported route action: {safe_action}"
    return {
        "original_question": original_question,
        "effective_question": effective_question,
        "top_k": top_k,
        "decompose_top_k": decompose_top_k,
        "messages": [HumanMessage(content=effective_question)],
        "governance_events": [],
        "framework_events_applied": 0,
        "iteration_count": 0,
        "tool_call_count": 0,
        "tool_error_count": 0,
        "last_tool_call_fingerprint": "",
        "consecutive_same_tool_call_count": 0,
        "last_tool_error_fingerprint": "",
        "consecutive_same_tool_error_count": 0,
        "agent_elapsed_ms": 0,
        "stop_reason": "",
        "pending_rag_result": {},
        "observations": [],
        "final_answer": final_answer,
        "references": [],
        "status": status,
        "reason": reason,
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
        "tool_trace": tool_trace,
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


def _build_stop_updates(reason: str) -> dict[str, str]:
    """把框架停止原因映射为 API 兼容状态更新。"""
    tool_errors = {
        AgentStopReason.FATAL_TOOL_ERROR.value,
        AgentStopReason.APPROVAL_REQUIRED.value,
        AgentStopReason.MISSING_IDEMPOTENCY_KEY.value,
        AgentStopReason.RUNTIME_UNAVAILABLE.value,
    }
    if reason in tool_errors:
        status = "tool_error"
        final_answer = "工具调用被治理策略或运行时终止，无法继续本次请求。"
    else:
        status = "model_error"
        final_answer = "本次请求达到 Agent 执行上限，已安全停止。请缩小问题范围后重试。"
    if reason == "agent_model_call_failed":
        final_answer = "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
    return {
        "status": status,
        "final_answer": final_answer,
        "reason": reason,
    }


def adapt_framework_agent_node(state: AgentGraphState) -> dict[str, Any]:
    """把 create_agent 事件映射为平铺状态的显式更新。"""
    projection = project_framework_events(state)
    events = list(state.get("governance_events", []))
    metrics = framework_metrics(events)
    successful_models = sum(
        item.get("kind") == "model" and item.get("ok") for item in events
    )
    updates: dict[str, Any] = {
        **projection,
        **metrics,
        "answer_llm_used": successful_models > 0,
        "llm_call_count": successful_models,
    }
    if state.get("pending_rag_result"):
        return updates
    reason = str(state.get("stop_reason") or "")
    if reason:
        updates.update(_build_stop_updates(reason))
        return updates
    message = _last_final_ai(state)
    answer = _normalize_ai_content(message.content if message else "")
    updates["final_answer"] = answer or "模型未生成可用的最终回答。"
    updates["references"] = []
    updates["status"] = "ok" if answer else "model_error"
    updates["reason"] = (
        "Agent model produced final answer"
        if answer
        else "Agent model returned empty final answer"
    )
    updates["stop_reason"] = AgentStopReason.COMPLETED.value
    return updates


def rag_subgraph_node(state: AgentGraphState) -> dict[str, Any]:
    """消费框架路径的知识库 observation，复用原 RAG 子图。"""
    settings = get_settings()
    projection = project_framework_events(state)
    observation = next(
        (
            item
            for item in reversed(projection["observations"])
            if item.get("tool_name") == "search_knowledge_base"
            and item.get("success")
        ),
        None,
    )
    if observation is None or not isinstance(observation.get("output"), dict):
        raise RuntimeError("RAG path requires a successful knowledge base observation")
    chunks = observation["output"].get("chunks")
    if not isinstance(chunks, list):
        raise RuntimeError("Knowledge base observation chunks must be a list")
    rag_result = run_rag_subgraph(
        question=str(state.get("effective_question") or "").strip(),
        top_k=int(state.get("top_k", settings.top_k)),
        decompose_top_k=int(
            state.get("decompose_top_k", settings.decompose_top_k)
        ),
        retrieved_chunks=chunks,
        trace_id=str(state.get("trace_id") or ""),
    )
    return {
        **projection,
        **build_rag_answer_updates({**state, **projection}, rag_result),
    }


def compose_node(state: AgentGraphState) -> dict[str, Any]:
    """返回现有 Answer Composer 的平铺状态更新。"""
    return build_compose_answer_updates(state)


def finalize_node(state: AgentGraphState) -> dict[str, Any]:
    """生成一致的 LLM 统计、trace、API 结果与持久化快照。"""
    llm_call_count = resolve_llm_call_count(
        decompose_llm_used=bool(state.get("decompose_llm_used")),
        answer_llm_used=bool(state.get("answer_llm_used")),
        current_count=int(state.get("llm_call_count", 0) or 0),
    )
    tool_trace = upsert_llm_usage_trace(
        list(state.get("tool_trace", [])),
        decompose_llm_used=bool(state.get("decompose_llm_used")),
        answer_llm_used=bool(state.get("answer_llm_used")),
        llm_call_count=llm_call_count,
    )
    snapshot = {
        **state,
        "llm_call_count": llm_call_count,
        "tool_trace": tool_trace,
    }
    result = build_agent_result(snapshot).model_dump(mode="json")
    original_question = str(state.get("original_question") or "")
    effective_question = str(state.get("effective_question") or "")
    maybe_save_agent_trace(
        trace_id=str(state.get("trace_id") or ""),
        session_id=state.get("session_id"),
        question=original_question,
        rewritten_question=(
            effective_question
            if effective_question != original_question
            else ""
        ),
        final_answer=result.get("answer", ""),
        status=result.get("status", ""),
        reason=result.get("reason", ""),
        tool_trace=result.get("tool_trace", []),
    )
    return {
        "llm_call_count": llm_call_count,
        "tool_trace": tool_trace,
        "result": result,
    }


__all__ = [
    "adapt_framework_agent_node",
    "compose_node",
    "finalize_node",
    "initialize_node",
    "rag_subgraph_node",
    "route_node",
]
