"""Agent Runtime 结果构造、统计和追踪存储辅助函数。"""

from typing import Any

from app.agent.state import AgentState
from app.config import Settings
from app.storage.traces import save_agent_trace
from app.common.logging import get_logger
from app.tools.audit import sanitize_tool_input
from app.tools.redaction import redact_sensitive_data
from app.tools.registry import tool_registry


logger = get_logger("chat_service")

_INTERNAL_ERROR_MESSAGE = "Agent 内部执行异常，请稍后重试"
_INTERNAL_ERROR_MARKER = "internal_error"


def _safe_trace_tool_args(tool_name: str, value: object) -> object:
    """按工具契约脱敏并截断 Router/Planner trace 中的参数。"""
    args = value if isinstance(value, dict) else {}
    spec = tool_registry.get(tool_name)
    if spec is None:
        return redact_sensitive_data(args)
    return sanitize_tool_input(spec, args)


def _safe_decision(decision: dict | None) -> dict:
    """把外部传入的路由决策规整为可安全读取的字典。"""
    return decision if isinstance(decision, dict) else {}


def get_route_decision_for_trace(decision: dict) -> str:
    """根据路由 action 生成 trace 使用的 route_decision。"""
    action = decision.get("action", "")

    if action == "search_knowledge_base":
        return "rag"

    if action in {"clarify", "direct_answer"}:
        return action

    if action:
        return action

    return "unknown"


def map_agent_status_to_public_status(status: str) -> str:
    """将 Agent 内部状态映射为 API 使用的大写业务状态。"""
    mapping = {
        "ok": "OK",
        "clarify": "NEED_CLARIFICATION",
        "direct_answer": "OK",
        "tool_error": "TOOL_ERROR",
        "model_error": "MODEL_ERROR",
        "no_evidence": "NO_RELEVANT_INFO",
        "error": "INTERNAL_ERROR",
        "running": "INTERNAL_ERROR",
    }
    return mapping.get(str(status or ""), "INTERNAL_ERROR")


def summarize_observations_for_result(state: AgentState) -> str:
    """从 observations 中提取 API 响应使用的工具结果摘要。"""
    if state.status == "clarify":
        return "未调用工具，问题需要澄清"

    if state.status == "direct_answer":
        return "未调用工具，直接回答"

    if not state.observations:
        return state.reason or ""

    summaries = []
    for observation in state.observations:
        if observation.tool_name == "search_knowledge_base":
            output = observation.output if isinstance(observation.output, dict) else {}
            chunks = output.get("chunks", [])
            count = output.get("count", len(chunks) if isinstance(chunks, list) else 0)
            summaries.append(f"知识库检索完成，召回 {count} 个 chunk")
            continue

        if observation.summary:
            summaries.append(observation.summary)

    return "；".join(summaries[-3:]) if summaries else state.reason or ""


def build_decide_trace(decision: dict, router_llm_used: bool) -> dict[str, Any]:
    """构造 Agent 开始阶段的路由决策 trace。"""
    tool_name = str(decision.get("tool_name") or "")
    sanitized_args = _safe_trace_tool_args(
        tool_name,
        decision.get("tool_args", {}),
    )
    return {
        "step": "decide",
        "route_decision": get_route_decision_for_trace(decision),
        "decision_type": decision.get("decision", ""),
        "action": decision.get("action", ""),
        "tool_name": tool_name,
        "tool_args": sanitized_args,
        "sanitized_tool_args": sanitized_args,
        "reason": decision.get("reason", ""),
        "router_source": decision.get("router_source", ""),
        "router_confidence": decision.get("router_confidence"),
        "router_fallback_reason": decision.get("router_fallback_reason", ""),
        "router_llm_used": bool(router_llm_used),
    }


def build_plan_trace(state: AgentState) -> dict[str, Any]:
    """构造 Planner 产物 trace，便于观察计划步骤。"""
    return {
        "step": "plan",
        "status": "success",
        "plan_size": len(state.plan),
        "steps": [
            {
                "step_id": step.step_id,
                "action": step.action,
                "tool_name": step.tool_name,
                "tool_args": _safe_trace_tool_args(
                    step.tool_name,
                    step.tool_args,
                ),
                "sanitized_tool_args": _safe_trace_tool_args(
                    step.tool_name,
                    step.tool_args,
                ),
                "reason": step.reason,
                "status": step.status,
            }
            for step in state.plan
        ],
    }


def normalize_llm_call_count(state: AgentState) -> None:
    """补齐 Router LLM 计数，并保留 answer 侧可能已经累计的更精确计数。"""
    answer_side_count = max(
        int(bool(state.decompose_llm_used)) + int(bool(state.answer_llm_used)),
        int(state.llm_call_count or 0),
    )
    state.llm_call_count = int(bool(state.router_llm_used)) + answer_side_count


def append_llm_usage_trace(state: AgentState) -> None:
    """把本次 Runtime 的 LLM 使用情况写入工具 trace。"""
    usage_trace = {
        "step": "llm_usage",
        "router_llm_used": bool(state.router_llm_used),
        "decompose_llm_used": bool(state.decompose_llm_used),
        "answer_llm_used": bool(state.answer_llm_used),
        "llm_call_count": int(state.llm_call_count or 0),
    }

    for index, step in enumerate(state.tool_trace):
        if step.get("step") == "llm_usage":
            state.tool_trace[index] = usage_trace
            return

    state.tool_trace.append(usage_trace)


def build_v2_result(state: AgentState, decision: dict) -> dict:
    """将 AgentState 转换成 API 主流程使用的结果字典。"""
    public_status = map_agent_status_to_public_status(state.status)
    need_clarification = state.status == "clarify"
    rag_trace = next(
        (
            step
            for step in reversed(state.tool_trace or [])
            if step.get("step") == "compose_final_answer"
            and step.get("mode") == "rag"
        ),
        {},
    )
    timings = rag_trace.get("timings")

    return {
        "status": public_status,
        "answer": state.final_answer or "",
        "references": state.references or [],
        "retrieval_confidence": rag_trace.get("retrieval_confidence", 0.0),
        "answer_confidence": rag_trace.get("answer_confidence", 0.0),
        "need_clarification": need_clarification,
        "clarification_question": state.final_answer if need_clarification else "",
        "reason": state.reason or "",
        "action": decision.get("action", ""),
        "tool_result_summary": summarize_observations_for_result(state),
        "tool_trace": state.tool_trace or [],
        "router_llm_used": bool(state.router_llm_used),
        "decompose_llm_used": bool(state.decompose_llm_used),
        "answer_llm_used": bool(state.answer_llm_used),
        "llm_call_count": int(state.llm_call_count or 0),
        "timings": dict(timings) if isinstance(timings, dict) else {},
    }


def build_internal_error_result(
    exc: Exception,
    decision: dict | None = None,
    *,
    runtime_label: str = "Agent Runtime",
    trace_step: str = "agent_runtime_error",
) -> dict:
    """构造 Agent Runtime 顶层异常时的脱敏兼容错误结果。

    ``exc`` 仅供调用方写入内部日志和 bad case；其内容不得进入返回结果。
    """
    decision = _safe_decision(decision)
    return {
        "status": "INTERNAL_ERROR",
        "answer": _INTERNAL_ERROR_MESSAGE,
        "references": [],
        "retrieval_confidence": 0.0,
        "answer_confidence": 0.0,
        "need_clarification": False,
        "clarification_question": "",
        "reason": _INTERNAL_ERROR_MESSAGE,
        "action": decision.get("action", ""),
        "tool_result_summary": "",
        "tool_trace": [
            {
                "step": trace_step,
                "status": "error",
                "error": _INTERNAL_ERROR_MARKER,
            }
        ],
        "router_llm_used": False,
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
    }


def _default_bad_case(trace_id: str, question: str) -> dict:
    """构造 Agent Runtime 使用的 bad case 默认结构。"""
    return {
        "trace_id": trace_id,
        "question": question,
        "answer": "",
        "references": [],
        "route": "",
        "error_type": "",
        "note": "",
    }


def _write_internal_bad_case(
    write_bad_cases,
    bad_case: dict | None,
    trace_id: str,
    question: str,
    exc: Exception,
    *,
    runtime_route: str = "agent_runtime",
    runtime_label: str = "Agent Runtime",
) -> None:
    """Runtime 顶层异常时写入 bad case，写入失败不继续抛错。"""
    if not callable(write_bad_cases):
        return

    record = dict(bad_case or _default_bad_case(trace_id, question))
    record["answer"] = f"系统内部执行 {runtime_label} 时发生错误，无法生成可靠回答。"
    record["references"] = []
    record["route"] = runtime_route
    record["error_type"] = "internal_error"
    record["note"] = str(exc)

    try:
        write_bad_cases(record)
    except Exception:
        return


def maybe_save_agent_trace(
    *,
    trace_id: str,
    session_id: str | None,
    question: str,
    rewritten_question: str = "",
    final_answer: str = "",
    status: str = "",
    reason: str = "",
    tool_trace: list[dict] | None = None,
    total_cost_ms: int = 0,
) -> None:
    """按配置保存 Agent trace，持久化失败不影响主流程。"""
    if not Settings.ENABLE_AGENT_TRACE_STORE:
        return

    try:
        save_agent_trace(
            trace_id=trace_id,
            session_id=session_id,
            question=question,
            rewritten_question=rewritten_question,
            final_answer=final_answer,
            status=status,
            reason=reason,
            tool_trace=tool_trace or [],
            total_cost_ms=total_cost_ms,
        )
    except Exception as exc:
        logger.warning("保存 Agent trace 失败，忽略持久化错误: %s", exc)


__all__ = [
    "append_llm_usage_trace",
    "build_decide_trace",
    "build_internal_error_result",
    "build_plan_trace",
    "build_v2_result",
    "get_route_decision_for_trace",
    "map_agent_status_to_public_status",
    "maybe_save_agent_trace",
    "normalize_llm_call_count",
    "summarize_observations_for_result",
]
