"""Agent 运行时编排模块。

串联路由、计划、执行、答案合成和追踪存储，生成 API 可消费的结果。
"""

from typing import Any

from app.agent.answer import compose_final_answer
from app.agent.executor import execute_plan
from app.agent.planner import build_plan_from_decision
from app.agent.state import AgentState
from app.config import Settings
from app.storage.traces import save_agent_trace
from app.common.logging import get_logger


logger = get_logger("chat_service")


def _route_agent_action(question: str, top_k: int, trace_id: str = "") -> dict:
    """局部导入 Router，避免 Runtime 与路由模块形成循环依赖。"""
    from app.agent import route_agent_action

    return route_agent_action(question, top_k, trace_id=trace_id)


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
    """将 Agent v2 内部状态映射为 API 使用的大写业务状态。"""
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
    """构造 v2 开始阶段的路由决策 trace。"""
    return {
        "step": "decide",
        "route_decision": get_route_decision_for_trace(decision),
        "decision_type": decision.get("decision", ""),
        "action": decision.get("action", ""),
        "tool_name": decision.get("tool_name", ""),
        "tool_args": decision.get("tool_args", {}),
        "reason": decision.get("reason", ""),
        "router_source": decision.get("router_source", ""),
        "router_confidence": decision.get("router_confidence"),
        "router_fallback_reason": decision.get("router_fallback_reason", ""),
        "router_llm_used": bool(router_llm_used),
    }


def build_plan_trace(state: AgentState) -> dict[str, Any]:
    """构造 v2 Planner 产物 trace，便于观察计划步骤。"""
    return {
        "step": "plan",
        "status": "success",
        "plan_size": len(state.plan),
        "steps": [
            {
                "step_id": step.step_id,
                "action": step.action,
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
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

    return {
        "status": public_status,
        "answer": state.final_answer or "",
        "references": state.references or [],
        "retrieval_confidence": 0.0,
        "answer_confidence": 0.0,
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
    }


def build_internal_error_result(
    exc: Exception,
    decision: dict | None = None,
    *,
    runtime_label: str = "Agent v2",
    trace_step: str = "agent_v2_error",
) -> dict:
    """构造 Agent Runtime 顶层异常时的兼容错误结果。"""
    decision = _safe_decision(decision)
    return {
        "status": "INTERNAL_ERROR",
        "answer": f"系统内部执行 {runtime_label} 时发生错误，无法生成可靠回答。",
        "references": [],
        "retrieval_confidence": 0.0,
        "answer_confidence": 0.0,
        "need_clarification": False,
        "clarification_question": "",
        "reason": str(exc),
        "action": decision.get("action", ""),
        "tool_result_summary": "",
        "tool_trace": [
            {
                "step": trace_step,
                "status": "error",
                "error": str(exc),
            }
        ],
        "router_llm_used": False,
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
    }


def _default_bad_case(trace_id: str, question: str) -> dict:
    """构造 Agent v2 使用的 bad case 默认结构。"""
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
    runtime_route: str = "agent_v2",
    runtime_label: str = "Agent v2",
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
        logger.warning("保存 Agent v2 trace 失败，忽略持久化错误: %s", exc)


def run_agent_v2(
    write_bad_cases,
    question: str,
    top_k: int,
    decompose_top_k: int,
    decision: dict | None = None,
    trace_id: str = "",
    session_id: str | None = None,
    bad_case: dict | None = None,
) -> dict:
    """串联 Router、Planner、Executor 和 Answer Composer，执行一次 Agent v2 流程。"""
    bad_case = bad_case or _default_bad_case(trace_id, question)
    decision_for_error = _safe_decision(decision)

    try:
        if decision is None:
            decision = _route_agent_action(question, top_k, trace_id=trace_id)

        decision = _safe_decision(decision)
        decision_for_error = decision

        router_llm_used = bool(
            decision.get(
                "router_llm_used",
                decision.get("router_source") in {"llm", "rule_based_fallback"},
            )
        )

        state = AgentState(
            trace_id=trace_id,
            session_id=session_id,
            original_question=question,
            rewritten_question="",
            top_k=top_k,
            decompose_top_k=decompose_top_k,
            max_steps=Settings.AGENT_MAX_STEPS,
            router_llm_used=router_llm_used,
        )

        state.tool_trace.append(build_decide_trace(decision, router_llm_used))

        state.plan = build_plan_from_decision(
            decision=decision,
            question=question,
            top_k=top_k,
        )
        state.tool_trace.append(build_plan_trace(state))

        state = execute_plan(state)
        state = compose_final_answer(state)

        normalize_llm_call_count(state)
        append_llm_usage_trace(state)

        result = build_v2_result(state, decision)
        maybe_save_agent_trace(
            trace_id=state.trace_id,
            session_id=state.session_id,
            question=state.original_question,
            rewritten_question=state.rewritten_question,
            final_answer=result.get("answer", ""),
            status=result.get("status", ""),
            reason=result.get("reason", ""),
            tool_trace=result.get("tool_trace", []),
        )
        return result

    except Exception as exc:
        _write_internal_bad_case(
            write_bad_cases=write_bad_cases,
            bad_case=bad_case,
            trace_id=trace_id,
            question=question,
            exc=exc,
        )
        result = build_internal_error_result(exc, decision_for_error)
        maybe_save_agent_trace(
            trace_id=trace_id,
            session_id=session_id,
            question=question,
            final_answer=result.get("answer", ""),
            status=result.get("status", ""),
            reason=result.get("reason", ""),
            tool_trace=result.get("tool_trace", []),
        )
        return result


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
    "run_agent_v2",
    "summarize_observations_for_result",
]
