"""Agent Runtime 结果构造、统计和追踪存储辅助函数。"""

from typing import Any

from pydantic import TypeAdapter, ValidationError

from app.agent.guardrails import (
    AgentStopReason,
    build_agent_tool_error_fingerprint,
    classify_tool_failure,
    stop_reason_for_tool_failure,
)
from app.common.logging import get_logger
from app.config import Settings
from app.storage.traces import save_agent_trace


logger = get_logger("chat_service")

_INTEGER_ADAPTER = TypeAdapter(int)


def sum_governance_event_costs(events: list[dict[str, Any]]) -> int:
    """汇总模型和工具治理事件记录的非负耗时。"""
    return sum(max(0, int(item.get("cost_ms", 0) or 0)) for item in events)


def count_consecutive_events(
    events: list[dict[str, Any]],
    *,
    key: str,
    value: str,
) -> int:
    """统计已计数工具事件尾部连续出现的指定字段值。"""
    count = 0
    for event in reversed(events):
        if event.get("kind") != "tool" or not event.get("counted_call"):
            continue
        if str(event.get(key) or "") != value:
            break
        count += 1
    return count


def build_tool_governance_event(
    *,
    tool_call: dict[str, Any],
    result: dict[str, Any],
    call_fingerprint: str,
    prior_events: list[dict[str, Any]],
    cost_ms: int,
    known_tool: bool,
) -> dict[str, Any]:
    """根据标准工具结果计算错误分类、预算和最小治理事件。"""
    tool_args = tool_call.get("args") or {}
    ok = result.get("ok") is True
    error_code = str(result.get("error_code") or "")
    failure_kind = ""
    stop_reason = ""
    error_fingerprint = ""

    if not ok:
        failure = classify_tool_failure(
            error_code=error_code, retryable=result.get("retryable", False)
        )
        failure_kind = failure.value
        error_fingerprint = build_agent_tool_error_fingerprint(
            call_fingerprint=call_fingerprint, error_code=error_code
        )
        immediate_stop = stop_reason_for_tool_failure(
            error_code=error_code, failure_kind=failure
        )
        error_count = (
            sum(
                event.get("kind") == "tool"
                and event.get("counted_call")
                and not event.get("ok")
                for event in prior_events
            )
            + 1
        )
        repeated_errors = (
            count_consecutive_events(
                prior_events, key="error_fingerprint", value=error_fingerprint
            )
            + 1
        )
        if immediate_stop:
            stop_reason = immediate_stop.value
        elif error_count >= Settings.AGENT_MAX_TOOL_ERRORS:
            stop_reason = AgentStopReason.MAX_TOOL_ERRORS.value
        elif repeated_errors >= Settings.AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS:
            stop_reason = AgentStopReason.REPEATED_TOOL_ERROR.value

    elapsed_ms = sum_governance_event_costs(prior_events) + max(0, cost_ms)
    if not stop_reason and elapsed_ms >= Settings.AGENT_MAX_TOTAL_COST_MS:
        stop_reason = AgentStopReason.MAX_TOTAL_COST.value

    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    error_source = "tool_runtime"
    if not runtime:
        error_source = (
            "structured_tool_validation" if known_tool else "structured_tool_adapter"
        )

    return {
        "kind": "tool",
        "name": str(tool_call.get("name") or ""),
        "args": tool_args,
        "result": result,
        "ok": ok,
        "counted_call": True,
        "call_fingerprint": call_fingerprint,
        "error_fingerprint": error_fingerprint,
        "error_code": error_code,
        "failure_kind": failure_kind,
        "error_source": error_source,
        "handler_executed": int(runtime.get("attempt_count", 0) or 0) > 0,
        "stop_reason": stop_reason,
        "cost_ms": max(0, int(cost_ms)),
        "tool_call_id": str(tool_call.get("id") or ""),
    }


def get_route_decision_for_trace(decision: dict) -> str:
    """根据路由 action 生成 trace 使用的 route_decision。"""
    action = decision.get("action", "")

    if action in {"clarify", "direct_answer"}:
        return action

    if action:
        return action

    return "unknown"


def build_decide_trace(
    decision: dict,
    *,
    route_cost_ms: int = 0,
) -> dict[str, Any]:
    """将最小内部路由协议扩展为稳定的外部兼容 trace。"""
    return {
        "step": "decide",
        "route_decision": get_route_decision_for_trace(decision),
        "decision_type": decision.get("decision", ""),
        "action": decision.get("action", ""),
        "tool_name": "",
        "tool_args": {},
        "sanitized_tool_args": {},
        "reason": decision.get("reason", ""),
        "router_source": decision.get("router_source", ""),
        "router_confidence": decision.get("router_confidence"),
        "router_fallback_reason": "",
        "router_llm_used": False,
        "route_cost_ms": max(0, int(route_cost_ms or 0)),
    }


def resolve_llm_call_count(
    *,
    decompose_llm_used: bool,
    answer_llm_used: bool,
    current_count: int,
) -> int:
    """解析 Agent Model 与 RAG 回答侧已经累计的精确调用数。"""
    return max(
        int(bool(decompose_llm_used)) + int(bool(answer_llm_used)),
        max(0, int(current_count or 0)),
    )


def upsert_llm_usage_trace(
    tool_trace: list[dict[str, Any]],
    *,
    decompose_llm_used: bool,
    answer_llm_used: bool,
    llm_call_count: int,
) -> list[dict[str, Any]]:
    """返回包含唯一最新 LLM usage 项的新工具 trace。"""
    usage_trace = {
        "step": "llm_usage",
        "router_llm_used": False,
        "decompose_llm_used": bool(decompose_llm_used),
        "answer_llm_used": bool(answer_llm_used),
        "llm_call_count": max(0, int(llm_call_count or 0)),
    }
    updated: list[dict[str, Any]] = []
    usage_written = False
    for step in tool_trace:
        if not isinstance(step, dict):
            continue
        if step.get("step") == "llm_usage":
            if not usage_written:
                updated.append(usage_trace)
                usage_written = True
            continue
        updated.append(dict(step))
    if not usage_written:
        updated.append(usage_trace)
    return updated


def validate_runtime_counts(
    *,
    top_k: Any,
    decompose_top_k: Any,
) -> tuple[int, int]:
    """保留公开 Runtime 数字输入原有的边界约束。"""

    def validate(name: str, value: Any, lower: int, upper: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            normalized = _INTEGER_ADAPTER.validate_python(value)
        except ValidationError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if normalized < lower or normalized > upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
        return normalized

    return (
        validate("top_k", top_k, 1, 10),
        validate("decompose_top_k", decompose_top_k, 1, 20),
    )


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
