"""Agent 工具治理规则、预算计算与稳定指纹。"""

import hashlib
import json
from enum import StrEnum
from typing import Any

from app.config import AppSettings, get_settings
from app.tools.errors import ToolErrorCode


class AgentToolFailureKind(StrEnum):
    """工具执行失败对 Agent 的稳定语义。"""

    RECOVERABLE = "recoverable"
    TRANSIENT_EXHAUSTED = "transient_exhausted"
    TERMINAL_GOVERNANCE = "terminal_governance"
    TERMINAL_RUNTIME = "terminal_runtime"


class AgentStopReason(StrEnum):
    """Agent 对外保持兼容的停止原因。"""

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations_reached"
    MAX_TOOL_CALLS = "max_tool_calls_reached"
    MAX_TOOL_ERRORS = "max_tool_errors_reached"
    MAX_TOTAL_COST = "max_total_cost_reached"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    REPEATED_TOOL_ERROR = "repeated_tool_error"
    FATAL_TOOL_ERROR = "fatal_tool_error"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"


_RECOVERABLE = (
    ToolErrorCode.UNKNOWN_TOOL.value,
    ToolErrorCode.INVALID_ARGUMENTS.value,
    ToolErrorCode.EXECUTION_ERROR.value,
    ToolErrorCode.INVALID_RESULT.value,
    ToolErrorCode.OUTPUT_VALIDATION_ERROR.value,
)

_TRANSIENT = (
    ToolErrorCode.TIMEOUT.value,
    ToolErrorCode.DEPENDENCY_UNAVAILABLE.value,
    ToolErrorCode.RATE_LIMITED.value,
)

_GOVERNANCE = (
    ToolErrorCode.PERMISSION_DENIED.value,
    ToolErrorCode.POLICY_DENIED.value,
    ToolErrorCode.AUDIT_WRITE_FAILED.value,
    ToolErrorCode.INVALID_INVOCATION_CONTEXT.value,
)


def classify_tool_failure(*, error_code: str) -> AgentToolFailureKind:
    """把 Runtime 错误映射为恢复或终止语义；未知错误 fail closed。"""
    code = str(error_code or "").strip().lower()
    if code in _RECOVERABLE:
        return AgentToolFailureKind.RECOVERABLE
    if code in _TRANSIENT:
        return AgentToolFailureKind.TRANSIENT_EXHAUSTED
    if code in _GOVERNANCE:
        return AgentToolFailureKind.TERMINAL_GOVERNANCE
    return AgentToolFailureKind.TERMINAL_RUNTIME


def stop_reason_for_tool_failure(
    *, error_code: str, failure_kind: AgentToolFailureKind
) -> AgentStopReason | None:
    """返回单次错误要求的立即停止原因。"""
    code = str(error_code or "").strip().lower()
    if code == ToolErrorCode.RUNTIME_UNAVAILABLE.value:
        return AgentStopReason.RUNTIME_UNAVAILABLE
    if failure_kind not in {
        AgentToolFailureKind.RECOVERABLE,
        AgentToolFailureKind.TRANSIENT_EXHAUSTED,
    }:
        return AgentStopReason.FATAL_TOOL_ERROR
    return None


def _digest(payload: object) -> str:
    """对有限 JSON 值生成稳定摘要，并明确处理不支持类型。"""
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        encoded = json.dumps(
            {"unsupported_type": type(payload).__name__}, sort_keys=True
        )
    return f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"


def build_unknown_tool_call_fingerprint(*, tool_name: str, tool_args: object) -> str:
    """为模型 JSON 中的未知工具生成有限、确定性的指纹。"""
    return _digest({"tool_name": str(tool_name or ""), "tool_args": tool_args})


def build_agent_tool_error_fingerprint(
    *, call_fingerprint: str, error_code: str
) -> str:
    """绑定调用指纹和稳定错误码。"""
    return _digest(
        {
            "call_fingerprint": call_fingerprint,
            "error_code": str(error_code or "").strip().lower(),
        }
    )


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
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """根据标准工具结果计算错误分类、预算和最小治理事件。"""
    resolved_settings = settings or get_settings()
    tool_args = tool_call.get("args") or {}
    ok = result.get("ok") is True
    error_code = str(result.get("error_code") or "")
    failure_kind = ""
    stop_reason = ""
    error_fingerprint = ""

    if not ok:
        failure = classify_tool_failure(
            error_code=error_code,
        )
        failure_kind = failure.value
        error_fingerprint = build_agent_tool_error_fingerprint(
            call_fingerprint=call_fingerprint,
            error_code=error_code,
        )
        immediate_stop = stop_reason_for_tool_failure(
            error_code=error_code,
            failure_kind=failure,
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
                prior_events,
                key="error_fingerprint",
                value=error_fingerprint,
            )
            + 1
        )
        if immediate_stop:
            stop_reason = immediate_stop.value
        elif error_count >= resolved_settings.agent_max_tool_errors:
            stop_reason = AgentStopReason.MAX_TOOL_ERRORS.value
        elif (
            repeated_errors
            >= resolved_settings.agent_max_consecutive_same_tool_errors
        ):
            stop_reason = AgentStopReason.REPEATED_TOOL_ERROR.value

    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    error_source = "tool_runtime"
    if (
        known_tool
        and error_code == ToolErrorCode.INVALID_ARGUMENTS.value
        and int(runtime.get("attempt_count", 0) or 0) == 0
    ):
        error_source = "structured_tool_validation"
    elif not runtime:
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


__all__ = [
    "AgentStopReason",
    "AgentToolFailureKind",
    "build_agent_tool_error_fingerprint",
    "build_tool_governance_event",
    "build_unknown_tool_call_fingerprint",
    "classify_tool_failure",
    "count_consecutive_events",
    "stop_reason_for_tool_failure",
]
