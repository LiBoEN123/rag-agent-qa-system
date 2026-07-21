"""Agent 工具错误分类、稳定指纹与循环预算纯函数。"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum, StrEnum
from typing import Any
from uuid import UUID

from app.tools.errors import ToolErrorCode


class AgentToolFailureKind(StrEnum):
    """Agent 对 ToolRuntime 失败采取恢复或停止行为的稳定分类。"""

    RECOVERABLE = "recoverable"
    TRANSIENT_EXHAUSTED = "transient_exhausted"
    TERMINAL_GOVERNANCE = "terminal_governance"
    APPROVAL_REQUIRED = "approval_required"
    TERMINAL_RUNTIME = "terminal_runtime"


class AgentStopReason(StrEnum):
    """迭代式 Agent 的稳定停止原因。"""

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations_reached"
    MAX_TOOL_CALLS = "max_tool_calls_reached"
    MAX_TOOL_ERRORS = "max_tool_errors_reached"
    MAX_TOTAL_COST = "max_total_cost_reached"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    REPEATED_TOOL_ERROR = "repeated_tool_error"
    FATAL_TOOL_ERROR = "fatal_tool_error"
    APPROVAL_REQUIRED = "approval_required"
    MISSING_IDEMPOTENCY_KEY = "missing_idempotency_key"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"


class AgentGuardrailCode(StrEnum):
    """只属于 Agent 编排层、不会进入 ToolRuntime 协议的稳定错误码。"""

    AGENT_BATCH_ABORTED = "agent_batch_aborted"


_RECOVERABLE_CODES = frozenset(
    {
        ToolErrorCode.UNKNOWN_TOOL.value,
        ToolErrorCode.INVALID_ARGUMENTS.value,
        ToolErrorCode.EXECUTION_ERROR.value,
        ToolErrorCode.INVALID_RESULT.value,
        ToolErrorCode.OUTPUT_VALIDATION_ERROR.value,
    }
)
_TRANSIENT_CODES = frozenset(
    {
        ToolErrorCode.TIMEOUT.value,
        ToolErrorCode.DEPENDENCY_UNAVAILABLE.value,
        ToolErrorCode.RATE_LIMITED.value,
    }
)
_TERMINAL_GOVERNANCE_CODES = frozenset(
    {
        ToolErrorCode.PERMISSION_DENIED.value,
        ToolErrorCode.POLICY_DENIED.value,
        ToolErrorCode.AUDIT_WRITE_FAILED.value,
        ToolErrorCode.INVALID_INVOCATION_CONTEXT.value,
        ToolErrorCode.MISSING_IDEMPOTENCY_KEY.value,
    }
)


def classify_tool_failure(
    *,
    error_code: str,
    retryable: bool,
) -> AgentToolFailureKind:
    """把 ToolRuntime 错误映射为 Agent 行为；未知错误保守终止。"""
    normalized = str(error_code or "").strip().lower()
    if normalized in _RECOVERABLE_CODES:
        return AgentToolFailureKind.RECOVERABLE
    if normalized in _TRANSIENT_CODES:
        # ToolRuntime 已经完成契约允许的重试；Agent 不复制自动重试。
        return AgentToolFailureKind.TRANSIENT_EXHAUSTED
    if normalized in _TERMINAL_GOVERNANCE_CODES:
        return AgentToolFailureKind.TERMINAL_GOVERNANCE
    if normalized == ToolErrorCode.APPROVAL_REQUIRED.value:
        return AgentToolFailureKind.APPROVAL_REQUIRED
    if normalized == ToolErrorCode.RUNTIME_UNAVAILABLE.value:
        return AgentToolFailureKind.TERMINAL_RUNTIME
    # 空码或未来新增错误都不能导致异常或无限恢复循环。retryable 只作为
    # 协议信息保留，未知错误仍采用 fail-closed 默认值。
    del retryable
    return AgentToolFailureKind.TERMINAL_RUNTIME


def stop_reason_for_tool_failure(
    *,
    error_code: str,
    failure_kind: AgentToolFailureKind,
) -> AgentStopReason | None:
    """返回单次失败要求的立即停止原因；可恢复失败返回 ``None``。"""
    normalized = str(error_code or "").strip().lower()
    if normalized == ToolErrorCode.APPROVAL_REQUIRED.value:
        return AgentStopReason.APPROVAL_REQUIRED
    if normalized == ToolErrorCode.MISSING_IDEMPOTENCY_KEY.value:
        return AgentStopReason.MISSING_IDEMPOTENCY_KEY
    if normalized == ToolErrorCode.RUNTIME_UNAVAILABLE.value:
        return AgentStopReason.RUNTIME_UNAVAILABLE
    if failure_kind in {
        AgentToolFailureKind.TERMINAL_GOVERNANCE,
        AgentToolFailureKind.APPROVAL_REQUIRED,
        AgentToolFailureKind.TERMINAL_RUNTIME,
    }:
        return AgentStopReason.FATAL_TOOL_ERROR
    return None


def _canonicalize(value: Any) -> Any:
    """把任意常见参数规整为不含对象地址的稳定 JSON 值。"""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"$float": "nan"}
        if math.isinf(value):
            return {"$float": "infinity" if value > 0 else "-infinity"}
        return value
    if isinstance(value, Decimal):
        return {"$decimal": format(value, "f")}
    if isinstance(value, (datetime, date, time)):
        return {"$datetime": value.isoformat()}
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {
                str(key): _canonicalize(item)
                for key, item in value.items()
            }
        pairs = [
            [_canonicalize(key), _canonicalize(item)]
            for key, item in value.items()
        ]
        return {
            "$mapping": sorted(
                pairs,
                key=lambda pair: _stable_json(pair[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return {"$set": sorted(items, key=_stable_json)}
    if isinstance(value, Sequence):
        return [_canonicalize(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _canonicalize(model_dump(mode="json"))
        except (TypeError, ValueError):
            pass
    value_type = type(value)
    return {"$unsupported_type": f"{value_type.__module__}.{value_type.__qualname__}"}


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_agent_tool_call_fingerprint(
    *,
    tool_name: str,
    tool_args: object,
) -> str:
    """生成只绑定工具名和语义参数、可安全写入 trace 的 SHA-256。"""
    payload = {
        "tool_name": str(tool_name or "").strip(),
        "tool_args": _canonicalize(tool_args),
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_agent_tool_error_fingerprint(
    *,
    call_fingerprint: str,
    error_code: str,
) -> str:
    """生成绑定调用指纹和稳定错误码的 SHA-256。"""
    payload = {
        "call_fingerprint": str(call_fingerprint or ""),
        "error_code": str(error_code or "").strip().lower(),
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def advance_consecutive_fingerprint(
    *,
    previous_fingerprint: str,
    previous_count: int,
    current_fingerprint: str,
) -> tuple[str, int]:
    """纯函数更新连续相同指纹计数。"""
    normalized_count = max(0, int(previous_count or 0))
    if current_fingerprint and current_fingerprint == previous_fingerprint:
        return current_fingerprint, normalized_count + 1
    return current_fingerprint, 1 if current_fingerprint else 0


def is_agent_elapsed_budget_exhausted(
    *,
    elapsed_ms: int,
    limit_ms: int,
) -> bool:
    """判断协作式累计耗时预算是否已用尽。"""
    return max(0, int(elapsed_ms or 0)) >= max(1, int(limit_ms or 1))


TERMINAL_AGENT_STOP_REASONS = frozenset(
    reason.value
    for reason in AgentStopReason
    if reason is not AgentStopReason.COMPLETED
)


__all__ = [
    "AgentGuardrailCode",
    "AgentStopReason",
    "AgentToolFailureKind",
    "TERMINAL_AGENT_STOP_REASONS",
    "advance_consecutive_fingerprint",
    "build_agent_tool_call_fingerprint",
    "build_agent_tool_error_fingerprint",
    "classify_tool_failure",
    "is_agent_elapsed_budget_exhausted",
    "stop_reason_for_tool_failure",
]
