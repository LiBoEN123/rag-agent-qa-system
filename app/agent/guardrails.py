"""Agent 专属错误分类与未知工具 JSON 指纹。"""

import hashlib
import json
from enum import StrEnum

from app.tools.errors import ToolErrorCode


class AgentToolFailureKind(StrEnum):
    """ToolRuntime 失败对 Agent 的稳定语义。"""

    RECOVERABLE = "recoverable"
    TRANSIENT_EXHAUSTED = "transient_exhausted"
    TERMINAL_GOVERNANCE = "terminal_governance"
    APPROVAL_REQUIRED = "approval_required"
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
    APPROVAL_REQUIRED = "approval_required"
    MISSING_IDEMPOTENCY_KEY = "missing_idempotency_key"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"


_RECOVERABLE = (
    ToolErrorCode.UNKNOWN_TOOL.value,
    ToolErrorCode.INVALID_ARGUMENTS.value,
    ToolErrorCode.EXECUTION_ERROR.value,
    ToolErrorCode.INVALID_RESULT.value,
    ToolErrorCode.OUTPUT_VALIDATION_ERROR.value,
    ToolErrorCode.SQL_INVALID_STATEMENT.value,
    ToolErrorCode.SQL_MULTIPLE_STATEMENTS.value,
    ToolErrorCode.SQL_WRITE_FORBIDDEN.value,
    ToolErrorCode.SQL_TABLE_NOT_ALLOWED.value,
    ToolErrorCode.SQL_DANGEROUS_FUNCTION.value,
    ToolErrorCode.SQL_RESULT_TOO_WIDE.value,
    ToolErrorCode.SQL_EXECUTION_ERROR.value,
    ToolErrorCode.MCP_EXECUTION_ERROR.value,
    ToolErrorCode.MCP_UNSUPPORTED_CONTENT.value,
    ToolErrorCode.MCP_OUTPUT_TOO_LARGE.value,
)

_TRANSIENT = (
    ToolErrorCode.TIMEOUT.value,
    ToolErrorCode.DEPENDENCY_UNAVAILABLE.value,
    ToolErrorCode.RATE_LIMITED.value,
    ToolErrorCode.SQL_TIMEOUT.value,
    ToolErrorCode.SQL_CONNECTION_ERROR.value,
    ToolErrorCode.MCP_TOOL_TIMEOUT.value,
    ToolErrorCode.MCP_TRANSPORT_ERROR.value,
)

_GOVERNANCE = (
    ToolErrorCode.PERMISSION_DENIED.value,
    ToolErrorCode.SQL_PERMISSION_DENIED.value,
    ToolErrorCode.POLICY_DENIED.value,
    ToolErrorCode.AUDIT_WRITE_FAILED.value,
    ToolErrorCode.INVALID_INVOCATION_CONTEXT.value,
    ToolErrorCode.MISSING_IDEMPOTENCY_KEY.value,
    ToolErrorCode.MCP_NOT_CONFIGURED.value,
    ToolErrorCode.MCP_PERMISSION_DENIED.value,
    ToolErrorCode.MCP_TOOL_NOT_ALLOWED.value,
)


def classify_tool_failure(*, error_code: str, retryable: bool) -> AgentToolFailureKind:
    """把 Runtime 错误映射为恢复或终止语义；未知错误 fail closed。"""
    del retryable
    code = str(error_code or "").strip().lower()
    if code in _RECOVERABLE:
        return AgentToolFailureKind.RECOVERABLE
    if code in _TRANSIENT:
        return AgentToolFailureKind.TRANSIENT_EXHAUSTED
    if code in _GOVERNANCE:
        return AgentToolFailureKind.TERMINAL_GOVERNANCE
    if code == ToolErrorCode.APPROVAL_REQUIRED.value:
        return AgentToolFailureKind.APPROVAL_REQUIRED
    return AgentToolFailureKind.TERMINAL_RUNTIME


def stop_reason_for_tool_failure(
    *, error_code: str, failure_kind: AgentToolFailureKind
) -> AgentStopReason | None:
    """返回单次错误要求的立即停止原因。"""
    code = str(error_code or "").strip().lower()
    if code == ToolErrorCode.APPROVAL_REQUIRED.value:
        return AgentStopReason.APPROVAL_REQUIRED
    if code == ToolErrorCode.MISSING_IDEMPOTENCY_KEY.value:
        return AgentStopReason.MISSING_IDEMPOTENCY_KEY
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
