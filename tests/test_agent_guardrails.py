"""Agent 项目特有错误分类与未知工具指纹测试。"""

import pytest

from app.agent.guardrails import (
    AgentStopReason,
    AgentToolFailureKind,
    build_agent_tool_error_fingerprint,
    build_unknown_tool_call_fingerprint,
    classify_tool_failure,
    stop_reason_for_tool_failure,
)
from app.tools.errors import ToolErrorCode


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        (ToolErrorCode.UNKNOWN_TOOL, AgentToolFailureKind.RECOVERABLE),
        (ToolErrorCode.INVALID_ARGUMENTS, AgentToolFailureKind.RECOVERABLE),
        (ToolErrorCode.EXECUTION_ERROR, AgentToolFailureKind.RECOVERABLE),
        (ToolErrorCode.INVALID_RESULT, AgentToolFailureKind.RECOVERABLE),
        (ToolErrorCode.OUTPUT_VALIDATION_ERROR, AgentToolFailureKind.RECOVERABLE),
        (ToolErrorCode.TIMEOUT, AgentToolFailureKind.TRANSIENT_EXHAUSTED),
        (ToolErrorCode.DEPENDENCY_UNAVAILABLE, AgentToolFailureKind.TRANSIENT_EXHAUSTED),
        (ToolErrorCode.RATE_LIMITED, AgentToolFailureKind.TRANSIENT_EXHAUSTED),
        (ToolErrorCode.PERMISSION_DENIED, AgentToolFailureKind.TERMINAL_GOVERNANCE),
        (ToolErrorCode.POLICY_DENIED, AgentToolFailureKind.TERMINAL_GOVERNANCE),
        (ToolErrorCode.AUDIT_WRITE_FAILED, AgentToolFailureKind.TERMINAL_GOVERNANCE),
        (
            ToolErrorCode.INVALID_INVOCATION_CONTEXT,
            AgentToolFailureKind.TERMINAL_GOVERNANCE,
        ),
        (ToolErrorCode.APPROVAL_REQUIRED, AgentToolFailureKind.APPROVAL_REQUIRED),
        (ToolErrorCode.RUNTIME_UNAVAILABLE, AgentToolFailureKind.TERMINAL_RUNTIME),
    ],
)
def test_classify_all_tool_error_codes(error_code, expected):
    assert classify_tool_failure(error_code=error_code.value, retryable=False) is expected


@pytest.mark.parametrize("error_code", ["future_error", "", "  UNKNOWN_NEW "])
def test_unknown_error_code_fails_closed_without_raising(error_code):
    assert classify_tool_failure(error_code=error_code, retryable=True) is (
        AgentToolFailureKind.TERMINAL_RUNTIME
    )


def test_unknown_tool_fingerprint_is_stable_for_model_json_only():
    first = build_unknown_tool_call_fingerprint(
        tool_name="future_tool", tool_args={"query": "A", "top_k": 5}
    )
    second = build_unknown_tool_call_fingerprint(
        tool_name="future_tool", tool_args={"top_k": 5, "query": "A"}
    )
    assert first == second
    assert first.startswith("sha256:")
    assert first != build_unknown_tool_call_fingerprint(
        tool_name="other_tool", tool_args={"query": "A", "top_k": 5}
    )


def test_unknown_tool_fingerprint_has_explicit_unsupported_fallback():
    class Unsupported:
        pass

    assert build_unknown_tool_call_fingerprint(
        tool_name="future_tool", tool_args=Unsupported()
    ) == build_unknown_tool_call_fingerprint(
        tool_name="future_tool", tool_args=Unsupported()
    )


def test_error_fingerprint_binds_call_and_normalized_code():
    call = build_unknown_tool_call_fingerprint(
        tool_name="future_tool", tool_args={"value": 1}
    )
    first = build_agent_tool_error_fingerprint(
        call_fingerprint=call, error_code="execution_error"
    )
    assert first == build_agent_tool_error_fingerprint(
        call_fingerprint=call, error_code=" EXECUTION_ERROR "
    )
    assert first != build_agent_tool_error_fingerprint(
        call_fingerprint=call, error_code="timeout"
    )


@pytest.mark.parametrize(
    ("code", "kind", "expected"),
    [
        (
            "permission_denied",
            AgentToolFailureKind.TERMINAL_GOVERNANCE,
            AgentStopReason.FATAL_TOOL_ERROR,
        ),
        (
            "approval_required",
            AgentToolFailureKind.APPROVAL_REQUIRED,
            AgentStopReason.APPROVAL_REQUIRED,
        ),
        (
            "runtime_unavailable",
            AgentToolFailureKind.TERMINAL_RUNTIME,
            AgentStopReason.RUNTIME_UNAVAILABLE,
        ),
        ("execution_error", AgentToolFailureKind.RECOVERABLE, None),
    ],
)
def test_stop_reason_mapping(code, kind, expected):
    assert stop_reason_for_tool_failure(error_code=code, failure_kind=kind) is expected
