"""Agent 工具错误分类、指纹、连续计数和预算纯函数测试。"""

import pytest

from app.agent.guardrails import (
    AgentStopReason,
    AgentToolFailureKind,
    TERMINAL_AGENT_STOP_REASONS,
    advance_consecutive_fingerprint,
    build_agent_tool_call_fingerprint,
    build_agent_tool_error_fingerprint,
    classify_tool_failure,
    is_agent_elapsed_budget_exhausted,
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
        (
            ToolErrorCode.OUTPUT_VALIDATION_ERROR,
            AgentToolFailureKind.RECOVERABLE,
        ),
        (ToolErrorCode.TIMEOUT, AgentToolFailureKind.TRANSIENT_EXHAUSTED),
        (
            ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            AgentToolFailureKind.TRANSIENT_EXHAUSTED,
        ),
        (ToolErrorCode.RATE_LIMITED, AgentToolFailureKind.TRANSIENT_EXHAUSTED),
        (
            ToolErrorCode.PERMISSION_DENIED,
            AgentToolFailureKind.TERMINAL_GOVERNANCE,
        ),
        (
            ToolErrorCode.POLICY_DENIED,
            AgentToolFailureKind.TERMINAL_GOVERNANCE,
        ),
        (
            ToolErrorCode.AUDIT_WRITE_FAILED,
            AgentToolFailureKind.TERMINAL_GOVERNANCE,
        ),
        (
            ToolErrorCode.INVALID_INVOCATION_CONTEXT,
            AgentToolFailureKind.TERMINAL_GOVERNANCE,
        ),
        (
            ToolErrorCode.MISSING_IDEMPOTENCY_KEY,
            AgentToolFailureKind.TERMINAL_GOVERNANCE,
        ),
        (
            ToolErrorCode.APPROVAL_REQUIRED,
            AgentToolFailureKind.APPROVAL_REQUIRED,
        ),
        (
            ToolErrorCode.RUNTIME_UNAVAILABLE,
            AgentToolFailureKind.TERMINAL_RUNTIME,
        ),
    ],
)
def test_classify_all_tool_error_codes(error_code, expected):
    assert (
        classify_tool_failure(error_code=error_code.value, retryable=False)
        is expected
    )


@pytest.mark.parametrize("error_code", ["future_error", "", "  UNKNOWN_NEW "])
def test_unknown_error_code_fails_closed_without_raising(error_code):
    assert classify_tool_failure(error_code=error_code, retryable=True) is (
        AgentToolFailureKind.TERMINAL_RUNTIME
    )


def test_call_fingerprint_is_stable_and_ignores_json_key_order():
    first = build_agent_tool_call_fingerprint(
        tool_name="search_knowledge_base",
        tool_args={"query": "A", "top_k": 5},
    )
    second = build_agent_tool_call_fingerprint(
        tool_name="search_knowledge_base",
        tool_args={"top_k": 5, "query": "A"},
    )

    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == 71
    assert "query" not in first
    assert "A" not in first


def test_call_fingerprint_changes_with_tool_or_arguments():
    baseline = build_agent_tool_call_fingerprint(
        tool_name="tool_a",
        tool_args={"value": 1},
    )
    assert baseline != build_agent_tool_call_fingerprint(
        tool_name="tool_a",
        tool_args={"value": 2},
    )
    assert baseline != build_agent_tool_call_fingerprint(
        tool_name="tool_b",
        tool_args={"value": 1},
    )


def test_non_dict_and_unsupported_arguments_have_stable_fingerprints():
    class Unsupported:
        pass

    assert build_agent_tool_call_fingerprint(
        tool_name="tool",
        tool_args=["a", 1],
    ) == build_agent_tool_call_fingerprint(
        tool_name="tool",
        tool_args=["a", 1],
    )
    assert build_agent_tool_call_fingerprint(
        tool_name="tool",
        tool_args=Unsupported(),
    ) == build_agent_tool_call_fingerprint(
        tool_name="tool",
        tool_args=Unsupported(),
    )


def test_error_fingerprint_binds_call_and_stable_error_code():
    call = build_agent_tool_call_fingerprint(
        tool_name="tool",
        tool_args={"value": 1},
    )
    first = build_agent_tool_error_fingerprint(
        call_fingerprint=call,
        error_code="execution_error",
    )

    assert first == build_agent_tool_error_fingerprint(
        call_fingerprint=call,
        error_code=" EXECUTION_ERROR ",
    )
    assert first != build_agent_tool_error_fingerprint(
        call_fingerprint=call,
        error_code="timeout",
    )


def test_consecutive_counter_increments_and_resets():
    assert advance_consecutive_fingerprint(
        previous_fingerprint="same",
        previous_count=1,
        current_fingerprint="same",
    ) == ("same", 2)
    assert advance_consecutive_fingerprint(
        previous_fingerprint="same",
        previous_count=9,
        current_fingerprint="different",
    ) == ("different", 1)
    assert advance_consecutive_fingerprint(
        previous_fingerprint="same",
        previous_count=2,
        current_fingerprint="",
    ) == ("", 0)


def test_elapsed_budget_uses_cooperative_greater_than_or_equal_semantics():
    assert not is_agent_elapsed_budget_exhausted(elapsed_ms=999, limit_ms=1000)
    assert is_agent_elapsed_budget_exhausted(elapsed_ms=1000, limit_ms=1000)
    assert is_agent_elapsed_budget_exhausted(elapsed_ms=1001, limit_ms=1000)


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
            "missing_idempotency_key",
            AgentToolFailureKind.TERMINAL_GOVERNANCE,
            AgentStopReason.MISSING_IDEMPOTENCY_KEY,
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


def test_terminal_stop_reason_set_contains_budgets_and_governance():
    assert AgentStopReason.MAX_TOOL_ERRORS.value in TERMINAL_AGENT_STOP_REASONS
    assert AgentStopReason.MAX_TOTAL_COST.value in TERMINAL_AGENT_STOP_REASONS
    assert AgentStopReason.APPROVAL_REQUIRED.value in TERMINAL_AGENT_STOP_REASONS
    assert AgentStopReason.COMPLETED.value not in TERMINAL_AGENT_STOP_REASONS
