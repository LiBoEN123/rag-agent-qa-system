"""Local StructuredTool 需要保留的项目治理、审计与输出契约。"""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.config import AppSettings
from app.tools.audit import (
    ToolAuditEvent,
    ToolAuditEventType,
    ToolAuditSink,
    sanitize_tool_input,
    summarize_tool_output,
    utc_timestamp,
)
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolRiskLevel, ToolSpec
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.policy import (
    ToolPolicyDecision,
    ToolPolicyDecisionType,
    ToolPolicyEngine,
    build_tool_args_fingerprint,
)
from app.tools.result import tool_error
from app.tools.runtime import (
    ProjectToolExecutorError,
    ProjectToolTimeoutError,
    run_with_timeout,
)


logger = logging.getLogger(__name__)
_WRITE_RISKS = {
    ToolRiskLevel.INTERNAL_WRITE,
    ToolRiskLevel.EXTERNAL_WRITE,
    ToolRiskLevel.DESTRUCTIVE,
}
_GOVERNANCE_ERRORS = {
    ToolErrorCode.PERMISSION_DENIED,
    ToolErrorCode.SQL_PERMISSION_DENIED,
    ToolErrorCode.APPROVAL_REQUIRED,
    ToolErrorCode.MISSING_IDEMPOTENCY_KEY,
    ToolErrorCode.INVALID_INVOCATION_CONTEXT,
    ToolErrorCode.POLICY_DENIED,
    ToolErrorCode.AUDIT_WRITE_FAILED,
}
_PUBLIC_ERROR_CODES = _GOVERNANCE_ERRORS | {
    ToolErrorCode.INVALID_ARGUMENTS,
    ToolErrorCode.TIMEOUT,
    ToolErrorCode.SQL_INVALID_STATEMENT,
    ToolErrorCode.SQL_MULTIPLE_STATEMENTS,
    ToolErrorCode.SQL_WRITE_FORBIDDEN,
    ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
    ToolErrorCode.SQL_DANGEROUS_FUNCTION,
    ToolErrorCode.SQL_RESULT_TOO_WIDE,
}


# 保存单次工具调用的输入、上下文、策略决定、审计状态和尝试次数。
@dataclass(slots=True)
class _Invocation:
    started_at: float
    tool_input: dict[str, Any]
    context: ToolInvocationContext
    tool_call_id: str
    decision: ToolPolicyDecision
    sanitized_input: Any
    audit_write_failed: bool = False
    attempt_count: int = 0


# 保存框架重试模式及当前调用状态，供一次工具执行流程共享。
@dataclass(slots=True)
class _ExecutionScope:
    framework_retry: bool
    invocation: _Invocation | None = None


_CURRENT_SCOPE: ContextVar[_ExecutionScope | None] = ContextVar(
    "project_tool_execution_scope", default=None
)


class ProjectToolFailure(Exception):
    """把已清洗的项目结果交给 LangChain retry/error middleware。"""

    # 携带统一工具失败结果和重试标记，供框架重试中间件判断。
    def __init__(self, result: dict[str, Any], *, retryable: bool) -> None:
        super().__init__(str(result.get("error_code") or "tool_failure"))
        self.result = result
        self.retryable = retryable


def enter_framework_execution() -> Token[_ExecutionScope | None]:
    """为一次 ToolNode 调用建立跨 retry attempt 的极小状态。"""
    return _CURRENT_SCOPE.set(_ExecutionScope(framework_retry=True))


# 使用上下文令牌恢复进入框架工具执行前的作用域。
def exit_framework_execution(token: Token[_ExecutionScope | None]) -> None:
    _CURRENT_SCOPE.reset(token)


# 判断当前作用域是否已创建项目工具调用记录。
def framework_execution_started() -> bool:
    scope = _CURRENT_SCOPE.get()
    return bool(scope and scope.invocation)


# 仅将明确标记为可重试的项目工具失败交给框架重试。
def retry_project_failure(exc: Exception) -> bool:
    return isinstance(exc, ProjectToolFailure) and exc.retryable


def project_failure_content(exc: Exception, _request: Any) -> str | None:
    """只把项目已经清洗的最终失败写入 ToolMessage。"""
    if not isinstance(exc, ProjectToolFailure):
        return None
    return json.dumps(exc.result, ensure_ascii=False, sort_keys=True, default=str)


# 根据幂等声明和重试上限计算工具允许的最大尝试次数。
def _max_attempts(spec: ToolSpec) -> int:
    return 1 + spec.max_retries if spec.idempotent else 1


# 汇总工具版本、尝试次数、执行耗时和超时信息。
def _runtime(spec: ToolSpec, call: _Invocation, *, timed_out: bool = False) -> dict:
    return {
        "tool_version": spec.version,
        "attempt_count": call.attempt_count,
        "max_attempts": _max_attempts(spec),
        "elapsed_ms": max(0, int((time.perf_counter() - call.started_at) * 1000)),
        "timeout": spec.timeout,
        "timed_out": timed_out,
    }


# 将策略决定转换为包含风险、审批、参数指纹和缺失权限的治理信息。
def _governance(decision: ToolPolicyDecision) -> dict:
    return {
        "decision": decision.decision.value,
        "risk_level": decision.risk_level.value,
        "approval_required": decision.approval_required,
        "args_fingerprint": decision.args_fingerprint,
        "missing_permissions": list(decision.missing_permissions),
    }


# 仅对允许公开的错误码保留限长消息，其余返回固定错误描述。
def _safe_error(code: ToolErrorCode, message: str) -> str:
    if code in _PUBLIC_ERROR_CODES:
        return (message or f"工具执行未成功（{code.value}）")[:500]
    return f"工具执行未成功（{code.value}）"


# 构造并写入工具审计事件，写入异常时记录错误类型并返回失败标记。
def _emit(
    event: ToolAuditEventType,
    spec: ToolSpec,
    call: _Invocation,
    sink: ToolAuditSink,
    *,
    code: ToolErrorCode | None = None,
    message: str = "",
    success: bool = False,
    output_summary: Any = None,
) -> bool:
    audit_event = ToolAuditEvent(
        timestamp=utc_timestamp(),
        event_type=event,
        tool_call_id=call.tool_call_id,
        trace_id=call.context.trace_id,
        session_id=call.context.session_id,
        principal_id=call.context.principal_id,
        principal_type=call.context.principal_type.value,
        source=call.context.source,
        tool_name=spec.name,
        tool_version=spec.version,
        risk_level=spec.risk_level.value,
        policy_decision=call.decision.decision.value,
        args_fingerprint=call.decision.args_fingerprint,
        attempt_count=call.attempt_count,
        elapsed_ms=max(0, int((time.perf_counter() - call.started_at) * 1000)),
        success=success,
        error_code=code.value if code else "",
        error_message=_safe_error(code, message) if code else "",
        sanitized_input=call.sanitized_input,
        sanitized_output_summary=output_summary,
    )
    try:
        sink.write(audit_event)
        return True
    except Exception as exc:
        logger.error(
            "tool audit write failed tool=%s event=%s error_type=%s",
            spec.name,
            event.value,
            type(exc).__name__,
        )
        return False


# 构造统一工具错误结果，附加运行信息、治理决定和审计写入状态。
def _error_result(
    spec: ToolSpec,
    call: _Invocation,
    code: ToolErrorCode,
    message: str,
    *,
    retryable: bool = False,
    timed_out: bool = False,
    public: bool = False,
) -> dict[str, Any]:
    result = tool_error(
        spec.name,
        call.tool_input,
        message[:500] if public else _safe_error(code, message),
        error_code=code.value,
        retryable=retryable,
        runtime=_runtime(spec, call, timed_out=timed_out),
        governance=_governance(call.decision),
    )
    result.update(tool_call_id=call.tool_call_id, audit={"write_failed": call.audit_write_failed})
    return result


# 记录失败审计事件，累计审计异常状态并返回统一错误结果。
def _failure(
    spec: ToolSpec,
    call: _Invocation,
    sink: ToolAuditSink,
    code: ToolErrorCode,
    message: str,
    *,
    retryable: bool = False,
    timed_out: bool = False,
    public: bool = False,
    event: ToolAuditEventType = ToolAuditEventType.FAILED,
) -> dict[str, Any]:
    written = _emit(event, spec, call, sink, code=code, message=message)
    call.audit_write_failed = not written or call.audit_write_failed
    return _error_result(
        spec, call, code, message,
        retryable=retryable, timed_out=timed_out, public=public,
    )


# 建立调用记录并执行策略及审计检查，未获放行时提前返回错误结果。
def _start_invocation(
    scope: _ExecutionScope,
    spec: ToolSpec,
    tool_args: dict[str, Any],
    context: ToolInvocationContext,
    tool_call_id: str,
    settings: AppSettings,
    policy: ToolPolicyEngine,
    sink: ToolAuditSink,
) -> dict[str, Any] | None:
    tool_input = spec.args_model.model_construct(**tool_args).model_dump(mode="json")
    fingerprint = build_tool_args_fingerprint(
        tool_name=spec.name, tool_version=spec.version, args=tool_input
    )
    decision = policy.evaluate(
        spec=spec, args=tool_input, context=context, args_fingerprint=fingerprint
    )
    call = _Invocation(
        started_at=time.perf_counter(),
        tool_input=tool_input,
        context=context,
        tool_call_id=tool_call_id or str(uuid4()),
        decision=decision,
        sanitized_input=sanitize_tool_input(spec, tool_input, settings=settings),
    )
    scope.invocation = call
    call.audit_write_failed = not _emit(ToolAuditEventType.REQUESTED, spec, call, sink)
    if call.audit_write_failed and spec.risk_level in _WRITE_RISKS:
        return _error_result(
            spec, call, ToolErrorCode.AUDIT_WRITE_FAILED,
            "写工具执行前审计记录失败，handler 未执行",
        )
    if decision.decision is ToolPolicyDecisionType.ALLOW:
        return None
    event = (
        ToolAuditEventType.APPROVAL_REQUIRED
        if decision.decision is ToolPolicyDecisionType.REQUIRE_APPROVAL
        else ToolAuditEventType.DENIED
    )
    code = decision.error_code or ToolErrorCode.POLICY_DENIED
    written = _emit(event, spec, call, sink, code=code, message=decision.reason)
    call.audit_write_failed = not written or call.audit_write_failed
    return _error_result(spec, call, code, decision.reason)


def execute_project_tool(
    *,
    spec: ToolSpec,
    tool_args: dict[str, Any],
    context: ToolInvocationContext,
    tool_call_id: str = "",
    settings: AppSettings,
    policy_engine: ToolPolicyEngine,
    audit_sink: ToolAuditSink,
) -> dict[str, Any]:
    """执行已经由 StructuredTool/Pydantic 校验的一次本地工具调用。"""
    scope = _CURRENT_SCOPE.get() or _ExecutionScope(framework_retry=False)
    if scope.invocation is None:
        try:
            blocked = _start_invocation(
                scope, spec, tool_args, context, tool_call_id,
                settings, policy_engine, audit_sink,
            )
        except (TypeError, ValueError):
            return invalid_arguments_result(
                spec=spec, tool_args=tool_args, context=context,
                tool_call_id=tool_call_id, settings=settings, audit_sink=audit_sink,
            )
        if blocked is not None:
            return blocked
    call = scope.invocation
    assert call is not None
    call.attempt_count += 1
    written = _emit(ToolAuditEventType.STARTED, spec, call, audit_sink)
    call.audit_write_failed = not written or call.audit_write_failed
    try:
        raw = run_with_timeout(
            lambda: spec.handler(call.tool_input, context),
            timeout=spec.timeout,
            max_workers=settings.tool_runtime_max_workers,
        )
    except ProjectToolTimeoutError:
        return _failure(
            spec, call, audit_sink, ToolErrorCode.TIMEOUT,
            f"工具 {spec.name} 执行超时", retryable=True, timed_out=True,
            event=ToolAuditEventType.TIMED_OUT,
        )
    except ProjectToolExecutorError:
        return _failure(
            spec, call, audit_sink, ToolErrorCode.RUNTIME_UNAVAILABLE,
            "工具执行器当前不可用",
        )
    except ToolExecutionError as exc:
        eligible = (
            spec.idempotent
            and exc.retryable
            and exc.code is not ToolErrorCode.TIMEOUT
            and exc.code not in _GOVERNANCE_ERRORS
        )
        if scope.framework_retry and eligible and call.attempt_count < _max_attempts(spec):
            written = _emit(
                ToolAuditEventType.RETRYING,
                spec,
                call,
                audit_sink,
                code=exc.code,
                message=str(exc),
            )
            call.audit_write_failed = not written or call.audit_write_failed
            raise ProjectToolFailure(
                _error_result(spec, call, exc.code, str(exc), retryable=True),
                retryable=True,
            ) from None
        result = _failure(
            spec,
            call,
            audit_sink,
            exc.code,
            str(exc),
            retryable=exc.retryable,
            timed_out=exc.code is ToolErrorCode.TIMEOUT,
            event=(
                ToolAuditEventType.TIMED_OUT
                if exc.code is ToolErrorCode.TIMEOUT
                else ToolAuditEventType.FAILED
            ),
        )
        if scope.framework_retry and eligible:
            raise ProjectToolFailure(result, retryable=True) from None
        return result
    except Exception:
        return _failure(
            spec, call, audit_sink, ToolErrorCode.EXECUTION_ERROR,
            "工具 handler 执行失败",
        )

    if not isinstance(raw, dict):
        return _failure(
            spec, call, audit_sink, ToolErrorCode.INVALID_RESULT,
            "tool handler result must be a dict",
        )
    success = raw.get("status") == "success" and raw.get("ok") is True
    failure = raw.get("status") == "error" and raw.get("ok") is False
    if failure:
        return _failure(
            spec, call, audit_sink, ToolErrorCode.EXECUTION_ERROR,
            str(raw.get("error") or "tool returned an error result"), public=True,
        )
    if not success:
        return _failure(
            spec, call, audit_sink, ToolErrorCode.INVALID_RESULT,
            "tool result status is inconsistent",
        )
    output = raw.get("output")
    if spec.output_model is not None:
        try:
            output = spec.output_model.model_validate(output).model_dump(mode="json")
        except ValidationError:
            return _failure(
                spec, call, audit_sink, ToolErrorCode.OUTPUT_VALIDATION_ERROR,
                "工具输出不符合声明的 schema",
            )
    result = dict(raw)
    result.update(
        status="success",
        ok=True,
        tool_name=spec.name,
        tool_input=call.tool_input,
        output=output,
        error="",
        error_code="",
        retryable=False,
    )
    written = _emit(
        ToolAuditEventType.SUCCEEDED,
        spec,
        call,
        audit_sink,
        success=True,
        output_summary=summarize_tool_output(spec, output, settings=settings),
    )
    call.audit_write_failed = not written or call.audit_write_failed
    result.update(
        runtime=_runtime(spec, call),
        governance=_governance(call.decision),
        tool_call_id=call.tool_call_id,
        audit={"write_failed": call.audit_write_failed},
    )
    return result


def invalid_arguments_result(
    *,
    spec: ToolSpec,
    tool_args: dict[str, Any],
    context: ToolInvocationContext,
    tool_call_id: str,
    settings: AppSettings,
    audit_sink: ToolAuditSink,
) -> dict[str, Any]:
    """把 ToolNode 的 validation error 映射为稳定项目结果并审计。"""
    decision = ToolPolicyDecision(
        decision=ToolPolicyDecisionType.DENY,
        reason="工具参数校验失败",
        risk_level=spec.risk_level,
        error_code=ToolErrorCode.INVALID_ARGUMENTS,
    )
    call = _Invocation(
        started_at=time.perf_counter(),
        tool_input=tool_args if isinstance(tool_args, dict) else {},
        context=context,
        tool_call_id=tool_call_id or str(uuid4()),
        decision=decision,
        sanitized_input=sanitize_tool_input(spec, tool_args, settings=settings),
    )
    requested = _emit(ToolAuditEventType.REQUESTED, spec, call, audit_sink)
    denied = _emit(
        ToolAuditEventType.DENIED,
        spec,
        call,
        audit_sink,
        code=ToolErrorCode.INVALID_ARGUMENTS,
        message=decision.reason,
    )
    call.audit_write_failed = not (requested and denied)
    return _error_result(spec, call, ToolErrorCode.INVALID_ARGUMENTS, decision.reason)


__all__ = [
    "ProjectToolFailure",
    "enter_framework_execution",
    "execute_project_tool",
    "exit_framework_execution",
    "framework_execution_started",
    "invalid_arguments_result",
    "project_failure_content",
    "retry_project_failure",
]
