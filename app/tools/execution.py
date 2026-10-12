"""只读 Local Tool 的权限、审计和结果校验；调度与重试复用 LangChain。"""

import asyncio
import logging
import time
from uuid import uuid4

from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.config import AppSettings
from app.tools.audit import (
    ToolAuditEvent, ToolAuditEventType, ToolAuditSink,
    sanitize_tool_input, summarize_tool_output, utc_timestamp,
)
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.policy import ToolPolicyEngine, build_tool_args_fingerprint
from app.tools.result import tool_result

logger = logging.getLogger(__name__)
_GOVERNANCE_ERRORS = {
    ToolErrorCode.PERMISSION_DENIED, ToolErrorCode.INVALID_INVOCATION_CONTEXT,
    ToolErrorCode.POLICY_DENIED, ToolErrorCode.AUDIT_WRITE_FAILED,
}


class ProjectToolFailure(Exception):
    """仅把显式可重试的只读业务错误交给原生 RunnableRetry。"""

    def __init__(self, error: ToolExecutionError):
        super().__init__(error.code.value)
        self.error = error


async def execute_project_tool(
    *, spec: ToolSpec, tool_args: dict, context: ToolInvocationContext,
    settings: AppSettings, policy_engine: ToolPolicyEngine, audit_sink: ToolAuditSink,
    tool_call_id: str = "",
) -> dict:
    """一次调用只保存局部审计数据；无 ContextVar、执行器或自研重试循环。"""
    started = time.perf_counter()
    call_id = tool_call_id or str(uuid4())
    attempt_count = 0
    audit_failed = False
    code = None
    error = ""
    retryable = False
    output = None
    fingerprint = ""
    decision = "deny"
    missing_permissions = ()
    tool_input = tool_args
    try:
        tool_input = spec.args_model.model_validate(tool_args).model_dump(mode="json")
        fingerprint = build_tool_args_fingerprint(
            tool_name=spec.name, tool_version=spec.version, args=tool_input,
        )
        policy = policy_engine.evaluate(
            spec=spec, args=tool_input, context=context, args_fingerprint=fingerprint,
        )
        decision = policy.decision.value
        missing_permissions = policy.missing_permissions
        code, error = policy.error_code, policy.reason if policy.error_code else ""
    except (ValidationError, TypeError, ValueError):
        code, error = ToolErrorCode.INVALID_ARGUMENTS, "工具参数校验失败"
    sanitized_input = sanitize_tool_input(spec, tool_input, settings=settings)

    def emit(event, *, success=False, error_code=None, summary=None):
        nonlocal audit_failed
        try:
            audit_sink.write(ToolAuditEvent(
                timestamp=utc_timestamp(), event_type=event, tool_call_id=call_id,
                trace_id=context.trace_id, session_id=context.session_id,
                principal_id=context.principal_id, principal_type=context.principal_type.value,
                source=context.source, tool_name=spec.name, tool_version=spec.version,
                risk_level="read_only", policy_decision=decision,
                args_fingerprint=fingerprint, attempt_count=attempt_count,
                elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
                success=success, error_code=error_code.value if error_code else "",
                error_message=f"工具执行未成功（{error_code.value}）" if error_code else "",
                sanitized_input=sanitized_input, sanitized_output_summary=summary,
            ))
        except Exception as exc:
            audit_failed = True
            logger.error("tool audit write failed tool=%s event=%s error_type=%s",
                         spec.name, event.value, type(exc).__name__)

    emit(ToolAuditEventType.REQUESTED)
    if code:
        emit(ToolAuditEventType.DENIED, error_code=code)
    else:
        # 原生 RunnableLambda.ainvoke 在线程中执行同步业务函数；不持有私有线程池。
        business = RunnableLambda(lambda args: spec.handler(args, context))

        async def attempt(args):
            nonlocal attempt_count
            attempt_count += 1
            emit(ToolAuditEventType.STARTED)
            try:
                async with asyncio.timeout(spec.timeout):
                    raw = await business.ainvoke(args)
            except ToolExecutionError as exc:
                if exc.retryable and exc.code not in _GOVERNANCE_ERRORS | {ToolErrorCode.TIMEOUT}:
                    if attempt_count <= spec.max_retries:
                        emit(ToolAuditEventType.RETRYING, error_code=exc.code)
                    raise ProjectToolFailure(exc) from None
                raise
            if not isinstance(raw, dict) or not (
                (raw.get("status") == "success" and raw.get("ok") is True)
                or (raw.get("status") == "error" and raw.get("ok") is False)
            ):
                raise ToolExecutionError("无效工具结果", code=ToolErrorCode.INVALID_RESULT)
            if not raw["ok"]:
                raise ToolExecutionError("工具返回失败")
            try:
                return spec.output_model.model_validate(raw.get("output")).model_dump(mode="json")
            except ValidationError:
                raise ToolExecutionError(
                    "工具输出不符合声明的 schema", code=ToolErrorCode.OUTPUT_VALIDATION_ERROR,
                ) from None

        try:
            output = await RunnableLambda(attempt).with_retry(
                retry_if_exception_type=(ProjectToolFailure,),
                stop_after_attempt=1 + spec.max_retries,
                exponential_jitter_params={
                    "initial": settings.tool_retry_base_delay_ms / 1000.0,
                    "max": settings.tool_retry_max_delay_ms / 1000.0,
                    "exp_base": 2.0,
                    "jitter": settings.tool_retry_jitter_ratio,
                },
            ).ainvoke(tool_input)
        except TimeoutError:
            code, error, retryable = ToolErrorCode.TIMEOUT, f"工具 {spec.name} 执行超时", True
        except (ProjectToolFailure, ToolExecutionError) as exc:
            failure = exc.error if isinstance(exc, ProjectToolFailure) else exc
            code, error, retryable = failure.code, str(failure), failure.retryable
        except Exception:
            code, error = ToolErrorCode.EXECUTION_ERROR, "工具 handler 执行失败"
        emit(
            ToolAuditEventType.TIMED_OUT if code is ToolErrorCode.TIMEOUT else
            ToolAuditEventType.FAILED if code else ToolAuditEventType.SUCCEEDED,
            success=code is None, error_code=code,
            summary=summarize_tool_output(spec, output, settings=settings) if code is None else None,
        )

    if code and code not in _GOVERNANCE_ERRORS | {ToolErrorCode.INVALID_ARGUMENTS, ToolErrorCode.TIMEOUT}:
        error = f"工具执行未成功（{code.value}）"
    return tool_result(
        spec.name, tool_input, output, error=error[:500],
        error_code=code.value if code else "", retryable=retryable,
        runtime={"tool_version": spec.version, "attempt_count": attempt_count,
                 "max_attempts": 1 + spec.max_retries, "timeout": spec.timeout,
                 "elapsed_ms": max(0, int((time.perf_counter() - started) * 1000)),
                 "timed_out": code is ToolErrorCode.TIMEOUT},
        governance={"decision": decision, "risk_level": "read_only",
                    "args_fingerprint": fingerprint, "missing_permissions": list(missing_permissions)},
        tool_call_id=call_id, audit={"write_failed": audit_failed},
    )
