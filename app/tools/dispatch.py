"""工具分发与安全兼容日志模块。"""

import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.tools.audit import sanitize_tool_input, summarize_tool_output
from app.tools.context import (
    ToolInvocationContext,
    build_default_tool_context,
)
from app.tools.errors import ToolErrorCode
from app.tools.redaction import redact_sensitive_data
from app.tools.registry import tool_registry
from app.tools.result import tool_error
from app.tools.runtime import execute_registered_tool


TOOL_CALLS_PATH = Path("logs") / "tool_calls.jsonl"
logger = logging.getLogger(__name__)


def append_tool_log(record: dict) -> None:
    """脱敏并限制体量后把兼容调用摘要追加写入 JSONL。"""
    safe_record = redact_sensitive_data(record)
    TOOL_CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOOL_CALLS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def _trace_conflict_result(
    tool_name: str,
    tool_args: dict,
) -> dict[str, Any]:
    """同时传入不一致 trace 时返回明确上下文错误。"""
    result = tool_error(
        tool_name,
        tool_args,
        "context.trace_id 与兼容 trace_id 参数不一致",
        error_code=ToolErrorCode.INVALID_INVOCATION_CONTEXT.value,
        runtime={
            "tool_version": "",
            "attempt_count": 0,
            "max_attempts": 0,
            "elapsed_ms": 0,
            "timeout": None,
            "timed_out": False,
            "attempts": [],
        },
    )
    result["tool_call_id"] = str(uuid4())
    result["governance"] = {
        "decision": "deny",
        "risk_level": "read_only",
        "approval_required": False,
        "args_fingerprint": "",
    }
    result["audit"] = {"write_failed": False}
    return result


def dispatch_tool(
    tool_name: str,
    tool_args: dict | None = None,
    trace_id: str = "",
    *,
    context: ToolInvocationContext | None = None,
) -> dict:
    """通过统一治理 Runtime 调用工具并记录安全摘要。"""
    compatible_args = tool_args if isinstance(tool_args, dict) else {}
    if (
        context is not None
        and trace_id.strip()
        and context.trace_id
        and trace_id.strip() != context.trace_id
    ):
        return _trace_conflict_result(tool_name, compatible_args)

    invocation_context = context or build_default_tool_context(
        trace_id=trace_id,
        source="direct_dispatch",
    )
    effective_trace_id = invocation_context.trace_id
    start = time.perf_counter()
    try:
        result = execute_registered_tool(
            registry=tool_registry,
            tool_name=tool_name,
            tool_args=tool_args,
            context=invocation_context,
        )
    except Exception as exc:
        result = tool_error(
            tool_name,
            compatible_args,
            str(exc),
            error_code=ToolErrorCode.EXECUTION_ERROR.value,
        )

    runtime = result.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    governance = result.get("governance")
    governance = governance if isinstance(governance, dict) else {}
    spec = tool_registry.get(tool_name)
    if spec is None:
        sanitized_args = redact_sensitive_data(compatible_args)
        output_summary: Any = {
            "type": type(result.get("output")).__name__,
        }
    else:
        sanitized_args = sanitize_tool_input(spec, compatible_args)
        output_summary = summarize_tool_output(spec, result.get("output"))

    try:
        append_tool_log({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trace_id": effective_trace_id,
            "tool_call_id": result.get("tool_call_id", ""),
            "tool_name": tool_name,
            "sanitized_tool_args": sanitized_args,
            "output_summary": output_summary,
            "tool_version": runtime.get("tool_version", ""),
            "error_code": result.get("error_code", ""),
            "retryable": result.get("retryable", False),
            "policy_decision": governance.get("decision", ""),
            "risk_level": governance.get("risk_level", ""),
            "approval_required": governance.get("approval_required", False),
            "args_fingerprint": governance.get("args_fingerprint", ""),
            "attempt_count": runtime.get("attempt_count", 0),
            "max_attempts": runtime.get("max_attempts", 0),
            "timed_out": runtime.get("timed_out", False),
            "timeout": runtime.get("timeout"),
            "elapsed_ms": max(
                0,
                int((time.perf_counter() - start) * 1000),
            ),
            "success": result.get("ok", False),
        })
    except Exception as exc:
        logger.error(
            "safe tool call summary write failed tool=%s error_type=%s",
            tool_name,
            type(exc).__name__,
        )
    return result


__all__ = ["TOOL_CALLS_PATH", "append_tool_log", "dispatch_tool"]
