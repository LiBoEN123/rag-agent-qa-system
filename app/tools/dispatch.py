"""统一工具分发入口。"""

from typing import Any
from uuid import uuid4

from app.tools.context import (
    ToolInvocationContext,
    build_default_tool_context,
)
from app.tools.errors import ToolErrorCode
from app.tools.registry import get_tool_registry
from app.tools.result import tool_error
from app.tools.runtime import execute_registered_tool


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
    """通过统一治理 Runtime 调用工具并返回标准结果。"""
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
    try:
        return execute_registered_tool(
            registry=get_tool_registry(),
            tool_name=tool_name,
            tool_args=tool_args,
            context=invocation_context,
        )
    except Exception as exc:
        return tool_error(
            tool_name,
            compatible_args,
            str(exc),
            error_code=ToolErrorCode.EXECUTION_ERROR.value,
        )


__all__ = ["dispatch_tool"]
