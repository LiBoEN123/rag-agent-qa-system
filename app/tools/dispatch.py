"""工具分发与日志模块。

通过注册表调用指定工具，并将参数、结果、耗时和追踪编号写入 JSONL。
"""

import json
import time
from pathlib import Path

from app.tools.context import ToolInvocationContext
from app.tools.registry import tool_registry
from app.tools.result import tool_error
from app.tools.runtime import execute_registered_tool


TOOL_CALLS_PATH = Path("logs") / "tool_calls.jsonl"


def append_tool_log(record: dict) -> None:
    """将一条工具调用记录追加写入 JSONL 日志。"""
    TOOL_CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOOL_CALLS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def dispatch_tool(
    tool_name: str,
    tool_args: dict | None = None,
    trace_id: str = "",
) -> dict:
    """调用注册工具并记录耗时、结果和追踪编号。"""
    tool_args = tool_args or {}
    start = time.perf_counter()
    try:
        result = execute_registered_tool(
            registry=tool_registry,
            tool_name=tool_name,
            tool_args=tool_args,
            context=ToolInvocationContext(trace_id=trace_id),
        )
    except Exception as exc:
        result = tool_error(tool_name, tool_args, str(exc))

    runtime = result.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    elapsed_ms = max(0, int((time.perf_counter() - start) * 1000))

    append_tool_log(
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trace_id": trace_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": result,
            "tool_version": runtime.get("tool_version", ""),
            "error_code": result.get("error_code", ""),
            "retryable": result.get("retryable", False),
            "attempt_count": runtime.get("attempt_count", 0),
            "max_attempts": runtime.get("max_attempts", 0),
            "timed_out": runtime.get("timed_out", False),
            "timeout": runtime.get("timeout"),
            "elapsed_ms": elapsed_ms,
            "success": result.get("ok", False),
        }
    )
    return result
