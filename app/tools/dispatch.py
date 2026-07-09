"""Tool dispatch and JSONL call logging."""

import json
import time
from pathlib import Path

from app.tools.result import tool_error


TOOL_CALLS_PATH = Path("logs") / "tool_calls.jsonl"


def append_tool_log(record: dict) -> None:
    TOOL_CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOOL_CALLS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def dispatch_tool(
    tool_name: str,
    tool_args: dict | None = None,
    trace_id: str = "",
):
    tool_args = tool_args or {}
    start = time.time()
    try:
        from app.tools.registry import tool_registry

        result = tool_registry.dispatch(tool_name, tool_args)
    except Exception as exc:
        result = tool_error(tool_name, tool_args, str(exc))

    append_tool_log(
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trace_id": trace_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": result,
            "elapsed_ms": int((time.time() - start) * 1000),
            "success": result.get("ok", False),
        }
    )
    return result
