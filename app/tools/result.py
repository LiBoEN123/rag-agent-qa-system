"""工具结果的唯一构造与模型内容投影入口。"""

import json
from typing import Any


def tool_result(
    tool_name: str, tool_input: dict, output: Any = None, *, error: str = "",
    error_code: str = "", retryable: bool = False, runtime: dict | None = None,
    governance: dict | None = None, tool_call_id: str = "", audit: dict | None = None,
) -> dict:
    """内部合同保留审计和治理字段；模型只接收 tool_content 的业务投影。"""
    ok = not error_code
    result = {
        "status": "success" if ok else "error", "ok": ok,
        "tool_name": tool_name, "tool_input": tool_input, "output": output,
        "error": error, "error_code": error_code, "retryable": retryable,
        "runtime": runtime or {}, "governance": governance or {},
    }
    if tool_call_id:
        result["tool_call_id"] = tool_call_id
    if audit is not None:
        result["audit"] = audit
    return result


def tool_success(tool_name: str, tool_input: dict, output: Any, **metadata) -> dict:
    return tool_result(tool_name, tool_input, output, **metadata)


def tool_error(
    tool_name: str, tool_input: dict, error: str, *,
    error_code: str = "execution_error", retryable: bool = False, **metadata,
) -> dict:
    return tool_result(tool_name, tool_input, error=error, error_code=error_code,
                       retryable=retryable, **metadata)


def tool_content(result: dict) -> str:
    """只序列化模型需要的业务结果；内部数据随 ToolMessage.artifact 传递。"""
    return json.dumps({key: result[key] for key in (
        "status", "ok", "tool_name", "output", "error", "error_code", "retryable",
    )}, ensure_ascii=False)
