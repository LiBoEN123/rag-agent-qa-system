"""工具结果协议模块。

为所有工具提供字段一致的成功与失败返回结构。
"""

from typing import Any


def tool_success(tool_name: str, tool_input: dict, output: Any) -> dict:
    """构造统一的工具成功返回结构。"""
    return {
        "status": "success",
        "ok": True,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "output": output,
        "error": "",
    }


def tool_error(tool_name: str, tool_input: dict, error: str) -> dict:
    """构造统一的工具失败返回结构。"""
    return {
        "status": "error",
        "ok": False,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "output": None,
        "error": error,
    }
