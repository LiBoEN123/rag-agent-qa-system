"""Shared tool result envelopes."""

from typing import Any


def tool_success(tool_name: str, tool_input: dict, output: Any) -> dict:
    return {
        "status": "success",
        "ok": True,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "output": output,
        "error": "",
    }


def tool_error(tool_name: str, tool_input: dict, error: str) -> dict:
    return {
        "status": "error",
        "ok": False,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "output": None,
        "error": error,
    }
