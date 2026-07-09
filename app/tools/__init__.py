"""Public tool API and backwards-compatible entry points."""

from pathlib import Path

from app.tools.calculator import safe_eval_expr, simple_calculator
from app.tools.knowledge_base import (
    get_chunk_text,
    list_documents,
    search_knowledge_base,
)
from app.tools.result import tool_error, tool_success
from app.tools import compare_documents as _compare
from app.tools import dispatch as _dispatch


TOOL_CALLS_PATH = Path("logs") / "tool_calls.jsonl"
_call_compare_documents_llm = _compare._call_compare_documents_llm


def append_tool_log(record: dict) -> None:
    _dispatch.TOOL_CALLS_PATH = TOOL_CALLS_PATH
    _dispatch.append_tool_log(record)


def dispatch_tool(
    tool_name: str,
    tool_args: dict | None = None,
    trace_id: str = "",
):
    _dispatch.TOOL_CALLS_PATH = TOOL_CALLS_PATH
    return _dispatch.dispatch_tool(tool_name, tool_args, trace_id=trace_id)


def compare_documents_tool(args: dict) -> dict:
    # Keep monkeypatching app.tools._call_compare_documents_llm compatible.
    _compare._call_compare_documents_llm = _call_compare_documents_llm
    return _compare.compare_documents_tool(args)
