"""工具系统公共入口。

统一暴露内置工具、分发器和文档对比能力，并同步可替换的运行时依赖。
"""

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


__all__ = [
    "TOOL_CALLS_PATH",
    "append_tool_log",
    "compare_documents_tool",
    "dispatch_tool",
    "get_chunk_text",
    "list_documents",
    "safe_eval_expr",
    "search_knowledge_base",
    "simple_calculator",
    "tool_error",
    "tool_success",
]


TOOL_CALLS_PATH = Path("logs") / "tool_calls.jsonl"
_call_compare_documents_llm = _compare._call_compare_documents_llm


def append_tool_log(record: dict) -> None:
    """使用门面层配置的路径追加工具调用日志。"""
    _dispatch.TOOL_CALLS_PATH = TOOL_CALLS_PATH
    _dispatch.append_tool_log(record)


def dispatch_tool(
    tool_name: str,
    tool_args: dict | None = None,
    trace_id: str = "",
):
    """通过统一入口分发工具，并同步调用日志路径。"""
    _dispatch.TOOL_CALLS_PATH = TOOL_CALLS_PATH
    return _dispatch.dispatch_tool(tool_name, tool_args, trace_id=trace_id)


def compare_documents_tool(args: dict) -> dict:
    """执行文档对比工具，并应用门面层替换的 LLM 调用函数。"""
    # Keep monkeypatching app.tools._call_compare_documents_llm compatible.
    _compare._call_compare_documents_llm = _call_compare_documents_llm
    return _compare.compare_documents_tool(args)
