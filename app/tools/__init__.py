"""工具系统公共入口。

统一暴露内置工具、分发器和文档对比能力。
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
    "get_langchain_tool",
    "get_langchain_tool_map",
    "get_langchain_tool_names",
    "get_langchain_tools",
    "list_documents",
    "safe_eval_expr",
    "search_knowledge_base",
    "simple_calculator",
    "tool_error",
    "tool_success",
]


TOOL_CALLS_PATH = Path("logs") / "tool_calls.jsonl"


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
    """执行文档对比工具。"""
    return _compare.compare_documents_tool(args)


# 放在门面函数定义之后导入，避免 registry 初始化时反向读取未完成的 app.tools。
from app.tools.langchain_tools import (  # noqa: E402
    get_langchain_tool,
    get_langchain_tool_map,
    get_langchain_tool_names,
    get_langchain_tools,
)
