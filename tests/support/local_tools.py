"""本地 ToolSpec 的直接 BaseTool.invoke 测试辅助。"""

from typing import Any

from app.agent.framework_tools import build_local_tool
from app.config import AppSettings, get_settings
from app.tools.audit import ToolAuditSink
from app.tools.context import ToolInvocationContext, build_default_tool_context
from app.tools.registry import ToolRegistry


def invoke_local_tool(
    registry: ToolRegistry,
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    context: ToolInvocationContext | None = None,
    settings: AppSettings | None = None,
    audit_sink: ToolAuditSink | None = None,
) -> dict[str, Any]:
    """构造无 Runtime 注入的同款工具，并经 BaseTool.invoke 执行。"""
    spec = registry.get(tool_name)
    if spec is None:
        raise LookupError(f"unknown local tool: {tool_name}")
    resolved_settings = settings or get_settings()
    result = build_local_tool(
        spec,
        context=context
        or build_default_tool_context(source="direct_test", settings=resolved_settings),
        settings=resolved_settings,
        audit_sink=audit_sink,
        runtime_injection=False,
    ).invoke(tool_args)
    if not isinstance(result, dict):
        raise TypeError("local tool did not return a result mapping")
    return result


__all__ = ["invoke_local_tool"]
