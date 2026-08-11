"""按请求权限提供启动期缓存的 MCP StructuredTool。"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.config import get_settings
from app.tools.context import ToolInvocationContext, has_permission


@dataclass(frozen=True, slots=True)
class CachedMcpTool:
    """一个已经完成 discovery、安全包装和名称隔离的 MCP Tool。"""

    server_alias: str
    remote_tool_name: str
    tool: BaseTool

    @property
    def required_permission(self) -> str:
        """返回访问该 Server 工具所需的精确项目权限。"""
        return f"mcp:{self.server_alias}"


class McpToolProvider:
    """只从不可变 cache 中按 server permission 选择工具。"""

    def __init__(self, entries: list[CachedMcpTool]) -> None:
        """固化 discovery 结果，并拒绝缓存中的重复工具名称。"""
        self._entries = tuple(entries)
        names = [entry.tool.name for entry in self._entries]
        if len(names) != len(set(names)):
            raise ValueError("MCP Tool cache 包含重复名称")

    def list_tools(self, *, context: ToolInvocationContext) -> list[BaseTool]:
        """只返回当前调用上下文具有对应 Server 权限的工具。"""
        return [
            entry.tool
            for entry in self._entries
            if has_permission(context.permissions, entry.required_permission)
        ]

    def catalog(self) -> list[dict[str, str]]:
        """返回不含 URL、Header 或 Client 对象的安全工具目录。"""
        return [
            {
                "server_alias": entry.server_alias,
                "tool_name": entry.tool.name,
                "remote_tool_name": entry.remote_tool_name,
                "description": " ".join(entry.tool.description.split())[:240],
            }
            for entry in self._entries
        ]


_provider_lock = threading.Lock()
_provider: McpToolProvider | None = None


def install_mcp_tool_provider(provider: McpToolProvider) -> None:
    """在 discovery 全部成功后原子安装进程级 cache。"""
    global _provider
    with _provider_lock:
        _provider = provider


def clear_mcp_tool_provider() -> None:
    """清空项目 MCP Tool cache；adapter 当前不持有永久 session。"""
    global _provider
    with _provider_lock:
        _provider = None


def get_mcp_tool_provider() -> McpToolProvider | None:
    """在线程锁保护下读取当前安装的进程级 MCP Provider。"""
    with _provider_lock:
        return _provider


def list_mcp_tools_for_context(
    context: ToolInvocationContext,
) -> list[BaseTool]:
    """默认关闭时保持空集合；显式启用却未初始化时 fail closed。"""
    provider = get_mcp_tool_provider()
    if provider is not None:
        return provider.list_tools(context=context)
    if get_settings().mcp_tools_enabled:
        raise RuntimeError("MCP runtime is not initialized")
    return []


def get_mcp_tool_catalog() -> list[dict[str, str]]:
    """返回当前缓存的安全工具目录；未初始化时返回空列表。"""
    provider = get_mcp_tool_provider()
    return provider.catalog() if provider is not None else []


__all__ = [
    "CachedMcpTool",
    "McpToolProvider",
    "clear_mcp_tool_provider",
    "get_mcp_tool_catalog",
    "get_mcp_tool_provider",
    "install_mcp_tool_provider",
    "list_mcp_tools_for_context",
]
