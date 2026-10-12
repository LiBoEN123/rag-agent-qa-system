"""按来源权限提供启动期缓存的只读 MCP 工具。"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.tools.context import ToolInvocationContext, has_permission
from app.tools.mcp.config import McpServerConfig


@dataclass(frozen=True, slots=True)
class CachedMcpTool:
    """一个已经完成 discovery、安全包装和名称隔离的 MCP Tool。"""

    server_alias: str
    remote_tool_name: str
    tool: BaseTool
    server_config: McpServerConfig


class McpToolProvider:
    """仅为来源服务提供按 Server 权限隔离的只读工具缓存。"""

    def __init__(
        self,
        entries: list[CachedMcpTool],
    ) -> None:
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
            if entry.server_config.enabled
            and has_permission(context.permissions, f"mcp:{entry.server_alias}")
        ]

    def catalog(self) -> list[dict[str, str | bool]]:
        """返回不含 URL、Header 或 Client 对象的安全工具目录。"""
        return [
            {
                "server": entry.server_alias,
                "enabled": entry.server_config.enabled,
                "tool_name": entry.tool.name,
                "remote_tool": entry.remote_tool_name,
                "description": " ".join(entry.tool.description.split())[:240],
            }
            for entry in self._entries
        ]


_provider_lock = threading.Lock()
_provider: McpToolProvider | None = None


def install_mcp_tool_provider(provider: McpToolProvider) -> McpToolProvider:
    """在 discovery 全部成功后原子安装 cache；并发初始化复用首次安装结果。"""
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = provider
        return _provider


def clear_mcp_tool_provider() -> None:
    """清空项目 MCP Tool cache；adapter 当前不持有永久 session。"""
    global _provider
    with _provider_lock:
        _provider = None


def get_mcp_tool_provider() -> McpToolProvider | None:
    """在线程锁保护下读取当前安装的进程级 MCP Provider。"""
    with _provider_lock:
        return _provider


__all__ = [
    "CachedMcpTool",
    "McpToolProvider",
    "clear_mcp_tool_provider",
    "get_mcp_tool_provider",
    "install_mcp_tool_provider",
]
