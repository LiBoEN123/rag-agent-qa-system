"""按统一 MCP Access Policy 提供启动期缓存的 StructuredTool。"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.config import AppSettings, get_settings
from app.tools.context import ToolInvocationContext
from app.tools.mcp.access import can_use_mcp_server
from app.tools.mcp.config import McpServerConfig


@dataclass(frozen=True, slots=True)
class CachedMcpTool:
    """一个已经完成 discovery、安全包装和名称隔离的 MCP Tool。"""

    server_alias: str
    remote_tool_name: str
    tool: BaseTool
    server_config: McpServerConfig

    @property
    def required_permission(self) -> str:
        """返回访问该 Server 工具所需的精确项目权限。"""
        return f"mcp:{self.server_alias}"


class McpToolProvider:
    """只从不可变 cache 中按统一 Server Access Policy 选择工具。"""

    def __init__(
        self,
        entries: list[CachedMcpTool],
        *,
        settings: AppSettings,
    ) -> None:
        """固化 discovery 结果，并拒绝缓存中的重复工具名称。"""
        self._entries = tuple(entries)
        self._settings = settings
        names = [entry.tool.name for entry in self._entries]
        if len(names) != len(set(names)):
            raise ValueError("MCP Tool cache 包含重复名称")

    def list_tools(self, *, context: ToolInvocationContext) -> list[BaseTool]:
        """只返回当前调用上下文具有对应 Server 权限的工具。"""
        return [
            entry.tool
            for entry in self._entries
            if can_use_mcp_server(
                context=context,
                server_alias=entry.server_alias,
                server_config=entry.server_config,
                settings=self._settings,
            )
        ]

    def catalog(self) -> list[dict[str, str | bool]]:
        """返回不含 URL、Header 或 Client 对象的安全工具目录。"""
        return [
            {
                "server": entry.server_alias,
                "enabled": entry.server_config.enabled,
                "enabled_for_default_principal": (
                    entry.server_config.enabled_for_default_principal
                ),
                "tool_name": entry.tool.name,
                "remote_tool": entry.remote_tool_name,
                "risk_level": str(
                    (entry.tool.metadata or {}).get("risk_level") or "read_only"
                ),
                "approval_required": bool(
                    (entry.tool.metadata or {}).get("approval_required", False)
                ),
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


def list_mcp_tools_for_context(
    context: ToolInvocationContext,
    *,
    settings: AppSettings | None = None,
) -> list[BaseTool]:
    """默认关闭时保持空集合；显式启用却未初始化时 fail closed。"""
    provider = get_mcp_tool_provider()
    if provider is not None:
        return provider.list_tools(context=context)
    resolved = settings or get_settings()
    if resolved.mcp_tools_enabled:
        raise RuntimeError("MCP runtime is not initialized")
    return []


def get_mcp_tool_catalog() -> list[dict[str, str | bool]]:
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
