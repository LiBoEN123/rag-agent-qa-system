"""MultiServerMCPClient 的启动期 discovery cache 与关闭生命周期。"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.common.logging import get_logger
from app.config import AppSettings
from app.tools.mcp.bridge import McpToolCallInterceptor
from app.tools.mcp.config import build_mcp_connections, load_mcp_config
from app.tools.mcp.discovery import discover_mcp_tools
from app.tools.mcp.provider import (
    McpToolProvider,
    clear_mcp_tool_provider,
    install_mcp_tool_provider,
)
from app.tools.registry import get_tool_registry


logger = get_logger("chat_service")


@dataclass(frozen=True, slots=True)
class McpRuntime:
    """一个应用实例复用的 adapter client 和不可变 Provider cache。"""

    client: MultiServerMCPClient | None
    provider: McpToolProvider


_runtime_lock = threading.Lock()
_runtime: McpRuntime | None = None


def get_mcp_runtime() -> McpRuntime | None:
    """在线程锁保护下读取当前应用实例的 MCP Runtime。"""
    with _runtime_lock:
        return _runtime


async def initialize_mcp_runtime(settings: AppSettings) -> McpRuntime | None:
    """显式启用时读取静态配置并完成一次 fail-closed discovery。"""
    global _runtime
    if not settings.mcp_tools_enabled:
        reset_mcp_runtime()
        logger.info("[startup] mcp_tools=disabled")
        return None
    existing = get_mcp_runtime()
    if existing is not None:
        return existing
    config = load_mcp_config(settings.mcp_servers_config_path)
    connections = build_mcp_connections(
        config,
        tool_timeout_seconds=settings.mcp_tool_timeout_seconds,
    )
    if not connections:
        provider = McpToolProvider([], settings=settings)
        runtime = _install_runtime(
            McpRuntime(client=None, provider=provider),
        )
        logger.info("[startup] mcp_tools=enabled no active servers")
        return runtime

    interceptor = McpToolCallInterceptor(
        settings,
        server_configs=config.servers,
    )
    client = MultiServerMCPClient(
        connections,
        tool_interceptors=[interceptor],
        tool_name_prefix=True,
        handle_tool_errors=False,
    )
    entries = await discover_mcp_tools(
        client,
        config=config,
        settings=settings,
        local_tool_names=set(get_tool_registry().names()),
    )
    provider = McpToolProvider(entries, settings=settings)
    runtime = _install_runtime(McpRuntime(client=client, provider=provider))
    logger.info(
        "[startup] mcp_tools=enabled servers=%s tools=%s",
        len(connections),
        len(entries),
    )
    return runtime


def _install_runtime(candidate: McpRuntime) -> McpRuntime:
    """原子安装 Runtime 与匹配 Provider，避免并发初始化时 cache 不一致。"""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = candidate
        else:
            candidate = _runtime
    install_mcp_tool_provider(candidate.provider)
    return candidate


async def close_mcp_runtime() -> None:
    """清理项目 cache；当前 stateless adapter 没有永久 Tool session。"""
    reset_mcp_runtime()


def reset_mcp_runtime() -> None:
    """同步清理无 session 的项目 MCP Runtime，供关闭与测试隔离使用。"""
    global _runtime
    with _runtime_lock:
        _runtime = None
    clear_mcp_tool_provider()


__all__ = [
    "McpRuntime",
    "close_mcp_runtime",
    "get_mcp_runtime",
    "initialize_mcp_runtime",
    "reset_mcp_runtime",
]
