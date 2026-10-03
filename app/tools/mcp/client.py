"""通过 MultiServerMCPClient 完成启动期 discovery 并安装唯一 Provider cache。"""

from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.common.logging import get_logger
from app.config import AppSettings
from app.tools.mcp.bridge import McpToolCallInterceptor
from app.tools.mcp.config import build_mcp_connections, load_mcp_config
from app.tools.mcp.discovery import discover_mcp_tools
from app.tools.mcp.provider import (
    McpToolProvider,
    clear_mcp_tool_provider,
    get_mcp_tool_provider,
    install_mcp_tool_provider,
)


logger = get_logger("chat_service")


async def initialize_mcp_runtime(settings: AppSettings) -> McpToolProvider | None:
    """显式启用时读取静态配置并完成一次 fail-closed discovery。"""
    if not settings.mcp_tools_enabled:
        reset_mcp_runtime()
        logger.info("[startup] mcp_tools=disabled")
        return None
    existing = get_mcp_tool_provider()
    if existing is not None:
        return existing
    config = load_mcp_config(settings.mcp_servers_config_path)
    connections = build_mcp_connections(
        config,
        tool_timeout_seconds=settings.mcp_tool_timeout_seconds,
    )
    if not connections:
        provider = install_mcp_tool_provider(McpToolProvider([]))
        logger.info("[startup] mcp_tools=enabled no active servers")
        return provider

    interceptor = McpToolCallInterceptor(
        settings,
        server_configs=config.servers,
    )
    client = MultiServerMCPClient(
        connections,
        tool_interceptors=[interceptor],
        tool_name_prefix=True,
        handle_tool_errors=True,
    )
    entries = await discover_mcp_tools(
        client,
        config=config,
        settings=settings,
    )
    provider = install_mcp_tool_provider(McpToolProvider(entries))
    logger.info(
        "[startup] mcp_tools=enabled servers=%s tools=%s",
        len(connections),
        len(entries),
    )
    return provider


async def close_mcp_runtime() -> None:
    """清理项目 cache；当前 stateless adapter 没有永久 Tool session。"""
    reset_mcp_runtime()


def reset_mcp_runtime() -> None:
    """同步清理 Provider cache，供关闭、评估切换与测试隔离使用。"""
    clear_mcp_tool_provider()


__all__ = [
    "close_mcp_runtime",
    "initialize_mcp_runtime",
    "reset_mcp_runtime",
]
