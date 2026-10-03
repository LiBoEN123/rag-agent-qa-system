"""官方 adapter Tool discovery、allowlist、annotation 和冲突校验。"""

from __future__ import annotations

import asyncio
import re

from langchain_core.tools import BaseTool

from app.config import AppSettings
from app.tools.mcp.config import McpConfig
from app.tools.mcp.provider import CachedMcpTool
from app.common.redaction import redact_sensitive_text


_EXPOSED_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_DESCRIPTION_LIMIT = 2000


class McpDiscoveryError(RuntimeError):
    """表示启动期 MCP Tool discovery 必须 fail closed。"""


async def discover_mcp_tools(
    client,
    *,
    config: McpConfig,
    settings: AppSettings,
) -> list[CachedMcpTool]:
    """逐 Server discovery，只缓存精确 allowlist 中的 read-only tool。"""
    cached: list[CachedMcpTool] = []
    exposed_names: set[str] = set()
    for alias, server in config.servers.items():
        if not server.enabled:
            continue
        if len(server.allowed_tools) > settings.mcp_max_tools_per_server:
            raise McpDiscoveryError(
                f"MCP Server {alias!r} 的 allowlist 超过每 Server 工具上限"
            )
        try:
            async with asyncio.timeout(settings.mcp_discovery_timeout_seconds):
                discovered = await client.get_tools(server_name=alias)
        except TimeoutError:
            raise McpDiscoveryError(
                f"MCP Server {alias!r} discovery timed out"
            ) from None
        except Exception:
            raise McpDiscoveryError(f"MCP Server {alias!r} discovery failed") from None

        by_exposed_name: dict[str, BaseTool] = {}
        for raw_tool in discovered:
            if not isinstance(raw_tool, BaseTool):
                raise McpDiscoveryError("MCP adapter 返回了非 LangChain Tool")
            if raw_tool.name in by_exposed_name:
                raise McpDiscoveryError(
                    f"MCP Server {alias!r} discovery 返回重复工具名"
                )
            by_exposed_name[raw_tool.name] = raw_tool

        for remote_name in server.allowed_tools:
            exposed_name = f"{alias}_{remote_name}"
            if not _EXPOSED_TOOL_NAME_PATTERN.fullmatch(exposed_name):
                raise McpDiscoveryError(
                    f"MCP Tool 名称加前缀后不符合项目名称合同: {exposed_name!r}"
                )
            raw_tool = by_exposed_name.get(exposed_name)
            if raw_tool is None:
                raise McpDiscoveryError(
                    f"MCP allowlist 工具不可用: {alias!r}/{remote_name!r}"
                )
            annotations = raw_tool.metadata or {}
            # Firecrawl scrape 的只读参数例外在 interceptor 中强制执行。
            scrape = alias == "firecrawl" and remote_name == "firecrawl_scrape"
            if annotations.get("destructiveHint") is True or (
                annotations.get("readOnlyHint") is False and not scrape
            ):
                raise McpDiscoveryError(
                    f"MCP Tool 明确声明为非只读或 destructive: {exposed_name!r}"
                )
            if not callable(raw_tool.coroutine) or raw_tool.args_schema is None:
                raise McpDiscoveryError(
                    f"MCP Tool 缺少 async coroutine 或 args_schema: {exposed_name!r}"
                )
            if exposed_name in exposed_names:
                raise McpDiscoveryError(
                    f"MCP source Tool 名称冲突: {exposed_name!r}"
                )
            exposed_names.add(exposed_name)
            raw_tool.description = redact_sensitive_text(raw_tool.description)[:_DESCRIPTION_LIMIT]
            raw_tool.metadata = {"mcp_server": alias, "remote_tool_name": remote_name}
            cached.append(CachedMcpTool(
                server_alias=alias, remote_tool_name=remote_name,
                server_config=server, tool=raw_tool,
            ))
    return cached


__all__ = ["McpDiscoveryError", "discover_mcp_tools"]
