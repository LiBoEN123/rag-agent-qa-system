"""MCP Server 的统一请求级访问策略。"""

from __future__ import annotations

from app.config import AppSettings
from app.tools.context import PrincipalType, ToolInvocationContext, has_permission
from app.tools.mcp.config import McpServerConfig


def can_use_mcp_server(
    *,
    context: ToolInvocationContext,
    server_alias: str,
    server_config: McpServerConfig,
    settings: AppSettings,
) -> bool:
    """集中决定一个请求主体能否看到并执行指定 MCP Server 的工具。"""
    if not server_config.enabled:
        return False
    if has_permission(context.permissions, f"mcp:{server_alias}"):
        return True
    if (
        context.principal_id == settings.tool_default_principal_id
        and context.principal_type == PrincipalType.SERVICE
        and server_config.enabled_for_default_principal
    ):
        return True
    return False


__all__ = ["can_use_mcp_server"]
