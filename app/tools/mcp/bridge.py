"""入库来源 MCP 调用：精确授权、超时、结果净化与摘要审计。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp import types

from app.common.logging import get_logger
from app.config import AppSettings
from app.tools.audit import (
    ToolAuditEvent,
    ToolAuditEventType,
    ToolAuditSink,
    get_default_tool_audit_sink,
    utc_timestamp,
)
from app.tools.context import ToolInvocationContext, has_permission
from app.tools.errors import ToolErrorCode
from app.tools.mcp.config import McpServerConfig
from app.tools.mcp.serialization import sanitize_mcp_call_result, summarize_mcp_output
from app.tools.policy import build_tool_args_fingerprint
from app.tools.result import tool_error, tool_result


logger = get_logger("chat_service")
MCP_BRIDGE_VERSION = "1.0.0"
_FIRECRAWL_READ_ONLY_ARGS = frozenset(
    {
        "url",
        "formats",
        "onlyMainContent",
        "removeBase64Images",
    }
)


@dataclass(slots=True)
class _DirectRuntime:
    """一次来源调用的上下文和净化结果；不经过 Agent 消息协议。"""

    context: ToolInvocationContext
    tool_call_id: str
    result: dict[str, Any] | None = None


class McpToolCallInterceptor:
    """在官方 adapter 转换前执行来源授权、审计、超时和 payload 安全策略。"""

    def __init__(
        self,
        settings: AppSettings,
        *,
        server_configs: Mapping[str, McpServerConfig],
        audit_sink: ToolAuditSink | None = None,
    ) -> None:
        self._settings = settings
        self._server_configs = dict(server_configs)
        self._audit_sink = audit_sink or get_default_tool_audit_sink(settings)

    async def __call__(self, request: MCPToolCallRequest, handler):
        started = time.perf_counter()
        runtime = request.runtime
        if not isinstance(runtime, _DirectRuntime):
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text="MCP source context required")
                ],
                isError=True,
            )
        context = runtime.context
        name = f"{request.server_name}_{request.name}"
        server = self._server_configs.get(request.server_name)
        fingerprint = ""
        code = ""
        error = ""
        output = None
        attempts = 0
        audit_failed = False
        try:
            fingerprint = build_tool_args_fingerprint(
                tool_name=name,
                tool_version=MCP_BRIDGE_VERSION,
                args=request.args,
            )
        except (TypeError, ValueError):
            code, error = (
                ToolErrorCode.INVALID_ARGUMENTS.value,
                "Invalid MCP source arguments",
            )
        if (
            server is None
            or not server.enabled
            or not has_permission(
                context.permissions,
                f"mcp:{request.server_name}",
            )
        ):
            code, error = (
                ToolErrorCode.MCP_PERMISSION_DENIED.value,
                "MCP source permission denied",
            )
        elif request.name not in server.allowed_tools:
            code, error = (
                ToolErrorCode.MCP_TOOL_NOT_ALLOWED.value,
                "MCP source tool is not allowed",
            )
        elif (
            request.server_name == "firecrawl"
            and request.name == "firecrawl_scrape"
            and not set(request.args).issubset(_FIRECRAWL_READ_ONLY_ARGS)
        ):
            code, error = (
                ToolErrorCode.INVALID_ARGUMENTS.value,
                "Firecrawl scrape only allows read-only URL fetching",
            )
        decision = "deny" if code else "allow"

        def audit(event_type: ToolAuditEventType) -> None:
            nonlocal audit_failed
            try:
                self._audit_sink.write(
                    ToolAuditEvent(
                        timestamp=utc_timestamp(),
                        event_type=event_type,
                        tool_call_id=runtime.tool_call_id,
                        trace_id=context.trace_id,
                        session_id=context.session_id,
                        principal_id=context.principal_id,
                        principal_type=context.principal_type.value,
                        source=context.source,
                        tool_name=name,
                        tool_version=MCP_BRIDGE_VERSION,
                        risk_level="read_only",
                        policy_decision=decision,
                        args_fingerprint=fingerprint,
                        attempt_count=attempts,
                        elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
                        success=event_type is ToolAuditEventType.SUCCEEDED,
                        error_code=code,
                        error_message="MCP source fetch failed" if code else "",
                        sanitized_input={
                            "mode": "summary",
                            "arg_count": len(request.args),
                            "fields": sorted(str(key) for key in request.args)[
                                : self._settings.tool_audit_max_items
                            ],
                        },
                        sanitized_output_summary=summarize_mcp_output(output),
                        server_alias=request.server_name,
                    )
                )
            except Exception as exc:
                audit_failed = True
                logger.error(
                    "MCP source audit write failed tool=%s event=%s error_type=%s",
                    name,
                    event_type.value,
                    type(exc).__name__,
                )

        audit(ToolAuditEventType.REQUESTED)
        safe_result = None
        if code:
            audit(ToolAuditEventType.DENIED)
        else:
            attempts = 1
            audit(ToolAuditEventType.STARTED)
            try:
                async with asyncio.timeout(self._settings.mcp_tool_timeout_seconds):
                    raw_result = await handler(request)
            except TimeoutError:
                code, error = (
                    ToolErrorCode.MCP_TOOL_TIMEOUT.value,
                    "MCP source timed out",
                )
            except Exception:
                code, error = (
                    ToolErrorCode.MCP_TRANSPORT_ERROR.value,
                    "MCP service unavailable",
                )
            else:
                outcome = sanitize_mcp_call_result(
                    raw_result,
                    server_alias=request.server_name,
                    remote_tool_name=request.name,
                )
                if not outcome.ok or outcome.result is None:
                    code, error = outcome.error_code, outcome.error
                else:
                    safe_result = outcome.result
                    output = outcome.ingestion_output
                    if safe_result.isError:
                        code = ToolErrorCode.MCP_EXECUTION_ERROR.value
                        error = "MCP tool execution failed"
                        if outcome.text.strip():
                            error = f"{error}: {outcome.text}"[:2_000]
            audit(
                ToolAuditEventType.TIMED_OUT
                if code == ToolErrorCode.MCP_TOOL_TIMEOUT.value
                else ToolAuditEventType.FAILED
                if code
                else ToolAuditEventType.SUCCEEDED
            )
        runtime.result = tool_result(
            name,
            request.args,
            output if not code else None,
            error=error,
            error_code=code,
            tool_call_id=runtime.tool_call_id,
            audit={"write_failed": audit_failed},
        )
        return (
            safe_result
            if safe_result is not None
            else types.CallToolResult(
                content=[types.TextContent(type="text", text=error)],
                isError=True,
            )
        )


async def invoke_mcp_tool(
    tool: BaseTool,
    args: dict[str, Any],
    *,
    context: ToolInvocationContext,
) -> dict[str, Any]:
    """通过官方 BaseTool 异步入口调用来源工具，返回 interceptor 的安全结果。"""
    metadata = tool.metadata or {}
    server = metadata.get("mcp_server")
    remote = metadata.get("remote_tool_name")
    if not server or not remote or tool.name != f"{server}_{remote}":
        return tool_error(
            tool.name,
            args,
            "Configured MCP source tool is unavailable",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )
    runtime = _DirectRuntime(context, str(uuid4()))
    try:
        await tool.ainvoke({**args, "runtime": runtime})
    except Exception:
        return tool_error(
            tool.name,
            args,
            "MCP source tool execution failed",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )
    if runtime.result is None:
        return tool_error(
            tool.name,
            args,
            "MCP source result is unavailable",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )
    return runtime.result
