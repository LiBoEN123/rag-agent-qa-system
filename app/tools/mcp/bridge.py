"""MCP Tool 的项目治理、薄同步 facade 与直接调用入口。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
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
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolRiskLevel
from app.tools.errors import ToolErrorCode
from app.tools.mcp.access import can_use_mcp_server
from app.tools.mcp.config import McpServerConfig
from app.tools.mcp.serialization import (
    McpSerializationProfile,
    attach_mcp_execution_metadata,
    normalize_mcp_tool_message,
    sanitize_mcp_call_result,
    summarize_mcp_output,
)
from app.tools.policy import build_tool_args_fingerprint
from app.tools.result import tool_error


logger = get_logger("chat_service")
MCP_BRIDGE_VERSION = "1.0.0"


# 计算从指定计时起点经过的毫秒数，并保证结果非负。
def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


# 从框架运行时提取项目工具调用上下文，类型不匹配时返回空值。
def _runtime_context(runtime: object | None) -> ToolInvocationContext | None:
    context = getattr(runtime, "context", None)
    return context if isinstance(context, ToolInvocationContext) else None


# 读取运行时工具调用标识，并将缺失值统一为空字符串。
def _runtime_call_id(runtime: object | None) -> str:
    return str(getattr(runtime, "tool_call_id", "") or "")


# 生成仅包含参数数量和限量字段名的审计摘要。
def _args_summary(args: dict[str, Any], settings: AppSettings) -> dict[str, Any]:
    return {
        "mode": "summary",
        "arg_count": len(args),
        "fields": sorted(str(key) for key in args)[: settings.tool_audit_max_items],
    }


# 构造 MCP 调用的桥接版本、尝试次数、耗时和超时元数据。
def _result_runtime(
    *,
    started_at: float,
    settings: AppSettings,
    attempt_count: int,
    timed_out: bool,
) -> dict[str, Any]:
    return {
        "tool_version": MCP_BRIDGE_VERSION,
        "attempt_count": attempt_count,
        "max_attempts": 1,
        "elapsed_ms": _elapsed_ms(started_at),
        "timeout": settings.mcp_tool_timeout_seconds,
        "timed_out": timed_out,
    }


# 构造 MCP 调用的准入决定、风险级别、审批要求和参数指纹。
def _governance(
    args_fingerprint: str,
    *,
    allowed: bool,
    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY,
    approval_required: bool = False,
    decision: str = "",
) -> dict[str, Any]:
    return {
        "decision": decision or ("allow" if allowed else "deny"),
        "risk_level": risk_level.value,
        "approval_required": approval_required,
        "args_fingerprint": args_fingerprint,
        "missing_permissions": [],
    }


# 按服务配置将需审批的远程工具归为外部写入，其余归为只读。
def _tool_risk_level(
    server_config: McpServerConfig | None,
    remote_tool_name: str,
) -> ToolRiskLevel:
    if (
        server_config is not None
        and remote_tool_name in server_config.approval_required_tools
    ):
        return ToolRiskLevel.EXTERNAL_WRITE
    return ToolRiskLevel.READ_ONLY


# 使用错误文本构造标记为失败的原生 MCP 工具结果。
def _error_result(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        isError=True,
    )


class McpToolCallInterceptor:
    """在官方 adapter 转换前施加权限、审计、超时与 payload 安全策略。"""

    # 保存运行配置、各 MCP 服务配置和调用审计写入器。
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

    # 写入带调用身份、参数及输出摘要的 MCP 审计事件，并返回写入状态。
    def _audit(
        self,
        *,
        event_type: ToolAuditEventType,
        request: MCPToolCallRequest,
        context: ToolInvocationContext,
        exposed_tool_name: str,
        args_fingerprint: str,
        started_at: float,
        attempt_count: int = 0,
        success: bool = False,
        error_code: str = "",
        output: object = None,
        policy_decision: str = "allow",
        risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY,
    ) -> bool:
        event = ToolAuditEvent(
            timestamp=utc_timestamp(),
            event_type=event_type,
            tool_call_id=_runtime_call_id(request.runtime),
            trace_id=context.trace_id,
            session_id=context.session_id,
            principal_id=context.principal_id,
            principal_type=context.principal_type.value,
            source=context.source,
            tool_name=exposed_tool_name,
            tool_version=MCP_BRIDGE_VERSION,
            risk_level=risk_level.value,
            policy_decision=policy_decision,
            args_fingerprint=args_fingerprint,
            attempt_count=attempt_count,
            elapsed_ms=_elapsed_ms(started_at),
            success=success,
            error_code=error_code,
            error_message=("" if not error_code else "MCP tool call failed"),
            sanitized_input=_args_summary(request.args, self._settings),
            sanitized_output_summary=summarize_mcp_output(output),
            server_alias=request.server_name,
        )
        try:
            self._audit_sink.write(event)
            return True
        except Exception as exc:
            logger.error(
                "MCP tool audit write failed tool=%s event=%s error_type=%s",
                exposed_tool_name,
                event_type.value,
                type(exc).__name__,
            )
            return False

    # 为 MCP 结果附加运行、治理、审计、错误信息和不可信内容标记。
    def _response(
        self,
        result: types.CallToolResult,
        *,
        request: MCPToolCallRequest,
        started_at: float,
        args_fingerprint: str,
        risk_level: ToolRiskLevel,
        attempt_count: int,
        audit_write_failed: bool,
        truncated: bool = False,
        error: str = "",
        error_code: str = "",
        timed_out: bool = False,
        allowed: bool = True,
        approval_required: bool = False,
        decision: str = "",
    ) -> types.CallToolResult:
        metadata = {
            "runtime": _result_runtime(
                started_at=started_at,
                settings=self._settings,
                attempt_count=attempt_count,
                timed_out=timed_out,
            ),
            "governance": _governance(
                args_fingerprint,
                allowed=allowed,
                risk_level=risk_level,
                approval_required=approval_required,
                decision=decision,
            ),
            "audit": {"write_failed": audit_write_failed},
            "output": {
                "content_is_untrusted": True,
                "server": request.server_name,
                "tool": request.name,
                "truncated": truncated,
            },
            "error": error,
            "error_code": error_code,
        }
        return attach_mcp_execution_metadata(result, metadata)

    # 检查权限和审批要求，在超时限制内调用远程工具并清理结果、记录审计。
    async def __call__(self, request: MCPToolCallRequest, handler):
        started_at = time.perf_counter()
        exposed_name = f"{request.server_name}_{request.name}"
        context = _runtime_context(request.runtime)
        server_config = self._server_configs.get(request.server_name)
        risk_level = _tool_risk_level(server_config, request.name)
        try:
            args_fingerprint = build_tool_args_fingerprint(
                tool_name=exposed_name,
                tool_version=MCP_BRIDGE_VERSION,
                args=request.args,
            )
        except (TypeError, ValueError):
            args_fingerprint = ""

        if (
            context is None
            or server_config is None
            or not can_use_mcp_server(
                context=context,
                server_alias=request.server_name,
                server_config=server_config,
                settings=self._settings,
            )
        ):
            audit_failed = False
            if context is not None:
                audit_writes = [
                    self._audit(
                        event_type=event_type,
                        request=request,
                        context=context,
                        exposed_tool_name=exposed_name,
                        args_fingerprint=args_fingerprint,
                        started_at=started_at,
                        error_code=ToolErrorCode.MCP_PERMISSION_DENIED.value,
                        policy_decision="deny",
                        risk_level=risk_level,
                    )
                    for event_type in (
                        ToolAuditEventType.REQUESTED,
                        ToolAuditEventType.DENIED,
                    )
                ]
                audit_failed = not all(audit_writes)
            return self._response(
                _error_result("MCP tool permission denied"),
                request=request,
                started_at=started_at,
                args_fingerprint=args_fingerprint,
                risk_level=risk_level,
                attempt_count=0,
                audit_write_failed=audit_failed,
                error="MCP tool permission denied",
                error_code=ToolErrorCode.MCP_PERMISSION_DENIED.value,
                allowed=False,
            )

        if risk_level is ToolRiskLevel.EXTERNAL_WRITE:
            requested_written = self._audit(
                event_type=ToolAuditEventType.REQUESTED,
                request=request,
                context=context,
                exposed_tool_name=exposed_name,
                args_fingerprint=args_fingerprint,
                started_at=started_at,
                policy_decision="require_approval",
                risk_level=risk_level,
            )
            if not requested_written:
                return self._response(
                    _error_result("MCP write tool audit is unavailable"),
                    request=request,
                    started_at=started_at,
                    args_fingerprint=args_fingerprint,
                    risk_level=risk_level,
                    attempt_count=0,
                    audit_write_failed=True,
                    error="MCP write tool audit is unavailable",
                    error_code=ToolErrorCode.AUDIT_WRITE_FAILED.value,
                    allowed=False,
                    decision="deny",
                )
            approval_written = self._audit(
                event_type=ToolAuditEventType.APPROVAL_REQUIRED,
                request=request,
                context=context,
                exposed_tool_name=exposed_name,
                args_fingerprint=args_fingerprint,
                started_at=started_at,
                error_code=ToolErrorCode.APPROVAL_REQUIRED.value,
                policy_decision="require_approval",
                risk_level=risk_level,
            )
            return self._response(
                _error_result("MCP external write requires human approval"),
                request=request,
                started_at=started_at,
                args_fingerprint=args_fingerprint,
                risk_level=risk_level,
                attempt_count=0,
                audit_write_failed=not approval_written,
                error="MCP external write requires human approval",
                error_code=ToolErrorCode.APPROVAL_REQUIRED.value,
                allowed=False,
                approval_required=True,
                decision="require_approval",
            )

        requested_written = self._audit(
            event_type=ToolAuditEventType.REQUESTED,
            request=request,
            context=context,
            exposed_tool_name=exposed_name,
            args_fingerprint=args_fingerprint,
            started_at=started_at,
            risk_level=risk_level,
        )
        started_written = self._audit(
            event_type=ToolAuditEventType.STARTED,
            request=request,
            context=context,
            exposed_tool_name=exposed_name,
            args_fingerprint=args_fingerprint,
            started_at=started_at,
            attempt_count=1,
            risk_level=risk_level,
        )
        initial_audit_failed = not (requested_written and started_written)
        error = ""
        error_code = ""
        timed_out = False
        truncated = False
        audit_output = None
        try:
            async with asyncio.timeout(self._settings.mcp_tool_timeout_seconds):
                raw_result = await handler(request)
        except TimeoutError:
            error = "MCP tool timed out"
            error_code = ToolErrorCode.MCP_TOOL_TIMEOUT.value
            timed_out = True
            safe_result = _error_result(error)
            event_type = ToolAuditEventType.TIMED_OUT
        except Exception:
            error = "MCP service unavailable"
            error_code = ToolErrorCode.MCP_TRANSPORT_ERROR.value
            safe_result = _error_result(error)
            event_type = ToolAuditEventType.FAILED
        else:
            profile = getattr(
                request.runtime,
                "mcp_serialization_profile",
                McpSerializationProfile.AGENT,
            )
            outcome = sanitize_mcp_call_result(
                raw_result,
                settings=self._settings,
                server_alias=request.server_name,
                remote_tool_name=request.name,
                profile=profile,
            )
            if not outcome.ok or outcome.result is None:
                error = outcome.error
                error_code = outcome.error_code
                safe_result = _error_result(error)
                event_type = ToolAuditEventType.FAILED
            else:
                safe_result = outcome.result
                truncated = outcome.truncated
                audit_output = outcome.ingestion_output or {
                    "content": outcome.text,
                    "content_is_untrusted": True,
                    "truncated": outcome.truncated,
                }
                if safe_result.isError:
                    error = "MCP tool execution failed"
                    if outcome.text.strip():
                        error = f"{error}: {outcome.text}"
                    limit = min(
                        self._settings.mcp_max_text_chars,
                        2_000,
                    )
                    if len(error) > limit:
                        error = error[: max(0, limit - 3)] + "..."
                    error_code = ToolErrorCode.MCP_EXECUTION_ERROR.value
                    event_type = ToolAuditEventType.FAILED
                else:
                    event_type = ToolAuditEventType.SUCCEEDED
                    if outcome.ingestion_output is not None and hasattr(
                        request.runtime, "mcp_ingestion_output"
                    ):
                        request.runtime.mcp_ingestion_output = outcome.ingestion_output

        final_written = self._audit(
            event_type=event_type,
            request=request,
            context=context,
            exposed_tool_name=exposed_name,
            args_fingerprint=args_fingerprint,
            started_at=started_at,
            attempt_count=1,
            success=not error_code,
            error_code=error_code,
            output=audit_output,
            risk_level=risk_level,
        )
        return self._response(
            safe_result,
            request=request,
            started_at=started_at,
            args_fingerprint=args_fingerprint,
            risk_level=risk_level,
            attempt_count=1,
            audit_write_failed=bool(initial_audit_failed or not final_written),
            truncated=truncated,
            error=error,
            error_code=error_code,
            timed_out=timed_out,
        )


# 为桥接层调用失败构造未执行远程工具的统一错误结果。
def _bridge_failure(tool_name: str, args: dict[str, Any], message: str) -> dict:
    result = tool_error(
        tool_name,
        args,
        message,
        error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        runtime={
            "tool_version": MCP_BRIDGE_VERSION,
            "attempt_count": 0,
            "max_attempts": 1,
            "elapsed_ms": 0,
            "timeout": None,
            "timed_out": False,
        },
        governance=_governance("", allowed=False),
    )
    result["audit"] = {"write_failed": False}
    result["tool_call_id"] = ""
    return result


def build_governed_mcp_tool(
    raw_tool: BaseTool,
    *,
    server_alias: str,
    remote_tool_name: str,
    description: str,
    approval_required: bool = False,
) -> BaseTool:
    """在官方 tool 本身只补充同步入口与项目 metadata。"""
    exposed_name = f"{server_alias}_{remote_tool_name}"
    raw_coroutine = raw_tool.coroutine
    if raw_tool.name != exposed_name or not callable(raw_coroutine):
        raise ValueError("MCP adapter tool 名称或 coroutine 无效")
    if not hasattr(raw_tool, "func"):
        raise ValueError("MCP adapter tool 不支持同步 facade")

    # 同步执行 MCP 协程，已有事件循环时返回桥接错误。
    def invoke_mcp(
        runtime: ToolRuntime[ToolInvocationContext],
        **tool_args: Any,
    ):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(raw_coroutine(runtime=runtime, **tool_args))
        failure = _bridge_failure(
            exposed_name,
            tool_args,
            "MCP sync bridge cannot run inside an active event loop",
        )
        return json.dumps(failure, ensure_ascii=False, sort_keys=True), None

    raw_tool.description = description
    raw_tool.func = invoke_mcp
    raw_tool.metadata = {
        "tool_origin": "mcp",
        "mcp_server": server_alias,
        "remote_tool_name": remote_tool_name,
        "tool_version": MCP_BRIDGE_VERSION,
        "risk_level": (
            ToolRiskLevel.EXTERNAL_WRITE.value
            if approval_required
            else ToolRiskLevel.READ_ONLY.value
        ),
        "approval_required": approval_required,
    }
    return raw_tool


# 为直接调用携带工具上下文、调用标识、序列化模式及入库输出。
@dataclass(slots=True)
class _DirectRuntime:
    context: ToolInvocationContext
    tool_call_id: str
    mcp_serialization_profile: McpSerializationProfile
    mcp_ingestion_output: dict[str, Any] | None = None


async def invoke_mcp_tool(
    tool: BaseTool,
    args: dict[str, Any],
    *,
    context: ToolInvocationContext,
    serialization_profile: McpSerializationProfile = McpSerializationProfile.AGENT,
) -> dict[str, Any]:
    """通过公开 BaseTool API 直接调用 cached governed MCP tool。"""
    metadata = tool.metadata or {}
    if metadata.get("tool_origin") != "mcp":
        return _bridge_failure(tool.name, args, "Configured MCP tool is unavailable")
    try:
        profile = McpSerializationProfile(serialization_profile)
    except (TypeError, ValueError):
        return _bridge_failure(tool.name, args, "MCP serialization profile is invalid")
    if (
        profile is McpSerializationProfile.INGESTION
        and metadata.get("risk_level") != ToolRiskLevel.READ_ONLY.value
    ):
        return _bridge_failure(
            tool.name, args, "MCP ingestion requires a read-only tool"
        )

    call_id = str(uuid4())
    runtime = _DirectRuntime(context, call_id, profile)
    tool_call = {
        "type": "tool_call",
        "name": tool.name,
        "args": {**args, "runtime": runtime},
        "id": call_id,
    }
    try:
        message = await tool.ainvoke(tool_call)
    except Exception:
        return _bridge_failure(tool.name, args, "MCP tool execution failed")
    if not isinstance(message, ToolMessage):
        return _bridge_failure(tool.name, args, "MCP tool returned an invalid result")
    result = normalize_mcp_tool_message(message, tool_name=tool.name, args=args)
    if result is None:
        return _bridge_failure(tool.name, args, "MCP tool returned an invalid result")
    if profile is McpSerializationProfile.INGESTION and result.get("ok") is True:
        if runtime.mcp_ingestion_output is None:
            return _bridge_failure(
                tool.name, args, "MCP ingestion metadata is unavailable"
            )
        result["output"] = runtime.mcp_ingestion_output
    return result
