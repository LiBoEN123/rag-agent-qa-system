"""MCP Tool 的 LangChain 拦截、同步兼容和项目结果归一化。"""

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
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from pydantic import BaseModel, ConfigDict, Field, create_model

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
from app.tools.policy import build_tool_args_fingerprint
from app.tools.result import tool_error, tool_success
from app.tools.mcp.serialization import (
    serialize_mcp_call_result,
    summarize_mcp_output,
)


logger = get_logger("chat_service")
MCP_BRIDGE_VERSION = "1.0.0"


class GovernedMcpStructuredTool(StructuredTool):
    """执行时注入 Runtime，同时向模型保留 adapter 的原始 args schema。"""

    remote_args_schema: Any = Field(exclude=True)

    @property
    def tool_call_schema(self):
        """不把项目 Runtime 字段暴露给模型，也不重写 MCP JSON Schema。"""
        schema = self.remote_args_schema
        if isinstance(schema, dict) and self.description:
            return {**schema, "description": self.description}
        return schema


def _elapsed_ms(started_at: float) -> int:
    """计算从性能计时起点到当前时刻经过的非负毫秒数。"""
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _runtime_context(runtime: object | None) -> ToolInvocationContext | None:
    """从 LangChain Runtime 中安全提取项目工具调用上下文。"""
    context = getattr(runtime, "context", None)
    return context if isinstance(context, ToolInvocationContext) else None


def _runtime_call_id(runtime: object | None) -> str:
    """读取 Runtime 中的 Tool Call ID，缺失时返回空字符串。"""
    value = getattr(runtime, "tool_call_id", "")
    return str(value or "")


def _args_summary(args: dict[str, Any], settings: AppSettings) -> dict[str, Any]:
    """只记录参数字段名和数量，不把 MCP 参数值写入审计。"""
    return {
        "mode": "summary",
        "arg_count": len(args),
        "fields": sorted(str(key) for key in args)[: settings.tool_audit_max_items],
    }


def _result_runtime(
    *,
    started_at: float,
    settings: AppSettings,
    attempt_count: int,
    timed_out: bool,
    error_code: str = "",
) -> dict[str, Any]:
    """构造与本地工具结果字典兼容的单次 MCP 执行信息。"""
    elapsed = _elapsed_ms(started_at)
    attempts = []
    if attempt_count:
        attempts.append(
            {
                "attempt": 1,
                "elapsed_ms": elapsed,
                "error_code": error_code,
                "timed_out": timed_out,
            }
        )
    return {
        "tool_version": MCP_BRIDGE_VERSION,
        "attempt_count": attempt_count,
        "max_attempts": 1,
        "elapsed_ms": elapsed,
        "timeout": settings.mcp_tool_timeout_seconds,
        "timed_out": timed_out,
        "attempts": attempts,
    }


def _governance(
    args_fingerprint: str,
    *,
    allowed: bool,
    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY,
    approval_required: bool = False,
    decision: str = "",
) -> dict[str, Any]:
    """构造 MCP 调用的标准治理判定摘要。"""
    return {
        "decision": decision or ("allow" if allowed else "deny"),
        "risk_level": risk_level.value,
        "approval_required": approval_required,
        "args_fingerprint": args_fingerprint,
        "missing_permissions": [],
    }


def _tool_risk_level(
    server_config: McpServerConfig | None,
    remote_tool_name: str,
) -> ToolRiskLevel:
    """把部署者明确指定的审批工具视为外部写。"""
    if (
        server_config is not None
        and remote_tool_name in server_config.approval_required_tools
    ):
        return ToolRiskLevel.EXTERNAL_WRITE
    return ToolRiskLevel.READ_ONLY


def _standard_result(
    *,
    exposed_tool_name: str,
    args: dict[str, Any],
    settings: AppSettings,
    started_at: float,
    args_fingerprint: str,
    call_id: str,
    output: dict[str, Any] | None = None,
    error: str = "",
    error_code: str = "",
    attempt_count: int = 1,
    timed_out: bool = False,
    allowed: bool = True,
    audit_write_failed: bool = False,
    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY,
    approval_required: bool = False,
    policy_decision: str = "",
) -> dict[str, Any]:
    """把 MCP 成功或失败统一封装为项目标准工具结果字典。"""
    runtime = _result_runtime(
        started_at=started_at,
        settings=settings,
        attempt_count=attempt_count,
        timed_out=timed_out,
        error_code=error_code,
    )
    governance = _governance(
        args_fingerprint,
        allowed=allowed,
        risk_level=risk_level,
        approval_required=approval_required,
        decision=policy_decision,
    )
    if error_code:
        result = tool_error(
            exposed_tool_name,
            args,
            error,
            error_code=error_code,
            retryable=False,
            runtime=runtime,
            governance=governance,
        )
    else:
        result = tool_success(
            exposed_tool_name,
            args,
            output or {},
            runtime=runtime,
            governance=governance,
        )
    result["tool_call_id"] = call_id
    result["audit"] = {"write_failed": audit_write_failed}
    return result


class McpToolCallInterceptor:
    """在官方 adapter 调用边界施加权限、超时、审计和输出安全策略。"""

    def __init__(
        self,
        settings: AppSettings,
        *,
        server_configs: Mapping[str, McpServerConfig],
        audit_sink: ToolAuditSink | None = None,
    ) -> None:
        """保存 MCP 治理设置，并选用注入或项目默认审计接收器。"""
        self._settings = settings
        self._server_configs = dict(server_configs)
        self._audit_sink = audit_sink or get_default_tool_audit_sink(settings)

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
        """写入不含完整参数和输出的审计事件，并返回写入是否成功。"""
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

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler,
    ) -> ToolMessage:
        """对一次远端调用执行权限、审批、审计和输出治理。"""
        started_at = time.perf_counter()
        exposed_name = f"{request.server_name}_{request.name}"
        call_id = _runtime_call_id(request.runtime)
        context = _runtime_context(request.runtime)
        try:
            args_fingerprint = build_tool_args_fingerprint(
                tool_name=exposed_name,
                tool_version=MCP_BRIDGE_VERSION,
                args=request.args,
            )
        except (TypeError, ValueError):
            args_fingerprint = ""

        server_config = self._server_configs.get(request.server_name)
        risk_level = _tool_risk_level(server_config, request.name)
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
            result = _standard_result(
                exposed_tool_name=exposed_name,
                args=request.args,
                settings=self._settings,
                started_at=started_at,
                args_fingerprint=args_fingerprint,
                call_id=call_id,
                error="MCP tool permission denied",
                error_code=ToolErrorCode.MCP_PERMISSION_DENIED.value,
                attempt_count=0,
                allowed=False,
                risk_level=risk_level,
            )
            if context is not None:
                requested_written = self._audit(
                    event_type=ToolAuditEventType.REQUESTED,
                    request=request,
                    context=context,
                    exposed_tool_name=exposed_name,
                    args_fingerprint=args_fingerprint,
                    started_at=started_at,
                    error_code=ToolErrorCode.MCP_PERMISSION_DENIED.value,
                    policy_decision="deny",
                    risk_level=risk_level,
                )
                denied_written = self._audit(
                    event_type=ToolAuditEventType.DENIED,
                    request=request,
                    context=context,
                    exposed_tool_name=exposed_name,
                    args_fingerprint=args_fingerprint,
                    started_at=started_at,
                    error_code=ToolErrorCode.MCP_PERMISSION_DENIED.value,
                    policy_decision="deny",
                    risk_level=risk_level,
                )
                result["audit"]["write_failed"] = not (
                    requested_written and denied_written
                )
            return _result_message(result, exposed_name, call_id)

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
                result = _standard_result(
                    exposed_tool_name=exposed_name,
                    args=request.args,
                    settings=self._settings,
                    started_at=started_at,
                    args_fingerprint=args_fingerprint,
                    call_id=call_id,
                    error="MCP write tool audit is unavailable",
                    error_code=ToolErrorCode.AUDIT_WRITE_FAILED.value,
                    attempt_count=0,
                    allowed=False,
                    audit_write_failed=True,
                    risk_level=risk_level,
                    policy_decision="deny",
                )
                return _result_message(result, exposed_name, call_id)
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
            result = _standard_result(
                exposed_tool_name=exposed_name,
                args=request.args,
                settings=self._settings,
                started_at=started_at,
                args_fingerprint=args_fingerprint,
                call_id=call_id,
                error="MCP external write requires human approval",
                error_code=ToolErrorCode.APPROVAL_REQUIRED.value,
                attempt_count=0,
                allowed=False,
                audit_write_failed=not approval_written,
                risk_level=risk_level,
                approval_required=True,
                policy_decision="require_approval",
            )
            return _result_message(result, exposed_name, call_id)

        requested_written = self._audit(
            event_type=ToolAuditEventType.REQUESTED,
            request=request,
            context=context,
            exposed_tool_name=exposed_name,
            args_fingerprint=args_fingerprint,
            started_at=started_at,
        )
        started_written = self._audit(
            event_type=ToolAuditEventType.STARTED,
            request=request,
            context=context,
            exposed_tool_name=exposed_name,
            args_fingerprint=args_fingerprint,
            started_at=started_at,
            attempt_count=1,
        )
        initial_audit_failed = not (requested_written and started_written)
        try:
            async with asyncio.timeout(self._settings.mcp_tool_timeout_seconds):
                raw_result = await handler(request)
        except TimeoutError:
            error_code = ToolErrorCode.MCP_TOOL_TIMEOUT.value
            result = _standard_result(
                exposed_tool_name=exposed_name,
                args=request.args,
                settings=self._settings,
                started_at=started_at,
                args_fingerprint=args_fingerprint,
                call_id=call_id,
                error="MCP tool timed out",
                error_code=error_code,
                timed_out=True,
                audit_write_failed=initial_audit_failed,
            )
            final_written = self._audit(
                event_type=ToolAuditEventType.TIMED_OUT,
                request=request,
                context=context,
                exposed_tool_name=exposed_name,
                args_fingerprint=args_fingerprint,
                started_at=started_at,
                attempt_count=1,
                error_code=error_code,
            )
        except Exception:
            error_code = ToolErrorCode.MCP_TRANSPORT_ERROR.value
            result = _standard_result(
                exposed_tool_name=exposed_name,
                args=request.args,
                settings=self._settings,
                started_at=started_at,
                args_fingerprint=args_fingerprint,
                call_id=call_id,
                error="MCP service unavailable",
                error_code=error_code,
                audit_write_failed=initial_audit_failed,
            )
            final_written = self._audit(
                event_type=ToolAuditEventType.FAILED,
                request=request,
                context=context,
                exposed_tool_name=exposed_name,
                args_fingerprint=args_fingerprint,
                started_at=started_at,
                attempt_count=1,
                error_code=error_code,
            )
        else:
            outcome = serialize_mcp_call_result(
                raw_result,
                settings=self._settings,
                server_alias=request.server_name,
                remote_tool_name=request.name,
            )
            if outcome.ok:
                result = _standard_result(
                    exposed_tool_name=exposed_name,
                    args=request.args,
                    settings=self._settings,
                    started_at=started_at,
                    args_fingerprint=args_fingerprint,
                    call_id=call_id,
                    output=outcome.output,
                    audit_write_failed=initial_audit_failed,
                )
                final_written = self._audit(
                    event_type=ToolAuditEventType.SUCCEEDED,
                    request=request,
                    context=context,
                    exposed_tool_name=exposed_name,
                    args_fingerprint=args_fingerprint,
                    started_at=started_at,
                    attempt_count=1,
                    success=True,
                    output=outcome.output,
                )
            else:
                result = _standard_result(
                    exposed_tool_name=exposed_name,
                    args=request.args,
                    settings=self._settings,
                    started_at=started_at,
                    args_fingerprint=args_fingerprint,
                    call_id=call_id,
                    error=outcome.error,
                    error_code=outcome.error_code,
                    audit_write_failed=initial_audit_failed,
                )
                final_written = self._audit(
                    event_type=ToolAuditEventType.FAILED,
                    request=request,
                    context=context,
                    exposed_tool_name=exposed_name,
                    args_fingerprint=args_fingerprint,
                    started_at=started_at,
                    attempt_count=1,
                    error_code=outcome.error_code,
                )
        result["audit"]["write_failed"] = bool(
            result["audit"]["write_failed"] or not final_written
        )
        return _result_message(result, exposed_name, call_id)


def _result_message(result: dict[str, Any], name: str, call_id: str) -> ToolMessage:
    """把标准结果序列化为与原调用 ID 配对的 LangChain ToolMessage。"""
    return ToolMessage(
        content=json.dumps(result, ensure_ascii=False, sort_keys=True),
        tool_call_id=call_id,
        name=name,
        status="success" if result.get("ok") is True else "error",
    )


def _parse_adapter_result(
    value: object,
    *,
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """从官方 adapter 的 content_and_artifact 返回值读取项目标准结果。"""
    content = value[0] if isinstance(value, tuple) and len(value) == 2 else value
    if isinstance(content, dict) and "ok" in content:
        return content
    if isinstance(content, ToolMessage) and isinstance(content.content, str):
        try:
            payload = json.loads(content.content)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    return tool_error(
        tool_name,
        args,
        "MCP tool returned an invalid adapter result",
        error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
    )


def _bridge_failure(tool_name: str, args: dict[str, Any], message: str) -> dict:
    """构造未进入远端执行时的安全、稳定 MCP bridge 失败结果。"""
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
            "attempts": [],
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
) -> StructuredTool:
    """为 adapter coroutine 增加项目同步入口和有限 metadata。"""
    exposed_name = f"{server_alias}_{remote_tool_name}"
    raw_coroutine = raw_tool.coroutine
    if not callable(raw_coroutine):
        raise ValueError("MCP adapter tool 缺少异步 coroutine")
    raw_args_schema = raw_tool.args_schema
    if isinstance(raw_args_schema, type) and issubclass(
        raw_args_schema,
        BaseModel,
    ):
        runtime_schema = create_model(
            f"{raw_args_schema.__name__}McpRuntime",
            __base__=raw_args_schema,
            __config__=ConfigDict(arbitrary_types_allowed=True),
            runtime=(ToolRuntime[ToolInvocationContext], ...),
        )
    elif isinstance(raw_args_schema, dict):
        runtime_schema = create_model(
            f"{exposed_name.title().replace('_', '')}McpRuntime",
            __config__=ConfigDict(
                arbitrary_types_allowed=True,
                extra="allow",
            ),
            runtime=(ToolRuntime[ToolInvocationContext], ...),
        )
    else:
        raise ValueError("MCP adapter tool args_schema 类型无效")

    async def ainvoke_mcp(
        runtime: ToolRuntime[ToolInvocationContext],
        **tool_args: Any,
    ) -> dict[str, Any]:
        """通过官方 adapter coroutine 异步执行并解析 MCP Tool 结果。"""
        try:
            value = await raw_coroutine(runtime=runtime, **tool_args)
        except Exception:
            return _bridge_failure(
                exposed_name,
                tool_args,
                "MCP tool execution failed",
            )
        return _parse_adapter_result(
            value,
            tool_name=exposed_name,
            args=tool_args,
        )

    def invoke_mcp(
        runtime: ToolRuntime[ToolInvocationContext],
        **tool_args: Any,
    ) -> dict[str, Any]:
        """为同步 ToolNode 提供不嵌套事件循环的最小异步桥接。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(ainvoke_mcp(runtime=runtime, **tool_args))
        return _bridge_failure(
            exposed_name,
            tool_args,
            "MCP sync bridge cannot run inside an active event loop",
        )

    return GovernedMcpStructuredTool(
        name=exposed_name,
        description=description,
        args_schema=runtime_schema,
        remote_args_schema=raw_args_schema,
        func=invoke_mcp,
        coroutine=ainvoke_mcp,
        metadata={
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
        },
    )


@dataclass(frozen=True, slots=True)
class _DirectRuntime:
    """为冒烟脚本直接调用受治理工具提供最小 Runtime 数据。"""

    context: ToolInvocationContext
    tool_call_id: str


async def invoke_mcp_tool(
    tool: BaseTool,
    args: dict[str, Any],
    *,
    context: ToolInvocationContext,
) -> dict[str, Any]:
    """供冒烟工具通过正式 bridge 调用 cached governed MCP tool。"""
    if (tool.metadata or {}).get("tool_origin") != "mcp":
        return _bridge_failure(tool.name, args, "Configured MCP tool is unavailable")
    coroutine = tool.coroutine
    if not callable(coroutine):
        return _bridge_failure(tool.name, args, "Configured MCP tool is unavailable")
    runtime = _DirectRuntime(context=context, tool_call_id=str(uuid4()))
    try:
        result = await coroutine(runtime=runtime, **args)
    except Exception:
        return _bridge_failure(tool.name, args, "MCP tool execution failed")
    if isinstance(result, dict) and "ok" in result:
        return result
    return _bridge_failure(tool.name, args, "MCP tool returned an invalid result")


__all__ = [
    "MCP_BRIDGE_VERSION",
    "McpToolCallInterceptor",
    "build_governed_mcp_tool",
    "invoke_mcp_tool",
]
