"""ToolSpec 到框架动态工具的薄适配。"""

import json
from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ConfigDict, create_model

from app.agent.guardrails import build_unknown_tool_call_fingerprint
from app.agent.model import get_agent_tool_provider
from app.tools import runtime as tool_runtime
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.errors import ToolErrorCode
from app.tools.mcp.provider import list_mcp_tools_for_context
from app.tools.policy import build_tool_args_fingerprint
from app.tools.providers import ToolProvider
from app.tools.registry import ToolRegistry
from app.tools.result import tool_error


def _provider_registry(provider: ToolProvider) -> ToolRegistry:
    """取得 Provider 所属 Registry，供唯一执行入口复用。"""
    registry = getattr(provider, "registry", None)
    if not isinstance(registry, ToolRegistry):
        raise ValueError("Agent ToolProvider 必须暴露 ToolSpec 所属的 ToolRegistry")
    return registry


def find_last_ai_message(
    messages: list[object], *, required: bool = False
) -> AIMessage | None:
    """返回最近的 AIMessage，并可在缺失时立即失败。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    if required:
        raise RuntimeError("tool governance requires the requesting AIMessage")
    return None


def _tool_request(
    *,
    name: str,
    args: dict[str, Any],
    runtime: ToolRuntime[ToolInvocationContext],
    tool: BaseTool,
) -> ToolCallRequest:
    """根据框架注入 Runtime 构造项目治理需要的调用视图。"""
    return ToolCallRequest(
        tool_call={
            "name": name,
            "args": args,
            "id": runtime.tool_call_id or "",
            "type": "tool_call",
        },
        tool=tool,
        state=runtime.state,
        runtime=runtime,
    )


def _tool_message(result: object, *, name: str, call_id: str) -> ToolMessage:
    """把工具函数结果包装为供治理层解析的配对消息。"""
    return ToolMessage(
        content=json.dumps(result, ensure_ascii=False, sort_keys=True, default=str),
        tool_call_id=call_id,
        name=name,
    )


def _build_tool(
    spec: ToolSpec,
    registry: ToolRegistry,
    governance: Any,
) -> StructuredTool:
    """将单个 ToolSpec 转换为带 Runtime 注入的框架工具。"""
    if spec.args_model is None:
        raise ValueError(f"tool {spec.name!r} 缺少 Pydantic args_model")
    runtime_schema = create_model(
        f"{spec.args_model.__name__}Runtime",
        __base__=spec.args_model,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        runtime=(ToolRuntime[ToolInvocationContext], ...),
    )

    structured_tool: StructuredTool | None = None

    def invoke_registered_tool(
        runtime: ToolRuntime[ToolInvocationContext],
        **tool_args: Any,
    ) -> tuple[str, object]:
        """通过项目唯一执行入口进入 Runtime，不直接接触 handler。"""
        context = runtime.context
        if not isinstance(context, ToolInvocationContext):
            raise RuntimeError("ToolRuntime did not inject ToolInvocationContext")
        assert structured_tool is not None
        request = _tool_request(
            name=spec.name,
            args=tool_args,
            runtime=runtime,
            tool=structured_tool,
        )
        message = governance.govern_tool_call(
            request,
            lambda _request: _tool_message(
                tool_runtime.execute_registered_tool(
                    registry=registry,
                    tool_name=spec.name,
                    tool_args=tool_args,
                    context=context,
                ),
                name=spec.name,
                call_id=runtime.tool_call_id or "",
            ),
        )
        return str(message.content), message.artifact

    async def ainvoke_registered_tool(
        runtime: ToolRuntime[ToolInvocationContext],
        **tool_args: Any,
    ) -> tuple[str, object]:
        """异步 Agent 复用本地工具的受控同步执行边界。"""
        context = runtime.context
        if not isinstance(context, ToolInvocationContext):
            raise RuntimeError("ToolRuntime did not inject ToolInvocationContext")
        assert structured_tool is not None
        request = _tool_request(
            name=spec.name,
            args=tool_args,
            runtime=runtime,
            tool=structured_tool,
        )
        async def execute(_request):
            result = tool_runtime.execute_registered_tool(
                registry=registry,
                tool_name=spec.name,
                tool_args=tool_args,
                context=context,
            )
            return _tool_message(
                result,
                name=spec.name,
                call_id=runtime.tool_call_id or "",
            )

        message = await governance.agovern_tool_call(request, execute)
        return str(message.content), message.artifact

    structured_tool = StructuredTool.from_function(
        func=invoke_registered_tool,
        coroutine=ainvoke_registered_tool,
        name=spec.name,
        description=spec.description,
        args_schema=runtime_schema,
        metadata={"tool_version": spec.version},
        response_format="content_and_artifact",
    )
    return structured_tool


def _build_governed_external_tool(
    tool: BaseTool,
    governance: Any,
) -> StructuredTool:
    """为已带执行治理的 MCP StructuredTool 补充 Agent 循环治理。"""
    raw_func = getattr(tool, "func", None)
    raw_coroutine = getattr(tool, "coroutine", None)
    if not callable(raw_func) or not callable(raw_coroutine):
        raise ValueError(f"external tool {tool.name!r} 缺少同步或异步入口")
    structured_tool: StructuredTool | None = None

    def invoke_external_tool(
        runtime: ToolRuntime[ToolInvocationContext],
        **tool_args: Any,
    ) -> tuple[str, object]:
        assert structured_tool is not None
        request = _tool_request(
            name=tool.name,
            args=tool_args,
            runtime=runtime,
            tool=structured_tool,
        )
        message = governance.govern_tool_call(
            request,
            lambda _request: _tool_message(
                raw_func(runtime=runtime, **tool_args),
                name=tool.name,
                call_id=runtime.tool_call_id or "",
            ),
        )
        return str(message.content), message.artifact

    async def ainvoke_external_tool(
        runtime: ToolRuntime[ToolInvocationContext],
        **tool_args: Any,
    ) -> tuple[str, object]:
        assert structured_tool is not None
        request = _tool_request(
            name=tool.name,
            args=tool_args,
            runtime=runtime,
            tool=structured_tool,
        )

        async def execute(_request):
            result = await raw_coroutine(runtime=runtime, **tool_args)
            return _tool_message(
                result,
                name=tool.name,
                call_id=runtime.tool_call_id or "",
            )

        message = await governance.agovern_tool_call(request, execute)
        return str(message.content), message.artifact

    structured_tool = StructuredTool.from_function(
        func=invoke_external_tool,
        coroutine=ainvoke_external_tool,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        metadata=dict(tool.metadata or {}),
        response_format="content_and_artifact",
    )
    return structured_tool


def build_framework_tools(
    provider: ToolProvider,
    *,
    context: ToolInvocationContext,
    governance: Any | None = None,
) -> list[BaseTool]:
    """按本次请求 Provider 构造供 create_agent 使用的动态工具。"""
    if governance is None:
        from app.agent.framework_middleware import (
            ProjectToolGovernanceMiddleware,
        )

        governance = ProjectToolGovernanceMiddleware()
    registry = _provider_registry(provider)
    specs: Sequence[ToolSpec] = provider.list_specs(context=context)
    tools: list[BaseTool] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, ToolSpec) or registry.get(spec.name) is not spec:
            raise ValueError("Provider ToolSpec 必须属于其声明的 Registry")
        if spec.name in seen:
            raise ValueError(f"tool {spec.name!r} 重复，禁止静默覆盖")
        seen.add(spec.name)
        tools.append(_build_tool(spec, registry, governance))
    return tools


def build_agent_framework_tools(
    provider: ToolProvider,
    *,
    context: ToolInvocationContext,
    governance: Any,
) -> list[BaseTool]:
    """合并本地 Registry tools 与启动期缓存的受治理 MCP tools。"""
    tools = build_framework_tools(
        provider,
        context=context,
        governance=governance,
    )
    seen = {tool.name for tool in tools}
    for tool in list_mcp_tools_for_context(context):
        if tool.name in seen:
            raise ValueError(f"tool {tool.name!r} 在本地与 MCP 工具源中重复")
        seen.add(tool.name)
        tools.append(_build_governed_external_tool(tool, governance))
    return tools


def resolve_agent_tools(
    context: ToolInvocationContext,
    *,
    governance: Any,
) -> list[BaseTool]:
    """从唯一 Provider 入口解析本次请求可见的完整工具集合。"""
    return build_agent_framework_tools(
        get_agent_tool_provider(),
        context=context,
        governance=governance,
    )


def resolve_tool_message(output: object, call_id: str) -> ToolMessage:
    """读取治理包装器生成且与调用 ID 匹配的普通 ToolMessage。"""
    if not isinstance(output, ToolMessage) or output.tool_call_id != call_id:
        raise TypeError("governed tool must return its paired ToolMessage")
    return output


def parse_tool_result(
    message: ToolMessage,
    *,
    known: bool,
    name: str,
    args: dict,
) -> dict[str, Any]:
    """读取标准 ToolMessage；框架校验错误映射为项目稳定错误码。"""
    try:
        value = (
            json.loads(message.content) if isinstance(message.content, str) else None
        )
    except (TypeError, ValueError):
        value = None
    if isinstance(value, dict) and "ok" in value:
        return value
    code = ToolErrorCode.INVALID_ARGUMENTS if known else ToolErrorCode.UNKNOWN_TOOL
    return tool_error(name, args, str(message.content), error_code=code.value)


def tool_call_fingerprint(request: ToolCallRequest, args: dict) -> str:
    """注册工具复用 Policy 指纹；未知工具仅处理模型 JSON。"""
    metadata = request.tool.metadata if request.tool is not None else None
    version = str((metadata or {}).get("tool_version") or "")
    if version:
        try:
            return build_tool_args_fingerprint(
                tool_name=request.tool_call["name"],
                tool_version=version,
                args=args,
            )
        except (TypeError, ValueError):
            pass
    return build_unknown_tool_call_fingerprint(
        tool_name=request.tool_call["name"], tool_args=args
    )


def governance_block(
    request: ToolCallRequest,
    reason: str,
    fingerprint: str,
    *,
    aborted: bool,
) -> tuple[ToolMessage, dict[str, Any]]:
    """为未执行的同批调用补齐唯一的配对 ToolMessage。"""
    code = "agent_batch_aborted" if aborted else reason
    args = request.tool_call.get("args") or {}
    result = tool_error(
        request.tool_call["name"],
        args,
        f"Agent governance stopped: {reason}",
        error_code=code,
    )
    result.update(
        {
            "handler_executed": False,
            "stop_reason": reason,
            "error_source": "agent_guardrail",
        }
    )
    message = ToolMessage(
        content=json.dumps(result, ensure_ascii=False, sort_keys=True),
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        status="error",
    )
    event = {
        "kind": "tool",
        "name": request.tool_call["name"],
        "args": args,
        "result": result,
        "ok": False,
        "counted_call": False,
        "call_fingerprint": fingerprint,
        "error_code": code,
        "stop_reason": reason,
        "cost_ms": 0,
    }
    return message, event
