"""ToolSpec 到框架动态工具的薄适配。"""

import json
from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from pydantic import ConfigDict, create_model

from app.agent.guardrails import build_unknown_tool_call_fingerprint
from app.agent.model import get_agent_tool_provider
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.errors import ToolErrorCode
from app.tools.policy import build_tool_args_fingerprint
from app.tools.providers import ToolProvider
from app.tools.registry import ToolRegistry
from app.tools.result import tool_error


def _provider_registry(provider: ToolProvider) -> ToolRegistry:
    """取得 Provider 所属 Registry，并拒绝绕过统一分发。"""
    registry = getattr(provider, "registry", None)
    if not isinstance(registry, ToolRegistry):
        raise ValueError("Agent ToolProvider 必须暴露用于统一分发的 ToolRegistry")
    return registry


def find_last_ai_message(
    messages: list[object], *, required: bool = False
) -> AIMessage | None:
    """返回最近的 AIMessage，并可在缺失时立即失败。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    if required:
        raise RuntimeError("RAG handoff requires the requesting AIMessage")
    return None


def _rag_handoff(
    result: dict[str, Any],
    runtime: ToolRuntime[ToolInvocationContext],
) -> Command:
    """把成功知识库结果和配对消息交接给父图。"""
    tool_message = ToolMessage(
        content=json.dumps(result, ensure_ascii=False, sort_keys=True, default=str),
        artifact=result,
        tool_call_id=runtime.tool_call_id or "",
        status="success",
    )
    return Command(
        graph=Command.PARENT,
        update={
            "messages": [
                find_last_ai_message(runtime.state.get("messages", []), required=True),
                tool_message,
            ],
            "pending_rag_result": result,
        },
    )


def _build_tool(spec: ToolSpec, registry: ToolRegistry) -> StructuredTool:
    """将单个 ToolSpec 转换为带 Runtime 注入的框架工具。"""
    if spec.args_model is None:
        raise ValueError(f"tool {spec.name!r} 缺少 Pydantic args_model")
    runtime_schema = create_model(
        f"{spec.args_model.__name__}Runtime",
        __base__=spec.args_model,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        runtime=(ToolRuntime[ToolInvocationContext], ...),
    )

    def invoke_registered_tool(
        runtime: ToolRuntime[ToolInvocationContext],
        **tool_args: Any,
    ) -> dict[str, Any] | Command:
        """通过项目 Registry/Runtime 执行，不直接接触 handler。"""
        context = runtime.context
        if not isinstance(context, ToolInvocationContext):
            raise RuntimeError("ToolRuntime did not inject ToolInvocationContext")
        result = registry.dispatch(spec.name, tool_args, context=context)
        output = result.get("output") if isinstance(result, dict) else None
        is_rag = (
            spec.name == "search_knowledge_base"
            and result.get("ok") is True
            and isinstance(output, dict)
            and isinstance(output.get("chunks"), list)
        )
        last_ai = find_last_ai_message(runtime.state.get("messages", []), required=True)
        assert last_ai is not None
        if is_rag and len(last_ai.tool_calls) == 1:
            return _rag_handoff(result, runtime)
        return result

    return StructuredTool.from_function(
        func=invoke_registered_tool,
        name=spec.name,
        description=spec.description,
        args_schema=runtime_schema,
        metadata={"tool_version": spec.version},
    )


def build_framework_tools(
    provider: ToolProvider,
    *,
    context: ToolInvocationContext,
) -> list[BaseTool]:
    """按本次请求 Provider 构造供 create_agent 使用的动态工具。"""
    registry = _provider_registry(provider)
    specs: Sequence[ToolSpec] = provider.list_specs(context=context)
    tools: list[BaseTool] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, ToolSpec) or registry.get(spec.name) is not spec:
            raise ValueError("Provider ToolSpec 必须属于其分发 Registry")
        if spec.name in seen:
            raise ValueError(f"tool {spec.name!r} 重复，禁止静默覆盖")
        seen.add(spec.name)
        tools.append(_build_tool(spec, registry))
    return tools


def resolve_tool_message(output: object, call_id: str) -> ToolMessage:
    """读取 ToolNode 已生成且与调用 ID 匹配的消息。"""
    if isinstance(output, ToolMessage):
        return output
    if not isinstance(output, Command) or not isinstance(output.update, dict):
        raise TypeError("ToolNode must return ToolMessage or Command")
    return next(
        item
        for item in output.update.get("messages", [])
        if isinstance(item, ToolMessage) and item.tool_call_id == call_id
    )


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


class RegistryToolsMiddleware(AgentMiddleware):
    """为模型与 ToolNode 从同一请求 Provider 解析动态工具。"""

    @staticmethod
    def _tools(context: ToolInvocationContext):
        """按当前请求上下文解析动态工具列表。"""
        return build_framework_tools(get_agent_tool_provider(), context=context)

    def wrap_model_call(self, request, handler):
        """向本次模型调用注入 Provider 提供的动态工具。"""
        return handler(request.override(tools=self._tools(request.runtime.context)))

    async def awrap_model_call(self, request, handler):
        """在异步模型调用中注入同一组请求级动态工具。"""
        return await handler(
            request.override(tools=self._tools(request.runtime.context))
        )

    def wrap_tool_call(self, request, handler):
        """以同一 Provider 语义解析模型实际请求的工具。"""
        tools = {tool.name: tool for tool in self._tools(request.runtime.context)}
        return handler(request.override(tool=tools.get(request.tool_call["name"])))

    async def awrap_tool_call(self, request, handler):
        """在异步工具调用中保持与模型可见工具相同的解析语义。"""
        tools = {tool.name: tool for tool in self._tools(request.runtime.context)}
        return await handler(
            request.override(tool=tools.get(request.tool_call["name"]))
        )
