"""ToolSpec 到 LangChain 动态工具的薄适配。"""

from collections.abc import Sequence
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ConfigDict, create_model

from app.agent.model import get_agent_tool_provider
from app.tools import runtime as tool_runtime
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.mcp.provider import list_mcp_tools_for_context
from app.tools.providers import ToolProvider
from app.tools.registry import ToolRegistry


def _provider_registry(provider: ToolProvider) -> ToolRegistry:
    """取得 Provider 所属 Registry，供唯一执行入口复用。"""
    registry = getattr(provider, "registry", None)
    if not isinstance(registry, ToolRegistry):
        raise ValueError("Agent ToolProvider 必须暴露 ToolSpec 所属的 ToolRegistry")
    return registry


def _build_tool(spec: ToolSpec, registry: ToolRegistry) -> StructuredTool:
    """将单个 ToolSpec 转换为只注入 Runtime 的框架工具。"""
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
    ) -> dict[str, Any]:
        """通过项目唯一执行入口进入 Runtime，不直接接触 handler。"""
        context = runtime.context
        if not isinstance(context, ToolInvocationContext):
            raise RuntimeError("ToolRuntime did not inject ToolInvocationContext")
        return tool_runtime.execute_registered_tool(
            registry=registry,
            tool_name=spec.name,
            tool_args=tool_args,
            context=context,
        )

    async def ainvoke_registered_tool(
        runtime: ToolRuntime[ToolInvocationContext],
        **tool_args: Any,
    ) -> dict[str, Any]:
        """异步 Agent 复用本地工具的受控同步执行边界。"""
        context = runtime.context
        if not isinstance(context, ToolInvocationContext):
            raise RuntimeError("ToolRuntime did not inject ToolInvocationContext")
        return tool_runtime.execute_registered_tool(
            registry=registry,
            tool_name=spec.name,
            tool_args=tool_args,
            context=context,
        )

    return StructuredTool.from_function(
        func=invoke_registered_tool,
        coroutine=ainvoke_registered_tool,
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
    """按本次请求 Provider 构造供 create_agent 使用的本地工具。"""
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
        tools.append(_build_tool(spec, registry))
    return tools


def build_agent_framework_tools(
    provider: ToolProvider,
    *,
    context: ToolInvocationContext,
) -> list[BaseTool]:
    """合并本地工具与已经过 MCP Bridge 治理的原生 BaseTool。"""
    tools = build_framework_tools(provider, context=context)
    seen = {tool.name for tool in tools}
    for tool in list_mcp_tools_for_context(context):
        if tool.name in seen:
            raise ValueError(f"tool {tool.name!r} 在本地与 MCP 工具源中重复")
        seen.add(tool.name)
        tools.append(tool)
    return tools


def resolve_agent_tools(context: ToolInvocationContext) -> list[BaseTool]:
    """从唯一 Provider 入口解析本次请求可见的完整工具集合。"""
    return build_agent_framework_tools(
        get_agent_tool_provider(),
        context=context,
    )
