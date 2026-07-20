"""LangChain Tool 适配层。

将 Provider 提供的 ``ToolSpec`` 按请求转换为 ``StructuredTool``，实际调用
仍统一转发给定义所属的 ``ToolRegistry``，不复制任何工具业务逻辑。
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from app.tools.context import (
    ToolInvocationContext,
    build_default_tool_context,
)
from app.tools.contracts import ToolSpec
from app.tools.providers import (
    LocalToolProvider,
    ToolProvider,
    get_default_tool_provider,
)
from app.tools.registry import ToolRegistry


def _build_langchain_tool(
    spec: ToolSpec,
    *,
    registry: ToolRegistry,
    context: ToolInvocationContext,
) -> StructuredTool:
    """根据注册项构造捕获完整请求上下文的 StructuredTool。"""
    if spec.args_model is None:
        raise ValueError(f"tool {spec.name!r} 缺少 Pydantic args_model")

    def invoke_registered_tool(**tool_args: Any) -> dict:
        """把已由 LangChain 校验的参数转发给项目统一工具分发入口。"""
        return registry.dispatch(spec.name, tool_args, context=context)

    return StructuredTool.from_function(
        func=invoke_registered_tool,
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_model,
    )


def _resolve_context(
    *,
    context: ToolInvocationContext | None,
    trace_id: str,
) -> ToolInvocationContext:
    """把兼容 trace_id 参数规整为完整且不可变的请求上下文。"""
    if context is not None:
        if trace_id.strip() and context.trace_id and trace_id.strip() != context.trace_id:
            raise ValueError("context.trace_id 与兼容 trace_id 参数不一致")
        return context
    return build_default_tool_context(
        trace_id=trace_id,
        source="langchain_tool_adapter",
    )


def build_langchain_tools(
    specs: Sequence[ToolSpec],
    *,
    registry: ToolRegistry,
    context: ToolInvocationContext,
) -> list[BaseTool]:
    """把一组注册 ToolSpec 转换为稳定有序的请求级 StructuredTool。"""
    tools: list[BaseTool] = []
    seen_names: set[str] = set()
    for spec in specs:
        if not isinstance(spec, ToolSpec):
            raise TypeError("specs 只能包含 ToolSpec")
        if spec.name in seen_names:
            raise ValueError(f"tool {spec.name!r} 重复，禁止静默覆盖")
        registered_spec = registry.get(spec.name)
        if registered_spec is not spec:
            raise ValueError(
                f"tool {spec.name!r} 不属于用于分发的 ToolRegistry"
            )
        seen_names.add(spec.name)
        tools.append(
            _build_langchain_tool(
                spec,
                registry=registry,
                context=context,
            )
        )
    return tools


def _resolve_provider_registry(
    provider: ToolProvider,
    registry: ToolRegistry | None,
) -> ToolRegistry:
    """解析 Provider 的分发注册表，并拒绝绕过 Registry 的 Provider。"""
    if registry is not None:
        return registry
    if isinstance(provider, LocalToolProvider):
        return provider.registry
    raise ValueError("非本地 ToolProvider 必须显式提供 ToolRegistry")


def get_langchain_tools(
    *,
    context: ToolInvocationContext | None = None,
    trace_id: str = "",
    provider: ToolProvider | None = None,
    registry: ToolRegistry | None = None,
) -> list[BaseTool]:
    """通过 Provider 发现并转换当前请求可用的 LangChain 工具。"""
    resolved_context = _resolve_context(context=context, trace_id=trace_id)
    resolved_provider = provider or get_default_tool_provider()
    resolved_registry = _resolve_provider_registry(
        resolved_provider,
        registry,
    )
    specs = resolved_provider.list_specs(context=resolved_context)
    return build_langchain_tools(
        specs,
        registry=resolved_registry,
        context=resolved_context,
    )


def get_langchain_tool_map(
    *,
    context: ToolInvocationContext | None = None,
    trace_id: str = "",
    provider: ToolProvider | None = None,
    registry: ToolRegistry | None = None,
) -> dict[str, BaseTool]:
    """返回以注册工具名为键的请求级 LangChain Tool 映射。"""
    return {
        langchain_tool.name: langchain_tool
        for langchain_tool in get_langchain_tools(
            context=context,
            trace_id=trace_id,
            provider=provider,
            registry=registry,
        )
    }


def get_langchain_tool(
    name: str,
    *,
    context: ToolInvocationContext | None = None,
    trace_id: str = "",
    provider: ToolProvider | None = None,
    registry: ToolRegistry | None = None,
) -> BaseTool | None:
    """按名称返回请求级 LangChain Tool；名称未注册时返回 ``None``。"""
    resolved_context = _resolve_context(context=context, trace_id=trace_id)
    resolved_provider = provider or get_default_tool_provider()
    resolved_registry = _resolve_provider_registry(
        resolved_provider,
        registry,
    )
    specs = resolved_provider.list_specs(context=resolved_context)
    tools = build_langchain_tools(
        specs,
        registry=resolved_registry,
        context=resolved_context,
    )
    return next((tool for tool in tools if tool.name == name), None)


def get_langchain_tool_names(
    *,
    context: ToolInvocationContext | None = None,
    provider: ToolProvider | None = None,
    registry: ToolRegistry | None = None,
) -> list[str]:
    """按稳定的注册顺序返回可适配为 LangChain Tool 的工具名称。"""
    return [
        tool.name
        for tool in get_langchain_tools(
            context=context,
            provider=provider,
            registry=registry,
        )
    ]


__all__ = [
    "build_langchain_tools",
    "get_langchain_tool",
    "get_langchain_tool_map",
    "get_langchain_tool_names",
    "get_langchain_tools",
]
