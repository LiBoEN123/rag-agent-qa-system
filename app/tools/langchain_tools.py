"""LangChain Tool 适配层。

将现有 ``ToolRegistry`` 中的工具按请求转换为 ``StructuredTool``，实际调用仍
统一转发给 ``dispatch_tool``，不复制任何工具业务逻辑。
"""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from app.tools.context import ToolInvocationContext
from app.tools.dispatch import dispatch_tool
from app.tools.registry import ToolSpec, tool_registry


def _build_langchain_tool(
    spec: ToolSpec,
    *,
    trace_id: str,
    context: ToolInvocationContext | None,
) -> StructuredTool:
    """根据注册项构造捕获完整请求上下文的 StructuredTool。"""
    if spec.args_model is None:
        raise ValueError(f"tool {spec.name!r} 缺少 Pydantic args_model")

    def invoke_registered_tool(**tool_args: Any) -> dict:
        """把已由 LangChain 校验的参数转发给项目统一工具分发入口。"""
        if context is not None:
            return dispatch_tool(spec.name, tool_args, context=context)
        return dispatch_tool(spec.name, tool_args, trace_id=trace_id)

    return StructuredTool.from_function(
        func=invoke_registered_tool,
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_model,
    )


def get_langchain_tools(
    *,
    context: ToolInvocationContext | None = None,
    trace_id: str = "",
) -> list[BaseTool]:
    """按注册顺序返回当前全部工具的请求级 LangChain Tool 实例。"""
    langchain_tools: list[BaseTool] = []
    for name in tool_registry.names():
        spec = tool_registry.get(name)
        if spec is None:
            raise RuntimeError(f"tool registry 中不存在已列出的工具: {name}")
        langchain_tools.append(
            _build_langchain_tool(
                spec,
                trace_id=trace_id,
                context=context,
            )
        )
    return langchain_tools


def get_langchain_tool_map(
    *,
    context: ToolInvocationContext | None = None,
    trace_id: str = "",
) -> dict[str, BaseTool]:
    """返回以注册工具名为键的请求级 LangChain Tool 映射。"""
    return {
        langchain_tool.name: langchain_tool
        for langchain_tool in get_langchain_tools(
            context=context,
            trace_id=trace_id,
        )
    }


def get_langchain_tool(
    name: str,
    *,
    context: ToolInvocationContext | None = None,
    trace_id: str = "",
) -> BaseTool | None:
    """按名称返回请求级 LangChain Tool；名称未注册时返回 ``None``。"""
    spec = tool_registry.get(name)
    if spec is None:
        return None
    return _build_langchain_tool(
        spec,
        trace_id=trace_id,
        context=context,
    )


def get_langchain_tool_names() -> list[str]:
    """按稳定的注册顺序返回可适配为 LangChain Tool 的工具名称。"""
    return tool_registry.names()
