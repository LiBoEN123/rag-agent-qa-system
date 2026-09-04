"""ToolSpec 到 LangChain ProjectStructuredTool 的薄适配。"""

from collections.abc import Sequence
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ConfigDict, Field, SkipValidation, create_model

from app.config import AppSettings, get_settings
from app.tools.audit import ToolAuditSink, get_default_tool_audit_sink
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.execution import execute_project_tool
from app.tools.mcp.provider import list_mcp_tools_for_context
from app.tools.policy import ToolPolicyEngine
from app.tools.providers import ToolProvider, get_default_tool_provider
from app.tools.registry import ToolRegistry


class ProjectStructuredTool(StructuredTool):
    """携带非公开 ToolSpec 的本地 StructuredTool。"""

    project_spec: SkipValidation[ToolSpec] = Field(exclude=True, repr=False)
    project_context: SkipValidation[ToolInvocationContext] = Field(exclude=True, repr=False)
    project_settings: SkipValidation[AppSettings] = Field(exclude=True, repr=False)
    project_policy_engine: SkipValidation[ToolPolicyEngine] = Field(exclude=True, repr=False)
    project_audit_sink: SkipValidation[ToolAuditSink] = Field(exclude=True, repr=False)


def build_local_tool(
    spec: ToolSpec,
    *,
    context: ToolInvocationContext,
    settings: AppSettings | None = None,
    policy_engine: ToolPolicyEngine | None = None,
    audit_sink: ToolAuditSink | None = None,
    runtime_injection: bool = True,
) -> ProjectStructuredTool:
    """构造直接调用 handler adapter 的本地工具。"""
    resolved_settings = settings or get_settings()
    resolved_policy = policy_engine or ToolPolicyEngine()
    resolved_audit = audit_sink or get_default_tool_audit_sink(resolved_settings)

    def execute(tool_args: dict[str, Any], runtime: Any = None) -> dict[str, Any]:
        invocation_context = context
        tool_call_id = ""
        if runtime is not None:
            invocation_context = runtime.context
            tool_call_id = str(runtime.tool_call_id or "")
        if not isinstance(invocation_context, ToolInvocationContext):
            raise RuntimeError("ToolRuntime did not inject ToolInvocationContext")
        return execute_project_tool(
            spec=spec,
            tool_args=tool_args,
            context=invocation_context,
            tool_call_id=tool_call_id,
            settings=resolved_settings,
            policy_engine=resolved_policy,
            audit_sink=resolved_audit,
        )

    if runtime_injection:
        runtime_schema = create_model(
            f"{spec.args_model.__name__}Runtime",
            __base__=spec.args_model,
            __config__=ConfigDict(arbitrary_types_allowed=True),
            runtime=(ToolRuntime[ToolInvocationContext], ...),
        )

        def invoke_project_tool(
            runtime: ToolRuntime[ToolInvocationContext],
            **tool_args: Any,
        ) -> dict[str, Any]:
            return execute(tool_args, runtime)

        args_schema = runtime_schema
    else:

        def invoke_project_tool(**tool_args: Any) -> dict[str, Any]:
            return execute(tool_args)

        args_schema = spec.args_model

    return ProjectStructuredTool.from_function(
        func=invoke_project_tool,
        name=spec.name,
        description=spec.description,
        args_schema=args_schema,
        metadata={"tool_version": spec.version},
        project_spec=spec,
        project_context=context,
        project_settings=resolved_settings,
        project_policy_engine=resolved_policy,
        project_audit_sink=resolved_audit,
    )


def build_framework_tools(
    provider: ToolProvider,
    *,
    context: ToolInvocationContext,
) -> list[BaseTool]:
    """按本次请求 Provider 构造供 create_agent 使用的本地工具。"""
    registry = getattr(provider, "registry", None)
    if not isinstance(registry, ToolRegistry):
        raise ValueError("Agent ToolProvider 必须暴露 ToolSpec 所属的 ToolRegistry")
    specs: Sequence[ToolSpec] = provider.list_specs(context=context)
    settings = get_settings()
    policy_engine = ToolPolicyEngine()
    audit_sink = get_default_tool_audit_sink(settings)
    tools: list[BaseTool] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, ToolSpec) or registry.get(spec.name) is not spec:
            raise ValueError("Provider ToolSpec 必须属于其声明的 Registry")
        if spec.name in seen:
            raise ValueError(f"tool {spec.name!r} 重复，禁止静默覆盖")
        seen.add(spec.name)
        tools.append(
            build_local_tool(
                spec, context=context, settings=settings,
                policy_engine=policy_engine, audit_sink=audit_sink,
            )
        )
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
    return build_agent_framework_tools(get_default_tool_provider(), context=context)


__all__ = [
    "ProjectStructuredTool",
    "build_agent_framework_tools",
    "build_framework_tools",
    "build_local_tool",
    "resolve_agent_tools",
]
