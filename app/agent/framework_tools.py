"""缓存与请求无关的 LangChain 本地工具，并合并受治理 MCP 工具。"""

from functools import lru_cache
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ConfigDict, Field, SkipValidation, create_model

from app.config import AppSettings, get_settings
from app.tools.audit import ToolAuditSink, get_default_tool_audit_sink
from app.tools.catalog import get_local_tool_specs
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.execution import execute_project_tool
from app.tools.mcp.provider import list_mcp_tools_for_context
from app.tools.policy import ToolPolicyEngine


class ProjectStructuredTool(StructuredTool):
    """携带非公开项目治理声明的原生工具，不保存请求上下文。"""

    project_spec: SkipValidation[ToolSpec] = Field(exclude=True, repr=False)
    project_settings: SkipValidation[AppSettings] = Field(exclude=True, repr=False)
    project_policy_engine: SkipValidation[ToolPolicyEngine] = Field(
        exclude=True, repr=False
    )
    project_audit_sink: SkipValidation[ToolAuditSink] = Field(exclude=True, repr=False)


def build_local_tool(
    spec: ToolSpec,
    *,
    settings: AppSettings | None = None,
    policy_engine: ToolPolicyEngine | None = None,
    audit_sink: ToolAuditSink | None = None,
) -> ProjectStructuredTool:
    """只使用 ToolNode 注入的 runtime.context 调用项目执行边界。"""
    resolved_settings = settings or get_settings()
    resolved_policy = policy_engine or ToolPolicyEngine()
    resolved_audit = audit_sink or get_default_tool_audit_sink(resolved_settings)
    runtime_schema = create_model(
        f"{spec.args_model.__name__}Runtime",
        __base__=spec.args_model,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        runtime=(ToolRuntime[ToolInvocationContext], ...),
    )

    # 校验框架注入的调用上下文，并交给项目工具执行入口处理。
    def invoke_project_tool(
        runtime: ToolRuntime[ToolInvocationContext], **tool_args: Any
    ) -> dict[str, Any]:
        if not isinstance(getattr(runtime, "context", None), ToolInvocationContext):
            raise RuntimeError("ToolRuntime did not inject ToolInvocationContext")
        return execute_project_tool(
            spec=spec,
            tool_args=tool_args,
            context=runtime.context,
            tool_call_id=str(runtime.tool_call_id or ""),
            settings=resolved_settings,
            policy_engine=resolved_policy,
            audit_sink=resolved_audit,
        )

    return ProjectStructuredTool.from_function(
        func=invoke_project_tool,
        name=spec.name,
        description=spec.description,
        args_schema=runtime_schema,
        metadata={"tool_version": spec.version},
        project_spec=spec,
        project_settings=resolved_settings,
        project_policy_engine=resolved_policy,
        project_audit_sink=resolved_audit,
    )


@lru_cache(maxsize=4)
def get_local_tools(settings: AppSettings) -> tuple[ProjectStructuredTool, ...]:
    """以完整配置缓存不可变工具集合，所有 Local permissions 共用。"""
    return tuple(
        build_local_tool(spec, settings=settings)
        for spec in get_local_tool_specs(settings)
    )


def local_tool_contract(tool: ProjectStructuredTool) -> dict[str, Any]:
    """导出原生工具的声明 schema 与治理元数据，保留历史输入别名。"""
    contract = tool.project_spec.to_contract_dict()
    # 注入模型唯一基类就是 StructuredTool 的原始参数模型；不含 runtime。
    args_model = tool.get_input_schema().__bases__[0]
    contract.update(name=tool.name, description=tool.description, args_schema=args_model.model_json_schema())
    return contract


def get_local_tool_contracts(settings: AppSettings) -> list[dict[str, Any]]:
    """以稳定顺序派生全量本地 manifest 合同，调用方获得独立副本。"""
    return [local_tool_contract(tool) for tool in get_local_tools(settings)]


def resolve_agent_tools(
    context: ToolInvocationContext, settings: AppSettings | None = None
) -> list[BaseTool]:
    """本地工具全部注册；MCP 继续使用当前请求的独立可见性路径。"""
    tools: list[BaseTool] = list(get_local_tools(settings or get_settings()))
    seen = {tool.name for tool in tools}
    for tool in list_mcp_tools_for_context(context):
        if tool.name in seen:
            raise ValueError(f"tool {tool.name!r} 在本地与 MCP 工具源中重复")
        seen.add(tool.name)
        tools.append(tool)
    return tools
