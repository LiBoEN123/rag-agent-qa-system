"""缓存与请求无关的 LangChain 本地工具，并合并受治理 MCP 工具。"""

from functools import lru_cache
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

from app.config import AppSettings, get_settings
from app.contracts.chat import ToolMode
from app.tools.audit import ToolAuditSink, get_default_tool_audit_sink
from app.tools.catalog import get_local_tool_specs
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.execution import execute_project_tool
from app.tools.mcp.provider import list_mcp_tools_for_context
from app.tools.policy import ToolPolicyEngine
from app.tools.result import tool_content


def build_local_tool(
    spec: ToolSpec,
    *,
    settings: AppSettings | None = None,
    policy_engine: ToolPolicyEngine | None = None,
    audit_sink: ToolAuditSink | None = None,
) -> StructuredTool:
    """只使用 ToolNode 注入的 runtime.context 调用项目执行边界。"""
    resolved_settings = settings or get_settings()
    resolved_policy = policy_engine or ToolPolicyEngine()
    resolved_audit = audit_sink or get_default_tool_audit_sink(resolved_settings)
    async def invoke_project_tool(
        runtime: ToolRuntime[ToolInvocationContext], **tool_args: Any
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(getattr(runtime, "context", None), ToolInvocationContext):
            raise RuntimeError("ToolRuntime did not inject ToolInvocationContext")
        result = await execute_project_tool(
            spec=spec, tool_args=tool_args, context=runtime.context,
            tool_call_id=str(runtime.tool_call_id or ""), settings=resolved_settings,
            policy_engine=resolved_policy, audit_sink=resolved_audit,
        )
        return tool_content(result), result

    return StructuredTool.from_function(
        coroutine=invoke_project_tool,
        name=spec.name,
        description=spec.description,
        # JSON schema preserves business constraints without admitting runtime as input.
        # execute_project_tool validates this same Pydantic model exactly once.
        args_schema=spec.args_model.model_json_schema(),
        response_format="content_and_artifact",
        metadata={"tool_version": spec.version, "required_permissions": spec.required_permissions},
    )


@lru_cache(maxsize=4)
def get_local_tools(settings: AppSettings) -> tuple[StructuredTool, ...]:
    """以完整配置缓存不可变工具集合，所有 Local permissions 共用。"""
    return tuple(
        build_local_tool(spec, settings=settings)
        for spec in get_local_tool_specs(settings)
    )


def local_tool_contract(spec: ToolSpec) -> dict[str, Any]:
    """唯一契约导出入口；业务模型引用保留原始 schema 的 title、别名和约束。"""
    return {
        "name": spec.name,
        "version": spec.version,
        "description": spec.description,
        "args_schema": spec.args_model.model_json_schema(),
        "output_schema": spec.output_model.model_json_schema(),
        "args_model": spec.args_model.__name__,
        "output_model": spec.output_model.__name__,
        "required_permissions": list(spec.required_permissions),
        "timeout": spec.timeout,
        "max_retries": spec.max_retries,
        "sensitive_arg_fields": list(spec.sensitive_arg_fields),
        "audit_output_mode": spec.audit_output_mode.value,
    }


def get_local_tool_contracts(settings: AppSettings) -> list[dict[str, Any]]:
    """以稳定顺序派生全量本地 manifest 合同，调用方获得独立副本。"""
    return [local_tool_contract(spec) for spec in get_local_tool_specs(settings)]


def resolve_agent_tools(
    context: ToolInvocationContext, settings: AppSettings | None = None
) -> list[BaseTool]:
    """本地工具全部注册；MCP 继续使用当前请求的独立可见性路径。"""
    resolved_settings = settings or get_settings()
    tools: list[BaseTool] = list(get_local_tools(resolved_settings))
    if context.tool_mode is ToolMode.RAG_ONLY:
        return tools
    seen = {tool.name for tool in tools}
    for tool in list_mcp_tools_for_context(context, settings=resolved_settings):
        if tool.name in seen:
            raise ValueError(f"tool {tool.name!r} 在本地与 MCP 工具源中重复")
        seen.add(tool.name)
        tools.append(tool)
    return tools
