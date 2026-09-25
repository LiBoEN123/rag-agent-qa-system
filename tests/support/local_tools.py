"""通过 LangChain public tool path 注入显式 Runtime 的本地测试辅助。"""

import asyncio

from collections.abc import Sequence
from typing import Any

from langchain.tools import ToolRuntime

from app.agent.framework_tools import build_local_tool
from app.config import AppSettings, get_settings
from app.tools.audit import ToolAuditSink
from app.tools.context import ToolInvocationContext, build_default_tool_context
from app.tools.contracts import ToolSpec


def tool_runtime(context: ToolInvocationContext) -> ToolRuntime[ToolInvocationContext]:
    return ToolRuntime(state={}, context=context, config={}, stream_writer=lambda _: None, tool_call_id="", store=None)


def invoke_local_tool(
    registry: Sequence[ToolSpec],
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    context: ToolInvocationContext | None = None,
    settings: AppSettings | None = None,
    audit_sink: ToolAuditSink | None = None,
) -> dict[str, Any]:
    """对独立声明集合调用原生 BaseTool.ainvoke，保持参数校验与项目执行。"""
    spec = next((item for item in registry if item.name == tool_name), None)
    if spec is None:
        raise LookupError(f"unknown local tool: {tool_name}")
    resolved_settings = settings or get_settings()
    context = context or build_default_tool_context(source="direct_test", settings=resolved_settings)
    tool = build_local_tool(spec, settings=resolved_settings, audit_sink=audit_sink)
    message = asyncio.run(tool.ainvoke({
        "name": tool.name, "id": "direct-test", "type": "tool_call",
        "args": {**tool_args, "runtime": tool_runtime(context)},
    }))
    return message.artifact


def local_tools(specs: Sequence[ToolSpec], settings: AppSettings | None = None):
    """为自定义工具测试构造普通 tuple，不增加任何注册或权限逻辑。"""
    return tuple(build_local_tool(spec, settings=settings) for spec in specs)


def visible_tools(tools, *, context):
    """用真实 middleware 检查请求可见工具，不复制权限算法。"""
    from langchain.agents.middleware import ModelRequest
    from langgraph.runtime import Runtime
    from app.agent.framework_agent import _DeferredAgentModel
    from app.agent.tool_visibility import ProjectToolVisibilityMiddleware

    request = ModelRequest(model=_DeferredAgentModel(), messages=[], tools=list(tools), runtime=Runtime(context=context))
    return ProjectToolVisibilityMiddleware.visible_request(request).tools
