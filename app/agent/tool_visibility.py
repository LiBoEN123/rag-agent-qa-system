"""LangChain 模型调用前的工具模式与权限过滤；执行授权仍由 Policy 负责。"""

from langchain.agents.middleware import AgentMiddleware, ModelRequest

from app.agent.framework_tools import ProjectStructuredTool
from app.contracts.chat import RAG_TOOL_NAMES, ToolMode
from app.tools.context import ToolInvocationContext, has_permission


class ProjectToolVisibilityMiddleware(AgentMiddleware):
    """先限制请求工具范围，再过滤本地权限；MCP 权限仍由独立路径处理。"""

    # 根据请求模式和调用者权限筛选模型可见工具。
    @staticmethod
    def visible_request(request: ModelRequest) -> ModelRequest:
        context = request.runtime.context
        permissions = (
            context.permissions if isinstance(context, ToolInvocationContext) else ()
        )
        rag_only = (
            isinstance(context, ToolInvocationContext)
            and context.tool_mode is ToolMode.RAG_ONLY
        )
        return request.override(
            tools=[
                tool
                for tool in request.tools
                if (not rag_only or getattr(tool, "name", None) in RAG_TOOL_NAMES)
                and (
                    not isinstance(tool, ProjectStructuredTool)
                    or all(
                        has_permission(permissions, permission)
                        for permission in tool.project_spec.required_permissions
                    )
                )
            ]
        )

    # 异步调用模型前，按权限更新请求中的可见工具列表。
    async def awrap_model_call(self, request, handler):
        return await handler(self.visible_request(request))
