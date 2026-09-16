"""LangChain 模型调用前的本地工具权限可见性；执行授权仍由 Policy 负责。"""

from langchain.agents.middleware import AgentMiddleware, ModelRequest

from app.agent.framework_tools import ProjectStructuredTool
from app.tools.context import ToolInvocationContext, has_permission


class ProjectToolVisibilityMiddleware(AgentMiddleware):
    """只过滤本地工具；MCP BaseTool 沿用其独立权限路径。"""

    # 根据调用者权限筛选模型可见的项目工具。
    @staticmethod
    def visible_request(request: ModelRequest) -> ModelRequest:
        context = request.runtime.context
        permissions = (
            context.permissions if isinstance(context, ToolInvocationContext) else ()
        )
        return request.override(
            tools=[
                tool
                for tool in request.tools
                if not isinstance(tool, ProjectStructuredTool)
                or all(
                    has_permission(permissions, permission)
                    for permission in tool.project_spec.required_permissions
                )
            ]
        )

    # 异步调用模型前，按权限更新请求中的可见工具列表。
    async def awrap_model_call(self, request, handler):
        return await handler(self.visible_request(request))
