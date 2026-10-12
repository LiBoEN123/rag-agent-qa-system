"""LangChain 模型调用前的工具模式与权限过滤；执行授权仍由 Policy 负责。"""

from langchain.agents.middleware import AgentMiddleware, ModelRequest

from app.contracts.chat import RAG_TOOL_NAMES, ToolMode
from app.tools.context import ToolInvocationContext, has_permission


class ProjectToolVisibilityMiddleware(AgentMiddleware):
    """按请求工具模式和权限过滤本地工具。"""

    # 根据请求模式和调用者权限筛选模型可见工具。
    @staticmethod
    def visible_request(request: ModelRequest) -> ModelRequest:
        context = request.runtime.context
        if not isinstance(context, ToolInvocationContext):
            return request.override(tools=[])
        permissions = context.permissions
        rag_only = context.tool_mode is ToolMode.RAG_ONLY
        return request.override(
            tools=[
                tool
                for tool in request.tools
                if (not rag_only or tool.name in RAG_TOOL_NAMES)
                and (
                    all(
                        has_permission(permissions, permission)
                        for permission in (tool.metadata or {}).get("required_permissions", ())
                    )
                )
            ]
        )

    # 异步调用模型前，按权限更新请求中的可见工具列表。
    async def awrap_model_call(self, request, handler):
        return await handler(self.visible_request(request))
