"""LangChain ``create_agent`` 唯一 Agent Loop 的关键版本契约。"""

from langchain.agents.middleware import AgentMiddleware

from app.agent.framework_agent import (
    _GovernedBoundModel,
    build_framework_agent,
)
from app.agent.framework_middleware import (
    ProjectToolGovernanceMiddleware,
)
from app.agent.tool_visibility import ProjectToolVisibilityMiddleware


def test_production_agent_is_framework_model_tool_graph():
    """生产 Runtime 直接包含 create_agent 标准节点与调用限制。"""
    node_names = set(build_framework_agent().get_graph().nodes)

    assert {"model", "tools"} <= node_names
    assert "ModelCallLimitMiddleware.before_model" in node_names
    assert "ProjectToolCallLimitMiddleware.after_model" in node_names


def test_project_governance_uses_native_tool_middleware_hooks():
    """项目工具治理只覆盖 LangChain 异步原生 Hook。"""
    for middleware_type in (ProjectToolGovernanceMiddleware,):
        assert middleware_type.wrap_tool_call is AgentMiddleware.wrap_tool_call
        assert middleware_type.awrap_tool_call is not AgentMiddleware.awrap_tool_call
        assert not hasattr(middleware_type, "govern_tool_call")
        assert not hasattr(middleware_type, "agovern_tool_call")
    for hook in ("before_model", "after_model", "after_agent"):
        assert hook not in ProjectToolGovernanceMiddleware.__dict__
        assert f"a{hook}" in ProjectToolGovernanceMiddleware.__dict__
    assert "wrap_model_call" not in ProjectToolVisibilityMiddleware.__dict__
    assert "invoke" not in _GovernedBoundModel.__dict__
