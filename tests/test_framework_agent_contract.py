"""LangChain ``create_agent`` 唯一 Agent Loop 的关键版本契约。"""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from app.agent.framework_agent import (
    build_framework_agent,
)
from app.agent.framework_middleware import ProjectToolGovernanceMiddleware


class _ProbeModel:
    profile: dict = {}

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def invoke(self, messages):
        del messages
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "handoff_probe",
                    "args": {"value": 7},
                    "id": "handoff-probe-call",
                    "type": "tool_call",
                }
            ],
        )


def test_production_agent_is_framework_model_tool_graph():
    """生产 Runtime 直接包含 create_agent 标准节点与调用限制。"""
    node_names = set(build_framework_agent().get_graph().nodes)

    assert {"model", "tools"} <= node_names
    assert "ModelCallLimitMiddleware.before_model" in node_names
    assert "ToolCallLimitMiddleware.after_model" in node_names


def test_project_governance_uses_native_tool_middleware_hooks():
    """项目工具治理直接覆盖 LangChain 同步与异步原生 Hook。"""
    middleware_type = ProjectToolGovernanceMiddleware

    assert middleware_type.wrap_tool_call is not AgentMiddleware.wrap_tool_call
    assert middleware_type.awrap_tool_call is not AgentMiddleware.awrap_tool_call
    assert not hasattr(middleware_type, "govern_tool_call")
    assert not hasattr(middleware_type, "agovern_tool_call")
