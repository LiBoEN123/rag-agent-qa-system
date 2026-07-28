"""确定性 Router 和 Agent 包公共面的架构回归测试。"""

import importlib.util

import pytest

import app.agent as agent_package
from app.agent.router import route_agent_action
from app.graph import nodes as graph_nodes
from app.services.chat_service import get_default_chat_service_dependencies


REMOVED_ROUTER_EXPORTS = (
    "call_router_llm",
    "parse_router_json_output",
    "route_agent_action_by_llm",
    "validate_route_decision",
    "normalize_router_confidence",
    "normalize_router_reason",
    "normalize_chunk_id_value",
    "ROUTER_ALLOWED_TOOLS",
    "ROUTER_TOOL_SPECS",
    "ROUTER_ALLOWED_ACTIONS",
    "ROUTER_ACTION_TO_DECISION",
    "HIGH_CONFIDENCE_RULE_ACTIONS",
    "add_router_metadata",
    "route_agent_action_rule_based",
)


def test_llm_router_module_is_removed():
    """生产包中不再存在旧 LLM Router 模块。"""
    assert importlib.util.find_spec("app.agent.llm_router") is None


@pytest.mark.parametrize("name", REMOVED_ROUTER_EXPORTS)
def test_agent_package_does_not_export_removed_router_api(name):
    """app.agent 不再聚合导出已删除的 Router 内部 API。"""
    assert not hasattr(agent_package, name)


def test_agent_package_has_no_business_function_reexports():
    """包级入口不再充当内部业务函数聚合器。"""
    assert agent_package.__all__ == []
    assert not hasattr(agent_package, "route_agent_action")
    assert not hasattr(agent_package, "format_tool_answer")
    assert not hasattr(agent_package, "Settings")


def test_current_router_surface_remains_importable_from_explicit_module():
    """当前生产只保留明确模块路径下的最小 Router 函数。"""
    assert callable(route_agent_action)


def test_only_graph_node_owns_the_router_in_chat_runtime():
    """ChatService 不再持有 Router，父图 route 节点是唯一所有者。"""
    dependencies = get_default_chat_service_dependencies()

    assert not hasattr(dependencies, "route_agent_action")
    assert graph_nodes.route_agent_action is route_agent_action
