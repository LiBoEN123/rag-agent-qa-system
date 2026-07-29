"""单一平铺 AgentGraphState 的状态合同测试。"""

from copy import deepcopy
from typing import is_typeddict

import pytest

from app.agent import state as agent_state_types
from app.agent.framework_agent import FrameworkAgentState
from app.agent.runtime import validate_runtime_counts
from app.agent.state import AgentObservation
from app.agent.tool_events import build_tool_observation
from app.graph.nodes import initialize_node
from app.graph.state import AgentGraphState


def test_agent_observation_is_plain_typed_dict():
    """工具 observation 只包含稳定的普通 Python 数据。"""
    observation: AgentObservation = {
        "step_id": 1,
        "tool_name": "search_knowledge_base",
        "success": True,
        "summary": "检索到 5 个 chunk",
        "output": {"count": 5},
        "error": "",
        "cost_ms": 123,
    }

    assert is_typeddict(AgentObservation)
    assert observation["success"] is True
    assert observation["output"]["count"] == 5
    assert observation["error"] == ""
    assert observation["cost_ms"] == 123


def test_graph_state_contains_business_fields_without_nested_state():
    """父图 Schema 是唯一完整状态，不再声明嵌套业务对象。"""
    fields = AgentGraphState.__annotations__
    expected_business_fields = {
        "observations",
        "final_answer",
        "references",
        "status",
        "reason",
        "decompose_llm_used",
        "answer_llm_used",
        "llm_call_count",
        "tool_trace",
    }

    assert expected_business_fields <= fields.keys()
    assert "agent_state" not in fields
    assert "rewritten_question" not in fields
    assert not hasattr(agent_state_types, "AgentState")


def test_initialize_returns_independent_flat_business_lists():
    """不同初始化调用返回独立列表，且不会构造第二层状态。"""
    graph_input = {
        "original_question": "问题",
        "effective_question": "问题",
        "decision": {"action": "agent"},
    }

    first = initialize_node(graph_input)
    second = initialize_node(graph_input)
    first["observations"].append(
        {
            "step_id": 1,
            "tool_name": "simple_calculator",
            "success": True,
            "summary": "完成",
            "output": {"result": 3},
            "error": "",
            "cost_ms": 1,
        }
    )

    assert first["status"] == "running"
    assert first["top_k"] == 5
    assert first["decompose_top_k"] == 10
    assert len(first["observations"]) == 1
    assert second["observations"] == []
    assert first["references"] is not second["references"]
    assert first["tool_trace"] is not second["tool_trace"]
    assert "agent_state" not in first


def test_flat_state_uses_only_checkpoint_safe_business_values():
    """初始化业务字段由普通容器、标量和 LangChain Message 组成。"""
    updates = initialize_node(
        {
            "original_question": "问题",
            "effective_question": "改写问题",
            "trace_id": "trace-state",
            "decision": {"action": "direct_answer"},
        }
    )
    snapshot = deepcopy(updates)
    static_governance_fields = {
        "principal_id",
        "principal_type",
        "roles",
        "permissions",
        "tenant_id",
        "request_source",
        "idempotency_key",
    }

    assert static_governance_fields.isdisjoint(updates)
    assert updates == snapshot
    assert all(not hasattr(value, "model_dump") for value in updates.values())


def test_graph_state_excludes_static_governance_and_legacy_fields():
    fields = AgentGraphState.__annotations__
    static_governance_fields = {
        "principal_id",
        "principal_type",
        "roles",
        "permissions",
        "tenant_id",
        "request_source",
        "idempotency_key",
    }

    assert static_governance_fields.isdisjoint(fields)
    assert "fallback_to_" + "legacy" not in fields
    assert "decision" not in FrameworkAgentState.__annotations__


@pytest.mark.parametrize(
    ("top_k", "decompose_top_k"),
    [(0, 10), (11, 10), (5, 0), (5, 21), (True, 10), (5.5, 10)],
)
def test_runtime_count_constraints_reject_invalid_inputs(
    top_k,
    decompose_top_k,
):
    """删除 Pydantic 状态后，Runtime 入口仍保留原数字边界。"""
    with pytest.raises(ValueError):
        validate_runtime_counts(
            top_k=top_k,
            decompose_top_k=decompose_top_k,
        )


def test_tool_observation_builder_preserves_numeric_constraints():
    with pytest.raises(ValueError, match="step_id"):
        build_tool_observation(
            step_id=0,
            tool_name="simple_calculator",
            result={"ok": True},
            cost_ms=0,
        )

    with pytest.raises(ValueError, match="cost_ms"):
        build_tool_observation(
            step_id=1,
            tool_name="simple_calculator",
            result={"ok": True},
            cost_ms=-1,
        )
