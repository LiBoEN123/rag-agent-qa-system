import pytest
from pydantic import ValidationError

from app.agent.framework_agent import FrameworkAgentState
from app.agent.state import AgentObservation, AgentState
from app.graph.state import AgentGraphState


def test_agent_observation_defaults():
    """工具 observation 保留稳定字段和安全默认值。"""
    observation = AgentObservation(
        step_id=1,
        tool_name="search_knowledge_base",
        success=True,
        summary="检索到 5 个 chunk",
        output={"count": 5},
        cost_ms=123,
    )

    assert observation.success is True
    assert observation.output["count"] == 5
    assert observation.error == ""
    assert observation.cost_ms == 123


def test_agent_state_defaults_without_plan_protocol():
    """AgentState 不再保存计划或旧步骤上限。"""
    state = AgentState(original_question="普通本科四年制最长学习年限是多少？")

    assert state.original_question == "普通本科四年制最长学习年限是多少？"
    assert state.rewritten_question == ""
    assert state.top_k == 5
    assert state.decompose_top_k == 10
    assert state.status == "running"
    assert state.observations == []
    assert state.references == []
    assert state.tool_trace == []
    assert state.llm_call_count == 0
    assert "router_llm_used" not in AgentState.model_fields
    assert "plan" not in AgentState.model_fields
    assert "max_steps" not in AgentState.model_fields


def test_agent_state_list_defaults_are_not_shared():
    """可变列表字段不会在不同状态对象之间共享。"""
    first = AgentState(original_question="问题1")
    second = AgentState(original_question="问题2")
    first.observations.append(
        AgentObservation(
            step_id=1,
            tool_name="simple_calculator",
            success=True,
        )
    )

    assert len(first.observations) == 1
    assert second.observations == []


def test_agent_state_serialization_round_trip():
    """AgentState 的 observation 与兼容结果字段可序列化往返。"""
    state = AgentState(
        original_question="普通本科四年制最长学习年限是多少？",
        trace_id="req_test",
        observations=[
            AgentObservation(
                step_id=1,
                tool_name="search_knowledge_base",
                success=True,
                summary="检索到 5 个 chunk",
                output={"count": 5},
                cost_ms=123,
            )
        ],
    )

    restored = AgentState.model_validate(state.model_dump())

    assert restored.original_question == state.original_question
    assert restored.observations[0].summary == state.observations[0].summary


def test_agent_state_excludes_static_governance_fields_from_checkpoints():
    state = AgentState(original_question="问题")
    payload = state.model_dump(mode="json")
    restored = AgentState.model_validate(payload)
    static_governance_fields = {
        "principal_id",
        "principal_type",
        "roles",
        "permissions",
        "tenant_id",
        "request_source",
        "idempotency_key",
    }

    assert static_governance_fields.isdisjoint(AgentState.model_fields)
    assert static_governance_fields.isdisjoint(payload)
    assert restored.original_question == "问题"
    assert all(not hasattr(value, "write") for value in payload.values())


def test_graph_state_excludes_static_governance_and_legacy_fallback_fields():
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
    assert "fallback_to_legacy" not in fields
    assert "decision" not in FrameworkAgentState.__annotations__


def test_agent_state_field_constraints():
    """关键数字字段仍执行 Pydantic 约束。"""
    with pytest.raises(ValidationError):
        AgentObservation(step_id=0)

    with pytest.raises(ValidationError):
        AgentState(original_question="问题", top_k=0)
