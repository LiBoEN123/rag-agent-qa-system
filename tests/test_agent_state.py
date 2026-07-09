import pytest
from pydantic import ValidationError

from app.agent.state import AgentObservation, AgentPlanStep, AgentState


def test_agent_plan_step_defaults():
    """验证计划步骤模型默认状态。"""
    step = AgentPlanStep(
        step_id=1,
        action="tool_call",
        tool_name="search_knowledge_base",
        tool_args={"query": "测试问题", "top_k": 5},
        reason="先检索知识库",
    )

    assert step.step_id == 1
    assert step.action == "tool_call"
    assert step.tool_name == "search_knowledge_base"
    assert step.tool_args["query"] == "测试问题"
    assert step.status == "pending"


def test_agent_observation_defaults():
    """验证观察结果模型默认错误信息。"""
    obs = AgentObservation(
        step_id=1,
        tool_name="search_knowledge_base",
        success=True,
        summary="检索到 5 个 chunk",
        output={"count": 5},
        cost_ms=123,
    )

    assert obs.success is True
    assert obs.output["count"] == 5
    assert obs.error == ""
    assert obs.cost_ms == 123


def test_agent_state_defaults():
    """验证 AgentState 初始化时使用安全默认值。"""
    state = AgentState(original_question="普通本科四年制最长学习年限是多少？")

    assert state.original_question == "普通本科四年制最长学习年限是多少？"
    assert state.rewritten_question == ""
    assert state.top_k == 5
    assert state.decompose_top_k == 10
    assert state.status == "running"
    assert state.max_steps == 5
    assert state.plan == []
    assert state.observations == []
    assert state.references == []
    assert state.tool_trace == []
    assert state.llm_call_count == 0


def test_agent_state_can_append_plan_and_observation():
    """验证状态对象可以累积计划步骤和观察结果。"""
    state = AgentState(original_question="普通本科四年制最长学习年限是多少？")
    step = AgentPlanStep(
        step_id=1,
        action="tool_call",
        tool_name="search_knowledge_base",
        tool_args={"query": "普通本科四年制最长学习年限是多少？", "top_k": 5},
        reason="先检索知识库",
    )
    obs = AgentObservation(
        step_id=1,
        tool_name="search_knowledge_base",
        success=True,
        summary="检索到 5 个 chunk",
        output={"count": 5},
        cost_ms=123,
    )

    state.plan.append(step)
    state.observations.append(obs)

    assert len(state.plan) == 1
    assert len(state.observations) == 1
    assert state.plan[0].tool_name == "search_knowledge_base"
    assert state.observations[0].success is True


def test_agent_state_list_defaults_are_not_shared():
    """验证 list/dict 字段不会在多个状态对象之间共享。"""
    state1 = AgentState(original_question="问题1")
    state2 = AgentState(original_question="问题2")

    state1.plan.append(
        AgentPlanStep(
            step_id=1,
            action="tool_call",
            tool_name="search_knowledge_base",
        )
    )

    assert len(state1.plan) == 1
    assert len(state2.plan) == 0


def test_agent_state_serialization_round_trip():
    """验证 AgentState 可以序列化为 dict 后再恢复。"""
    state = AgentState(
        original_question="普通本科四年制最长学习年限是多少？",
        trace_id="req_test",
    )
    state.plan.append(
        AgentPlanStep(
            step_id=1,
            action="tool_call",
            tool_name="search_knowledge_base",
            tool_args={"query": "普通本科四年制最长学习年限是多少？", "top_k": 5},
            reason="先检索知识库",
            status="success",
        )
    )
    state.observations.append(
        AgentObservation(
            step_id=1,
            tool_name="search_knowledge_base",
            success=True,
            summary="检索到 5 个 chunk",
            output={"count": 5},
            cost_ms=123,
        )
    )

    data = state.model_dump()
    restored = AgentState.model_validate(data)

    assert restored.original_question == state.original_question
    assert restored.plan[0].tool_name == state.plan[0].tool_name
    assert restored.observations[0].summary == state.observations[0].summary


def test_agent_state_field_constraints():
    """验证关键数字字段的约束生效。"""
    with pytest.raises(ValidationError):
        AgentPlanStep(step_id=0, action="tool_call")

    with pytest.raises(ValidationError):
        AgentState(original_question="问题", top_k=0)
