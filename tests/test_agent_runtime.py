"""LangGraph 复用的 Agent Runtime 支持函数测试。"""

from app.agent.runtime import (
    build_internal_error_result,
    build_v2_result,
    map_agent_status_to_public_status,
)
from app.agent.state import AgentState


def test_map_agent_status_to_public_status():
    assert map_agent_status_to_public_status("ok") == "OK"
    assert map_agent_status_to_public_status("clarify") == "NEED_CLARIFICATION"
    assert map_agent_status_to_public_status("direct_answer") == "OK"
    assert map_agent_status_to_public_status("tool_error") == "TOOL_ERROR"
    assert map_agent_status_to_public_status("model_error") == "MODEL_ERROR"
    assert map_agent_status_to_public_status("no_evidence") == "NO_RELEVANT_INFO"
    assert map_agent_status_to_public_status("error") == "INTERNAL_ERROR"


def test_build_result_preserves_public_protocol():
    state = AgentState(
        original_question="请补充对象",
        status="clarify",
        final_answer="请说明具体对象。",
        reason="问题存在指代不明",
    )

    result = build_v2_result(state, {"action": "clarify"})

    assert result["status"] == "NEED_CLARIFICATION"
    assert result["need_clarification"] is True
    assert result["clarification_question"] == "请说明具体对象。"
    assert result["action"] == "clarify"
    assert result["references"] == []


def test_internal_error_result_keeps_compatible_schema():
    result = build_internal_error_result(
        RuntimeError("graph exploded"),
        {"action": "direct_answer"},
        runtime_label="LangGraph Agent",
        trace_step="agent_langgraph_error",
    )

    assert result["status"] == "INTERNAL_ERROR"
    assert result["reason"] == "graph exploded"
    assert result["action"] == "direct_answer"
    assert result["references"] == []
    assert result["tool_trace"] == [
        {
            "step": "agent_langgraph_error",
            "status": "error",
            "error": "graph exploded",
        }
    ]
