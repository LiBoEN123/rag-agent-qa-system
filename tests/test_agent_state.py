"""无 Parent Graph 后的 Agent 状态与输入合同测试。"""

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from app.agent.framework_agent import FrameworkAgentState
from app.agent.runtime import (
    prepare_agent_runtime_invocation,
    validate_runtime_counts,
)
from app.agent.tool_events import build_tool_trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parent_agent_graph_state_module_is_removed():
    """生产状态只剩 create_agent 内部 Schema 与最终 Result。"""
    assert not (PROJECT_ROOT / "app" / "graph" / "state.py").is_file()
    fields = FrameworkAgentState.__annotations__
    assert "knowledge_base_result" in fields
    assert "framework_events_applied" not in fields
    assert "pending_rag_result" not in fields
    assert "decision" not in fields


def test_runtime_input_uses_effective_question_and_request_rag_context():
    invocation = prepare_agent_runtime_invocation(
        original_question="它什么时候开始？",
        effective_question="奖学金申请什么时候开始？",
        top_k=4,
        decompose_top_k=8,
        trace_id="trace-state",
        session_id="session-state",
    )

    assert invocation.agent_input == {
        "messages": [HumanMessage(content="奖学金申请什么时候开始？")]
    }
    assert invocation.tool_context.effective_question == (
        "奖学金申请什么时候开始？"
    )
    assert invocation.tool_context.top_k == 4
    assert invocation.tool_context.decompose_top_k == 8


@pytest.mark.parametrize(
    ("top_k", "decompose_top_k"),
    [(0, 10), (11, 10), (5, 0), (5, 21), (True, 10), (5.5, 10)],
)
def test_runtime_count_constraints_reject_invalid_inputs(
    top_k,
    decompose_top_k,
):
    with pytest.raises(ValueError):
        validate_runtime_counts(
            top_k=top_k,
            decompose_top_k=decompose_top_k,
        )


def test_tool_trace_builder_preserves_numeric_constraints():
    with pytest.raises(ValueError, match="step_id"):
        build_tool_trace(
            step_id=0,
            tool_name="list_documents",
            tool_args={},
            result={"ok": True},
            cost_ms=0,
        )

    with pytest.raises(ValueError, match="cost_ms"):
        build_tool_trace(
            step_id=1,
            tool_name="list_documents",
            tool_args={},
            result={"ok": True},
            cost_ms=-1,
        )

    trace = build_tool_trace(
        step_id=1,
        tool_name="list_documents",
        tool_args={},
        result={"ok": True},
        cost_ms=0,
    )
    assert trace["step_id"] == 1
    assert trace["cost_ms"] == 0
