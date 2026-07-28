"""LangGraph 复用的 Agent Runtime 支持函数测试。"""

from copy import deepcopy

from app.agent.results import (
    build_agent_result,
    build_internal_error_result,
)
from app.agent.runtime import (
    resolve_llm_call_count,
    upsert_llm_usage_trace,
)
from app.graph import nodes as graph_nodes


def test_build_result_preserves_public_protocol_without_mutation():
    state = {
        "status": "clarify",
        "final_answer": "请说明具体对象。",
        "references": [],
        "reason": "问题存在指代不明",
        "decision": {"action": "clarify"},
        "observations": [],
        "tool_trace": [{"step": "decide"}],
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
    }
    original = deepcopy(state)

    result = build_agent_result(state).model_dump(mode="json")

    assert state == original
    assert result["status"] == "NEED_CLARIFICATION"
    assert result["need_clarification"] is True
    assert result["clarification_question"] == "请说明具体对象。"
    assert result["action"] == "clarify"
    assert result["references"] == []


def test_resolve_llm_call_count_keeps_exact_or_minimum_usage():
    assert (
        resolve_llm_call_count(
            decompose_llm_used=True,
            answer_llm_used=True,
            current_count=3,
        )
        == 3
    )
    assert (
        resolve_llm_call_count(
            decompose_llm_used=True,
            answer_llm_used=True,
            current_count=0,
        )
        == 2
    )


def test_upsert_llm_usage_trace_is_pure_and_idempotent():
    trace = [
        {"step": "decide"},
        {
            "step": "llm_usage",
            "router_llm_used": False,
            "decompose_llm_used": False,
            "answer_llm_used": False,
            "llm_call_count": 0,
        },
        {"step": "llm_usage", "llm_call_count": 99},
    ]
    original = deepcopy(trace)

    updated = upsert_llm_usage_trace(
        trace,
        decompose_llm_used=True,
        answer_llm_used=True,
        llm_call_count=2,
    )
    repeated = upsert_llm_usage_trace(
        updated,
        decompose_llm_used=True,
        answer_llm_used=True,
        llm_call_count=2,
    )

    assert trace == original
    assert updated is not trace
    assert repeated == updated
    assert sum(item.get("step") == "llm_usage" for item in repeated) == 1
    assert repeated[-1]["llm_call_count"] == 2


def test_internal_error_result_keeps_compatible_schema():
    result = build_internal_error_result(
        RuntimeError("graph exploded"),
        runtime_label="LangGraph Agent",
        trace_step="agent_langgraph_error",
    )

    assert result["status"] == "INTERNAL_ERROR"
    assert result["answer"] == "Agent 内部执行异常，请稍后重试"
    assert result["reason"] == "Agent 内部执行异常，请稍后重试"
    assert result["action"] == ""
    assert result["references"] == []
    assert result["tool_trace"] == [
        {
            "step": "agent_langgraph_error",
            "status": "error",
            "error": "internal_error",
        }
    ]


def test_finalize_node_uses_one_trace_snapshot_without_mutating_input(
    monkeypatch,
):
    state = {
        "trace_id": "trace-finalize",
        "session_id": "session-finalize",
        "original_question": "原问题",
        "effective_question": "改写问题",
        "decision": {"action": "agent"},
        "status": "ok",
        "final_answer": "最终回答",
        "references": [],
        "reason": "completed",
        "observations": [],
        "tool_trace": [{"step": "decide"}],
        "decompose_llm_used": False,
        "answer_llm_used": True,
        "llm_call_count": 1,
    }
    original = deepcopy(state)
    saved = []
    monkeypatch.setattr(
        graph_nodes,
        "maybe_save_agent_trace",
        lambda **kwargs: saved.append(kwargs),
    )

    updates = graph_nodes.finalize_node(state)

    assert state == original
    assert updates["tool_trace"] == updates["result"]["tool_trace"]
    assert updates["result"]["tool_trace"] is not updates["tool_trace"]
    assert sum(
        item.get("step") == "llm_usage"
        for item in updates["tool_trace"]
    ) == 1
    assert saved[0]["rewritten_question"] == "改写问题"
    assert saved[0]["tool_trace"] is updates["result"]["tool_trace"]
