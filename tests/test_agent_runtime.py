"""LangGraph 复用的 Agent Runtime 支持函数测试。"""

from copy import deepcopy

from app.agent.results import (
    build_internal_error_result,
    build_short_circuit_result,
    resolve_llm_call_count,
    upsert_llm_usage_trace,
)
from app.agent.runtime import (
    prepare_agent_runtime_invocation,
)


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


def test_runtime_short_circuit_uses_one_trace_snapshot():
    invocation = prepare_agent_runtime_invocation(
        original_question="你是谁？",
        effective_question="你是谁？",
        top_k=5,
        decompose_top_k=10,
        trace_id="trace-finalize",
        session_id="session-finalize",
    )

    result = build_short_circuit_result(invocation)

    assert sum(
        item.get("step") == "llm_usage"
        for item in result["tool_trace"]
    ) == 1
    assert result["status"] == "OK"
    assert result["answer"]
    assert result["action"] == "direct_answer"
