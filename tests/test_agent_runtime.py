"""LangGraph 复用的 Agent Runtime 支持函数测试。"""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.agent import framework_tools, runtime
from app.contracts.chat import ToolMode
from app.tools.context import ToolInvocationContext

from app.agent.results import (
    build_internal_error_result,
    build_short_circuit_result,
    upsert_llm_usage_trace,
)
from app.agent.runtime import (
    prepare_agent_runtime_invocation,
)


def test_upsert_llm_usage_trace_is_pure_and_idempotent():
    trace = [
        {"step": "decide"},
        {
            "step": "llm_usage",
            "router_llm_used": False,
            "answer_llm_used": False,
            "llm_call_count": 0,
        },
        {"step": "llm_usage", "llm_call_count": 99},
    ]
    original = deepcopy(trace)

    updated = upsert_llm_usage_trace(
        trace,
        answer_llm_used=True,
        llm_call_count=2,
    )
    repeated = upsert_llm_usage_trace(
        updated,
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


def test_rag_only_reuses_local_cache_without_mcp_discovery(monkeypatch, settings_factory):
    discovery = Mock(return_value=[])
    monkeypatch.setattr(runtime, "list_mcp_tools_for_context", discovery)
    monkeypatch.setattr(framework_tools, "list_mcp_tools_for_context", discovery)
    settings = settings_factory(mcp_tools_enabled=True)
    cached = runtime.get_compiled_agent_runtime(settings)
    context = ToolInvocationContext(tool_mode=ToolMode.RAG_ONLY, permissions=("knowledge_base:read", "mcp:github"))
    invocation = prepare_agent_runtime_invocation(
        original_question="查询本科生学习年限", effective_question="查询本科生学习年限",
        trace_id="mode-test", session_id="session-mode",
        tool_context=context, settings=settings,
    )
    assert invocation.agent is cached
    assert invocation.tool_context.tool_mode is ToolMode.RAG_ONLY
    assert invocation.tool_context.permissions == context.permissions
    assert framework_tools.resolve_agent_tools(context, settings) == list(framework_tools.get_local_tools(settings))
    discovery.assert_not_called()
    full_context = ToolInvocationContext(permissions=context.permissions)
    assert full_context.tool_mode is ToolMode.FULL
    assert runtime.get_compiled_agent_runtime(settings, tool_context=full_context) is cached
    discovery.assert_called_once_with(full_context, settings=settings)


@pytest.mark.parametrize(("question", "action"), [("", "clarify"), ("这个怎么弄", "clarify"), ("你是谁？", "direct_answer")])
def test_rag_only_preserves_router_short_circuits(monkeypatch, question, action):
    compile_agent = Mock(side_effect=AssertionError("short circuit must not build agent"))
    monkeypatch.setattr(runtime, "get_compiled_agent_runtime", compile_agent)
    invocation = prepare_agent_runtime_invocation(
        original_question=question, effective_question=question,
        trace_id="router-mode", session_id=None,
        tool_context=ToolInvocationContext(tool_mode=ToolMode.RAG_ONLY),
    )
    assert invocation.decision["action"] == action
    assert invocation.agent is None
    compile_agent.assert_not_called()
