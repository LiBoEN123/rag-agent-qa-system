"""直接 ``create_agent`` 构建与无 Parent Graph 架构测试。"""

from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.final_response import FinalAgentResponse
from app.agent import framework_agent
from app.tools.context import build_default_tool_context
from tests.support.agent_runtime import FakeToolCallingModel
from app.prompts import AGENT_SYSTEM_PROMPT
from app.agent.runtime import (
    get_compiled_agent_runtime,
    reset_agent_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_runtime_cache():
    reset_agent_runtime()
    yield
    reset_agent_runtime()


def test_create_agent_is_the_only_agent_graph():
    node_names = set(get_compiled_agent_runtime().get_graph().nodes)

    assert {"__start__", "model", "tools", "__end__"} <= node_names
    assert {
        "route",
        "initialize",
        "framework_agent",
        "adapt_framework",
        "rag_subgraph",
        "compose",
        "finalize",
    }.isdisjoint(node_names)


def test_compiled_create_agent_is_reused_for_same_settings():
    assert get_compiled_agent_runtime() is get_compiled_agent_runtime()


def test_parent_graph_modules_are_removed():
    graph_root = PROJECT_ROOT / "app" / "graph"

    assert list(graph_root.rglob("*.py")) == []


def test_builder_passes_retrieval_guidance_through_native_system_prompt(monkeypatch):
    create = Mock()
    monkeypatch.setattr(framework_agent, "create_agent", create)

    framework_agent.build_framework_agent(tools=[])

    create.assert_called_once()
    assert create.call_args.kwargs["system_prompt"] is AGENT_SYSTEM_PROMPT
    assert "search_knowledge_base" in AGENT_SYSTEM_PROMPT
    assert "ToolMessage" in AGENT_SYSTEM_PROMPT
    assert "修改 retrieval query" in AGENT_SYSTEM_PROMPT
    assert "不得用模型内部知识补齐" in AGENT_SYSTEM_PROMPT
    strategy = create.call_args.kwargs["response_format"]
    assert isinstance(strategy, ToolStrategy)
    assert strategy.schema is FinalAgentResponse
    assert strategy.handle_errors is True


@pytest.mark.asyncio
async def test_structured_call_does_not_consume_business_tool_budget(
    monkeypatch, settings_factory,
):
    """真实绑定和框架执行证明合成调用不会多计一次业务工具。"""
    settings = settings_factory(agent_max_tool_calls=1)
    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{
            "name": "list_documents", "args": {}, "id": "list",
        }]),
        FinalAgentResponse(outcome="answer", answer="列表为空"),
    ])
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    monkeypatch.setattr("app.rag.pipeline.preview_docs", lambda: [])
    observed_counts = []
    native_after = ToolCallLimitMiddleware.after_model

    def record(self, state, runtime):
        update = native_after(self, state, runtime)
        if update and "run_tool_call_count" in update:
            observed_counts.append(update["run_tool_call_count"]["__all__"])
        return update

    monkeypatch.setattr(ToolCallLimitMiddleware, "after_model", record)
    graph = framework_agent.build_framework_agent(settings)
    output = await graph.ainvoke(
        {"messages": [HumanMessage(content="列出文档")]},
        context=build_default_tool_context(settings=settings),
    )

    assert output["structured_response"].answer == "列表为空"
    assert model.binding_kwargs["tool_choice"] == "any"
    assert max(observed_counts) == 1
    assert [e["name"] for e in output["governance_events"] if e["kind"] == "tool"] == [
        "list_documents",
    ]


@pytest.mark.asyncio
async def test_schema_retries_remain_bounded_by_native_model_limit(monkeypatch, settings_factory):
    from types import SimpleNamespace

    from app.agent.results import build_framework_agent_result

    settings = settings_factory(agent_max_iterations=2, agent_max_tool_calls=1)
    model = FakeToolCallingModel([
        lambda model, messages: model.final_response(outcome="answer", answer=""),
        lambda model, messages: model.final_response(outcome="clarify", clarification_question=""),
    ])
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    output = await framework_agent.build_framework_agent(settings).ainvoke(
        {"messages": [HumanMessage(content="测试问题")]},
        context=build_default_tool_context(settings=settings),
    )
    result = build_framework_agent_result(SimpleNamespace(decision={"action": "agent"}, route_cost_ms=0), output)
    assert result["status"] == "MODEL_ERROR"
    assert result["llm_call_count"] == len(model.received_messages) == 2
    assert not any(event["kind"] == "tool" for event in output["governance_events"])


@pytest.mark.asyncio
async def test_business_limit_overrides_same_turn_structured_response(monkeypatch, settings_factory):
    from types import SimpleNamespace

    from app.agent.results import build_framework_agent_result

    settings = settings_factory(agent_max_tool_calls=1)

    def final_with_excess_call(model, messages):
        final = model.final_response(outcome="answer", answer="不应返回")
        final.tool_calls.append({"name": "list_documents", "args": {}, "id": "excess"})
        return final

    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{"name": "list_documents", "args": {}, "id": "first"}]),
        final_with_excess_call,
    ])
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    handler = Mock(return_value=[])
    monkeypatch.setattr("app.rag.pipeline.preview_docs", handler)
    output = await framework_agent.build_framework_agent(settings).ainvoke(
        {"messages": [HumanMessage(content="列出文档")]},
        context=build_default_tool_context(settings=settings),
    )
    result = build_framework_agent_result(SimpleNamespace(decision={"action": "agent"}, route_cost_ms=0), output)
    assert output["structured_response"].answer == "不应返回"
    assert result["status"] == "MODEL_ERROR"
    assert result["reason"] == "max_tool_calls_reached"
    assert "不应返回" not in str(result)
    handler.assert_called_once()
