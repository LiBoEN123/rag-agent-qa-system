"""Agent 工具调用和累计耗时预算的 Graph 集成测试。"""

import json

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.graph import nodes as graph_nodes
from app.graph.builder import build_agent_graph
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry
from app.tools.result import tool_success


class BudgetArgs(BaseModel):
    value: int


class BudgetModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self.response


def _setup(monkeypatch, *, model, handler, max_tool_calls=10, max_elapsed_ms=120000):
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="budget_tool",
            version="1.0.0",
            description="用于验证 Agent 预算消息完整性的测试工具",
            args_schema={"value": "integer"},
            output_schema={"value": "integer"},
            handler=handler,
            args_model=BudgetArgs,
        )
    )
    monkeypatch.setattr(graph_nodes, "get_agent_chat_model", lambda: model)
    monkeypatch.setattr(
        graph_nodes,
        "get_agent_tool_provider",
        lambda: LocalToolProvider(registry),
    )
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_ITERATIONS", 6)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOOL_CALLS", max_tool_calls)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOOL_ERRORS", 3)
    monkeypatch.setattr(
        graph_nodes.Settings,
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS",
        2,
    )
    monkeypatch.setattr(
        graph_nodes.Settings,
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS",
        2,
    )
    monkeypatch.setattr(
        graph_nodes.Settings,
        "AGENT_MAX_TOTAL_COST_MS",
        max_elapsed_ms,
    )
    return build_agent_graph().compile()


def _invoke(graph):
    return graph.invoke(
        {
            "original_question": "预算测试",
            "effective_question": "预算测试",
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": {"action": "agent"},
            "trace_id": "trace-budget",
            "session_id": "session-budget",
        },
        context=ToolInvocationContext(
            trace_id="trace-budget",
            session_id="session-budget",
            principal_id="budget-test",
            source="agent_budget_test",
        ),
    )


def test_tool_budget_limit_adds_messages_for_every_unexecuted_call(monkeypatch):
    handler_calls: list[int] = []

    def handler(args, context):
        handler_calls.append(args["value"])
        return tool_success("budget_tool", args, {"value": args["value"]})

    model = BudgetModel(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "budget_tool",
                    "args": {"value": 1},
                    "id": "budget-1",
                    "type": "tool_call",
                },
                {
                    "name": "budget_tool",
                    "args": {"value": 2},
                    "id": "budget-2",
                    "type": "tool_call",
                },
            ],
        )
    )
    graph = _setup(
        monkeypatch,
        model=model,
        handler=handler,
        max_tool_calls=1,
    )

    output = _invoke(graph)
    messages = [item for item in output["messages"] if isinstance(item, ToolMessage)]

    assert handler_calls == []
    assert output["tool_call_count"] == 0
    assert output["stop_reason"] == "max_tool_calls_reached"
    assert {item.tool_call_id for item in messages} == {"budget-1", "budget-2"}
    assert all(
        json.loads(item.content)["handler_executed"] is False
        for item in messages
    )
    assert all(
        json.loads(item.content)["error_source"] == "agent_guardrail"
        for item in messages
    )


def test_model_elapsed_budget_blocks_tool_without_real_sleep(monkeypatch):
    handler_calls: list[int] = []

    def handler(args, context):
        handler_calls.append(args["value"])
        return tool_success("budget_tool", args, {"value": args["value"]})

    model = BudgetModel(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "budget_tool",
                    "args": {"value": 1},
                    "id": "elapsed-call",
                    "type": "tool_call",
                }
            ],
        )
    )
    ticks = iter([10.0, 10.020])
    monkeypatch.setattr(graph_nodes, "_agent_perf_counter", lambda: next(ticks))
    graph = _setup(
        monkeypatch,
        model=model,
        handler=handler,
        max_elapsed_ms=10,
    )

    output = _invoke(graph)
    payload = json.loads(
        next(
            item
            for item in output["messages"]
            if isinstance(item, ToolMessage)
        ).content
    )
    iteration_trace = next(
        item
        for item in output["result"]["tool_trace"]
        if item.get("step") == "agent_iteration"
    )

    assert model.calls == 1
    assert handler_calls == []
    assert output["agent_elapsed_ms"] == 19
    assert output["stop_reason"] == "max_total_cost_reached"
    assert payload["stop_reason"] == "max_total_cost_reached"
    assert iteration_trace["model_cost_ms"] == 19
    assert iteration_trace["budget_snapshot"]["elapsed_limit_ms"] == 10
