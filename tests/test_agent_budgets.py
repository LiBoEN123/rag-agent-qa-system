"""Agent 工具调用和累计耗时预算的 Graph 集成测试。"""

import json

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.agent import framework_agent, framework_middleware, framework_tools
from app.config import get_settings
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry
from app.tools.result import tool_success
from tests.support.agent_runtime import build_agent_harness


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
    """按指定预算组装使用测试模型与工具的 Agent 图。"""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="budget_tool",
            version="1.0.0",
            description="用于验证 Agent 预算消息完整性的测试工具",
            output_schema={"value": "integer"},
            handler=handler,
            args_model=BudgetArgs,
        )
    )
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_agent_tool_provider",
        lambda: LocalToolProvider(registry),
    )
    settings = get_settings().model_copy(
        update={
            "agent_max_iterations": 6,
            "agent_max_tool_calls": max_tool_calls,
            "agent_max_tool_errors": 3,
            "agent_max_consecutive_same_tool_calls": 2,
            "agent_max_consecutive_same_tool_errors": 2,
            "agent_max_total_cost_ms": max_elapsed_ms,
        }
    )
    return build_agent_harness(settings)


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

    assert handler_calls == [1]
    assert output["tool_call_count"] == 1
    assert output["stop_reason"] == "max_tool_calls_reached"
    assert {item.tool_call_id for item in messages} == {"budget-1", "budget-2"}
    blocked = next(item for item in messages if item.tool_call_id == "budget-2")
    assert blocked.status == "error"
    assert str(blocked.content).startswith("Tool call limit exceeded")


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
    ticks = iter([10_000, 10_020])
    monkeypatch.setattr(framework_middleware, "_now_ms", lambda: next(ticks))
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
    assert output["agent_elapsed_ms"] == 20
    assert output["stop_reason"] == "max_total_cost_reached"
    assert payload["stop_reason"] == "max_total_cost_reached"
    assert iteration_trace["model_cost_ms"] == 20
