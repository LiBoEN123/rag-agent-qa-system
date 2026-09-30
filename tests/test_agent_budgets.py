"""Agent 工具调用和累计耗时预算的 Graph 集成测试。"""

import pytest
from pydantic import RootModel


from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.agent import framework_agent, framework_tools
from app.config import get_settings
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from tests.support.local_tools import local_tools
from app.tools.result import tool_success
from tests.support.agent_runtime import build_agent_harness


class BudgetArgs(BaseModel):
    value: int


class BudgetModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return self.response


def _setup(monkeypatch, *, model, handler, max_tool_calls=10, max_elapsed_ms=120000):
    """按指定预算组装使用测试模型与工具的 Agent 图。"""
    registry = []
    registry.append(
        ToolSpec(
            name="budget_tool",
            version="1.0.0",
            description="用于验证 Agent 预算消息完整性的测试工具",
            output_model=RootModel[dict],
            handler=handler,
            args_model=BudgetArgs,
        )
    )
    monkeypatch.setattr(framework_agent, "get_agent_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_local_tools",
        lambda settings: local_tools(registry),
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


async def _invoke(graph):
    return [state async for state in graph.astream(
        {
            "original_question": "预算测试",
            "effective_question": "预算测试",
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
    )][-1]


@pytest.mark.asyncio
async def test_tool_budget_limit_adds_messages_for_every_unexecuted_call(monkeypatch):
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

    output = await _invoke(graph)
    messages = [item for item in output["messages"] if isinstance(item, ToolMessage)]

    assert handler_calls == [1]
    assert sum(
        bool(event.get("counted_call"))
        for event in output.get("governance_events", [])
    ) == 1
    assert output["stop_reason"] == "max_tool_calls_reached"
    assert {item.tool_call_id for item in messages} == {"budget-1", "budget-2"}
    blocked = next(item for item in messages if item.tool_call_id == "budget-2")
    assert blocked.status == "error"
    assert str(blocked.content).startswith("Tool call limit exceeded")


