"""LangChain ``create_agent`` 唯一 Agent Loop 的关键版本契约。"""

from copy import deepcopy
from langchain_core.messages import AIMessage

from app.agent.framework_agent import (
    build_framework_agent,
)
from app.agent.tool_events import normalize_governance_events


class _ProbeModel:
    profile: dict = {}

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def invoke(self, messages):
        del messages
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "handoff_probe",
                    "args": {"value": 7},
                    "id": "handoff-probe-call",
                    "type": "tool_call",
                }
            ],
        )


def test_production_agent_is_framework_model_tool_graph():
    """生产 Runtime 直接包含 create_agent 标准节点与调用限制。"""
    node_names = set(build_framework_agent().get_graph().nodes)

    assert {"model", "tools"} <= node_names
    assert "ModelCallLimitMiddleware.before_model" in node_names
    assert "ToolCallLimitMiddleware.after_model" in node_names


def test_governance_event_normalization_is_pure_and_stable():
    """Runtime 出口的一次性事件归一化不修改治理事件。"""
    events = [
            {
                "kind": "model",
                "ok": True,
                "has_tool_calls": True,
                "requested_tool_call_count": 1,
                "cost_ms": 2,
            },
            {
                "kind": "tool",
                "name": "simple_calculator",
                "args": {"expression": "1+2"},
                "result": {
                    "ok": True,
                    "status": "success",
                    "output": {"result": 3},
                    "runtime": {"attempt_count": 1},
                },
                "ok": True,
                "counted_call": True,
                "cost_ms": 3,
                "tool_call_id": "calc-1",
            },
            {
                "kind": "tool",
                "name": "simple_calculator",
                "result": {"ok": False, "error": "blocked"},
                "counted_call": False,
                "error_code": "repeated_tool_call",
                "stop_reason": "repeated_tool_call",
                "tool_call_id": "calc-2",
            },
    ]
    original = deepcopy(events)

    observations, trace = normalize_governance_events(events)
    repeated = normalize_governance_events(events)

    assert events == original
    assert observations == [
        {
            "step_id": 1,
            "tool_name": "simple_calculator",
            "success": True,
            "summary": "计算工具执行完成",
            "output": {"result": 3},
            "error": "",
            "cost_ms": 3,
        }
    ]
    assert [item["step"] for item in trace] == [
        "agent_iteration",
        "tool_call",
        "tool_call",
    ]
    assert repeated == (observations, trace)
