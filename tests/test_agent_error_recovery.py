"""Agent 错误预算、循环防护与终止治理的真实工具链测试。"""

import json

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.agent import framework_agent, framework_tools
from app.config import get_settings
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolRiskLevel, ToolSpec
from app.tools.errors import ToolErrorCode
from tests.support.local_tools import local_tools
from app.tools.result import tool_error, tool_success
from tests.support.agent_runtime import build_agent_harness


class ValueArgs(BaseModel):
    value: int


class FakeModel:
    """按序返回工具调用并记录第二轮是否真实收到 ToolMessage。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.received_messages: list[list] = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.received_messages.append(list(messages))
        return self.responses.pop(0)
def _call(tool_name: str, value: int | None, call_id: str) -> dict:
    args = {} if value is None else {"value": value}
    return {
        "name": tool_name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def _spec(
    name: str,
    handler,
    *,
    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY,
    required_permissions: tuple[str, ...] = (),
    requires_approval: bool = False,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        version="1.0.0",
        description=f"执行 {name} 的受控 Agent 错误恢复测试工具",
        output_schema={"value": "integer"},
        handler=handler,
        args_model=ValueArgs,
        risk_level=risk_level,
        required_permissions=required_permissions,
        idempotent=True,
        requires_approval=requires_approval,
    )


def _invoke(
    monkeypatch,
    *,
    registry: list[ToolSpec],
    model: FakeModel,
    context: ToolInvocationContext | None = None,
    max_tool_errors: int = 3,
    max_same_calls: int = 2,
    max_same_errors: int = 2,
):
    """按指定错误预算组装并调用受控 Agent 图。"""
    provider = local_tools(registry)
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_local_tools",
        lambda settings: provider,
    )
    settings = get_settings().model_copy(
        update={
            "agent_max_iterations": 10,
            "agent_max_tool_calls": 20,
            "agent_max_tool_errors": max_tool_errors,
            "agent_max_consecutive_same_tool_calls": max_same_calls,
            "agent_max_consecutive_same_tool_errors": max_same_errors,
            "agent_max_total_cost_ms": 120000,
        }
    )
    resolved_context = context or ToolInvocationContext(
        trace_id="trace-guardrail",
        session_id="session-guardrail",
        principal_id="guardrail-test",
        permissions=("test:write", "protected:read"),
        source="agent_error_recovery_test",
        idempotency_key="idem-test",
    )
    return build_agent_harness(settings).invoke(
        {
            "original_question": "执行测试工具",
            "effective_question": "执行测试工具",
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": {"action": "agent", "router_llm_used": False},
            "trace_id": resolved_context.trace_id,
            "session_id": resolved_context.session_id,
        },
        context=resolved_context,
    )


def _payloads(output) -> dict[str, dict]:
    return {
        message.tool_call_id: json.loads(message.content)
        for message in output["messages"]
        if isinstance(message, ToolMessage)
    }


def test_max_tool_errors_stops_on_third_failure_without_another_model_call(
    monkeypatch,
):
    calls: list[int] = []

    def handler(args, context):
        calls.append(args["value"])
        return tool_error(
            "failing_tool",
            args,
            "controlled failure",
            error_code=ToolErrorCode.EXECUTION_ERROR.value,
        )

    registry = []
    registry.append(_spec("failing_tool", handler))
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[_call("failing_tool", 1, "e1")]),
            AIMessage(content="", tool_calls=[_call("failing_tool", 2, "e2")]),
            AIMessage(content="", tool_calls=[_call("failing_tool", 3, "e3")]),
        ]
    )

    output = _invoke(monkeypatch, registry=registry, model=model)

    assert calls == [1, 2, 3]
    assert len(model.received_messages) == 3
    assert output["tool_error_count"] == 3
    assert output["stop_reason"] == "max_tool_errors_reached"
    assert output["status"] == "model_error"
    assert output["result"]["status"] == "MODEL_ERROR"


def test_third_identical_call_is_blocked_before_structured_tool(monkeypatch):
    calls: list[int] = []

    def handler(args, context):
        calls.append(args["value"])
        return tool_success("repeatable_tool", args, {"value": args["value"]})

    registry = []
    registry.append(_spec("repeatable_tool", handler))
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[_call("repeatable_tool", 1, "r1")]),
            AIMessage(content="", tool_calls=[_call("repeatable_tool", 1, "r2")]),
            AIMessage(content="", tool_calls=[_call("repeatable_tool", 1, "r3")]),
        ]
    )

    output = _invoke(monkeypatch, registry=registry, model=model)
    payload = _payloads(output)["r3"]

    assert calls == [1, 1]
    assert output["tool_call_count"] == 2
    assert output["consecutive_same_tool_call_count"] == 3
    assert output["stop_reason"] == "repeated_tool_call"
    assert payload["error_source"] == "agent_guardrail"
    assert payload["handler_executed"] is False
    assert payload["stop_reason"] == "repeated_tool_call"


def test_same_tool_with_different_arguments_does_not_trigger_call_guard(
    monkeypatch,
):
    calls: list[int] = []

    def handler(args, context):
        calls.append(args["value"])
        return tool_success("varying_tool", args, {"value": args["value"]})

    registry = []
    registry.append(_spec("varying_tool", handler))
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[_call("varying_tool", 1, "v1")]),
            AIMessage(content="", tool_calls=[_call("varying_tool", 2, "v2")]),
            AIMessage(content="done"),
        ]
    )

    output = _invoke(monkeypatch, registry=registry, model=model)

    assert calls == [1, 2]
    assert output["stop_reason"] == "completed"
    assert output["consecutive_same_tool_call_count"] == 1


def test_second_identical_error_is_saved_then_stops(monkeypatch):
    calls: list[int] = []

    def handler(args, context):
        calls.append(args["value"])
        return tool_error(
            "same_error_tool",
            args,
            "same controlled failure",
            error_code=ToolErrorCode.EXECUTION_ERROR.value,
        )

    registry = []
    registry.append(_spec("same_error_tool", handler))
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[_call("same_error_tool", 1, "s1")]),
            AIMessage(content="", tool_calls=[_call("same_error_tool", 1, "s2")]),
        ]
    )

    output = _invoke(monkeypatch, registry=registry, model=model)
    payload = _payloads(output)["s2"]

    assert calls == [1, 1]
    assert len(model.received_messages) == 2
    assert output["stop_reason"] == "repeated_tool_error"
    assert payload["failure_kind"] == "recoverable"
    assert payload["stop_reason"] == "repeated_tool_error"


def test_same_error_code_with_different_arguments_is_not_same_error_loop(
    monkeypatch,
):
    def handler(args, context):
        return tool_error(
            "different_error_calls",
            args,
            "controlled failure",
            error_code=ToolErrorCode.EXECUTION_ERROR.value,
        )

    registry = []
    registry.append(_spec("different_error_calls", handler))
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[_call("different_error_calls", 1, "d1")],
            ),
            AIMessage(
                content="",
                tool_calls=[_call("different_error_calls", 2, "d2")],
            ),
            AIMessage(content="handled"),
        ]
    )

    output = _invoke(monkeypatch, registry=registry, model=model)

    assert len(model.received_messages) == 3
    assert output["consecutive_same_tool_error_count"] == 1
    assert output["stop_reason"] == "completed"


def test_approval_required_stops_and_aborts_later_batch_call(monkeypatch):
    approval_calls: list[int] = []
    later_calls: list[int] = []

    def approval_handler(args, context):
        approval_calls.append(args["value"])
        return tool_success("approval_tool", args, {"value": args["value"]})

    def later_handler(args, context):
        later_calls.append(args["value"])
        return tool_success("later_tool", args, {"value": args["value"]})

    registry = []
    registry.append(
        _spec(
            "approval_tool",
            approval_handler,
            risk_level=ToolRiskLevel.INTERNAL_WRITE,
            required_permissions=("test:write",),
            requires_approval=True,
        )
    )
    registry.append(_spec("later_tool", later_handler))
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _call("approval_tool", 1, "approval"),
                    _call("later_tool", 2, "later"),
                ],
            )
        ]
    )

    output = _invoke(monkeypatch, registry=registry, model=model)
    payloads = _payloads(output)

    assert approval_calls == []
    assert later_calls == []
    assert len(model.received_messages) == 1
    assert output["stop_reason"] == "approval_required"
    assert payloads["approval"]["error_code"] == "approval_required"
    assert payloads["approval"]["handler_executed"] is False
    assert payloads["later"]["error_code"] == "agent_batch_aborted"
    assert payloads["later"]["handler_executed"] is False


def test_missing_idempotency_key_stops_without_handler_or_model_retry(
    monkeypatch,
):
    calls: list[int] = []

    def handler(args, context):
        calls.append(args["value"])
        return tool_success("write_tool", args, {"value": args["value"]})

    registry = []
    registry.append(
        _spec(
            "write_tool",
            handler,
            risk_level=ToolRiskLevel.INTERNAL_WRITE,
            required_permissions=("test:write",),
        )
    )
    model = FakeModel(
        [AIMessage(content="", tool_calls=[_call("write_tool", 1, "write")])]
    )
    context = ToolInvocationContext(
        trace_id="trace-missing-idem",
        principal_id="writer",
        permissions=("test:write",),
        source="agent_error_recovery_test",
    )

    output = _invoke(
        monkeypatch,
        registry=registry,
        model=model,
        context=context,
    )

    assert calls == []
    assert len(model.received_messages) == 1
    assert output["stop_reason"] == "missing_idempotency_key"
    assert _payloads(output)["write"]["handler_executed"] is False


def test_runtime_unavailable_is_terminal(monkeypatch):
    from app.tools import execution
    from app.tools.runtime import ProjectToolExecutorError

    calls: list[int] = []

    def handler(args, context):
        calls.append(args["value"])
        return tool_success("runtime_tool", args, {"value": args["value"]})

    registry = []
    registry.append(_spec("runtime_tool", handler))
    model = FakeModel(
        [AIMessage(content="", tool_calls=[_call("runtime_tool", 1, "runtime")])]
    )

    def unavailable(*args, **kwargs):
        del args, kwargs
        raise ProjectToolExecutorError

    monkeypatch.setattr(execution, "run_with_timeout", unavailable)
    output = _invoke(monkeypatch, registry=registry, model=model)

    assert calls == []
    assert len(model.received_messages) == 1
    assert output["stop_reason"] == "runtime_unavailable"
    assert _payloads(output)["runtime"]["failure_kind"] == "terminal_runtime"


def test_hidden_tool_is_denied_and_public_batch_call_is_aborted(monkeypatch):
    protected_calls: list[int] = []
    normal_calls: list[int] = []

    def protected_handler(args, context):
        protected_calls.append(args["value"])
        return tool_success("protected_tool", args, {"value": args["value"]})

    def normal_handler(args, context):
        normal_calls.append(args["value"])
        return tool_success("normal_tool", args, {"value": args["value"]})

    registry = []
    registry.append(
        _spec(
            "protected_tool",
            protected_handler,
            required_permissions=("protected:read",),
        )
    )
    registry.append(_spec("normal_tool", normal_handler))
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _call("protected_tool", 1, "denied"),
                    _call("normal_tool", 2, "normal"),
                ],
            ),
            AIMessage(content="hidden tool handled"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-denied-batch",
        principal_id="reader",
        permissions=(),
        source="agent_error_recovery_test",
    )

    output = _invoke(
        monkeypatch,
        registry=registry,
        model=model,
        context=context,
    )
    payloads = _payloads(output)

    assert protected_calls == []
    assert normal_calls == []
    assert output["stop_reason"] == "fatal_tool_error"
    assert set(payloads) == {"denied", "normal"}
    assert payloads["denied"]["error_code"] == "permission_denied"
    assert payloads["normal"]["error_code"] == "agent_batch_aborted"
    assert payloads["normal"]["handler_executed"] is False


def test_structured_tool_validation_error_is_corrected_next_round(monkeypatch):
    calls: list[int] = []

    def handler(args, context):
        calls.append(args["value"])
        return tool_success("validated_tool", args, {"value": args["value"]})

    registry = []
    registry.append(_spec("validated_tool", handler))
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[_call("validated_tool", None, "bad")]),
            AIMessage(content="", tool_calls=[_call("validated_tool", 7, "fixed")]),
            AIMessage(content="corrected"),
        ]
    )

    output = _invoke(monkeypatch, registry=registry, model=model)
    first_payload = json.loads(model.received_messages[1][-1].content)

    assert calls == [7]
    assert first_payload["error_code"] == "invalid_arguments"
    assert first_payload["error_source"] == "structured_tool_validation"
    assert first_payload["handler_executed"] is False
    assert first_payload["error"] == "工具参数校验失败"
    assert output["result"]["answer"] == "corrected"


def test_execution_error_can_switch_tools(monkeypatch):
    executed: list[str] = []

    def first_handler(args, context):
        executed.append("first")
        return tool_error(
            "first_tool",
            args,
            "controlled failure",
            error_code=ToolErrorCode.EXECUTION_ERROR.value,
        )

    def second_handler(args, context):
        executed.append("second")
        return tool_success("second_tool", args, {"value": args["value"]})

    registry = []
    registry.append(_spec("first_tool", first_handler))
    registry.append(_spec("second_tool", second_handler))
    model = FakeModel(
        [
            AIMessage(content="", tool_calls=[_call("first_tool", 1, "first")]),
            AIMessage(content="", tool_calls=[_call("second_tool", 2, "second")]),
            AIMessage(content="used fallback tool"),
        ]
    )

    output = _invoke(monkeypatch, registry=registry, model=model)

    assert executed == ["first", "second"]
    assert output["result"]["answer"] == "used fallback tool"
