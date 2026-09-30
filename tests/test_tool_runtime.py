"""LangChain 原生本地工具执行与项目薄治理回归。"""

import asyncio
import json
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, RootModel

from tests.support.local_tools import tool_runtime
from app.agent import framework_agent
from app.agent.framework_agent import build_framework_agent
from app.agent.framework_tools import build_local_tool
from langchain_core.tools import StructuredTool
from app.config import AppSettings
from app.tools.audit import ToolAuditEvent, ToolAuditEventType
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.result import tool_error, tool_success
class EmptyArgs(BaseModel):
    pass


class RuntimeArgs(BaseModel):
    value: int


class RuntimeOutput(BaseModel):
    value: int


class StrictRuntimeArgs(RuntimeArgs):
    model_config = ConfigDict(extra="forbid")


class FakeAgentModel:
    profile: dict = {}

    def __init__(self, responses):
        self.responses = list(responses)

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    async def ainvoke(self, messages):
        del messages
        return self.responses.pop(0)


class CollectingSink:
    def __init__(self):
        self.events: list[ToolAuditEvent] = []

    def write(self, event: ToolAuditEvent) -> None:
        self.events.append(event)


class FailingSink:
    def write(self, event: ToolAuditEvent) -> None:
        del event
        raise OSError("unavailable")


def make_spec(
    handler: Callable[[dict, ToolInvocationContext], dict],
    **overrides: Any,
) -> ToolSpec:
    values: dict[str, Any] = {
        "name": "runtime_echo",
        "version": "2.1.0",
        "description": "Runtime 测试工具",
        "output_model": RuntimeOutput,
        "handler": handler,
        "args_model": RuntimeArgs,
        "timeout": 1.0,
        "max_retries": 0,

    }
    values.update(overrides)
    return ToolSpec(**values)


def invoke_with_framework(
    monkeypatch,
    settings: AppSettings,
    spec: ToolSpec,
    *,
    args: dict[str, Any],
    context: ToolInvocationContext | None = None,
    audit_sink=None,
) -> tuple[dict[str, Any], ToolMessage]:
    resolved_context = context or ToolInvocationContext(trace_id="runtime-test")
    model = FakeAgentModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": spec.name,
                        "args": args,
                        "id": "framework-tool-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    monkeypatch.setattr(framework_agent, "get_agent_model", lambda: model)
    tool = build_local_tool(
        spec,
        settings=settings,
        audit_sink=audit_sink,
    )
    agent = build_framework_agent(settings, tools=[tool])
    inputs = {"messages": [HumanMessage(content="run")]}
    async def stream():
        states = [state async for state in agent.astream(
            inputs, context=resolved_context, stream_mode="values"
        )]
        return states[-1]

    output = asyncio.run(stream())
    message = next(item for item in output["messages"] if isinstance(item, ToolMessage))
    assert not {"runtime", "governance", "audit", "tool_input"} & json.loads(message.content).keys()
    return message.artifact, message


def test_structured_tool_owns_validation_and_schema(settings_factory):
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, {"value": args["value"]})

    spec = make_spec(handler)
    tool = build_local_tool(
        spec,
        settings=settings_factory(),
    )
    assert type(tool) is StructuredTool
    assert tool.tool_call_schema["properties"] == (
        spec.args_model.model_json_schema()["properties"]
    )
    result = asyncio.run(tool.ainvoke({"name": tool.name, "id": "direct", "type": "tool_call", "args": {"value": "7", "runtime": tool_runtime(ToolInvocationContext(trace_id="direct"))}})).artifact
    assert result["tool_input"] == {"value": 7}
    assert calls == 1


@pytest.mark.parametrize("args_model", [RuntimeArgs, StrictRuntimeArgs, EmptyArgs])
@pytest.mark.parametrize("mode", ["direct_async", "stream"])
def test_runtime_context_and_call_id_survive_all_input_shapes(
    monkeypatch, settings_factory, args_model, mode,
):
    """严格和空参数模型也必须经框架注入上下文，且不改变 LLM 参数。"""
    seen = []
    context = ToolInvocationContext(trace_id="injected-context", principal_id="caller")
    args = {} if args_model is EmptyArgs else {"value": "7"}
    expected = {} if args_model is EmptyArgs else {"value": 7}

    def handler(tool_args, tool_context):
        seen.append(tool_context)
        return tool_success("runtime_echo", tool_args, tool_args)

    spec = make_spec(handler, args_model=args_model, output_model=RootModel[dict])
    settings = settings_factory()
    tool = build_local_tool(spec, settings=settings)
    assert tool.tool_call_schema["properties"] == (
        args_model.model_json_schema()["properties"]
    )
    if mode.startswith("direct"):
        runtime = tool_runtime(context)
        runtime.tool_call_id = "direct-call"
        inputs = {**args, "runtime": runtime}
        result = asyncio.run(tool.ainvoke({"name": tool.name, "id": "direct-call", "type": "tool_call", "args": inputs})).artifact
        assert result["tool_call_id"] == "direct-call"
    else:
        result, message = invoke_with_framework(
            monkeypatch, settings, spec, args=args, context=context,
        )
        assert result["tool_call_id"] == message.tool_call_id == "framework-tool-call"
        assert message.status == "success"
    assert result["ok"] is True
    assert result["tool_input"] == result["output"] == expected
    assert len(seen) == 1 and seen[0] is context


def test_strict_input_rejects_extra_business_args_before_handler(settings_factory):
    calls = []
    tool = build_local_tool(
        make_spec(lambda args, context: calls.append(args), args_model=StrictRuntimeArgs),
        settings=settings_factory(),
    )
    message = asyncio.run(tool.ainvoke({"name": tool.name, "id": "strict", "type": "tool_call", "args": {"value": 1, "unexpected": True, "runtime": tool_runtime(ToolInvocationContext())}}))
    assert message.artifact["error_code"] == "invalid_arguments"
    assert calls == []


def test_framework_invalid_args_never_call_handler(
    monkeypatch,
    settings_factory,
):
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, args)

    sink = CollectingSink()
    result, message = invoke_with_framework(
        monkeypatch,
        settings_factory(),
        make_spec(handler, max_retries=1),
        args={"value": "invalid"},
        audit_sink=sink,
    )
    assert calls == 0
    assert result["error_code"] == "invalid_arguments"
    assert result["runtime"]["attempt_count"] == 0
    assert message.status == "error"
    assert [event.event_type for event in sink.events] == [
        ToolAuditEventType.REQUESTED, ToolAuditEventType.DENIED,
    ]
    assert all(event.tool_call_id == message.tool_call_id for event in sink.events)


def test_native_retry_retries_only_declared_failure(
    monkeypatch,
    settings_factory,
):
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ToolExecutionError(
                "secret upstream detail",
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                retryable=True,
            )
        return tool_success("runtime_echo", args, {"value": args["value"]})

    result, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(
            tool_retry_base_delay_ms=0,
            tool_retry_max_delay_ms=0,
            tool_retry_jitter_ratio=0,
        ),
        make_spec(handler, max_retries=1),
        args={"value": 3},
    )
    assert calls == 2
    assert result["ok"] is True
    assert result["runtime"]["attempt_count"] == 2
    assert result["runtime"]["max_attempts"] == 2


def test_non_retryable_failure_runs_once(
    monkeypatch,
    settings_factory,
):
    settings = settings_factory(
        tool_retry_base_delay_ms=0,
        tool_retry_max_delay_ms=0,
    )
    for retryable in (False,):
        calls = 0

        def handler(args, context):
            nonlocal calls
            calls += 1
            raise ToolExecutionError(
                "private failure",
                code=ToolErrorCode.RATE_LIMITED,
                retryable=retryable,
            )

        spec = make_spec(handler, max_retries=1)
        result, _ = invoke_with_framework(
            monkeypatch,
            settings,
            spec,
            args={"value": 1},
        )
        assert calls == 1
        assert result["error_code"] == "rate_limited"
        assert "private failure" not in result["error"]


def test_timeout_returns_early_and_never_retries(
    monkeypatch,
    settings_factory,
):
    calls = 0
    completed = threading.Event()

    def handler(args, context):
        nonlocal calls
        calls += 1
        time.sleep(0.12)
        completed.set()
        return tool_success("runtime_echo", args, args)

    result, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(tool_retry_base_delay_ms=0),
        make_spec(handler, timeout=0.02, max_retries=1),
        args={"value": 1},
    )
    assert result["runtime"]["elapsed_ms"] < 100
    assert result["error_code"] == "timeout"
    assert result["runtime"]["timed_out"] is True
    assert result["runtime"]["attempt_count"] == 1
    assert calls == 1
    assert completed.wait(timeout=1)


def test_permission_blocks_before_handler(
    monkeypatch,
    settings_factory,
):
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, args)

    denied, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(),
        make_spec(handler, required_permissions=("document:write",)),
        args={"value": 1},
    )
    assert denied["error_code"] == "permission_denied"

    assert calls == 0


def test_read_audit_failure_is_recorded_and_fails_open(
    monkeypatch,
    settings_factory,
):
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, {"value": args["value"]})

    read, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(),
        make_spec(handler),
        args={"value": 1},
        audit_sink=FailingSink(),
    )
    assert read["ok"] is True
    assert read["audit"]["write_failed"] is True
    assert calls == 1


def test_retry_audit_and_output_validation(
    monkeypatch,
    settings_factory,
):
    calls = 0
    sink = CollectingSink()

    def retry_handler(args, context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ToolExecutionError(
                "temporary",
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                retryable=True,
            )
        return tool_success("runtime_echo", args, {"value": args["value"]})

    result, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(
            tool_retry_base_delay_ms=0,
            tool_retry_max_delay_ms=0,
        ),
        make_spec(retry_handler, max_retries=1),
        args={"value": 1},
        audit_sink=sink,
    )
    assert result["ok"] is True
    assert [event.event_type for event in sink.events] == [
        ToolAuditEventType.REQUESTED,
        ToolAuditEventType.STARTED,
        ToolAuditEventType.RETRYING,
        ToolAuditEventType.STARTED,
        ToolAuditEventType.SUCCEEDED,
    ]

    calls = 0

    def invalid_output(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, {"value": "invalid"})

    result, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(),
        make_spec(invalid_output, output_model=RuntimeOutput, max_retries=1),
        args={"value": 1},
    )
    assert result["error_code"] == "output_validation_error"
    assert calls == 1


def test_handler_error_dict_keeps_contract_without_retry(
    monkeypatch,
    settings_factory,
):
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_error("runtime_echo", args, "business failure")

    result, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(),
        make_spec(handler, max_retries=1),
        args={"value": 1},
    )
    assert calls == 1
    assert result["error_code"] == "execution_error"
    assert {
        "status",
        "ok",
        "tool_name",
        "tool_input",
        "output",
        "error",
        "error_code",
        "retryable",
        "runtime",
        "governance",
    } <= result.keys()
