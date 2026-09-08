"""LangChain 原生本地工具执行与项目薄治理回归。"""

import json
import threading
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from tests.support.local_tools import tool_runtime
from app.agent import framework_agent
from app.agent.framework_agent import build_framework_agent
from app.agent.framework_tools import ProjectStructuredTool, build_local_tool
from app.config import AppSettings
from app.tools.audit import ToolAuditEvent, ToolAuditEventType
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolRiskLevel, ToolSpec
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.result import tool_error, tool_success


class RuntimeArgs(BaseModel):
    value: int


class RuntimeOutput(BaseModel):
    value: int


class FakeAgentModel:
    profile: dict = {}

    def __init__(self, responses):
        self.responses = list(responses)

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def invoke(self, messages):
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
        "output_schema": {"value": "integer"},
        "handler": handler,
        "args_model": RuntimeArgs,
        "timeout": 1.0,
        "max_retries": 0,
        "idempotent": True,
    }
    if overrides.get("output_model") is not None and "output_schema" not in overrides:
        values.pop("output_schema")
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
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    tool = build_local_tool(
        spec,
        settings=settings,
        audit_sink=audit_sink,
    )
    output = build_framework_agent(settings, tools=[tool]).invoke(
        {"messages": [HumanMessage(content="run")]},
        context=resolved_context,
    )
    message = next(item for item in output["messages"] if isinstance(item, ToolMessage))
    return json.loads(message.content), message


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
    assert isinstance(tool, ProjectStructuredTool)
    assert tool.project_spec is spec
    assert tool.tool_call_schema.model_json_schema()["properties"] == (
        spec.args_model.model_json_schema()["properties"]
    )
    result = tool.invoke({"value": "7", "runtime": tool_runtime(ToolInvocationContext(trace_id="direct"))})
    assert result["tool_input"] == {"value": 7}
    assert calls == 1


def test_framework_invalid_args_never_call_handler(
    monkeypatch,
    settings_factory,
):
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, args)

    result, message = invoke_with_framework(
        monkeypatch,
        settings_factory(),
        make_spec(handler, max_retries=1),
        args={"value": "invalid"},
    )
    assert calls == 0
    assert result["error_code"] == "invalid_arguments"
    assert result["runtime"]["attempt_count"] == 0
    assert message.status == "error"


def test_tool_retry_middleware_retries_only_declared_failure(
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


def test_non_retryable_and_non_idempotent_fail_once(
    monkeypatch,
    settings_factory,
):
    settings = settings_factory(
        tool_retry_base_delay_ms=0,
        tool_retry_max_delay_ms=0,
    )
    for idempotent, retryable in ((True, False), (False, True)):
        calls = 0

        def handler(args, context):
            nonlocal calls
            calls += 1
            raise ToolExecutionError(
                "private failure",
                code=ToolErrorCode.RATE_LIMITED,
                retryable=retryable,
            )

        spec = make_spec(handler, max_retries=0 if not idempotent else 1)
        if not idempotent:
            object.__setattr__(spec, "idempotent", False)
            object.__setattr__(spec, "max_retries", 1)
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

    started = time.perf_counter()
    result, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(tool_retry_base_delay_ms=0),
        make_spec(handler, timeout=0.02, max_retries=1),
        args={"value": 1},
    )
    assert time.perf_counter() - started < 0.1
    assert result["error_code"] == "timeout"
    assert result["runtime"]["timed_out"] is True
    assert result["runtime"]["attempt_count"] == 1
    assert calls == 1
    assert completed.wait(timeout=1)


def test_permission_and_approval_block_before_handler(
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

    approval_spec = make_spec(
        handler,
        risk_level=ToolRiskLevel.EXTERNAL_WRITE,
        required_permissions=("external:write",),
        requires_approval=True,
    )
    approval, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(),
        approval_spec,
        args={"value": 1},
        context=ToolInvocationContext(
            trace_id="approval",
            principal_id="svc",
            source="test",
            permissions=("external:write",),
            idempotency_key="idem-1",
        ),
    )
    assert approval["error_code"] == "approval_required"
    assert approval["governance"]["approval_required"] is True
    assert calls == 0


def test_write_audit_failure_closes_and_read_failure_opens(
    monkeypatch,
    settings_factory,
):
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, {"value": args["value"]})

    write_spec = make_spec(
        handler,
        risk_level=ToolRiskLevel.INTERNAL_WRITE,
        required_permissions=("document:write",),
    )
    context = ToolInvocationContext(
        trace_id="audit-write",
        principal_id="svc",
        source="test",
        permissions=("document:write",),
        idempotency_key="idem-1",
    )
    blocked, _ = invoke_with_framework(
        monkeypatch,
        settings_factory(),
        write_spec,
        args={"value": 1},
        context=context,
        audit_sink=FailingSink(),
    )
    assert blocked["error_code"] == "audit_write_failed"
    assert calls == 0

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
