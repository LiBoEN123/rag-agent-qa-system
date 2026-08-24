"""ToolRuntime 治理审计顺序和失败策略测试。"""

from pydantic import BaseModel

from app.tools.audit import ToolAuditEvent, ToolAuditEventType
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolRiskLevel, ToolSpec
from app.tools.registry import ToolRegistry
from app.tools.result import tool_success
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.runtime import ToolRuntime


class AuditArgs(BaseModel):
    value: int


class CollectingSink:
    def __init__(self):
        self.events: list[ToolAuditEvent] = []

    def write(self, event: ToolAuditEvent) -> None:
        self.events.append(event)


class FailingSink:
    def write(self, event: ToolAuditEvent) -> None:
        raise OSError("unavailable")


def _runtime(handler, *, sink, **spec_overrides) -> ToolRuntime:
    values = {
        "name": "audited_tool",
        "version": "1.0.0",
        "description": "审计测试工具",
        "output_schema": {},
        "handler": handler,
        "args_model": AuditArgs,
    }
    values.update(spec_overrides)
    registry = ToolRegistry()
    registry.register(ToolSpec(**values))
    return ToolRuntime(
        registry,
        max_workers=1,
        retry_base_delay_ms=0,
        retry_max_delay_ms=0,
        retry_jitter_ratio=0,
        audit_sink=sink,
    )


def test_success_and_permission_denial_have_expected_audit_order():
    sink = CollectingSink()
    runtime = _runtime(
        lambda args, context: tool_success("audited_tool", args, args),
        sink=sink,
        required_permissions=("knowledge_base:read",),
    )
    context = ToolInvocationContext(
        principal_id="svc",
        permissions=("knowledge_base:read",),
        source="test",
    )
    result = runtime.execute("audited_tool", {"value": 1}, context=context)
    assert result["ok"] is True
    assert [event.event_type for event in sink.events] == [
        ToolAuditEventType.REQUESTED,
        ToolAuditEventType.STARTED,
        ToolAuditEventType.SUCCEEDED,
    ]
    assert len({event.tool_call_id for event in sink.events}) == 1

    sink.events.clear()
    denied = runtime.execute(
        "audited_tool",
        {"value": 1},
        context=ToolInvocationContext(principal_id="svc", source="test"),
    )
    assert denied["error_code"] == "permission_denied"
    assert denied["runtime"]["attempt_count"] == 0
    assert [event.event_type for event in sink.events] == [
        ToolAuditEventType.REQUESTED,
        ToolAuditEventType.DENIED,
    ]
    runtime.close()


def test_read_audit_failure_opens_but_write_failure_closes():
    read_calls = 0

    def read_handler(args, context):
        nonlocal read_calls
        read_calls += 1
        return tool_success("audited_tool", args, args)

    read_runtime = _runtime(read_handler, sink=FailingSink())
    read_result = read_runtime.execute("audited_tool", {"value": 1})
    assert read_result["ok"] is True
    assert read_calls == 1
    assert read_result["audit"]["write_failed"] is True
    read_runtime.close()

    write_calls = 0

    def write_handler(args, context):
        nonlocal write_calls
        write_calls += 1
        return tool_success("audited_tool", args, args)

    write_runtime = _runtime(
        write_handler,
        sink=FailingSink(),
        risk_level=ToolRiskLevel.INTERNAL_WRITE,
        required_permissions=("document:write",),
    )
    write_result = write_runtime.execute(
        "audited_tool",
        {"value": 1},
        context=ToolInvocationContext(
            principal_id="svc",
            permissions=("document:write",),
            source="test",
            idempotency_key="idem-1",
        ),
    )
    assert write_result["error_code"] == "audit_write_failed"
    assert write_result["runtime"]["attempt_count"] == 0
    assert write_calls == 0
    write_runtime.close()


def test_retry_audit_order_starts_each_submitted_attempt():
    calls = 0
    sink = CollectingSink()

    def handler(args, context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ToolExecutionError(
                "temporary",
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                retryable=True,
            )
        return tool_success("audited_tool", args, args)

    runtime = _runtime(handler, sink=sink, max_retries=1)
    result = runtime.execute("audited_tool", {"value": 1})
    assert result["ok"] is True
    assert [event.event_type for event in sink.events] == [
        ToolAuditEventType.REQUESTED,
        ToolAuditEventType.STARTED,
        ToolAuditEventType.RETRYING,
        ToolAuditEventType.STARTED,
        ToolAuditEventType.SUCCEEDED,
    ]
    runtime.close()
