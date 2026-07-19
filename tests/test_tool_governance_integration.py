"""工具治理 Runtime、旧分发入口和 LangChain 适配集成测试。"""

import json

from pydantic import BaseModel

from app.agent.executor import build_tool_trace_step
from app.agent.state import AgentObservation, AgentPlanStep
from app.tools import dispatch_tool
from app.tools.audit import ToolAuditEvent
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolRiskLevel, ToolSpec
from app.tools.registry import ToolRegistry
from app.tools.result import tool_success
from app.tools.runtime import ToolRuntime


class IntegrationArgs(BaseModel):
    value: int


class Sink:
    def __init__(self):
        self.events: list[ToolAuditEvent] = []

    def write(self, event: ToolAuditEvent) -> None:
        self.events.append(event)


def _runtime(spec: ToolSpec, sink: Sink) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(spec)
    return ToolRuntime(
        registry,
        max_workers=1,
        retry_base_delay_ms=0,
        retry_max_delay_ms=0,
        retry_jitter_ratio=0,
        audit_sink=sink,
    )


def test_external_write_returns_structured_approval_without_handler_execution():
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("external_tool", args, args)

    sink = Sink()
    runtime = _runtime(
        ToolSpec(
            name="external_tool",
            version="1.0.0",
            description="外部写测试",
            args_schema={"value": "integer"},
            output_schema={},
            handler=handler,
            args_model=IntegrationArgs,
            risk_level=ToolRiskLevel.EXTERNAL_WRITE,
            required_permissions=("external:write",),
            requires_approval=True,
        ),
        sink,
    )
    result = runtime.execute(
        "external_tool",
        {"value": 1},
        context=ToolInvocationContext(
            principal_id="svc",
            permissions=("external:write",),
            source="test",
            idempotency_key="idem-1",
        ),
    )
    assert result["error_code"] == "approval_required"
    assert result["governance"]["decision"] == "require_approval"
    assert result["governance"]["args_fingerprint"].startswith("sha256:")
    assert result["runtime"]["attempt_count"] == 0
    assert calls == 0
    assert [event.event_type.value for event in sink.events] == [
        "requested",
        "approval_required",
    ]
    runtime.close()


def test_legacy_dispatch_uses_default_read_only_service_context(monkeypatch, tmp_path):
    from app.tools import dispatch as tool_dispatch

    monkeypatch.setattr(
        tool_dispatch,
        "TOOL_CALLS_PATH",
        tmp_path / "tool_calls.jsonl",
    )
    result = dispatch_tool(
        "simple_calculator",
        {"expression": "1 + 2"},
        trace_id="legacy-trace",
    )
    assert result["ok"] is True
    assert result["governance"]["decision"] == "allow"
    assert result["runtime"]["attempt_count"] == 1
    assert result["tool_call_id"]


def test_public_tool_trace_redacts_args_and_summarizes_chunk_output():
    chunk_text = "private chunk body " * 100
    step = AgentPlanStep(
        step_id=1,
        action="tool_call",
        tool_name="get_chunk_text",
        tool_args={"chunk_id": 1, "password": "raw-password"},
    )
    observation = AgentObservation(
        step_id=1,
        tool_name="get_chunk_text",
        success=True,
        output={"chunk": {"text": chunk_text, "token": "raw-token"}},
    )
    trace = build_tool_trace_step(
        step,
        {
            "status": "success",
            "ok": True,
            "output": observation.output,
            "governance": {"decision": "allow", "risk_level": "read_only"},
            "runtime": {"attempt_count": 1},
        },
        observation,
    )
    serialized = json.dumps(trace, ensure_ascii=False)
    assert "raw-password" not in serialized
    assert "raw-token" not in serialized
    assert chunk_text not in serialized
    assert trace["sanitized_tool_args"]["password"] == "***REDACTED***"
    assert trace["output_summary"]["mode"] == "none"
