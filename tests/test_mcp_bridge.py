"""MCP Tool async interceptor、同步 bridge、输出归一化和审计测试。"""

import asyncio
import json
from dataclasses import dataclass

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel

from app.agent.framework_tools import parse_tool_result
from app.tools.audit import ToolAuditEvent
from app.tools.context import ToolInvocationContext
from app.tools.mcp.bridge import (
    McpToolCallInterceptor,
    build_governed_mcp_tool,
)


class WeatherArgs(BaseModel):
    location: str


class RecordingSink:
    def __init__(self):
        self.events: list[ToolAuditEvent] = []

    def write(self, event: ToolAuditEvent):
        self.events.append(event)


@dataclass
class Runtime:
    context: ToolInvocationContext
    tool_call_id: str = "mcp-call-1"


def _runtime(*permissions):
    return Runtime(
        ToolInvocationContext(
            trace_id="trace-mcp",
            session_id="session-mcp",
            principal_id="mcp-user",
            permissions=permissions,
            source="mcp_test",
        )
    )


def _payload(message: ToolMessage):
    return json.loads(message.content)


def _request(runtime, **args):
    return MCPToolCallRequest(
        name="get_weather",
        args=args or {"location": "Beijing"},
        server_name="weather",
        runtime=runtime,
    )


def test_async_success_is_standard_untrusted_redacted_and_audited(
    settings_factory,
):
    sink = RecordingSink()
    interceptor = McpToolCallInterceptor(
        settings_factory(),
        audit_sink=sink,
    )

    async def handler(request):
        assert request.runtime.context.trace_id == "trace-mcp"
        return CallToolResult(
            content=[TextContent(type="text", text="Sunny")],
            structuredContent={
                "temperature": 20,
                "token": "remote-secret",
                "note": "cookie=session-secret",
            },
        )

    message = asyncio.run(
        interceptor(_request(_runtime("mcp:weather")), handler)
    )
    payload = _payload(message)

    assert payload["ok"] is True
    assert payload["tool_name"] == "weather_get_weather"
    assert payload["output"] == {
        "content": "Sunny",
        "content_is_untrusted": True,
        "server": "weather",
        "structured_content": {
            "temperature": 20,
            "token": "***REDACTED***",
            "note": "cookie=***REDACTED***",
        },
        "tool": "get_weather",
        "truncated": False,
    }
    assert payload["runtime"]["attempt_count"] == 1
    assert payload["governance"]["risk_level"] == "read_only"
    assert payload["governance"]["args_fingerprint"].startswith("sha256:")
    assert [event.event_type.value for event in sink.events] == [
        "requested",
        "started",
        "succeeded",
    ]
    assert all(event.server_alias == "weather" for event in sink.events)
    assert sink.events[0].sanitized_input == {
        "mode": "summary",
        "arg_count": 1,
        "fields": ["location"],
    }
    serialized_audit = json.dumps(
        [event.to_dict() for event in sink.events],
        ensure_ascii=False,
    )
    assert "remote-secret" not in serialized_audit
    assert "Beijing" not in serialized_audit


def test_permission_denial_does_not_call_remote_handler(settings_factory):
    sink = RecordingSink()
    interceptor = McpToolCallInterceptor(settings_factory(), audit_sink=sink)
    calls = 0

    async def handler(request):
        nonlocal calls
        del request
        calls += 1
        return CallToolResult(content=[])

    message = asyncio.run(interceptor(_request(_runtime()), handler))
    payload = _payload(message)

    assert payload["error_code"] == "mcp_permission_denied"
    assert payload["runtime"]["attempt_count"] == 0
    assert calls == 0
    assert [event.event_type.value for event in sink.events] == [
        "requested",
        "denied",
    ]


def test_timeout_transport_and_remote_errors_are_safe(settings_factory):
    settings = settings_factory(mcp_tool_timeout_seconds=0.001)
    runtime = _runtime("mcp:weather")

    async def slow(request):
        del request
        await asyncio.sleep(0.02)
        return CallToolResult(content=[])

    timed_out = _payload(
        asyncio.run(
            McpToolCallInterceptor(
                settings,
                audit_sink=RecordingSink(),
            )(_request(runtime), slow)
        )
    )
    assert timed_out["error_code"] == "mcp_tool_timeout"
    assert timed_out["runtime"]["timed_out"] is True

    async def transport_failure(request):
        del request
        raise ConnectionError("Authorization: Bearer transport-secret")

    transport = _payload(
        asyncio.run(
            McpToolCallInterceptor(
                settings_factory(),
                audit_sink=RecordingSink(),
            )(_request(runtime), transport_failure)
        )
    )
    assert transport["error_code"] == "mcp_transport_error"
    assert "transport-secret" not in json.dumps(transport)

    async def remote_failure(request):
        del request
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text="cookie=remote-secret")],
        )

    remote = _payload(
        asyncio.run(
            McpToolCallInterceptor(
                settings_factory(),
                audit_sink=RecordingSink(),
            )(_request(runtime), remote_failure)
        )
    )
    assert remote["error_code"] == "mcp_execution_error"
    assert "remote-secret" not in json.dumps(remote)


def test_plain_text_credentials_are_redacted(settings_factory):
    async def handler(request):
        del request
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        "Authorization: Bearer deployment-secret; "
                        "api_key=sk-private-value"
                    ),
                )
            ]
        )

    message = asyncio.run(
        McpToolCallInterceptor(
            settings_factory(),
            audit_sink=RecordingSink(),
        )(_request(_runtime("mcp:weather")), handler)
    )
    serialized = json.dumps(_payload(message), ensure_ascii=False)

    assert "deployment-secret" not in serialized
    assert "sk-private-value" not in serialized
    assert "***REDACTED***" in serialized


def test_unsupported_binary_and_large_outputs_are_bounded(settings_factory):
    runtime = _runtime("mcp:weather")

    async def image_handler(request):
        del request
        return CallToolResult(
            content=[
                ImageContent(
                    type="image",
                    data="base64-private-binary",
                    mimeType="image/png",
                )
            ]
        )

    unsupported = _payload(
        asyncio.run(
            McpToolCallInterceptor(
                settings_factory(),
                audit_sink=RecordingSink(),
            )(_request(runtime), image_handler)
        )
    )
    assert unsupported["error_code"] == "mcp_unsupported_content"
    assert "base64-private-binary" not in json.dumps(unsupported)

    async def large_handler(request):
        del request
        return CallToolResult(
            content=[TextContent(type="text", text="x" * 1000)],
            structuredContent={
                "items": [
                    {"value": "y" * 200, "api_key": "structured-secret"}
                    for _ in range(50)
                ]
            },
        )

    settings = settings_factory(
        mcp_max_text_chars=128,
        mcp_max_structured_chars=256,
    )
    bounded = _payload(
        asyncio.run(
            McpToolCallInterceptor(
                settings,
                audit_sink=RecordingSink(),
            )(_request(runtime), large_handler)
        )
    )
    output = bounded["output"]
    assert len(output["content"]) <= 128
    assert output["truncated"] is True
    assert len(
        json.dumps(
            output["structured_content"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    ) <= 256
    assert "structured-secret" not in json.dumps(output)


def _governed_tool(settings):
    interceptor = McpToolCallInterceptor(
        settings,
        audit_sink=RecordingSink(),
    )

    async def raw_coroutine(runtime=None, **args):
        request = _request(runtime, **args)

        async def handler(_request):
            return CallToolResult(
                content=[TextContent(type="text", text="Weather ready")]
            )

        return await interceptor(request, handler), None

    raw_tool = StructuredTool(
        name="weather_get_weather",
        description="Remote weather",
        args_schema=WeatherArgs,
        coroutine=raw_coroutine,
        response_format="content_and_artifact",
    )
    return build_governed_mcp_tool(
        raw_tool,
        server_alias="weather",
        remote_tool_name="get_weather",
        description="Safe weather",
    )


def test_governed_tool_supports_sync_and_native_async_paths(settings_factory):
    tool = _governed_tool(settings_factory())
    runtime = _runtime("mcp:weather")

    sync_result = tool.func(runtime=runtime, location="Beijing")
    async_result = asyncio.run(
        tool.coroutine(runtime=runtime, location="Beijing")
    )

    assert sync_result["ok"] is True
    assert async_result["ok"] is True
    assert sync_result["output"] == async_result["output"]
    assert "runtime" not in tool.tool_call_schema.model_fields
    parsed = parse_tool_result(
        ToolMessage(
            content=json.dumps(async_result),
            tool_call_id="parse-call",
        ),
        known=True,
        name=tool.name,
        args={"location": "Beijing"},
    )
    assert parsed["ok"] is True


def test_sync_bridge_fails_safely_inside_existing_event_loop(settings_factory):
    tool = _governed_tool(settings_factory())
    runtime = _runtime("mcp:weather")

    async def call_sync_from_loop():
        return tool.func(runtime=runtime, location="Beijing")

    result = asyncio.run(call_sync_from_loop())

    assert result["ok"] is False
    assert result["error_code"] == "mcp_execution_error"
    assert result["runtime"]["attempt_count"] == 0
