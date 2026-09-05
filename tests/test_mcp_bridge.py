"""MCP Tool async interceptor、同步 bridge、输出归一化和审计测试。"""

import asyncio
import json
from dataclasses import dataclass

from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import BaseModel

from app.agent.framework_middleware import parse_tool_result
from app.tools.audit import ToolAuditEvent
from app.tools.context import PrincipalType, ToolInvocationContext
from app.tools.mcp.bridge import (
    McpToolCallInterceptor,
    build_governed_mcp_tool,
    invoke_mcp_tool,
)
from app.tools.mcp.config import McpServerConfig
from app.tools.mcp.serialization import (
    MCP_INGESTION_MAX_TEXT_CHARS,
    McpSerializationProfile,
    normalize_mcp_tool_message,
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
    mcp_serialization_profile: McpSerializationProfile = (
        McpSerializationProfile.AGENT
    )
    mcp_ingestion_output: dict | None = None


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


def _payload(result: CallToolResult):
    """通过公开 adapter/BaseTool 路径读取 interceptor 的最终项目结果。"""

    class ResultSession:
        async def call_tool(self, name, args, progress_callback=None):
            del name, args, progress_callback
            return result

    remote = Tool(
        name="get_weather",
        description="weather",
        inputSchema={"type": "object", "properties": {}},
    )
    tool = convert_mcp_tool_to_langchain_tool(
        ResultSession(),
        remote,
        server_name="weather",
        tool_name_prefix=True,
        handle_tool_errors=True,
    )
    message = asyncio.run(
        tool.ainvoke(
            {
                "type": "tool_call",
                "name": tool.name,
                "args": {},
                "id": "adapter-test-call",
            }
        )
    )
    assert message.tool_call_id == "adapter-test-call"
    assert message.status == ("error" if result.isError else "success")
    if result.structuredContent is not None:
        assert message.artifact == {
            "structured_content": result.structuredContent,
        }
    payload = normalize_mcp_tool_message(
        message,
        tool_name="weather_get_weather",
        args={"location": "Beijing"},
    )
    assert payload is not None
    return payload


def _request(runtime, **args):
    return MCPToolCallRequest(
        name="get_weather",
        args=args or {"location": "Beijing"},
        server_name="weather",
        runtime=runtime,
    )


def _server_configs(**overrides):
    payload = {
        "transport": "http",
        "url": "https://weather.example.com/mcp",
        "allowed_tools": ["get_weather"],
    }
    payload.update(overrides)
    return {"weather": McpServerConfig.model_validate(payload)}


def test_async_success_is_standard_untrusted_redacted_and_audited(
    settings_factory,
):
    sink = RecordingSink()
    interceptor = McpToolCallInterceptor(
        settings_factory(),
        server_configs=_server_configs(),
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

    message = asyncio.run(interceptor(_request(_runtime("mcp:weather")), handler))
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
    interceptor = McpToolCallInterceptor(
        settings_factory(),
        server_configs=_server_configs(),
        audit_sink=sink,
    )
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


def test_external_write_stops_for_approval_before_remote_handler(
    settings_factory,
):
    sink = RecordingSink()
    interceptor = McpToolCallInterceptor(
        settings_factory(),
        server_configs=_server_configs(approval_required_tools=["get_weather"]),
        audit_sink=sink,
    )
    calls = 0

    async def handler(request):
        nonlocal calls
        del request
        calls += 1
        return CallToolResult(content=[])

    message = asyncio.run(interceptor(_request(_runtime("mcp:weather")), handler))
    payload = _payload(message)

    assert calls == 0
    assert payload["error_code"] == "approval_required"
    assert payload["runtime"]["attempt_count"] == 0
    assert payload["governance"] == {
        "decision": "require_approval",
        "risk_level": "external_write",
        "approval_required": True,
        "args_fingerprint": payload["governance"]["args_fingerprint"],
        "missing_permissions": [],
    }
    assert payload["governance"]["args_fingerprint"].startswith("sha256:")
    assert [event.event_type.value for event in sink.events] == [
        "requested",
        "approval_required",
    ]
    assert all(event.risk_level == "external_write" for event in sink.events)
    assert all(event.policy_decision == "require_approval" for event in sink.events)


def test_external_write_fails_closed_when_audit_is_disabled(settings_factory):
    interceptor = McpToolCallInterceptor(
        settings_factory(tool_audit_enabled=False),
        server_configs=_server_configs(approval_required_tools=["get_weather"]),
    )
    calls = 0

    async def handler(request):
        nonlocal calls
        del request
        calls += 1
        return CallToolResult(content=[])

    payload = _payload(
        asyncio.run(interceptor(_request(_runtime("mcp:weather")), handler))
    )

    assert calls == 0
    assert payload["error_code"] == "audit_write_failed"
    assert payload["governance"]["risk_level"] == "external_write"
    assert payload["governance"]["decision"] == "deny"
    assert payload["audit"]["write_failed"] is True


def test_default_service_principal_is_authorized_at_execution_boundary(
    settings_factory,
):
    settings = settings_factory(tool_default_principal_id="chat_api")
    interceptor = McpToolCallInterceptor(
        settings,
        server_configs=_server_configs(enabled_for_default_principal=True),
        audit_sink=RecordingSink(),
    )
    runtime = Runtime(
        ToolInvocationContext(
            principal_id="chat_api",
            principal_type=PrincipalType.SERVICE,
            permissions=("knowledge_base:read",),
        )
    )
    calls = 0

    async def handler(request):
        nonlocal calls
        del request
        calls += 1
        return CallToolResult(content=[TextContent(type="text", text="Sunny")])

    payload = _payload(asyncio.run(interceptor(_request(runtime), handler)))

    assert payload["ok"] is True
    assert calls == 1


def test_non_default_user_cannot_bypass_execution_with_default_visibility(
    settings_factory,
):
    settings = settings_factory(tool_default_principal_id="chat_api")
    interceptor = McpToolCallInterceptor(
        settings,
        server_configs=_server_configs(enabled_for_default_principal=True),
        audit_sink=RecordingSink(),
    )
    runtime = Runtime(
        ToolInvocationContext(
            principal_id="chat_api",
            principal_type=PrincipalType.USER,
        )
    )
    calls = 0

    async def handler(request):
        nonlocal calls
        del request
        calls += 1
        return CallToolResult(content=[])

    payload = _payload(asyncio.run(interceptor(_request(runtime), handler)))

    assert payload["error_code"] == "mcp_permission_denied"
    assert calls == 0


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
                server_configs=_server_configs(),
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
                server_configs=_server_configs(),
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
            content=[
                TextContent(type="text", text="remote reason"),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="github://owner/repo/error.txt",
                        text="cookie=remote-secret " + "x" * 1000,
                    ),
                ),
            ],
        )

    remote = _payload(
        asyncio.run(
            McpToolCallInterceptor(
                settings_factory(mcp_max_text_chars=128),
                server_configs=_server_configs(),
                audit_sink=RecordingSink(),
            )(_request(runtime), remote_failure)
        )
    )
    assert remote["error_code"] == "mcp_execution_error"
    assert remote["error"].startswith("MCP tool execution failed: remote reason")
    assert "***REDACTED***" in remote["error"]
    assert len(remote["error"]) <= 128
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
            server_configs=_server_configs(),
            audit_sink=RecordingSink(),
        )(_request(_runtime("mcp:weather")), handler)
    )
    serialized = json.dumps(_payload(message), ensure_ascii=False)

    assert "deployment-secret" not in serialized
    assert "sk-private-value" not in serialized
    assert "***REDACTED***" in serialized


def test_official_mixed_content_is_safely_normalized_and_audited(settings_factory):
    runtime = _runtime("mcp:weather")
    sink = RecordingSink()

    async def mixed_handler(request):
        del request
        return CallToolResult(
            content=[
                TextContent(type="text", text="plain text"),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=(
                            "https://alice:uri-password@example.com/readme"
                            "?token=query-secret#fragment"
                        ),
                        mimeType="text/plain",
                        text="embedded cookie=embedded-secret",
                    ),
                ),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="github://owner/repo/archive.bin",
                        mimeType="application/octet-stream",
                        blob="blob-raw-base64",
                    ),
                ),
                ResourceLink(
                    type="resource_link",
                    name="README",
                    uri=(
                        "https://alice:uri-password@example.com/README.md"
                        "?access_token=query-secret#fragment"
                    ),
                    mimeType="text/markdown",
                ),
                ImageContent(
                    type="image",
                    data="image-raw-base64",
                    mimeType="image/png",
                ),
                AudioContent(
                    type="audio",
                    data="audio-raw-base64",
                    mimeType="audio/wav",
                ),
            ],
            structuredContent={"token": "structured-secret", "count": 6},
        )

    payload = _payload(
        asyncio.run(
            McpToolCallInterceptor(
                settings_factory(),
                server_configs=_server_configs(),
                audit_sink=sink,
            )(_request(runtime), mixed_handler)
        )
    )
    serialized = json.dumps(payload, ensure_ascii=False)
    serialized_audit = json.dumps(
        [event.to_dict() for event in sink.events],
        ensure_ascii=False,
    )

    assert payload["ok"] is True
    assert payload["output"]["content_is_untrusted"] is True
    assert payload["output"]["structured_content"] == {
        "token": "***REDACTED***",
        "count": 6,
    }
    assert "plain text" in payload["output"]["content"]
    assert "embedded cookie=***REDACTED***" in payload["output"]["content"]
    assert '"type":"embedded_blob"' in payload["output"]["content"]
    assert f'"size_chars":{len("blob-raw-base64")}' in payload["output"]["content"]
    assert '"type":"resource_link"' in payload["output"]["content"]
    assert '"type":"image"' in payload["output"]["content"]
    assert f'"data_chars":{len("image-raw-base64")}' in payload["output"]["content"]
    assert '"type":"audio"' in payload["output"]["content"]
    assert f'"data_chars":{len("audio-raw-base64")}' in payload["output"]["content"]
    assert "https://example.com/README.md" in payload["output"]["content"]
    for secret in (
        "uri-password",
        "query-secret",
        "structured-secret",
        "blob-raw-base64",
        "image-raw-base64",
        "audio-raw-base64",
    ):
        assert secret not in serialized
        assert secret not in serialized_audit


def test_large_text_embedded_text_and_structured_outputs_are_bounded(
    settings_factory,
):
    runtime = _runtime("mcp:weather")

    async def large_handler(request):
        del request
        return CallToolResult(
            content=[
                TextContent(type="text", text="x" * 1000),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="github://owner/repo/large.txt",
                        text="y" * 1000,
                    ),
                ),
            ],
            structuredContent={
                "items": [
                    {"value": "z" * 200, "api_key": "structured-secret"}
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
                server_configs=_server_configs(),
                audit_sink=RecordingSink(),
            )(_request(runtime), large_handler)
        )
    )
    output = bounded["output"]
    assert len(output["content"]) <= 128
    assert output["truncated"] is True
    assert "content_blocks" not in output
    assert "serialization_profile" not in output
    assert (
        len(
            json.dumps(
                output["structured_content"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        <= 256
    )
    assert "structured-secret" not in json.dumps(output)


def test_ingestion_profile_preserves_complete_text_without_changing_agent_profile(
    settings_factory,
):
    body = "complete body " * 2000
    sink = RecordingSink()

    async def handler(request):
        del request
        return CallToolResult(
            content=[
                TextContent(type="text", text="description"),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="repo://owner/repo/sha/"
                        "f7fc20386c23e20d55f438222fab301cfccc4c65/contents/README.md",
                        mimeType="text/markdown",
                        text=body,
                    ),
                ),
            ]
        )

    settings = settings_factory(mcp_max_text_chars=128)
    interceptor = McpToolCallInterceptor(
        settings,
        server_configs=_server_configs(),
        audit_sink=sink,
    )
    agent = _payload(
        asyncio.run(interceptor(_request(_runtime("mcp:weather")), handler))
    )["output"]
    ingestion_runtime = _runtime("mcp:weather")
    ingestion_runtime.mcp_serialization_profile = McpSerializationProfile.INGESTION
    asyncio.run(interceptor(_request(ingestion_runtime), handler))
    ingestion = ingestion_runtime.mcp_ingestion_output
    assert ingestion is not None

    assert len(agent["content"]) == 128
    assert agent["truncated"] is True
    assert "content_blocks" not in agent
    assert ingestion["truncated"] is False
    assert ingestion["serialization_profile"] == "ingestion"
    assert ingestion["content_blocks"][0] == {
        "type": "text",
        "text": "description",
        "size_chars": 11,
    }
    assert ingestion["content_blocks"][1]["text"] == body
    assert ingestion["content_blocks"][1]["size_chars"] == len(body)
    assert "complete body" not in json.dumps(
        [event.to_dict() for event in sink.events],
        ensure_ascii=False,
    )


def test_oversized_ingestion_output_is_bounded_and_drops_block_body(settings_factory):
    body = "x" * (MCP_INGESTION_MAX_TEXT_CHARS + 1)
    runtime = _runtime("mcp:weather")
    runtime.mcp_serialization_profile = McpSerializationProfile.INGESTION

    async def handler(request):
        del request
        return CallToolResult(content=[TextContent(type="text", text=body)])

    asyncio.run(
        McpToolCallInterceptor(
            settings_factory(),
            server_configs=_server_configs(),
            audit_sink=RecordingSink(),
        )(_request(runtime), handler)
    )
    output = runtime.mcp_ingestion_output
    assert output is not None

    assert len(output["content"]) == MCP_INGESTION_MAX_TEXT_CHARS
    assert output["truncated"] is True
    assert output["content_blocks"] == [
        {
            "type": "text",
            "size_chars": MCP_INGESTION_MAX_TEXT_CHARS + 1,
            "truncated": True,
        }
    ]


def test_unknown_future_content_fails_without_payload_repr(settings_factory):
    class FutureContent:
        def __repr__(self):
            return "FutureContent(payload=unknown-secret)"

    async def handler(request):
        del request
        return CallToolResult.model_construct(content=[FutureContent()])

    payload = _payload(
        asyncio.run(
            McpToolCallInterceptor(
                settings_factory(),
                server_configs=_server_configs(),
                audit_sink=RecordingSink(),
            )(_request(_runtime("mcp:weather")), handler)
        )
    )
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["error_code"] == "mcp_unsupported_content"
    assert "FutureContent" in payload["error"]
    assert "unknown-secret" not in serialized


def _governed_tool(settings):
    interceptor = McpToolCallInterceptor(
        settings,
        server_configs=_server_configs(),
        audit_sink=RecordingSink(),
    )

    class WeatherSession:
        async def call_tool(self, name, args, progress_callback=None):
            del name, args, progress_callback
            return CallToolResult(
                content=[TextContent(type="text", text="Weather ready")]
            )

    raw_tool = convert_mcp_tool_to_langchain_tool(
        WeatherSession(),
        Tool(
            name="get_weather",
            description="Remote weather",
            inputSchema=WeatherArgs.model_json_schema(),
        ),
        tool_interceptors=[interceptor],
        server_name="weather",
        tool_name_prefix=True,
        handle_tool_errors=True,
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

    def call(call_id):
        return {
            "type": "tool_call",
            "name": tool.name,
            "args": {"location": "Beijing", "runtime": runtime},
            "id": call_id,
        }

    sync_message = tool.invoke(call("sync"))
    async_message = asyncio.run(tool.ainvoke(call("async")))
    sync_result = parse_tool_result(
        sync_message, known=True, name=tool.name, args={"location": "Beijing"}
    )
    async_result = parse_tool_result(
        async_message, known=True, name=tool.name, args={"location": "Beijing"}
    )

    assert sync_result["ok"] is True
    assert async_result["ok"] is True
    assert sync_result["output"] == async_result["output"]
    assert tool.args_schema == WeatherArgs.model_json_schema()
    assert "runtime" not in tool.tool_call_schema["properties"]
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


def test_direct_ingestion_profile_rejects_non_read_only_tool(settings_factory):
    tool = _governed_tool(settings_factory())
    tool.metadata["risk_level"] = "external_write"

    result = asyncio.run(
        invoke_mcp_tool(
            tool,
            {"location": "Beijing"},
            context=_runtime("mcp:weather").context,
            serialization_profile=McpSerializationProfile.INGESTION,
        )
    )

    assert result["ok"] is False
    assert result["error_code"] == "mcp_execution_error"
    assert result["runtime"]["attempt_count"] == 0


def test_sync_bridge_fails_safely_inside_existing_event_loop(settings_factory):
    tool = _governed_tool(settings_factory())
    runtime = _runtime("mcp:weather")

    async def call_sync_from_loop():
        return tool.invoke(
            {
                "type": "tool_call",
                "name": tool.name,
                "args": {"location": "Beijing", "runtime": runtime},
                "id": "sync-in-loop",
            }
        )

    message = asyncio.run(call_sync_from_loop())
    result = parse_tool_result(
        message, known=True, name=tool.name, args={"location": "Beijing"}
    )

    assert result["ok"] is False
    assert result["error_code"] == "mcp_execution_error"
    assert result["runtime"]["attempt_count"] == 0
