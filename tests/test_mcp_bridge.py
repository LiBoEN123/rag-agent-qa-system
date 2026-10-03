"""通过官方 adapter 验证来源调用、授权、超时、净化与摘要审计。"""

import asyncio
import json

import pytest
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

from app.tools.audit import ToolAuditEvent
from app.tools.context import ToolInvocationContext
from app.tools.mcp.bridge import McpToolCallInterceptor, invoke_mcp_tool
from app.tools.mcp.config import McpConfig
from app.tools.mcp.discovery import discover_mcp_tools
from app.tools.mcp.serialization import (
    MCP_INGESTION_MAX_TEXT_CHARS,
    sanitize_mcp_call_result,
)


class RecordingSink:
    def __init__(self):
        self.events: list[ToolAuditEvent] = []

    def write(self, event):
        self.events.append(event)


def _source_tool(
    settings, handler, sink, *, server="github", remote="get_file_contents"
):
    config = McpConfig.model_validate(
        {
            "version": 1,
            "servers": {
                server: {
                    "transport": "http",
                    "url": "https://example.com/mcp",
                    "allowed_tools": [remote],
                }
            },
        }
    )
    interceptor = McpToolCallInterceptor(
        settings, server_configs=config.servers, audit_sink=sink
    )

    class Session:
        async def call_tool(self, name, args, progress_callback=None):
            assert name == remote
            return await handler(args)

    raw = convert_mcp_tool_to_langchain_tool(
        Session(),
        Tool(name=remote, description="source", inputSchema={"type": "object"}),
        tool_interceptors=[interceptor],
        server_name=server,
        tool_name_prefix=True,
        handle_tool_errors=True,
    )

    class Client:
        async def get_tools(self, *, server_name):
            assert server_name == server
            return [raw]

    [entry] = asyncio.run(
        discover_mcp_tools(Client(), config=config, settings=settings)
    )
    assert entry.tool is raw
    return entry.tool


def _fetch(tool, args=None, *, permissions=("mcp:github",)):
    return asyncio.run(
        invoke_mcp_tool(
            tool,
            args or {},
            context=ToolInvocationContext(
                principal_id="source-service",
                permissions=permissions,
                trace_id="source-trace",
                source="source_test",
            ),
        )
    )


def test_source_result_is_untrusted_redacted_and_audited(settings_factory):
    sink = RecordingSink()

    async def handler(args):
        assert args == {"path": "README.md"}
        return CallToolResult(
            content=[
                TextContent(type="text", text="Authorization: Bearer body-secret")
            ],
            structuredContent={
                "temperature": 20,
                "token": "remote-secret",
                "note": "cookie=session-secret",
            },
        )

    result = _fetch(
        _source_tool(settings_factory(), handler, sink), {"path": "README.md"}
    )
    assert result["ok"] is True
    assert result["tool_name"] == "github_get_file_contents"
    output = result["output"]
    assert output["server"] == "github" and output["tool"] == "get_file_contents"
    assert output["content_is_untrusted"] is True
    assert output["truncated"] is False
    assert output["structured_content"] == {
        "temperature": 20,
        "token": "***REDACTED***",
        "note": "cookie=***REDACTED***",
    }
    assert output["content_blocks"][0]["text"] == output["content"]
    assert result["audit"] == {"write_failed": False}
    assert [e.event_type.value for e in sink.events] == [
        "requested",
        "started",
        "succeeded",
    ]
    assert [e.attempt_count for e in sink.events] == [0, 1, 1]
    assert all(
        e.server_alias == "github" and e.args_fingerprint.startswith("sha256:")
        for e in sink.events
    )
    assert sink.events[0].sanitized_input == {
        "mode": "summary",
        "arg_count": 1,
        "fields": ["path"],
    }
    audit = json.dumps([e.to_dict() for e in sink.events])
    for secret in ("body-secret", "remote-secret", "session-secret"):
        assert secret not in json.dumps(result) and secret not in audit
    assert "README.md" not in audit


@pytest.mark.parametrize("permissions", [(), ("mcp:firecrawl",)])
def test_source_permissions_fail_before_remote_handler(settings_factory, permissions):
    sink = RecordingSink()

    async def forbidden(args):
        pytest.fail("unauthorized source must not execute")

    result = _fetch(
        _source_tool(settings_factory(), forbidden, sink), permissions=permissions
    )
    assert result["error_code"] == "mcp_permission_denied"
    assert [e.event_type.value for e in sink.events] == ["requested", "denied"]
    assert all(e.attempt_count == 0 for e in sink.events)


@pytest.mark.parametrize("extra", [{}, {"actions": [{"type": "click"}]}])
def test_firecrawl_source_scrape_only_allows_read_only_arguments(
    settings_factory, extra
):
    calls = []
    sink = RecordingSink()

    async def handler(args):
        calls.append(args)
        return CallToolResult(content=[TextContent(type="text", text="page")])

    tool = _source_tool(
        settings_factory(), handler, sink, server="firecrawl", remote="firecrawl_scrape"
    )
    result = _fetch(
        tool,
        {"url": "https://example.com", "formats": ["markdown", "html"], **extra},
        permissions=("mcp:firecrawl",),
    )
    assert bool(calls) is (not extra)
    assert result["ok"] is (not extra)
    if extra:
        assert result["error_code"] == "invalid_arguments"
        assert sink.events[-1].attempt_count == 0


@pytest.mark.parametrize(
    "failure,code",
    [
        ("timeout", "mcp_tool_timeout"),
        ("transport", "mcp_transport_error"),
        ("remote", "mcp_execution_error"),
    ],
)
def test_timeout_transport_and_remote_errors_are_safe(settings_factory, failure, code):
    sink = RecordingSink()

    async def handler(args):
        if failure == "timeout":
            await asyncio.sleep(0.02)
        if failure == "transport":
            raise ConnectionError("Authorization: Bearer transport-secret")
        return CallToolResult(
            isError=True,
            content=[
                TextContent(
                    type="text", text="remote reason cookie=remote-secret " + "x" * 3000
                ),
            ],
        )

    result = _fetch(
        _source_tool(settings_factory(mcp_tool_timeout_seconds=0.001), handler, sink)
    )
    assert result["error_code"] == code
    assert result["ok"] is False
    assert len(result["error"]) <= 2000
    if failure == "remote":
        assert result["error"].startswith("MCP tool execution failed: remote reason")
        assert "***REDACTED***" in result["error"]
    serialized = json.dumps(result) + json.dumps([e.to_dict() for e in sink.events])
    assert "transport-secret" not in serialized and "remote-secret" not in serialized
    assert sink.events[-1].event_type.value == (
        "timed_out" if failure == "timeout" else "failed"
    )
    assert sink.events[-1].attempt_count == 1


def test_read_only_source_survives_audit_failure(settings_factory):
    class BrokenSink:
        def write(self, event):
            raise OSError("private audit details")

    async def handler(args):
        return CallToolResult(content=[TextContent(type="text", text="body")])

    result = _fetch(_source_tool(settings_factory(), handler, BrokenSink()))
    assert result["ok"] is True
    assert result["audit"] == {"write_failed": True}
    assert "private" not in json.dumps(result)


def test_official_mixed_content_preserves_safe_source_metadata(settings_factory):
    sink = RecordingSink()

    async def handler(args):
        return CallToolResult(
            content=[
                TextContent(type="text", text="plain text"),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="https://alice:uri-password@example.com/readme?token=query-secret#fragment",
                        mimeType="text/plain",
                        text="embedded cookie=embedded-secret",
                    ),
                ),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="repo://owner/repo/archive.bin",
                        mimeType="application/octet-stream",
                        blob="blob-raw-base64",
                    ),
                ),
                ResourceLink(
                    type="resource_link",
                    name="README",
                    uri="https://example.com/README.md?token=query-secret",
                    mimeType="text/markdown",
                ),
                ImageContent(
                    type="image", data="image-raw-base64", mimeType="image/png"
                ),
                AudioContent(
                    type="audio", data="audio-raw-base64", mimeType="audio/wav"
                ),
            ],
            structuredContent={"token": "structured-secret", "count": 6},
        )

    result = _fetch(_source_tool(settings_factory(), handler, sink))
    assert result["ok"] is True
    output = result["output"]
    assert output["content_blocks"][1]["uri"] == "https://example.com/readme"
    assert output["content_blocks"][1]["text"] == "embedded cookie=***REDACTED***"
    assert [b["type"] for b in output["content_blocks"]] == [
        "text",
        "embedded_resource_text",
        "embedded_blob",
        "resource_link",
        "image",
        "audio",
    ]
    serialized = json.dumps(result) + json.dumps([e.to_dict() for e in sink.events])
    for secret in (
        "uri-password",
        "query-secret",
        "structured-secret",
        "blob-raw-base64",
        "image-raw-base64",
        "audio-raw-base64",
    ):
        assert secret not in serialized


def test_ingestion_preserves_complete_text_and_structured_content(settings_factory):
    body = "complete body " * 2000
    nested = {"api_key": "nested-secret"}
    for _ in range(10):
        nested = {"child": nested}
    structured = {
        "markdown": body,
        "html": "<p>" + body + "</p>",
        "links": ["link"] * 101,
        "nested": nested,
    }
    sink = RecordingSink()

    async def handler(args):
        return CallToolResult(
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="repo://owner/repo/sha/" + "a" * 40 + "/contents/README.md",
                        mimeType="text/markdown",
                        text=body,
                    ),
                )
            ],
            structuredContent=structured,
        )

    result = _fetch(_source_tool(settings_factory(), handler, sink))
    output = result["output"]
    assert output["truncated"] is False
    assert output["content_blocks"][0]["text"] == body
    assert output["content_blocks"][0]["size_chars"] == len(body)
    for key in ("markdown", "html", "links"):
        assert output["structured_content"][key] == structured[key]
    nested = output["structured_content"]["nested"]
    for _ in range(10):
        nested = nested["child"]
    assert nested == {"api_key": "***REDACTED***"}
    audit = json.dumps([e.to_dict() for e in sink.events])
    assert body not in audit and "nested-secret" not in json.dumps(result)


def test_oversized_ingestion_text_drops_block_body():
    outcome = sanitize_mcp_call_result(
        CallToolResult(
            content=[
                TextContent(type="text", text="x" * (MCP_INGESTION_MAX_TEXT_CHARS + 1))
            ]
        ),
        server_alias="github",
        remote_tool_name="get_file_contents",
    )
    assert outcome.ok is True
    output = outcome.ingestion_output
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

    async def handler(args):
        return CallToolResult.model_construct(content=[FutureContent()])

    result = _fetch(_source_tool(settings_factory(), handler, RecordingSink()))
    assert result["error_code"] == "mcp_unsupported_content"
    assert "FutureContent" in result["error"]
    assert "unknown-secret" not in json.dumps(result)


def test_missing_source_runtime_fails_before_handler(settings_factory):
    interceptor = McpToolCallInterceptor(
        settings_factory(), server_configs={}, audit_sink=RecordingSink()
    )

    async def forbidden(request):
        pytest.fail("missing source context must fail closed")

    result = asyncio.run(
        interceptor(
            MCPToolCallRequest(
                name="get_file_contents",
                server_name="github",
                args={},
                runtime=None,
            ),
            forbidden,
        )
    )
    assert result.isError is True


@pytest.mark.parametrize("source", ["github", "firecrawl"])
def test_official_adapter_source_fetch_reaches_pending_candidate(
    monkeypatch, settings_factory, source
):
    from types import SimpleNamespace

    from app.ingestion import IngestionRequester, IngestionPrincipalType
    from app.sources.service import ExternalKnowledgeSourceService

    calls = []

    async def handler(args):
        calls.append(args)
        if source == "github":
            return CallToolResult(
                content=[
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="repo://owner/repo/sha/"
                            + "a" * 40
                            + "/contents/README.md",
                            mimeType="text/markdown",
                            text="source body",
                        ),
                    )
                ]
            )
        return CallToolResult(
            content=[],
            structuredContent={
                "markdown": "source body",
                "metadata": {"statusCode": 200, "url": "https://example.com"},
            },
        )

    tool = _source_tool(
        settings_factory(),
        handler,
        RecordingSink(),
        server=source,
        remote="get_file_contents" if source == "github" else "firecrawl_scrape",
    )
    provider = SimpleNamespace(list_tools=lambda *, context: [tool])
    monkeypatch.setattr("app.sources.service.get_mcp_tool_provider", lambda: provider)
    service = ExternalKnowledgeSourceService(
        requester=IngestionRequester(
            principal_id="source-operator",
            principal_type=IngestionPrincipalType.SERVICE,
            source="source_test",
        )
    )
    context = ToolInvocationContext(
        principal_id="source-operator", permissions=(f"mcp:{source}",)
    )
    if source == "github":
        candidate = asyncio.run(
            service.propose_github_file(
                owner="owner",
                repository="repo",
                path="README.md",
                context=context,
            )
        )
        assert calls == [{"owner": "owner", "repo": "repo", "path": "README.md"}]
        assert candidate.document.version == "a" * 40
    else:
        candidate = asyncio.run(
            service.propose_firecrawl_page(url="https://example.com", context=context)
        )
        assert calls == [
            {
                "url": "https://example.com",
                "formats": ["markdown", "html"],
                "onlyMainContent": True,
                "removeBase64Images": True,
            }
        ]
    assert candidate.document.content == "source body"
    assert candidate.status.value == "pending"
    assert candidate.document.trust_level.value == "untrusted_external"
