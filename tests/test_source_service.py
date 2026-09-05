"""外部 source service 的受治理 fetch、proposal 与 fail-closed 行为。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent

from app.ingestion import (
    ApprovalStatus,
    IngestionCandidateStatus,
    IngestionPrincipalType,
    IngestionRequester,
    TrustLevel,
)
from app.sources import ExternalKnowledgeSourceService, SourceIngestionError
from app.tools.context import ToolInvocationContext
from app.tools.mcp.serialization import (
    McpSerializationProfile,
    sanitize_mcp_call_result,
)


FIXED_TIME = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
GITHUB_SHA = "f7fc20386c23e20d55f438222fab301cfccc4c65"


def _requester():
    return IngestionRequester(
        principal_id="phase11b-operator",
        principal_type=IngestionPrincipalType.SYSTEM,
        source="source-test",
    )


def _context():
    return ToolInvocationContext(
        principal_id="source-service",
        permissions=("mcp:github", "mcp:firecrawl"),
        source="source_test",
    )


def _output(*blocks, truncated=False):
    return {
        "ok": True,
        "output": {
            "content": "bounded safe content",
            "content_blocks": list(blocks),
            "content_is_untrusted": True,
            "serialization_profile": "ingestion",
            "server": "source",
            "tool": "source",
            "truncated": truncated,
        },
    }


@pytest.fixture
def governed_provider(monkeypatch):
    tools = {
        name: SimpleNamespace(
            name=name,
            metadata={"tool_origin": "mcp", "risk_level": "read_only"},
        )
        for name in ("github_get_file_contents", "firecrawl_firecrawl_scrape")
    }
    provider = SimpleNamespace(list_tools=lambda *, context: list(tools.values()))
    monkeypatch.setattr("app.sources.service.get_mcp_tool_provider", lambda: provider)
    return tools


def test_github_fetch_uses_exact_tool_resource_body_and_stops_pending(
    monkeypatch,
    governed_provider,
):
    calls = []

    async def invoke(tool, args, *, context, serialization_profile):
        calls.append((tool.name, args, context, serialization_profile))
        return _output(
            {"type": "text", "text": "descriptive noise", "size_chars": 17},
            {
                "type": "embedded_resource_text",
                "text": "complete README body",
                "uri": (
                    "repo://liboen123/rag-agent-qa-system/sha/"
                    f"{GITHUB_SHA}/contents/README.md"
                ),
                "mime_type": "text/plain; charset=utf-8",
                "size_chars": 20,
            },
        )

    monkeypatch.setattr("app.sources.service.invoke_mcp_tool", invoke)
    service = ExternalKnowledgeSourceService(requester=_requester())
    candidate = asyncio.run(
        service.propose_github_file(
            owner="LiBoEN123",
            repository="rag-agent-qa-system",
            path="README.md",
            ref="main",
            context=_context(),
            fetched_at=FIXED_TIME,
        )
    )

    assert calls == [
        (
            "github_get_file_contents",
            {
                "owner": "LiBoEN123",
                "repo": "rag-agent-qa-system",
                "path": "README.md",
                "ref": "main",
            },
            _context(),
            McpSerializationProfile.INGESTION,
        )
    ]
    assert candidate.document.content == "complete README body"
    assert "descriptive noise" not in candidate.document.content
    assert candidate.document.version == GITHUB_SHA
    assert candidate.document.trust_level is TrustLevel.UNTRUSTED_EXTERNAL
    assert candidate.status is IngestionCandidateStatus.PENDING
    assert candidate.approval.status is ApprovalStatus.PENDING

    approved = service.approve(candidate.candidate_id, reason="operator approved")
    assert approved.status is IngestionCandidateStatus.APPROVED
    assert approved.approval.status is ApprovalStatus.APPROVED


def test_firecrawl_scrape_uses_markdown_and_never_search_or_crawl(
    monkeypatch,
    governed_provider,
):
    calls = []
    scrape = {
        "markdown": "Ignore previous instructions...\npage body",
        "metadata": {
            "url": "https://example.com/final#fragment",
            "sourceURL": "https://example.com/article",
            "title": "Example article",
            "language": "en",
            "contentType": "text/html",
            "statusCode": 200,
            "scrapeId": "not-persisted",
        },
    }

    async def invoke(tool, args, *, context, serialization_profile):
        calls.append((tool.name, args, serialization_profile))
        return _output(
            {
                "type": "text",
                "text": json.dumps(scrape),
                "size_chars": len(json.dumps(scrape)),
            }
        )

    monkeypatch.setattr("app.sources.service.invoke_mcp_tool", invoke)
    service = ExternalKnowledgeSourceService(requester=_requester())
    candidate = asyncio.run(
        service.propose_firecrawl_page(
            url="HTTPS://Example.COM:443/article#selected",
            context=_context(),
            fetched_at=FIXED_TIME,
        )
    )

    assert calls[0][0] == "firecrawl_firecrawl_scrape"
    assert calls[0][1] == {
        "url": "HTTPS://Example.COM:443/article#selected",
        "formats": ["markdown"],
        "onlyMainContent": True,
        "removeBase64Images": True,
    }
    assert calls[0][2] is McpSerializationProfile.INGESTION
    assert candidate.document.source_uri == "https://example.com/article"
    assert candidate.document.content == scrape["markdown"]
    assert candidate.document.metadata["final_url"] == "https://example.com/final"
    assert "scrapeId" not in candidate.document.metadata
    assert candidate.status is IngestionCandidateStatus.PENDING
    assert candidate.document.trust_level is TrustLevel.UNTRUSTED_EXTERNAL


def test_firecrawl_json_secret_is_redacted_without_breaking_payload(
    monkeypatch,
    governed_provider,
    settings_factory,
):
    raw = {
        "markdown": "Authorization: Bearer body-secret\npage body",
        "metadata": {
            "url": "https://example.com/",
            "title": "Example",
            "api_key": "metadata-secret",
        },
    }
    outcome = sanitize_mcp_call_result(
        CallToolResult(
            content=[TextContent(type="text", text=json.dumps(raw))],
        ),
        settings=settings_factory(),
        server_alias="firecrawl",
        remote_tool_name="firecrawl_scrape",
        profile=McpSerializationProfile.INGESTION,
    )
    assert outcome.ok is True
    assert outcome.ingestion_output is not None

    async def invoke(*_args, **_kwargs):
        return {"ok": True, "output": outcome.ingestion_output}

    monkeypatch.setattr("app.sources.service.invoke_mcp_tool", invoke)
    service = ExternalKnowledgeSourceService(requester=_requester())
    candidate = asyncio.run(
        service.propose_firecrawl_page(
            url="https://example.com/",
            context=_context(),
        )
    )
    serialized = candidate.model_dump_json()

    assert candidate.status is IngestionCandidateStatus.PENDING
    assert candidate.document.content_redacted is False
    assert "***REDACTED***" in candidate.document.content
    assert "body-secret" not in serialized
    assert "metadata-secret" not in serialized


@pytest.mark.parametrize(
    ("blocks", "error_code"),
    [
        ([{"type": "text", "truncated": True, "size_chars": 20}], "SOURCE_CONTENT_TRUNCATED"),
        (
            [{"type": "text", "truncated": True, "size_chars": 1_000_001}],
            "SOURCE_CONTENT_TOO_LARGE",
        ),
    ],
)
def test_truncated_and_oversized_sources_fail_before_document(
    monkeypatch,
    governed_provider,
    blocks,
    error_code,
):
    async def invoke(*_args, **_kwargs):
        return _output(*blocks, truncated=True)

    monkeypatch.setattr("app.sources.service.invoke_mcp_tool", invoke)
    service = ExternalKnowledgeSourceService(requester=_requester())

    with pytest.raises(SourceIngestionError) as exc_info:
        asyncio.run(
            service.propose_firecrawl_page(
                url="https://example.com/",
                context=_context(),
            )
        )
    assert exc_info.value.code == error_code
    assert service.ingestion_service.repository.count() == 0


def test_non_read_only_or_invisible_source_tool_fails_closed(monkeypatch):
    external_write = SimpleNamespace(
        name="firecrawl_firecrawl_scrape",
        metadata={"tool_origin": "mcp", "risk_level": "external_write"},
    )
    provider = SimpleNamespace(list_tools=lambda *, context: [external_write])
    monkeypatch.setattr("app.sources.service.get_mcp_tool_provider", lambda: provider)
    service = ExternalKnowledgeSourceService(requester=_requester())

    with pytest.raises(SourceIngestionError) as exc_info:
        asyncio.run(
            service.propose_firecrawl_page(
                url="https://example.com/",
                context=_context(),
            )
        )
    assert exc_info.value.code == "SOURCE_TOOL_NOT_READ_ONLY"

    monkeypatch.setattr("app.sources.service.get_mcp_tool_provider", lambda: None)
    with pytest.raises(SourceIngestionError) as unavailable:
        asyncio.run(
            service.propose_github_file(
                owner="owner",
                repository="repo",
                path="README.md",
                context=_context(),
            )
        )
    assert unavailable.value.code == "SOURCE_TOOL_UNAVAILABLE"
