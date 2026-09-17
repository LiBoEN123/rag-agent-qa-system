"""Offline command tests for the governed Online Tool smoke CLI."""

import json

import pytest

from app.tools.context import PrincipalType, ToolInvocationContext
from scripts import smoke_test_online_tools as smoke


class FakeRegistry:
    def __init__(self, result=None, error=None):
        self.result = result or {
            "tool_name": "fake",
            "ok": True,
            "status": "success",
            "output": {},
        }
        self.error = error
        self.calls = []

    def execute(self, tool_name, tool_args, *, context):
        self.calls.append((tool_name, tool_args, context))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def route_execution_through_fake_registry(monkeypatch):
    """让 CLI 测试替身覆盖诊断 BaseTool 调用而不访问网络。"""

    def fake_execute(*, registry, tool_name, tool_args=None, context=None):
        return registry.execute(tool_name, tool_args, context=context)

    monkeypatch.setattr(smoke, "invoke_local_tool", fake_execute)


def _context(*permissions):
    return ToolInvocationContext(
        trace_id="smoke",
        principal_id="test",
        principal_type=PrincipalType.SERVICE,
        permissions=permissions,
        source="test",
    )


@pytest.mark.parametrize(
    ("argv", "permission", "tool_name", "expected_args"),
    [
        (
            ["search", "--query", "news", "--max-results", "2"],
            "web:search",
            "web_search",
            {"query": "news", "max_results": 2, "freshness": "any"},
        ),
        (
            ["fetch", "--url", "https://example.test/page"],
            "web:fetch",
            "web_fetch",
            {"url": "https://example.test/page", "max_chars": 4000},
        ),
        (
            ["pdf", "--url", "https://example.test/report.pdf"],
            "web:fetch",
            "web_pdf_fetch",
            {
                "url": "https://example.test/report.pdf",
                "start_page": 1,
                "max_pages": 3,
                "max_chars": 8000,
            },
        ),
    ],
)
def test_subcommands_use_direct_base_tool_execution_entry(
    monkeypatch,
    argv,
    permission,
    tool_name,
    expected_args,
):
    monkeypatch.setattr(smoke, "build_context", lambda name: _context(permission))
    registry = FakeRegistry()

    code = smoke.run_command(smoke.build_parser().parse_args(argv), registry=registry)

    assert code == 0
    assert registry.calls[0][0] == tool_name
    assert registry.calls[0][1] == expected_args
    assert registry.calls[0][2].permissions == (permission,)


def test_permission_failure_returns_two_without_execution(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "build_context", lambda name: _context())
    registry = FakeRegistry()
    args = smoke.build_parser().parse_args(["search", "--query", "news"])

    assert smoke.run_command(args, registry=registry) == 2
    assert registry.calls == []
    assert "web:search" in capsys.readouterr().err


def test_execution_exception_is_redacted_and_returns_one(monkeypatch, capsys):
    monkeypatch.setattr(
        smoke,
        "build_context",
        lambda name: _context("web:fetch"),
    )
    registry = FakeRegistry(error=RuntimeError("secret detail"))
    args = smoke.build_parser().parse_args(
        ["fetch", "--url", "https://example.test"]
    )

    assert smoke.run_command(args, registry=registry) == 1
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.err
    assert "secret detail" not in captured.err


def test_debug_reraises_execution_exception(monkeypatch):
    monkeypatch.setattr(
        smoke,
        "build_context",
        lambda name: _context("web:fetch"),
    )
    registry = FakeRegistry(error=RuntimeError("debug detail"))
    args = smoke.build_parser().parse_args(
        ["pdf", "--url", "https://example.test/report.pdf", "--debug"]
    )

    with pytest.raises(RuntimeError, match="debug detail"):
        smoke.run_command(args, registry=registry)


def test_runtime_failure_result_returns_one(monkeypatch):
    monkeypatch.setattr(
        smoke,
        "build_context",
        lambda name: _context("web:search"),
    )
    registry = FakeRegistry(
        result={
            "tool_name": "web_search",
            "ok": False,
            "status": "error",
            "error_code": "provider_error",
            "error": "unavailable",
        }
    )
    args = smoke.build_parser().parse_args(["search", "--query", "news"])

    assert smoke.run_command(args, registry=registry) == 1


@pytest.mark.parametrize("command", ["fetch", "pdf"])
def test_body_preview_reads_document_evidence(command, capsys):
    smoke._print_result(command, {
        "ok": True, "status": "success",
        "output": {"evidence": [{"page_content": "正文" * 500}]},
    }, 1)
    assert json.loads(capsys.readouterr().out)["content_preview"] == ("正文" * 500)[:500]
