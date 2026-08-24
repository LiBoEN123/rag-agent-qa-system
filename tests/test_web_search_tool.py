"""DDGS Web Search 参数、规范化、错误和 ToolRuntime 集成测试。"""

import json
import threading
import time
from dataclasses import replace
from importlib import import_module

import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from pydantic import ValidationError

import app.tools.runtime as tool_runtime
from app.tools.audit import NullToolAuditSink
from app.tools.context import ToolInvocationContext
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.registry import ToolRegistry, get_tool_registry
from app.tools.result import tool_success
from app.tools.runtime import ToolRuntime, execute_registered_tool
from app.tools.schemas import WebSearchArgs, WebSearchOutput


web_search_module = import_module("app.tools.web_search")
web_search_handler = web_search_module.web_search


class CollectingAuditSink:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


@pytest.fixture(autouse=True)
def isolate_web_search(settings_env):
    """固定 Web 搜索配置并隔离共享工具运行时。"""
    tool_runtime.reset_tool_runtime()
    settings_env(
        web_search_region="us-en",
        web_search_safesearch="moderate",
        web_search_timeout=8,
        web_search_proxy="",
    )
    yield
    tool_runtime.reset_tool_runtime()


def _runtime(registry, *, audit_sink=None, sleep_func=lambda _delay: None):
    return ToolRuntime(
        registry,
        max_workers=2,
        retry_base_delay_ms=0,
        retry_max_delay_ms=0,
        retry_jitter_ratio=0,
        sleep_func=sleep_func,
        audit_sink=audit_sink or NullToolAuditSink(),
    )


def _web_spec():
    """返回默认注册表中的 Web 搜索工具契约。"""
    spec = get_tool_registry().get("web_search")
    assert spec is not None
    return spec


def _args(**overrides):
    values = {"query": "LangGraph releases", "max_results": 5, "freshness": "any"}
    values.update(overrides)
    return values


@pytest.mark.parametrize("query", ["", "   ", "x" * 501])
def test_web_search_args_reject_empty_or_too_long_query(query):
    with pytest.raises(ValidationError):
        WebSearchArgs(query=query)


def test_web_search_args_strip_query_and_apply_defaults():
    args = WebSearchArgs(query="  Python news  ")
    assert args.model_dump() == {
        "query": "Python news",
        "max_results": 5,
        "freshness": "any",
    }


@pytest.mark.parametrize("max_results", [0, 11])
def test_web_search_args_reject_max_results_outside_bounds(max_results):
    with pytest.raises(ValidationError):
        WebSearchArgs(query="python", max_results=max_results)


@pytest.mark.parametrize("max_results", [1, 10])
def test_web_search_args_accept_max_results_boundaries(max_results):
    assert WebSearchArgs(query="python", max_results=max_results).max_results == max_results


@pytest.mark.parametrize("freshness", ["any", "day", "week", "month", "year"])
def test_web_search_args_accept_freshness_values(freshness):
    assert WebSearchArgs(query="python", freshness=freshness).freshness == freshness


def test_web_search_args_reject_invalid_freshness():
    with pytest.raises(ValidationError):
        WebSearchArgs(query="python", freshness="hour")


@pytest.mark.parametrize(
    ("freshness", "timelimit"),
    [("any", None), ("day", "d"), ("week", "w"), ("month", "m"), ("year", "y")],
)
def test_ddgs_parameters_and_freshness_mapping(
    monkeypatch,
    settings_env,
    freshness,
    timelimit,
):
    """搜索参数与新鲜度选项正确映射到 DDGS 调用。"""
    calls = []

    class FakeDDGS:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def text(self, query, **kwargs):
            calls.append(("text", query, kwargs))
            return []

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    settings_env(
        web_search_region="cn-zh",
        web_search_safesearch="on",
        web_search_timeout=7,
        web_search_proxy="socks5://proxy.invalid:1080",
    )

    result = web_search_handler(
        _args(query="Python", max_results=3, freshness=freshness),
        ToolInvocationContext(),
    )

    assert result["ok"] is True
    assert calls == [
        (
            "init",
            {"proxy": "socks5://proxy.invalid:1080", "timeout": 7},
        ),
        (
            "text",
            "Python",
            {
                "region": "cn-zh",
                "safesearch": "on",
                "timelimit": timelimit,
                "max_results": 3,
                "backend": "auto",
            },
        ),
    ]


def test_web_search_normalizes_filters_deduplicates_and_limits(monkeypatch):
    raw = [
        {
            "title": "  Python &amp; LangGraph  ",
            "href": "https://example.com/a?x=1&amp;y=2",
            "body": "  First\n\t snippet &amp; details  ",
            "debug": "drop-me",
        },
        {"title": "duplicate", "href": "https://example.com/a?x=1&y=2", "body": "dup"},
        {"href": "https://example.com/no-title", "body": 123},
        {"title": "FTP", "href": "ftp://example.com/file", "body": "drop"},
        {"title": "Long", "href": "http://example.com/long", "body": "z" * 1200},
        "not-a-dict",
    ]

    class FakeDDGS:
        def __init__(self, **kwargs):
            del kwargs

        def text(self, query, **kwargs):
            del query, kwargs
            return raw

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    result = web_search_handler(_args(max_results=10), ToolInvocationContext())
    output = WebSearchOutput.model_validate(result["output"])

    assert output.count == 3
    assert [item.rank for item in output.results] == [1, 2, 3]
    assert [item.url for item in output.results] == [
        "https://example.com/a?x=1&y=2",
        "https://example.com/no-title",
        "http://example.com/long",
    ]
    assert output.results[0].title == "Python & LangGraph"
    assert output.results[0].snippet == "First snippet & details"
    assert output.results[1].title == ""
    assert output.results[1].snippet == ""
    assert len(output.results[2].snippet) == 1000
    assert set(output.results[0].model_dump()) == {"rank", "title", "url", "snippet"}


def test_web_search_zero_results_is_success(monkeypatch):
    class FakeDDGS:
        def __init__(self, **kwargs):
            del kwargs

        def text(self, query, **kwargs):
            del query, kwargs
            return []

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    result = web_search_handler(_args(query="nothing"), ToolInvocationContext())
    assert result["ok"] is True
    assert result["output"] == {
        "query": "nothing",
        "count": 0,
        "provider": "ddgs",
        "content_is_untrusted": True,
        "results": [],
    }


def test_web_search_rejects_invalid_top_level_result(monkeypatch):
    class FakeDDGS:
        def __init__(self, **kwargs):
            del kwargs

        def text(self, query, **kwargs):
            del query, kwargs
            return iter([])

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    with pytest.raises(ToolExecutionError) as raised:
        web_search_handler(_args(), ToolInvocationContext())
    assert raised.value.code is ToolErrorCode.INVALID_RESULT
    assert raised.value.retryable is False


@pytest.mark.parametrize(
    ("exception", "code", "retryable"),
    [
        (RatelimitException("secret proxy rate"), ToolErrorCode.RATE_LIMITED, True),
        (TimeoutException("secret proxy timeout"), ToolErrorCode.TIMEOUT, True),
        (
            DDGSException("secret proxy dependency"),
            ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            True,
        ),
        (ValueError("secret proxy argument"), ToolErrorCode.INVALID_ARGUMENTS, False),
    ],
)
def test_web_search_maps_public_ddgs_errors_without_leaking_details(
    monkeypatch,
    exception,
    code,
    retryable,
):
    calls = 0

    class FakeDDGS:
        def __init__(self, **kwargs):
            del kwargs

        def text(self, query, **kwargs):
            nonlocal calls
            del query, kwargs
            calls += 1
            raise exception

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    with pytest.raises(ToolExecutionError) as raised:
        web_search_handler(_args(), ToolInvocationContext())
    assert raised.value.code is code
    assert raised.value.retryable is retryable
    assert "secret" not in str(raised.value)
    assert "proxy" not in str(raised.value).lower()
    assert calls == 1


def test_runtime_executes_production_web_search_spec_with_permission(monkeypatch):
    calls = []

    class FakeDDGS:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def text(self, query, **kwargs):
            calls.append((query, kwargs))
            return [{"title": "Release", "href": "https://example.com", "body": "News"}]

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    registry = ToolRegistry()
    registry.register(_web_spec())
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_search",
            {"query": "  release news  ", "max_results": 2, "freshness": "week"},
            context=ToolInvocationContext(permissions=("web:search",)),
        )
    finally:
        runtime.close()

    assert result["ok"] is True
    assert result["tool_input"]["query"] == "release news"
    assert result["output"]["count"] == 1
    assert WebSearchOutput.model_validate(result["output"])
    assert result["runtime"]["attempt_count"] == 1
    assert len(calls) == 2


def test_execution_entry_uses_web_runtime_and_output_model(
    monkeypatch,
    settings_env,
):
    """唯一执行入口经过 Web Runtime，并返回符合模型的输出。"""
    class FakeDDGS:
        def __init__(self, **kwargs):
            del kwargs

        def text(self, query, **kwargs):
            del query, kwargs
            return [
                {
                    "title": "Registry result",
                    "href": "https://example.com/registry",
                    "body": "Registry to Runtime",
                }
            ]

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    settings_env(tool_audit_enabled=False)
    registry = ToolRegistry()
    registry.register(_web_spec())

    result = execute_registered_tool(
        registry=registry,
        tool_name="web_search",
        tool_args=_args(),
        context=ToolInvocationContext(permissions=("web:search",)),
    )

    assert result["ok"] is True
    assert result["output"]["results"][0]["title"] == "Registry result"
    assert result["runtime"]["attempt_count"] == 1
    assert result["governance"]["decision"] == "allow"


def test_direct_execution_entry_denies_permission_before_ddgs(
    monkeypatch,
    settings_env,
):
    """唯一执行入口在调用 DDGS 前先拒绝缺少权限的请求。"""
    class ForbiddenDDGS:
        def __init__(self, **kwargs):
            raise AssertionError(f"DDGS must not run: {kwargs}")

    monkeypatch.setattr(web_search_module, "DDGS", ForbiddenDDGS)
    settings_env(tool_audit_enabled=False)
    registry = ToolRegistry()
    registry.register(_web_spec())
    result = execute_registered_tool(
        registry=registry,
        tool_name="web_search",
        tool_args=_args(),
        context=ToolInvocationContext(),
    )
    assert result["error_code"] == "permission_denied"
    assert result["runtime"]["attempt_count"] == 0


def test_runtime_retries_dependency_error_once_then_succeeds(monkeypatch):
    calls = 0

    class FakeDDGS:
        def __init__(self, **kwargs):
            del kwargs

        def text(self, query, **kwargs):
            nonlocal calls
            del query, kwargs
            calls += 1
            if calls == 1:
                raise DDGSException("temporary")
            return [{"title": "OK", "href": "https://example.com/ok", "body": "done"}]

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    registry = ToolRegistry()
    registry.register(_web_spec())
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_search",
            _args(),
            context=ToolInvocationContext(permissions=("web:search",)),
        )
    finally:
        runtime.close()
    assert result["ok"] is True
    assert calls == 2
    assert result["runtime"]["attempt_count"] == 2
    assert result["runtime"]["max_attempts"] == 2
    assert [item["error_code"] for item in result["runtime"]["attempts"]] == [
        "dependency_unavailable",
        "",
    ]


def test_runtime_output_model_rejects_invalid_web_output():
    invalid_spec = replace(
        _web_spec(),
        handler=lambda args, context: tool_success(
            "web_search",
            args,
            {
                "query": args["query"],
                "count": 2,
                "provider": "ddgs",
                "content_is_untrusted": True,
                "results": [],
            },
        ),
    )
    registry = ToolRegistry()
    registry.register(invalid_spec)
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_search",
            _args(),
            context=ToolInvocationContext(permissions=("web:search",)),
        )
    finally:
        runtime.close()
    assert result["error_code"] == "output_validation_error"
    assert result["runtime"]["attempt_count"] == 1


def test_web_search_audit_summary_redacts_query_and_result_text(monkeypatch):
    secret_query = "private exact search phrase"

    class FakeDDGS:
        def __init__(self, **kwargs):
            del kwargs

        def text(self, query, **kwargs):
            del query, kwargs
            return [{"title": "Secret title", "href": "https://example.com", "body": "Secret body"}]

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    sink = CollectingAuditSink()
    registry = ToolRegistry()
    registry.register(_web_spec())
    runtime = _runtime(registry, audit_sink=sink)
    try:
        result = runtime.execute(
            "web_search",
            _args(query=secret_query),
            context=ToolInvocationContext(permissions=("web:search",)),
        )
    finally:
        runtime.close()
    serialized = json.dumps([event.to_dict() for event in sink.events])
    assert result["ok"] is True
    assert secret_query not in serialized
    assert "Secret title" not in serialized
    assert "Secret body" not in serialized
    assert '"count": 1' in serialized


def test_tool_runtime_timeout_still_bounds_web_search(monkeypatch):
    completed = threading.Event()

    class SlowDDGS:
        def __init__(self, **kwargs):
            del kwargs

        def text(self, query, **kwargs):
            del query, kwargs
            time.sleep(0.08)
            completed.set()
            return []

    monkeypatch.setattr(web_search_module, "DDGS", SlowDDGS)
    registry = ToolRegistry()
    registry.register(replace(_web_spec(), timeout=0.01))
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_search",
            _args(),
            context=ToolInvocationContext(permissions=("web:search",)),
        )
        assert completed.wait(timeout=1)
    finally:
        runtime.close()
    assert result["error_code"] == "timeout"
    assert result["runtime"]["attempt_count"] == 1
    assert result["runtime"]["timed_out"] is True
