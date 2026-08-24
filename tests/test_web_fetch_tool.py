"""安全 Web Fetch 参数、下载、提取、SSRF 与 ToolRuntime 集成测试。"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import replace
from importlib import import_module
from io import BytesIO
from pathlib import Path

import pytest
import requests
from pydantic import ValidationError
from requests import Response
from requests_hardened import Config, Manager
from requests_hardened.ip_filter import InvalidIPAddress
from requests_hardened.ip_filter_adapter import IPFilterAdapter

from app.config import (
    AppSettings,
)
import app.tools.runtime as tool_runtime
from app.tools.audit import NullToolAuditSink
from app.tools.context import ToolInvocationContext
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.registry import ToolRegistry, get_tool_registry
from app.tools.result import tool_success
from app.tools.runtime import ToolRuntime, execute_registered_tool
from app.tools.schemas import WebFetchArgs, WebFetchOutput


web_fetch_module = import_module("app.tools.web_fetch")
web_download_module = import_module("app.tools.web_download")
web_fetch_handler = web_fetch_module.web_fetch
ARTICLE_HTML = (
    Path(__file__).parent / "fixtures" / "web_fetch_article.html"
).read_text(encoding="utf-8")


class FakeResponse:
    """提供 requests.Response 所需流式表面的无网络响应。"""

    def __init__(
        self,
        body: bytes = ARTICLE_HTML.encode(),
        *,
        status_code: int = 200,
        content_type: str | None = "text/html; charset=utf-8",
        url: str = "https://example.com/article",
        content_length: str | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.body = body
        self.status_code = status_code
        self.url = url
        self.encoding = "utf-8"
        self.headers: dict[str, str] = {}
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        self.chunks = chunks
        self.iteration_count = 0
        self.closed = False

    def iter_content(self, chunk_size: int):
        """按测试指定块或真实 body 产生有限字节。"""
        self.iteration_count += 1
        if self.chunks is not None:
            yield from self.chunks
            return
        for index in range(0, len(self.body), chunk_size):
            yield self.body[index : index + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """记录 hardened Manager 返回 Session 的请求和关闭状态。"""

    def __init__(self, outcome: FakeResponse | Exception) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str, dict]] = []
        self.trust_env = True
        self.max_redirects = 30
        self.closed = False

    def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def close(self) -> None:
        self.closed = True


class CollectingAuditSink:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def isolate_web_fetch(settings_env):
    """固定安全配置并隔离共享 Runtime 与生产审计。"""
    tool_runtime.reset_tool_runtime()
    settings_env(
        web_fetch_connect_timeout=3.0,
        web_fetch_read_timeout=10.0,
        web_fetch_max_response_bytes=2_000_000,
        web_fetch_max_redirects=5,
        web_fetch_user_agent="rag-agent-qa-system/1.0",
        web_fetch_proxy="",
        web_trusted_proxy_mode=False,
        tool_audit_enabled=False,
    )
    yield
    tool_runtime.reset_tool_runtime()


def _web_fetch_spec():
    """返回默认注册表中的 Web 抓取工具契约。"""
    spec = get_tool_registry().get("web_fetch")
    assert spec is not None
    return spec


def _args(**overrides):
    values = {"url": "https://example.com/article", "max_chars": 12000}
    values.update(overrides)
    return values


def _runtime(registry: ToolRegistry, *, audit_sink=None) -> ToolRuntime:
    return ToolRuntime(
        registry,
        max_workers=2,
        retry_base_delay_ms=0,
        retry_max_delay_ms=0,
        retry_jitter_ratio=0,
        sleep_func=lambda _delay: None,
        audit_sink=audit_sink or NullToolAuditSink(),
    )


def _install_fake_manager(monkeypatch, outcomes):
    """用按 Manager 实例消费结果的 fake 替换网络边界。"""
    remaining = list(outcomes)
    managers = []

    class FakeManager:
        def __init__(self, config):
            self.config = config
            self.session = FakeSession(remaining.pop(0))
            managers.append(self)

        def get_session(self):
            return self.session

    monkeypatch.setattr(web_download_module, "Manager", FakeManager)
    return managers


@pytest.mark.parametrize("url", ["http://example.com/", "https://example.com/a"])
def test_web_fetch_args_accept_http_and_https(url):
    args = WebFetchArgs(url=url)
    assert str(args.url).startswith(url.rstrip("/"))
    assert args.max_chars == 12000


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///tmp/page.html",
        "https://user:secret@example.com/",
        "https://example.com:8080/",
        "https://example.com/\r\nInjected: yes",
        "",
    ],
)
def test_web_fetch_args_reject_unsafe_urls(url):
    with pytest.raises(ValidationError):
        WebFetchArgs(url=url)


def test_web_fetch_args_reject_overlong_url():
    with pytest.raises(ValidationError):
        WebFetchArgs(url="https://example.com/" + "x" * 2048)


@pytest.mark.parametrize("max_chars", [999, 20001, True])
def test_web_fetch_args_reject_invalid_max_chars(max_chars):
    with pytest.raises(ValidationError):
        WebFetchArgs(url="https://example.com/", max_chars=max_chars)


@pytest.mark.parametrize("max_chars", [1000, 20000])
def test_web_fetch_args_accept_max_char_boundaries(max_chars):
    assert (
        WebFetchArgs(url="https://example.com/", max_chars=max_chars).max_chars
        == max_chars
    )


def test_web_fetch_args_forbid_runtime_request_controls():
    with pytest.raises(ValidationError):
        WebFetchArgs(
            url="https://example.com/",
            headers={"Authorization": "secret"},
        )


def test_web_fetch_output_validates_url_status_and_content_length():
    values = {
        "requested_url": "https://example.com/",
        "final_url": "https://example.com/final",
        "status_code": 200,
        "content_type": "text/plain",
        "content": "正文",
        "content_length": 2,
        "truncated": False,
    }
    assert WebFetchOutput.model_validate(values).content_is_untrusted is True
    with pytest.raises(ValidationError):
        WebFetchOutput.model_validate({**values, "content_length": 1})
    with pytest.raises(ValidationError):
        WebFetchOutput.model_validate({**values, "status_code": 404})
    with pytest.raises(ValidationError):
        WebFetchOutput.model_validate({**values, "final_url": "file:///tmp/x"})


def test_web_fetch_configuration_accepts_safe_values():
    """Web 抓取配置接受满足安全约束的取值。"""
    AppSettings(
        _env_file=None,
        web_fetch_connect_timeout=3,
        web_fetch_read_timeout=10.0,
        web_fetch_max_response_bytes=2_000_000,
        web_fetch_max_redirects=5,
        web_fetch_user_agent="rag-agent-qa-system/1.0",
        web_fetch_proxy=None,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"web_fetch_connect_timeout": 0},
        {"web_fetch_read_timeout": True},
        {"web_fetch_read_timeout": 27},
        {"web_fetch_max_response_bytes": 0},
        {"web_fetch_max_response_bytes": True},
        {"web_fetch_max_redirects": -1},
        {"web_fetch_max_redirects": False},
        {"web_fetch_user_agent": ""},
        {"web_fetch_user_agent": "agent\nInjected"},
        {"web_fetch_proxy": 7},
    ],
)
def test_web_fetch_configuration_rejects_invalid_values(overrides):
    """Web 抓取配置拒绝越界、布尔冒充数值和非法文本。"""
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, **overrides)


def test_web_fetch_empty_proxy_is_normalized_to_none():
    """空代理被规范为未配置，非空代理会去除两端空白。"""
    assert AppSettings(_env_file=None, web_fetch_proxy="").web_fetch_proxy is None
    assert AppSettings(_env_file=None, web_fetch_proxy="   ").web_fetch_proxy is None
    assert AppSettings(
        _env_file=None,
        web_fetch_proxy=" http://proxy.example:8080 ",
    ).web_fetch_proxy == (
        "http://proxy.example:8080"
    )


def test_request_uses_hardened_manager_get_only_and_safe_defaults(monkeypatch):
    response = FakeResponse()
    managers = _install_fake_manager(monkeypatch, [response])

    result = web_fetch_handler(
        _args(url="https://example.com/article#section"), ToolInvocationContext()
    )

    manager = managers[0]
    config = manager.config
    session = manager.session
    assert config.ip_filter_enable is True
    assert config.ip_filter_allow_loopback_ips is False
    assert config.never_redirect is False
    assert config.default_timeout == (3.0, 10.0)
    assert config.user_agent_override == "rag-agent-qa-system/1.0"
    assert session.trust_env is False
    assert session.max_redirects == 5
    assert session.closed is True
    assert response.closed is True
    method, url, kwargs = session.calls[0]
    assert method == "GET"
    assert url == "https://example.com/article"
    assert kwargs["stream"] is True
    assert kwargs["allow_redirects"] is True
    assert kwargs["timeout"] == (3.0, 10.0)
    assert kwargs["headers"] == {
        "Accept": "text/html, application/xhtml+xml, text/plain",
        "User-Agent": "rag-agent-qa-system/1.0",
    }
    assert "proxies" not in kwargs
    assert not ({"Cookie", "Authorization", "Referer"} & kwargs["headers"].keys())
    assert result["output"]["requested_url"] == "https://example.com/article"


def test_explicit_server_proxy_is_used_but_never_returned(
    monkeypatch,
    settings_env,
):
    """显式服务端代理参与请求，但不会出现在工具返回中。"""
    proxy = "http://trusted-proxy.example:8080"
    settings_env(web_fetch_proxy=proxy)
    managers = _install_fake_manager(
        monkeypatch,
        [FakeResponse(body=b"plain body", content_type="text/plain")],
    )

    result = web_fetch_handler(_args(), ToolInvocationContext())

    kwargs = managers[0].session.calls[0][2]
    assert managers[0].config.ip_filter_enable is True
    assert kwargs["proxies"] == {"http": proxy, "https": proxy}
    assert proxy not in json.dumps(result)


def test_trusted_proxy_mode_delegates_ip_policy_and_uses_explicit_proxy(
    monkeypatch,
    settings_env,
):
    """受信代理模式委托 IP 策略，并仅使用显式代理。"""
    proxy = "http://127.0.0.1:7890"
    settings_env(
        web_trusted_proxy_mode=True,
        web_trusted_proxy_url=proxy,
    )
    managers = _install_fake_manager(
        monkeypatch,
        [FakeResponse(body=b"plain body", content_type="text/plain")],
    )

    result = web_fetch_handler(_args(), ToolInvocationContext())

    manager = managers[0]
    kwargs = manager.session.calls[0][2]
    assert manager.config.ip_filter_enable is False
    assert manager.config.ip_filter_allow_loopback_ips is False
    assert manager.session.trust_env is False
    assert kwargs["proxies"] == {"http": proxy, "https": proxy}
    assert proxy not in json.dumps(result)


def test_trusted_proxy_mode_without_proxy_fails_closed(settings_env):
    """受信代理模式缺少代理地址时以拒绝方式失败。"""
    with pytest.raises(ValidationError, match="WEB_TRUSTED_PROXY_URL"):
        settings_env(
            web_trusted_proxy_mode=True,
            web_trusted_proxy_url="",
        )


def test_proxy_failure_returns_stable_message_without_proxy(
    monkeypatch,
    settings_env,
):
    """代理失败返回稳定错误，且不泄露代理地址。"""
    proxy = "http://secret-proxy.example:8080"
    settings_env(web_fetch_proxy=proxy)
    _install_fake_manager(
        monkeypatch,
        [requests.exceptions.ProxyError(f"failed through {proxy}")],
    )

    with pytest.raises(ToolExecutionError) as raised:
        web_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.DEPENDENCY_UNAVAILABLE
    assert proxy not in str(raised.value)
    assert "proxy" not in str(raised.value).lower()


def test_content_length_limit_stops_before_stream_and_closes(
    monkeypatch,
    settings_env,
):
    """响应长度头超限时不读取流，并关闭响应与会话。"""
    settings_env(web_fetch_max_response_bytes=10)
    response = FakeResponse(content_length="11")
    managers = _install_fake_manager(monkeypatch, [response])

    with pytest.raises(ToolExecutionError) as raised:
        web_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.INVALID_RESULT
    assert response.iteration_count == 0
    assert response.closed is True
    assert managers[0].session.closed is True


@pytest.mark.parametrize("content_length", [None, "invalid"])
def test_streamed_body_limit_applies_without_valid_content_length(
    monkeypatch,
    settings_env,
    content_length,
):
    """缺少有效长度头时仍对流式响应实施字节上限。"""
    settings_env(web_fetch_max_response_bytes=10)
    response = FakeResponse(
        content_length=content_length,
        chunks=[b"123456", b"78901"],
    )
    managers = _install_fake_manager(monkeypatch, [response])

    with pytest.raises(ToolExecutionError) as raised:
        web_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.INVALID_RESULT
    assert response.iteration_count == 1
    assert response.closed is True
    assert managers[0].session.closed is True


@pytest.mark.parametrize("content_type", ["text/html", "application/xhtml+xml"])
def test_html_types_use_real_trafilatura(monkeypatch, content_type):
    _install_fake_manager(
        monkeypatch,
        [FakeResponse(content_type=f"{content_type}; charset=utf-8")],
    )

    result = web_fetch_handler(_args(), ToolInvocationContext())
    output = WebFetchOutput.model_validate(result["output"])

    assert output.content_type == content_type
    assert output.title == "受治理的网页正文测试文章"
    assert "Trafilatura 正文提取" in output.content
    assert output.published_date == "2026-07-22"
    assert output.author == ""
    assert output.content_is_untrusted is True
    assert output.content_length == len(output.content)
    assert "<html" not in output.content
    assert "站点导航" not in output.content


def test_plain_text_is_normalized_without_html_extraction(monkeypatch):
    _install_fake_manager(
        monkeypatch,
        [
            FakeResponse(
                body=b"first\r\n\r\n\r\nsecond\n",
                content_type="text/plain; charset=utf-8",
            )
        ],
    )

    output = web_fetch_handler(_args(), ToolInvocationContext())["output"]

    assert output["content"] == "first\n\nsecond"
    assert output["title"] == ""
    assert output["author"] == ""
    assert output["published_date"] == ""
    assert output["language"] == ""


@pytest.mark.parametrize(
    "content_type", ["application/pdf", "application/octet-stream", None]
)
def test_unsupported_or_missing_content_type_is_rejected(monkeypatch, content_type):
    _install_fake_manager(
        monkeypatch,
        [FakeResponse(content_type=content_type)],
    )

    with pytest.raises(ToolExecutionError) as raised:
        web_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.INVALID_RESULT
    assert raised.value.retryable is False


def test_empty_trafilatura_result_is_non_retryable(monkeypatch):
    _install_fake_manager(monkeypatch, [FakeResponse()])
    monkeypatch.setattr(web_fetch_module, "extract", lambda *args, **kwargs: None)

    with pytest.raises(ToolExecutionError) as raised:
        web_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.INVALID_RESULT
    assert raised.value.retryable is False


def test_extracted_content_is_truncated_at_requested_character_limit(monkeypatch):
    _install_fake_manager(monkeypatch, [FakeResponse()])
    monkeypatch.setattr(
        web_fetch_module,
        "extract",
        lambda *args, **kwargs: json.dumps(
            {"text": "正" * 1500, "title": "长文章"}, ensure_ascii=False
        ),
    )

    output = web_fetch_handler(_args(max_chars=1000), ToolInvocationContext())["output"]

    assert len(output["content"]) == 1000
    assert output["content_length"] == 1000
    assert output["truncated"] is True


@pytest.mark.parametrize(
    ("exception", "code", "retryable"),
    [
        (InvalidIPAddress("internal"), ToolErrorCode.POLICY_DENIED, False),
        (requests.exceptions.ConnectTimeout("late"), ToolErrorCode.TIMEOUT, True),
        (
            requests.exceptions.ConnectionError("dns failed"),
            ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            True,
        ),
    ],
)
def test_public_request_exceptions_are_safely_mapped(
    monkeypatch, exception, code, retryable
):
    _install_fake_manager(monkeypatch, [exception])

    with pytest.raises(ToolExecutionError) as raised:
        web_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is code
    assert raised.value.retryable is retryable
    assert "internal" not in str(raised.value)
    assert "dns failed" not in str(raised.value)


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (404, ToolErrorCode.EXECUTION_ERROR, False),
        (429, ToolErrorCode.RATE_LIMITED, True),
        (500, ToolErrorCode.DEPENDENCY_UNAVAILABLE, True),
    ],
)
def test_http_status_mapping(monkeypatch, status, code, retryable):
    _install_fake_manager(monkeypatch, [FakeResponse(status_code=status)])

    with pytest.raises(ToolExecutionError) as raised:
        web_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is code
    assert raised.value.retryable is retryable


@pytest.mark.parametrize(
    ("target", "resolved_ip", "family"),
    [
        ("http://127.0.0.1/", "127.0.0.1", socket.AF_INET),
        ("http://localhost/", "127.0.0.1", socket.AF_INET),
        ("http://[::1]/", "::1", socket.AF_INET6),
        ("http://169.254.169.254/", "169.254.169.254", socket.AF_INET),
        ("http://10.0.0.1/", "10.0.0.1", socket.AF_INET),
        ("http://192.168.1.1/", "192.168.1.1", socket.AF_INET),
    ],
)
def test_requests_hardened_blocks_internal_targets_before_transport(
    monkeypatch, target, resolved_ip, family
):
    """直接使用公开 Manager 验证内网目标在 socket 发送前被 IP 过滤。"""
    socket_address = (
        (resolved_ip, 80, 0, 0) if family == socket.AF_INET6 else (resolved_ip, 80)
    )
    monkeypatch.setattr(
        "requests_hardened.ip_filter.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", socket_address)
        ],
    )
    manager = Manager(
        Config(
            ip_filter_enable=True,
            ip_filter_allow_loopback_ips=False,
            never_redirect=False,
            default_timeout=(0.1, 0.1),
        )
    )
    with manager.get_session() as session:
        session.trust_env = False
        with pytest.raises(InvalidIPAddress):
            session.request("GET", target, stream=True)


class RedirectTestAdapter(IPFilterAdapter):
    """保留真实 IPFilterAdapter 判定，仅替换最终无网络传输。"""

    def __init__(self) -> None:
        super().__init__(is_https_proto=False, allow_loopback=False)
        self.transported_urls: list[str] = []

    def send(self, request, **kwargs):
        self.build_connection_pool_key_attributes(request, verify=True)
        self.transported_urls.append(request.url)
        response = Response()
        response.status_code = 302
        response.url = request.url
        response.request = request
        response.headers["Location"] = "http://127.0.0.1/private"
        response.raw = BytesIO()
        return response


def test_public_redirect_to_private_is_blocked_before_second_transport(monkeypatch):
    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host == "public.example":
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("93.184.216.34", port),
                )
            ]
        return original_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(
        "requests_hardened.ip_filter.socket.getaddrinfo", fake_getaddrinfo
    )
    manager = Manager(
        Config(
            ip_filter_enable=True,
            ip_filter_allow_loopback_ips=False,
            never_redirect=False,
            default_timeout=(0.1, 0.1),
        )
    )
    adapter = RedirectTestAdapter()
    with manager.get_session() as session:
        session.trust_env = False
        session.mount("http://", adapter)
        with pytest.raises(InvalidIPAddress):
            session.request("GET", "http://public.example/start")

    assert adapter.transported_urls == ["http://public.example/start"]


def test_ssrf_denial_never_falls_back_to_raw_requests(monkeypatch):
    _install_fake_manager(monkeypatch, [InvalidIPAddress("127.0.0.1")])
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("raw requests fallback is forbidden")
        ),
    )

    with pytest.raises(ToolExecutionError) as raised:
        web_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.POLICY_DENIED


def test_registry_runtime_executes_production_web_fetch_spec(monkeypatch):
    managers = _install_fake_manager(monkeypatch, [FakeResponse()])
    registry = ToolRegistry()
    registry.register(_web_fetch_spec())

    result = execute_registered_tool(
        registry=registry,
        tool_name="web_fetch",
        tool_args=_args(),
        context=ToolInvocationContext(permissions=("web:fetch",)),
    )

    assert result["ok"] is True
    assert result["runtime"]["attempt_count"] == 1
    assert result["governance"]["decision"] == "allow"
    assert WebFetchOutput.model_validate(result["output"])
    assert len(managers) == 1


def test_permission_denial_prevents_manager_construction(monkeypatch):
    class ForbiddenManager:
        def __init__(self, config):
            raise AssertionError(f"manager must not be constructed: {config}")

    monkeypatch.setattr(web_download_module, "Manager", ForbiddenManager)
    registry = ToolRegistry()
    registry.register(_web_fetch_spec())

    result = execute_registered_tool(
        registry=registry,
        tool_name="web_fetch",
        tool_args=_args(),
        context=ToolInvocationContext(),
    )

    assert result["error_code"] == "permission_denied"
    assert result["runtime"]["attempt_count"] == 0


def test_runtime_retries_temporary_dependency_once_then_succeeds(monkeypatch):
    managers = _install_fake_manager(
        monkeypatch,
        [requests.exceptions.ConnectionError("temporary"), FakeResponse()],
    )
    registry = ToolRegistry()
    registry.register(_web_fetch_spec())
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_fetch",
            _args(),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
    finally:
        runtime.close()

    assert result["ok"] is True
    assert len(managers) == 2
    assert [len(manager.session.calls) for manager in managers] == [1, 1]
    assert result["runtime"]["attempt_count"] == 2
    assert result["runtime"]["max_attempts"] == 2
    assert [item["error_code"] for item in result["runtime"]["attempts"]] == [
        "dependency_unavailable",
        "",
    ]


@pytest.mark.parametrize(
    "first_outcome",
    [
        InvalidIPAddress("blocked"),
        FakeResponse(status_code=404),
    ],
)
def test_runtime_does_not_retry_non_retryable_fetch_failures(
    monkeypatch, first_outcome
):
    managers = _install_fake_manager(monkeypatch, [first_outcome, FakeResponse()])
    registry = ToolRegistry()
    registry.register(_web_fetch_spec())
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_fetch",
            _args(),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
    finally:
        runtime.close()

    assert result["ok"] is False
    assert result["runtime"]["attempt_count"] == 1
    assert len(managers) == 1


@pytest.mark.parametrize("status", [429, 500])
def test_runtime_retries_rate_limit_and_server_error(monkeypatch, status):
    managers = _install_fake_manager(
        monkeypatch, [FakeResponse(status_code=status), FakeResponse()]
    )
    registry = ToolRegistry()
    registry.register(_web_fetch_spec())
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_fetch",
            _args(),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
    finally:
        runtime.close()

    assert result["ok"] is True
    assert result["runtime"]["attempt_count"] == 2
    assert len(managers) == 2


def test_runtime_output_model_rejects_invalid_fetch_output():
    invalid_spec = replace(
        _web_fetch_spec(),
        handler=lambda args, context: tool_success(
            "web_fetch",
            args,
            {
                "requested_url": args["url"],
                "final_url": args["url"],
                "status_code": 200,
                "content_type": "text/plain",
                "content": "正文",
                "content_length": 999,
                "truncated": False,
            },
        ),
    )
    registry = ToolRegistry()
    registry.register(invalid_spec)
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_fetch",
            _args(),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
    finally:
        runtime.close()

    assert result["error_code"] == "output_validation_error"
    assert result["runtime"]["attempt_count"] == 1


def test_runtime_total_timeout_still_bounds_fetch(monkeypatch):
    completed = threading.Event()

    class SlowSession(FakeSession):
        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            time.sleep(0.08)
            completed.set()
            return self.outcome

    class SlowManager:
        def __init__(self, config):
            self.session = SlowSession(FakeResponse())

        def get_session(self):
            return self.session

    monkeypatch.setattr(web_download_module, "Manager", SlowManager)
    registry = ToolRegistry()
    registry.register(replace(_web_fetch_spec(), timeout=0.01))
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_fetch",
            _args(),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
        assert completed.wait(timeout=1)
    finally:
        runtime.close()

    assert result["error_code"] == "timeout"
    assert result["runtime"]["attempt_count"] == 1
    assert result["runtime"]["timed_out"] is True


def test_fetch_audit_never_records_full_url_proxy_or_content(
    monkeypatch,
    settings_env,
):
    """抓取审计不记录完整 URL、代理地址或页面正文。"""
    proxy = "http://audit-secret-proxy.example:8080"
    url = "https://example.com/private/path?secret=1"
    settings_env(web_fetch_proxy=proxy)
    _install_fake_manager(
        monkeypatch,
        [
            FakeResponse(
                body=b"private page content",
                content_type="text/plain",
                url="https://example.com/final-private",
            )
        ],
    )
    sink = CollectingAuditSink()
    registry = ToolRegistry()
    registry.register(_web_fetch_spec())
    runtime = _runtime(registry, audit_sink=sink)
    try:
        result = runtime.execute(
            "web_fetch",
            _args(url=url),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
    finally:
        runtime.close()

    serialized = json.dumps([event.to_dict() for event in sink.events])
    assert result["ok"] is True
    assert url not in serialized
    assert proxy not in serialized
    assert "private page content" not in serialized
    assert "final-private" not in serialized
    assert "content_length" in serialized
