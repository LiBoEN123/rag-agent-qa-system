"""安全 Web PDF 参数、解析、共享下载与 ToolRuntime 集成测试。"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from importlib import import_module
from io import BytesIO
from pathlib import Path

import pytest
import requests
from pydantic import ValidationError
from pypdf import PdfReader, PdfWriter

from app.config import AppSettings
import app.tools.runtime as tool_runtime
from app.tools.audit import NullToolAuditSink
from app.tools.context import ToolInvocationContext
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.registry import ToolRegistry, get_tool_registry
from app.tools.result import tool_success
from app.tools.runtime import ToolRuntime
from app.tools.schemas import WebPdfFetchArgs, WebPdfFetchOutput
from scripts.generate_web_pdf_fixture import build_text_pdf


web_download_module = import_module("app.tools.web_download")
web_fetch_module = import_module("app.tools.web_fetch")
web_pdf_fetch_module = import_module("app.tools.web_pdf_fetch")
web_pdf_fetch_handler = web_pdf_fetch_module.web_pdf_fetch
PDF_BYTES = (
    Path(__file__).parent / "fixtures" / "web_pdf_text.pdf"
).read_bytes()


class FakeResponse:
    """提供共享 hardened 下载层所需流式表面的无网络响应。"""

    def __init__(
        self,
        body: bytes = PDF_BYTES,
        *,
        status_code: int = 200,
        content_type: str | None = "application/pdf",
        url: str = "https://example.com/document.pdf",
        content_length: str | None = None,
    ) -> None:
        self.body = body
        self.status_code = status_code
        self.url = url
        self.encoding = None
        self.headers: dict[str, str] = {}
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        self.iteration_count = 0
        self.closed = False

    def iter_content(self, chunk_size: int):
        self.iteration_count += 1
        for index in range(0, len(self.body), chunk_size):
            yield self.body[index : index + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """记录请求参数，并按测试安排返回响应或抛出异常。"""

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
def isolate_web_pdf_fetch(settings_env):
    """固定 PDF 与共享网络配置，并隔离 Runtime 和生产审计。"""
    tool_runtime.reset_tool_runtime()
    settings_env(
        web_fetch_connect_timeout=3.0,
        web_fetch_read_timeout=10.0,
        web_fetch_max_redirects=5,
        web_fetch_user_agent="rag-agent-qa-system/1.0",
        web_fetch_proxy="",
        web_pdf_max_response_bytes=10_000_000,
        web_pdf_max_document_pages=1000,
        tool_audit_enabled=False,
    )
    yield
    tool_runtime.reset_tool_runtime()


def _pdf_spec():
    """返回默认注册表中的 PDF 抓取工具契约。"""
    spec = get_tool_registry().get("web_pdf_fetch")
    assert spec is not None
    return spec


def _args(**overrides):
    values = {
        "url": "https://example.com/document.pdf",
        "start_page": 1,
        "max_pages": 10,
        "max_chars": 20000,
    }
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
    """在 HTML/PDF 共用的 Manager 边界安装确定性 fake。"""
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


def _writer_pdf(*, pages: int = 1, encrypted: bool = False) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    if encrypted:
        writer.encrypt("test-password")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _pdf_with_blank_middle_page() -> bytes:
    reader = PdfReader(BytesIO(PDF_BYTES), strict=False)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.add_blank_page(width=612, height=792)
    writer.add_page(reader.pages[1])
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("url", ["http://example.com/a.pdf", "https://example.com/a"])
def test_web_pdf_args_accept_http_and_https(url):
    args = WebPdfFetchArgs(url=url)
    assert str(args.url).startswith(url.rstrip("/"))
    assert (args.start_page, args.max_pages, args.max_chars) == (1, 10, 20000)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file.pdf",
        "file:///tmp/file.pdf",
        "https://user:secret@example.com/file.pdf",
        "https://example.com:8080/file.pdf",
        "data:application/pdf;base64,JVBERi0=",
        "",
    ],
)
def test_web_pdf_args_reject_unsafe_urls(url):
    with pytest.raises(ValidationError):
        WebPdfFetchArgs(url=url)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_page", 0),
        ("start_page", True),
        ("max_pages", 0),
        ("max_pages", 21),
        ("max_pages", True),
        ("max_chars", 999),
        ("max_chars", 30001),
        ("max_chars", True),
    ],
)
def test_web_pdf_args_reject_invalid_ranges(field, value):
    with pytest.raises(ValidationError):
        WebPdfFetchArgs(url="https://example.com/a.pdf", **{field: value})


def test_web_pdf_args_accept_boundaries_and_forbid_runtime_controls():
    args = WebPdfFetchArgs(
        url="https://example.com/a.pdf",
        start_page=1,
        max_pages=20,
        max_chars=30000,
    )
    assert (args.start_page, args.max_pages, args.max_chars) == (1, 20, 30000)
    with pytest.raises(ValidationError):
        WebPdfFetchArgs(
            url="https://example.com/a.pdf",
            headers={"Authorization": "secret"},
        )


def test_web_pdf_output_validates_lengths_pages_and_fixed_fields():
    values = {
        "requested_url": "https://example.com/a.pdf",
        "final_url": "https://example.com/final.pdf",
        "status_code": 200,
        "content_type": "application/pdf",
        "total_pages": 2,
        "start_page": 1,
        "pages_read": 2,
        "page_numbers": [1, 2],
        "content": "[Page 1]\none\n\n[Page 2]\ntwo",
        "content_length": len("[Page 1]\none\n\n[Page 2]\ntwo"),
        "truncated": False,
    }
    output = WebPdfFetchOutput.model_validate(values)
    assert output.extractor == "pypdf"
    assert output.content_is_untrusted is True
    with pytest.raises(ValidationError):
        WebPdfFetchOutput.model_validate({**values, "content_length": 1})
    with pytest.raises(ValidationError):
        WebPdfFetchOutput.model_validate(
            {**values, "pages_read": 2, "page_numbers": [2, 2]}
        )
    with pytest.raises(ValidationError):
        WebPdfFetchOutput.model_validate({**values, "unexpected": b"%PDF"})


def test_web_pdf_configuration_accepts_safe_values():
    """PDF 抓取配置接受满足安全与预算约束的取值。"""
    AppSettings(
        _env_file=None,
        web_pdf_max_response_bytes=10_000_000,
        web_pdf_max_document_pages=1000,
        web_pdf_tool_timeout_seconds=45,
        web_fetch_connect_timeout=3,
        web_fetch_read_timeout=10,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"web_pdf_max_response_bytes": 0},
        {"web_pdf_max_response_bytes": True},
        {"web_pdf_max_document_pages": 0},
        {"web_pdf_tool_timeout_seconds": 0},
        {"web_pdf_tool_timeout_seconds": 13},
    ],
)
def test_web_pdf_configuration_rejects_invalid_values(overrides):
    """PDF 抓取配置拒绝越界值和布尔冒充数值。"""
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    "content_type",
    ["application/pdf", "application/x-pdf", "application/octet-stream"],
)
def test_supported_pdf_content_types_require_and_accept_pdf_magic(
    monkeypatch, content_type
):
    managers = _install_fake_manager(
        monkeypatch,
        [FakeResponse(content_type=f"{content_type}; charset=binary")],
    )

    output = web_pdf_fetch_handler(_args(), ToolInvocationContext())["output"]

    assert output["content_type"] == content_type
    request_kwargs = managers[0].session.calls[0][2]
    assert request_kwargs["headers"]["Accept"] == (
        "application/pdf, application/x-pdf, application/octet-stream"
    )
    assert request_kwargs["stream"] is True


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("application/octet-stream", b"not a pdf"),
        ("application/pdf", b"<html>upstream error</html>"),
        ("application/x-pdf", b"broken"),
    ],
)
def test_pdf_content_types_without_magic_are_rejected(
    monkeypatch, content_type, body
):
    _install_fake_manager(
        monkeypatch,
        [FakeResponse(body=body, content_type=content_type)],
    )

    with pytest.raises(ToolExecutionError) as raised:
        web_pdf_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.INVALID_RESULT
    assert raised.value.retryable is False
    assert body.decode(errors="ignore") not in str(raised.value)


def test_html_content_type_is_rejected_before_pdf_parser(monkeypatch):
    _install_fake_manager(
        monkeypatch,
        [FakeResponse(body=b"%PDF-fake", content_type="text/html")],
    )
    monkeypatch.setattr(
        web_pdf_fetch_module,
        "PdfReader",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("PdfReader must not receive HTML")
        ),
    )

    with pytest.raises(ToolExecutionError) as raised:
        web_pdf_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.INVALID_RESULT


def test_two_page_pdf_preserves_markers_metadata_and_output_contract(monkeypatch):
    _install_fake_manager(monkeypatch, [FakeResponse()])

    result = web_pdf_fetch_handler(_args(), ToolInvocationContext())
    output = WebPdfFetchOutput.model_validate(result["output"])

    assert output.total_pages == 2
    assert output.page_numbers == [1, 2]
    assert output.pages_read == 2
    assert output.content == (
        "[Page 1]\nGoverned PDF page one\n\n"
        "[Page 2]\nGoverned PDF page two"
    )
    assert output.title == "Governed Web PDF Fixture"
    assert output.author == "Codex Tests"
    assert output.subject == "Public PDF extraction"
    assert output.content_length == len(output.content)
    assert output.truncated is False
    dumped = output.model_dump()
    assert "body" not in dumped
    assert "metadata" not in dumped


def test_start_page_and_max_pages_select_actual_pdf_pages(monkeypatch):
    _install_fake_manager(monkeypatch, [FakeResponse()])

    output = web_pdf_fetch_handler(
        _args(start_page=2, max_pages=1),
        ToolInvocationContext(),
    )["output"]

    assert output["page_numbers"] == [2]
    assert output["pages_read"] == 1
    assert output["content"] == "[Page 2]\nGoverned PDF page two"


def test_blank_page_is_skipped_without_renumbering_other_pages(monkeypatch):
    _install_fake_manager(
        monkeypatch,
        [FakeResponse(body=_pdf_with_blank_middle_page())],
    )

    output = web_pdf_fetch_handler(_args(max_pages=3), ToolInvocationContext())[
        "output"
    ]

    assert output["total_pages"] == 3
    assert output["page_numbers"] == [1, 3]
    assert "[Page 2]" not in output["content"]
    assert "[Page 3]\nGoverned PDF page two" in output["content"]


def test_truncation_happens_after_page_markers_are_composed(monkeypatch):
    long_pdf = build_text_pdf(
        ("A" * 1200, "B" * 1200),
        title="Long Fixture",
    )
    _install_fake_manager(monkeypatch, [FakeResponse(body=long_pdf)])

    output = web_pdf_fetch_handler(
        _args(max_chars=1000),
        ToolInvocationContext(),
    )["output"]

    assert output["content"].startswith("[Page 1]\n")
    assert len(output["content"]) == 1000
    assert output["content_length"] == 1000
    assert output["page_numbers"] == [1]
    assert output["truncated"] is True


@pytest.mark.parametrize(
    ("body", "expected_fragment"),
    [
        (_writer_pdf(), "OCR"),
        (_writer_pdf(encrypted=True), "Encrypted"),
        (b"%PDF-this is damaged", "malformed"),
    ],
)
def test_no_text_encrypted_and_damaged_pdf_are_non_retryable(
    monkeypatch, body, expected_fragment
):
    _install_fake_manager(monkeypatch, [FakeResponse(body=body)])

    with pytest.raises(ToolExecutionError) as raised:
        web_pdf_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.INVALID_RESULT
    assert raised.value.retryable is False
    assert expected_fragment in str(raised.value)


def test_unexpected_parser_error_is_redacted_and_non_retryable(monkeypatch):
    _install_fake_manager(monkeypatch, [FakeResponse()])

    def broken_reader(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("xref object 42 contains secret bytes")

    monkeypatch.setattr(web_pdf_fetch_module, "PdfReader", broken_reader)

    with pytest.raises(ToolExecutionError) as raised:
        web_pdf_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.INVALID_RESULT
    assert raised.value.retryable is False
    assert "xref" not in str(raised.value)
    assert "secret" not in str(raised.value)


def test_document_page_limit_and_start_page_range(
    monkeypatch,
    settings_env,
):
    """文档页数上限与起始页范围均受到校验。"""
    settings_env(web_pdf_max_document_pages=1)
    _install_fake_manager(monkeypatch, [FakeResponse(), FakeResponse()])

    with pytest.raises(ToolExecutionError) as too_many:
        web_pdf_fetch_handler(_args(), ToolInvocationContext())
    assert too_many.value.code is ToolErrorCode.INVALID_RESULT

    settings_env(web_pdf_max_document_pages=1000)
    with pytest.raises(ToolExecutionError) as out_of_range:
        web_pdf_fetch_handler(_args(start_page=3), ToolInvocationContext())
    assert out_of_range.value.code is ToolErrorCode.INVALID_ARGUMENTS
    assert out_of_range.value.retryable is False


def test_pdf_download_byte_limit_uses_shared_streaming_boundary(
    monkeypatch,
    settings_env,
):
    """PDF 下载通过共享流式边界实施字节上限。"""
    settings_env(web_pdf_max_response_bytes=len(PDF_BYTES) - 1)
    response = FakeResponse(content_length=str(len(PDF_BYTES)))
    managers = _install_fake_manager(monkeypatch, [response])

    with pytest.raises(ToolExecutionError) as raised:
        web_pdf_fetch_handler(_args(), ToolInvocationContext())

    assert raised.value.code is ToolErrorCode.INVALID_RESULT
    assert response.iteration_count == 0
    assert response.closed is True
    assert managers[0].session.closed is True


def test_html_and_pdf_tools_import_the_same_shared_download_entry():
    assert (
        web_fetch_module.download_public_web_resource
        is web_pdf_fetch_module.download_public_web_resource
        is web_download_module.download_public_web_resource
    )


def test_registry_runtime_executes_real_pdf_spec_and_pypdf(monkeypatch):
    managers = _install_fake_manager(monkeypatch, [FakeResponse()])
    registry = ToolRegistry()
    registry.register(_pdf_spec())

    result = registry.dispatch(
        "web_pdf_fetch",
        _args(),
        context=ToolInvocationContext(permissions=("web:fetch",)),
    )

    assert result["ok"] is True
    assert result["runtime"]["attempt_count"] == 1
    assert result["governance"]["decision"] == "allow"
    assert WebPdfFetchOutput.model_validate(result["output"])
    assert len(managers) == 1


def test_permission_denial_prevents_download_and_pdf_reader(monkeypatch):
    class ForbiddenManager:
        def __init__(self, config):
            raise AssertionError(f"manager must not be constructed: {config}")

    def forbidden_reader(*args, **kwargs):
        raise AssertionError(f"PdfReader must not be constructed: {args}, {kwargs}")

    monkeypatch.setattr(web_download_module, "Manager", ForbiddenManager)
    monkeypatch.setattr(web_pdf_fetch_module, "PdfReader", forbidden_reader)
    registry = ToolRegistry()
    registry.register(_pdf_spec())

    result = registry.dispatch(
        "web_pdf_fetch",
        _args(),
        context=ToolInvocationContext(),
    )

    assert result["error_code"] == "permission_denied"
    assert result["runtime"]["attempt_count"] == 0


def test_runtime_retries_network_dependency_once_then_succeeds(monkeypatch):
    managers = _install_fake_manager(
        monkeypatch,
        [requests.exceptions.ConnectionError("temporary"), FakeResponse()],
    )
    registry = ToolRegistry()
    registry.register(_pdf_spec())
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_pdf_fetch",
            _args(),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
    finally:
        runtime.close()

    assert result["ok"] is True
    assert len(managers) == 2
    assert result["runtime"]["attempt_count"] == 2
    assert result["runtime"]["max_attempts"] == 2
    assert [item["error_code"] for item in result["runtime"]["attempts"]] == [
        "dependency_unavailable",
        "",
    ]


@pytest.mark.parametrize(
    "body",
    [_writer_pdf(), _writer_pdf(encrypted=True), b"%PDF-damaged"],
)
def test_runtime_does_not_retry_invalid_pdf_results(monkeypatch, body):
    managers = _install_fake_manager(
        monkeypatch,
        [FakeResponse(body=body), FakeResponse()],
    )
    registry = ToolRegistry()
    registry.register(_pdf_spec())
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_pdf_fetch",
            _args(),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
    finally:
        runtime.close()

    assert result["error_code"] == "invalid_result"
    assert result["runtime"]["attempt_count"] == 1
    assert len(managers) == 1


def test_runtime_output_model_rejects_invalid_pdf_output():
    invalid_spec = replace(
        _pdf_spec(),
        handler=lambda args, context: tool_success(
            "web_pdf_fetch",
            args,
            {
                "requested_url": args["url"],
                "final_url": args["url"],
                "status_code": 200,
                "content_type": "application/pdf",
                "total_pages": 1,
                "start_page": 1,
                "pages_read": 1,
                "page_numbers": [1],
                "content": "[Page 1]\ntext",
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
            "web_pdf_fetch",
            _args(),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
    finally:
        runtime.close()

    assert result["error_code"] == "output_validation_error"
    assert result["runtime"]["attempt_count"] == 1


def test_runtime_total_timeout_bounds_pdf_download(monkeypatch):
    completed = threading.Event()

    class SlowSession(FakeSession):
        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            time.sleep(0.08)
            completed.set()
            return self.outcome

    class SlowManager:
        def __init__(self, config):
            del config
            self.session = SlowSession(FakeResponse())

        def get_session(self):
            return self.session

    monkeypatch.setattr(web_download_module, "Manager", SlowManager)
    registry = ToolRegistry()
    registry.register(replace(_pdf_spec(), timeout=0.01))
    runtime = _runtime(registry)
    try:
        result = runtime.execute(
            "web_pdf_fetch",
            _args(),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
        assert completed.wait(timeout=1)
    finally:
        runtime.close()

    assert result["error_code"] == "timeout"
    assert result["runtime"]["attempt_count"] == 1
    assert result["runtime"]["timed_out"] is True


def test_pdf_audit_never_records_url_body_metadata_or_proxy(
    monkeypatch,
    settings_env,
):
    """PDF 审计不记录 URL、正文、元数据或代理地址。"""
    proxy = "http://audit-secret-proxy.example:8080"
    url = "https://example.com/private.pdf?secret=1"
    body_marker = "Governed PDF page one"
    metadata_marker = "Governed Web PDF Fixture"
    settings_env(web_fetch_proxy=proxy)
    _install_fake_manager(monkeypatch, [FakeResponse()])
    sink = CollectingAuditSink()
    registry = ToolRegistry()
    registry.register(_pdf_spec())
    runtime = _runtime(registry, audit_sink=sink)
    try:
        result = runtime.execute(
            "web_pdf_fetch",
            _args(url=url),
            context=ToolInvocationContext(permissions=("web:fetch",)),
        )
    finally:
        runtime.close()

    serialized = json.dumps([event.to_dict() for event in sink.events])
    assert result["ok"] is True
    assert url not in serialized
    assert proxy not in serialized
    assert body_marker not in serialized
    assert metadata_marker not in serialized
    assert "%PDF" not in serialized
    assert "content_length" in serialized
    assert "total_pages" in serialized
