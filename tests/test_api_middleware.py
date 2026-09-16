"""API 层 HTTP 追踪中间件的兼容行为。"""

import asyncio

import pytest
from fastapi import Request, Response

from app.api import middleware


def _request(path: str, *, trace_id: str = "") -> Request:
    headers = []
    if trace_id:
        headers.append((b"x-trace-id", trace_id.encode("ascii")))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
    )


def test_chat_middleware_injects_trace_state_and_timing_headers(monkeypatch):
    request = _request("/chat/stream")
    response = Response()
    perf_values = iter((10.0, 10.125))

    monkeypatch.setattr(middleware, "create_trace_id", lambda: "req-generated")
    monkeypatch.setattr(middleware, "now_iso", lambda: "2026-07-29T10:00:00.000")
    monkeypatch.setattr(
        middleware.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    async def call_next(received_request):
        assert received_request is request
        return response

    actual = asyncio.run(
        middleware.trace_and_timing_middleware(request, call_next)
    )

    assert actual is response
    assert request.state.trace_id == "req-generated"
    assert request.state.request_enter_perf == 10.0
    assert request.state.request_enter_time == "2026-07-29T10:00:00.000"
    assert response.headers["X-Trace-Id"] == "req-generated"
    assert "X-Total-Cost-Ms" not in response.headers


def test_chat_middleware_preserves_incoming_trace_id(monkeypatch):
    request = _request("/chat/stream", trace_id="caller-trace")
    perf_values = iter((20.0, 20.001))

    monkeypatch.setattr(
        middleware.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    async def call_next(_request):
        return Response()

    response = asyncio.run(
        middleware.trace_and_timing_middleware(request, call_next)
    )

    assert request.state.trace_id == "caller-trace"
    assert response.headers["X-Trace-Id"] == "caller-trace"
    assert "X-Total-Cost-Ms" not in response.headers


def test_non_chat_path_is_passed_through_unchanged():
    request = _request("/health")
    expected = Response(status_code=204)

    async def call_next(received_request):
        assert received_request is request
        return expected

    actual = asyncio.run(
        middleware.trace_and_timing_middleware(request, call_next)
    )

    assert actual is expected
    assert "X-Trace-Id" not in actual.headers
    assert "X-Total-Cost-Ms" not in actual.headers
    assert not hasattr(request.state, "trace_id")


def test_chat_middleware_propagates_downstream_exception(monkeypatch):
    request = _request("/chat/stream")
    perf_values = iter((30.0, 30.010))
    expected = RuntimeError("downstream failed")

    monkeypatch.setattr(
        middleware.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    async def call_next(_request):
        raise expected

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            middleware.trace_and_timing_middleware(request, call_next)
        )

    assert exc_info.value is expected
