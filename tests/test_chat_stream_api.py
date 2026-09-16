"""SSE 编码、HTTP 语义与断开处理的隔离测试。"""

import asyncio
import json
from dataclasses import replace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from app.agent.results import create_agent_result
from app.api.chat import create_chat_router
from app.api.middleware import trace_and_timing_middleware
from app.api.streaming import ChatStreamSSEEncoder
from app.contracts.chat import AgentResultStatus
from app.services.chat_models import ChatServiceOutcome, ChatServiceRequest
from app.services.chat_service import ChatService, get_default_chat_service_dependencies
from app.services.chat_stream_models import ChatStreamEvent


@pytest.fixture(autouse=True)
def run_blocking_calls_inline_in_sandbox(monkeypatch):
    """隔离沙箱默认 executor 关闭挂起，不改变 HTTP/SSE 业务断言。"""

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)


def _outcome(*, answer="完整回答", status=AgentResultStatus.OK):
    return ChatServiceOutcome(
        http_status=200,
        code=0,
        message="success",
        question="测试问题",
        result=create_agent_result(
            status=status,
            answer=answer,
            reason="正常回答",
        ),
        response_status="ok",
    )


def _parse_sse(body: str):
    parsed = []
    for block in body.strip().split("\n\n"):
        fields = {}
        for line in block.splitlines():
            name, value = line.split(":", 1)
            fields[name] = value.strip()
        parsed.append(
            {
                "id": fields["id"],
                "event": fields["event"],
                "data": json.loads(fields["data"]),
            }
        )
    return parsed


def test_stream_event_model_rejects_unknown_public_event_name():
    with pytest.raises(ValueError, match="unsupported chat stream event"):
        ChatStreamEvent("internal_state", {})


def test_sse_encoder_assigns_sequence_trace_timestamp_and_strict_tool_projection():
    encoder = ChatStreamSSEEncoder(
        trace_id="trace-sse",
        timestamp_provider=lambda: "2026-08-02T12:00:00.000",
    )
    start = encoder.encode(
        ChatStreamEvent(
            "start",
            {
                "session_id_present": True,
                "session_id": "must-not-leak",
            },
        )
    )
    tool = encoder.encode(
        ChatStreamEvent(
            "tool",
            {
                "tool_name": "web_fetch",
                "phase": "completed",
                "status": "success",
                "cost_ms": 12,
                "tool_args": {"authorization": "Bearer private"},
                "raw_tool": {"body": "private page"},
                "secret": "sk-private",
            },
        )
    )
    events = _parse_sse((start + tool).decode("utf-8"))

    assert [event["id"] for event in events] == ["1", "2"]
    assert [event["event"] for event in events] == ["start", "tool"]
    assert [event["data"]["sequence"] for event in events] == [1, 2]
    assert all(
        event["data"]["trace_id"] == "trace-sse" for event in events
    )
    assert all(event["data"]["schema_version"] == 1 for event in events)
    assert all(event["data"]["timestamp"] for event in events)
    assert events[1]["data"]["data"] == {
        "tool_name": "web_fetch",
        "phase": "completed",
        "status": "success",
        "cost_ms": 12,
    }
    serialized = (start + tool).decode("utf-8")
    assert "must-not-leak" not in serialized
    assert "authorization" not in serialized
    assert "private page" not in serialized
    assert "sk-private" not in serialized


def test_final_encoder_uses_complete_existing_chat_response():
    encoder = ChatStreamSSEEncoder(
        trace_id="trace-final",
        timestamp_provider=lambda: "2026-08-02T12:00:00.000",
    )

    event = _parse_sse(
        encoder.encode(
            ChatStreamEvent(
                "final",
                {"outcome": _outcome(), "total_cost_ms": 321},
            )
        ).decode("utf-8")
    )[0]
    data = event["data"]["data"]

    assert data["http_status"] == 200
    assert data["total_cost_ms"] == 321
    assert data["response"]["code"] == 0
    assert data["response"]["message"] == "success"
    assert data["response"]["trace_id"] == "trace-final"
    assert data["response"]["data"]["answer"] == "完整回答"
    assert "llm_call_count" in data["response"]["data"]


class FakeStreamingService:
    def __init__(self):
        self.stream_requests = []
        self.closed = False

    async def stream(self, request):
        self.stream_requests.append(request)
        try:
            yield ChatStreamEvent(
                "start",
                {"session_id_present": bool(request.session_id)},
            )
            yield ChatStreamEvent("answer_delta", {"text": "完整回答"})
            yield ChatStreamEvent(
                "final",
                {"outcome": _outcome(), "total_cost_ms": 50},
            )
            yield ChatStreamEvent("done", {"reason": "completed"})
        finally:
            self.closed = True


def _app(service):
    application = FastAPI()
    application.middleware("http")(trace_and_timing_middleware)
    application.include_router(create_chat_router(service))
    return application


def _post(application: FastAPI, path: str, **kwargs) -> Response:
    """通过异步 ASGI transport 调用接口，避免同步 portal 线程。"""

    async def send() -> Response:
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://testserver",
        ) as client:
            return await client.post(path, **kwargs)

    return asyncio.run(send())


def test_stream_route_returns_sse_headers_and_ordered_final():
    service = FakeStreamingService()

    response = _post(
        _app(service),
        "/chat/stream",
        headers={"X-Trace-Id": "caller-trace"},
        json={
            "question": "测试问题",
            "top_k": 4,
            "decompose_top_k": 8,
            "session_id": "session-1",
        },
    )

    events = _parse_sse(response.text)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Trace-Id"] == "caller-trace"
    assert "X-Total-Cost-Ms" not in response.headers
    assert response.headers["Cache-Control"] == "no-cache"
    assert [event["event"] for event in events] == [
        "start",
        "answer_delta",
        "final",
        "done",
    ]
    assert [event["data"]["sequence"] for event in events] == [1, 2, 3, 4]
    assert events[-2]["data"]["data"]["response"]["data"]["answer"] == (
        "完整回答"
    )
    assert len(service.stream_requests) == 1
    request = service.stream_requests[0]
    assert request.question == "测试问题"
    assert (request.top_k, request.decompose_top_k, request.session_id) == (4, 8, "session-1")
    assert request.trace_id == "caller-trace"
    assert request.request_enter_time
    assert request.request_enter_perf > 0
    assert service.closed is True


def test_empty_question_is_sse_and_schema_error_is_json_422():
    def forbidden(*args, **kwargs):
        pytest.fail("empty question must not run business dependencies")

    service = ChatService(
        replace(
            get_default_chat_service_dependencies(),
            settings_provider=forbidden,
            stream_agent=forbidden,
            log_and_write_metrics=lambda record: None,
        )
    )
    application = _app(service)
    empty = _post(application, "/chat/stream", json={"question": "   "})
    invalid = _post(application, "/chat/stream", json={"top_k": 5})

    assert empty.status_code == 200
    assert empty.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(empty.text)
    assert [event["event"] for event in events] == ["start", "final", "done"]
    assert [event["data"]["sequence"] for event in events] == [1, 2, 3]
    assert all(
        event["data"]["trace_id"] == empty.headers["X-Trace-Id"] for event in events
    )
    final = events[1]["data"]["data"]
    assert final["http_status"] == 400
    assert final["response"]["code"] == "EMPTY_QUESTION"
    assert final["response"]["data"]["status"] == "clarify"
    assert events[-1]["data"]["data"] == {"reason": "completed"}
    assert invalid.status_code == 422
    assert invalid.headers["content-type"].startswith("application/json")


def test_chat_surface_only_exposes_stream():
    from app.api.application import create_app

    application = create_app(chat_service=FakeStreamingService())
    paths = application.openapi()["paths"]
    assert "/chat" not in paths
    assert "post" in paths["/chat/stream"]
    assert "400" not in paths["/chat/stream"]["post"]["responses"]
    assert _post(application, "/chat", json={"question": "测试问题"}).status_code == 404


def test_stream_protocol_failure_emits_generic_error_and_contiguous_done():
    class InvalidEventService(FakeStreamingService):
        async def stream(self, request):
            self.stream_requests.append(request)
            try:
                yield ChatStreamEvent(
                    "start",
                    {"session_id_present": False},
                )
                yield ChatStreamEvent(
                    "tool",
                    {
                        "tool_name": "web_fetch",
                        "phase": "invalid",
                        "status": "",
                        "secret": "must-not-leak",
                    },
                )
            finally:
                self.closed = True

    service = InvalidEventService()

    response = _post(
        _app(service),
        "/chat/stream",
        json={"question": "测试协议错误"},
    )

    events = _parse_sse(response.text)
    assert [event["event"] for event in events] == [
        "start",
        "error",
        "done",
    ]
    assert [event["data"]["sequence"] for event in events] == [1, 2, 3]
    assert events[1]["data"]["data"] == {
        "code": "STREAM_ERROR",
        "message": "流式处理失败",
    }
    assert events[2]["data"]["data"] == {"reason": "error"}
    assert "must-not-leak" not in response.text
    assert service.closed is True


def test_api_disconnect_stops_and_closes_service_generator_without_final_or_done():
    service = FakeStreamingService()
    endpoint = next(
        route.endpoint
        for route in create_chat_router(service).routes
        if route.path == "/chat/stream"
    )

    class State:
        trace_id = "trace-disconnect"
        request_enter_time = "2026-08-02T12:00:00.000"
        request_enter_perf = 10.0

    class Request:
        state = State()

        def __init__(self):
            self.checks = iter((False, False, True))

        async def is_disconnected(self):
            return next(self.checks, True)

    async def call_and_consume():
        response = await endpoint(
            type(
                "Payload",
                (),
                {
                    "question": "测试问题",
                    "top_k": 4,
                    "decompose_top_k": 8,
                    "session_id": "session-1",
                },
            )(),
            Request(),
        )
        chunks = [chunk async for chunk in response.body_iterator]
        return b"".join(chunks).decode("utf-8")

    body = asyncio.run(call_and_consume())

    assert service.stream_requests == [
        ChatServiceRequest(
            question="测试问题",
            top_k=4,
            decompose_top_k=8,
            session_id="session-1",
            trace_id="trace-disconnect",
            request_enter_time="2026-08-02T12:00:00.000",
            request_enter_perf=10.0,
        )
    ]
    assert "event: start" in body
    assert "event: final" not in body
    assert "event: done" not in body
    assert service.closed is True
