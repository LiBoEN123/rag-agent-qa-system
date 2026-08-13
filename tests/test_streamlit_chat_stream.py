"""Streamlit 聊天 SSE 客户端的离线合同测试。"""

import json

import pytest

from demo import streamlit_app


def _event_lines(
    event: str,
    sequence: int,
    data: dict,
    *,
    trace_id: str = "trace-ui",
) -> list[str]:
    envelope = {
        "schema_version": 1,
        "trace_id": trace_id,
        "sequence": sequence,
        "timestamp": "2026-08-02T12:00:00.000",
        "data": data,
    }
    return [
        f"id: {sequence}",
        f"event: {event}",
        f"data: {json.dumps(envelope, ensure_ascii=False)}",
        "",
    ]


def _successful_stream_lines() -> list[str]:
    response = {
        "code": 0,
        "message": "success",
        "trace_id": "trace-ui",
        "data": {
            "question": "问题",
            "answer": "完整回答",
            "status": "ok",
        },
    }
    return [
        *_event_lines("start", 1, {"session_id_present": True}),
        *_event_lines(
            "progress",
            2,
            {"stage": "agent", "status": "started"},
        ),
        *_event_lines("answer_delta", 3, {"text": "完整"}),
        *_event_lines("answer_delta", 4, {"text": "回答"}),
        *_event_lines(
            "final",
            5,
            {
                "http_status": 200,
                "response": response,
                "total_cost_ms": 123,
            },
        ),
        *_event_lines("done", 6, {"reason": "completed"}),
    ]


def test_iter_sse_events_supports_comments_bytes_and_multiline_data():
    lines = [
        b": ping",
        b"",
        b"id: 1",
        b"event: start",
        b'data: {"schema_version": 1,',
        b'data: "trace_id": "trace-ui"}',
        b"",
    ]

    events = list(streamlit_app.iter_sse_events(lines))

    assert events == [
        {
            "id": "1",
            "event": "start",
            "data": '{"schema_version": 1,\n"trace_id": "trace-ui"}',
        }
    ]


def test_consume_chat_stream_returns_authoritative_final_and_notifies_ui():
    seen: list[tuple[str, dict]] = []

    result = streamlit_app.consume_chat_stream_events(
        _successful_stream_lines(),
        on_event=lambda event, envelope: seen.append((event, envelope)),
    )

    response_json, status_code, error = result
    assert error is None
    assert status_code == 200
    assert response_json["data"]["answer"] == "完整回答"
    assert [event for event, _ in seen] == [
        "start",
        "progress",
        "answer_delta",
        "answer_delta",
        "final",
        "done",
    ]


def test_consume_chat_stream_rejects_sequence_gap_after_partial_delta():
    lines = [
        *_event_lines("start", 1, {"session_id_present": False}),
        *_event_lines("answer_delta", 2, {"text": "部分内容"}),
        *_event_lines("done", 4, {"reason": "completed"}),
    ]

    with pytest.raises(
        streamlit_app.ChatStreamProtocolError,
        match="sequence 不连续",
    ):
        streamlit_app.consume_chat_stream_events(lines)


def test_consume_protocol_error_stream_never_requires_fake_final():
    lines = [
        *_event_lines("start", 1, {"session_id_present": False}),
        *_event_lines(
            "error",
            2,
            {"code": "STREAM_ERROR", "message": "流式处理失败"},
        ),
        *_event_lines("done", 3, {"reason": "error"}),
    ]

    assert streamlit_app.consume_chat_stream_events(lines) == (
        None,
        200,
        "流式处理失败",
    )


def test_consume_rag_stream_allows_no_delta_and_uses_final_business_status():
    response = {
        "code": "NO_RELEVANT_INFO",
        "message": "我没有在文档中找到相关信息",
        "trace_id": "trace-ui",
        "data": {
            "question": "问题",
            "answer": "",
            "references": [],
            "status": "no_evidence",
        },
    }
    lines = [
        *_event_lines("start", 1, {"session_id_present": True}),
        *_event_lines(
            "progress",
            2,
            {"stage": "rag", "status": "started"},
        ),
        *_event_lines(
            "final",
            3,
            {
                "http_status": 200,
                "response": response,
                "total_cost_ms": 321,
            },
        ),
        *_event_lines("done", 4, {"reason": "completed"}),
    ]

    assert streamlit_app.consume_chat_stream_events(lines) == (
        response,
        200,
        None,
    )


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        content_type: str,
        lines: list[str] | None = None,
        json_data: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.lines = list(lines or [])
        self.json_data = json_data
        self.encoding = ""
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.closed = True

    def iter_lines(self, *, chunk_size, decode_unicode):
        assert chunk_size == 1
        assert decode_unicode is True
        return iter(self.lines)

    def json(self):
        if self.json_data is None:
            raise ValueError("not json")
        return self.json_data


def test_request_chat_stream_posts_once_and_does_not_fallback_to_sync(
    monkeypatch,
):
    response = FakeResponse(
        status_code=200,
        content_type="text/event-stream; charset=utf-8",
        lines=_successful_stream_lines(),
    )
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(streamlit_app.requests, "post", fake_post)
    seen = []

    result = streamlit_app.request_chat_stream(
        "http://backend.example/",
        {"question": "问题"},
        on_event=lambda event, envelope: seen.append(event),
    )

    assert result[0]["data"]["answer"] == "完整回答"
    assert result[1:] == (200, None)
    assert len(calls) == 1
    assert calls[0][0] == "http://backend.example/chat/stream"
    assert calls[0][1]["stream"] is True
    assert calls[0][1]["headers"] == {"Accept": "text/event-stream"}
    assert seen[-2:] == ["final", "done"]
    assert response.closed is True


def test_request_chat_stream_preserves_preflight_json_error(monkeypatch):
    response = FakeResponse(
        status_code=400,
        content_type="application/json",
        json_data={
            "code": "EMPTY_QUESTION",
            "message": "问题不能为空",
            "data": {},
        },
    )
    monkeypatch.setattr(
        streamlit_app.requests,
        "post",
        lambda url, **kwargs: response,
    )

    assert streamlit_app.request_chat_stream(
        "http://backend.example",
        {"question": ""},
    ) == (
        response.json_data,
        400,
        None,
    )
    assert response.closed is True


def test_build_chat_payload_keeps_session_and_shared_counts(monkeypatch):
    monkeypatch.setattr(
        streamlit_app.st,
        "session_state",
        {"session_id": "streamlit-session"},
    )

    assert streamlit_app.build_chat_payload("问题", 4, 8) == {
        "question": "问题",
        "top_k": 4,
        "decompose_top_k": 8,
        "session_id": "streamlit-session",
    }
