"""ChatService 流式编排、Session 与 telemetry 的隔离测试。"""

import asyncio
from types import SimpleNamespace

import pytest

from app.agent.results import create_agent_result
from app.api.responses import build_chat_response_from_outcome
from app.contracts.chat import AgentResultStatus
from app.agent.streaming import AgentRuntimeStreamEvent
from app.observability.metrics import (
    attach_query_rewrite_trace,
    build_query_rewrite_trace,
    collect_trace_metrics,
)
from app.services.chat_models import ChatServiceRequest
from app.services.chat_service import ChatService, ChatServiceDependencies
from app.services.chat_stream_models import ChatStreamEvent


@pytest.fixture(autouse=True)
def run_blocking_calls_inline_in_sandbox(monkeypatch):
    """隔离沙箱默认 executor 关闭挂起，不改变流式业务断言。"""

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)


class IncrementingClock:
    def __init__(self):
        self.value = 20.0

    def __call__(self):
        self.value += 0.001
        return self.value


def _result(
    *,
    status: AgentResultStatus = AgentResultStatus.OK,
    answer: str = "完整回答",
    references=None,
    **overrides,
):
    values = create_agent_result(
        status=status,
        answer=answer,
        references=list(references or []),
        reason="测试原因",
        action="agent",
    )
    values.update(overrides)
    return values


def _request(**overrides):
    values = {
        "question": "测试问题",
        "top_k": 4,
        "decompose_top_k": 8,
        "session_id": "session-1",
        "trace_id": "trace-1",
        "request_enter_time": "2026-08-02T10:00:00.000",
        "request_enter_perf": 19.5,
    }
    values.update(overrides)
    return ChatServiceRequest(**values)


def _build_service(runtime_events, *, sync_result=None):
    calls = {
        "messages": [],
        "metrics": [],
        "bad_cases": [],
        "runtime_closed": False,
        "stream_kwargs": [],
    }
    clock = IncrementingClock()
    resolved_sync_result = sync_result or next(
        (
            event.data["result"]
            for event in runtime_events
            if event.event == "result"
        ),
        _result(),
    )

    async def fake_stream(**kwargs):
        calls["stream_kwargs"].append(kwargs)
        try:
            for event in runtime_events:
                yield event
        finally:
            calls["runtime_closed"] = True

    def fake_rewrite(question, history, trace_id=""):
        del history, trace_id
        return {
            "original_question": question,
            "rewritten_question": question,
            "history_used": False,
            "rewritten": False,
            "need_check": False,
            "need_rewrite": False,
            "reason": "no rewrite",
            "rewrite_llm_used": False,
            "error": "",
        }

    dependencies = ChatServiceDependencies(
        settings_provider=lambda: SimpleNamespace(top_k=5, decompose_top_k=10),
        get_history=lambda session_id: [],
        rewrite_query_with_history=fake_rewrite,
        run_agent=lambda **kwargs: dict(resolved_sync_result),
        build_query_rewrite_trace=build_query_rewrite_trace,
        attach_query_rewrite_trace=attach_query_rewrite_trace,
        collect_trace_metrics=collect_trace_metrics,
        log_and_write_metrics=lambda record: calls["metrics"].append(dict(record)),
        write_bad_cases=lambda record: calls["bad_cases"].append(dict(record)),
        append_message=lambda session_id, role, content: calls["messages"].append(
            (session_id, role, content)
        ),
        now_iso=lambda: "2026-08-02T10:00:00.000",
        perf_counter=clock,
        stream_agent=fake_stream,
    )
    return ChatService(dependencies), calls


def _collect(service, request=None):
    async def collect():
        return [event async for event in service.stream(request or _request())]

    return asyncio.run(collect())


def test_direct_answer_streams_ordered_deltas_final_and_done_once():
    result = _result(answer="完整回答")
    service, calls = _build_service(
        [
            AgentRuntimeStreamEvent(
                "progress",
                {"stage": "routing", "status": "completed"},
            ),
            AgentRuntimeStreamEvent("answer_delta", {"text": "完整"}),
            AgentRuntimeStreamEvent("answer_delta", {"text": "回答"}),
            AgentRuntimeStreamEvent("result", {"result": result}),
        ]
    )

    events = _collect(service)
    event_names = [event.event for event in events]
    final = next(event for event in events if event.event == "final")

    assert event_names[0] == "start"
    assert event_names[-2:] == ["final", "done"]
    assert "".join(
        event.data["text"]
        for event in events
        if event.event == "answer_delta"
    ) == "完整回答"
    assert final.data["outcome"].result["answer"] == "完整回答"
    assert final.data["outcome"].code == 0
    assert calls["messages"] == [
        ("session-1", "user", "测试问题"),
        ("session-1", "assistant", "完整回答"),
    ]
    assert calls["runtime_closed"] is True
    assert len(calls["metrics"]) == 1
    metrics = calls["metrics"][0]
    assert metrics["streaming"] is True
    assert metrics["answer_delta_count"] == 2
    assert metrics["streamed_answer_chars"] == 4
    assert metrics["streamed_answer_matches_final"] is True
    assert metrics["client_disconnected"] is False
    assert calls["bad_cases"] == []


def test_rag_stream_has_safe_progress_tool_and_authoritative_final_without_delta():
    references = [
        {
            "reference_id": "Reference 1",
            "file_name": "policy.md",
            "source_path": "policy.md",
            "chunk_id": "chunk-1",
            "position": 1,
            "snippet": "安全摘要",
            "score": 0.9,
            "rerank_score": 0.95,
        }
    ]
    result = _result(answer="RAG 完整回答", references=references)
    service, calls = _build_service(
        [
            AgentRuntimeStreamEvent(
                "progress",
                {"stage": "rag", "status": "started"},
            ),
            AgentRuntimeStreamEvent(
                "tool",
                {
                    "tool_name": "search_knowledge_base",
                    "phase": "completed",
                    "status": "success",
                    "cost_ms": 15,
                },
            ),
            AgentRuntimeStreamEvent(
                "progress",
                {"stage": "rag", "status": "completed"},
            ),
            AgentRuntimeStreamEvent("result", {"result": result}),
        ]
    )

    events = _collect(service)
    final = next(event for event in events if event.event == "final")

    assert all(event.event != "answer_delta" for event in events)
    assert final.data["outcome"].result["answer"] == "RAG 完整回答"
    assert final.data["outcome"].result["references"] == references
    assert calls["metrics"][0]["answer_delta_count"] == 0
    assert calls["metrics"][0]["streamed_answer_matches_final"] is None


def test_closing_service_stream_marks_disconnect_and_does_not_save_partial_answer():
    result = _result(answer="不会完成")
    service, calls = _build_service(
        [
            AgentRuntimeStreamEvent("answer_delta", {"text": "部分"}),
            AgentRuntimeStreamEvent("result", {"result": result}),
        ]
    )

    async def consume_until_delta_then_close():
        stream = service.stream(_request())
        seen = []
        async for event in stream:
            seen.append(event)
            if event.event == "answer_delta":
                await stream.aclose()
                break
        return seen

    events = asyncio.run(consume_until_delta_then_close())

    assert events[-1].event == "answer_delta"
    assert all(event.event not in {"final", "done"} for event in events)
    assert calls["runtime_closed"] is True
    assert calls["messages"] == []
    assert calls["bad_cases"] == []
    assert len(calls["metrics"]) == 1
    assert calls["metrics"][0]["client_disconnected"] is True
    assert calls["metrics"][0]["status"] == "client_disconnected"


def test_sync_and_stream_final_use_identical_response_mapping():
    result = _result(
        answer="一致回答",
        answer_confidence=0.8,
        retrieval_confidence=0.7,
        answer_llm_used=True,
        llm_call_count=1,
    )
    service, _ = _build_service(
        [AgentRuntimeStreamEvent("result", {"result": result})],
        sync_result=result,
    )

    sync_outcome = service.handle(_request(session_id=""))
    stream_events = _collect(service, _request(session_id=""))
    stream_outcome = next(
        event.data["outcome"]
        for event in stream_events
        if event.event == "final"
    )
    sync_response = build_chat_response_from_outcome(
        sync_outcome,
        trace_id="trace-1",
    ).model_dump(mode="json")
    stream_response = build_chat_response_from_outcome(
        stream_outcome,
        trace_id="trace-1",
    ).model_dump(mode="json")

    assert stream_response == sync_response


@pytest.mark.parametrize(
    ("status", "expected_code", "expected_public_status"),
    [
        (
            AgentResultStatus.NEED_CLARIFICATION,
            "NEED_CLARIFICATION",
            "clarify",
        ),
        (
            AgentResultStatus.NO_RELEVANT_INFO,
            "NO_RELEVANT_INFO",
            "no_evidence",
        ),
        (AgentResultStatus.TOOL_ERROR, "TOOL_ERROR", "error"),
        (AgentResultStatus.MODEL_ERROR, "MODEL_ERROR", "error"),
        (AgentResultStatus.INTERNAL_ERROR, "INTERNAL_ERROR", "error"),
    ],
)
def test_business_statuses_stay_in_final_and_never_become_stream_error(
    status,
    expected_code,
    expected_public_status,
):
    result = _result(
        status=status,
        answer="不可公开答案" if status is not AgentResultStatus.OK else "",
        need_clarification=status is AgentResultStatus.NEED_CLARIFICATION,
        clarification_question=(
            "请补充具体对象"
            if status is AgentResultStatus.NEED_CLARIFICATION
            else ""
        ),
    )
    service, _ = _build_service(
        [AgentRuntimeStreamEvent("result", {"result": result})]
    )

    events = _collect(service)
    final = next(event for event in events if event.event == "final")
    outcome = final.data["outcome"]

    assert all(event.event != "error" for event in events)
    assert outcome.code == expected_code
    assert outcome.response_status == expected_public_status
    assert events[-1] == ChatStreamEvent("done", {"reason": "completed"})
