"""ChatService 流式编排、Session 与 telemetry 的隔离测试。"""

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agent.results import build_internal_error_result, create_agent_result
from app.api.responses import build_chat_response_from_outcome
from app.contracts.chat import AgentResultStatus
from app.agent.streaming import AgentRuntimeStreamEvent
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


def _build_service(
    runtime_events,
    *,
    stream_error=None,
    rewrite_llm_used=False,
    **overrides,
):
    calls = {
        "messages": [],
        "metrics": [],
        "bad_cases": [],
        "runtime_closed": False,
        "stream_kwargs": [],
    }
    clock = IncrementingClock()

    async def fake_stream(**kwargs):
        calls["stream_kwargs"].append(kwargs)
        try:
            for event in runtime_events:
                yield event
            if stream_error is not None:
                raise stream_error
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
            "rewrite_llm_used": rewrite_llm_used,
            "error": "",
        }

    dependencies = ChatServiceDependencies(
        settings_provider=lambda: SimpleNamespace(
            top_k=5, decompose_top_k=10,
            tool_default_principal_id="test",
            tool_default_permissions=("knowledge_base:read",),
        ),
        get_history=lambda session_id: [],
        rewrite_query_with_history=fake_rewrite,
        log_and_write_metrics=lambda record: calls["metrics"].append(dict(record)),
        write_bad_cases=lambda record: calls["bad_cases"].append(dict(record)),
        append_message=lambda session_id, role, content: calls["messages"].append(
            (session_id, role, content)
        ),
        now_iso=lambda: "2026-08-02T10:00:00.000",
        perf_counter=clock,
        stream_agent=fake_stream,
    )
    return ChatService(replace(dependencies, **overrides)), calls


def _collect(service, request=None):
    async def collect():
        return [event async for event in service.stream(request or _request())]

    return asyncio.run(collect())


def test_direct_answer_streams_ordered_deltas_final_and_done_once(monkeypatch):
    logs = []
    monkeypatch.setattr(
        "app.services.chat_service.logger.info",
        lambda msg, *args: logs.append(msg % args),
    )
    result = _result(
        answer="完整回答",
        tool_trace=[
            {
                "step": "decide",
                "decision_type": "direct_answer",
                "route_cost_ms": 7,
            }
        ],
    )
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
    assert (
        "".join(event.data["text"] for event in events if event.event == "answer_delta")
        == "完整回答"
    )
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
    assert metrics["route_cost_ms"] == 7
    assert metrics["llm_call_count"] == 0
    assert metrics["answer_delta_count"] == 2
    assert metrics["streamed_answer_chars"] == 4
    assert metrics["stream_event_count"] == len(events)
    assert metrics["first_event_ms"] is not None
    assert metrics["first_answer_delta_ms"] is not None
    assert metrics["streamed_answer_matches_final"] is True
    assert metrics["client_disconnected"] is False
    assert calls["bad_cases"] == []
    assert len(calls["stream_kwargs"]) == 1
    kwargs = calls["stream_kwargs"][0]
    assert {
        key: kwargs[key]
        for key in (
            "original_question",
            "effective_question",
            "top_k",
            "decompose_top_k",
            "trace_id",
            "session_id",
        )
    } == {
        "original_question": "测试问题",
        "effective_question": "测试问题",
        "top_k": 4,
        "decompose_top_k": 8,
        "trace_id": "trace-1",
        "session_id": "session-1",
    }
    assert "decision" not in kwargs
    assert callable(kwargs["write_bad_cases"])
    assert kwargs["bad_case"]["question"] == "测试问题"
    assert "rag_ready_check_cost_ms" not in metrics
    assert [line for line in logs if "runtime_start agent_runtime=" in line] == [
        "[trace_id=trace-1] [agent] runtime_start agent_runtime=create_agent"
    ]


def test_rag_stream_has_safe_progress_tool_and_authoritative_final_without_delta(
    monkeypatch,
):
    logs = []
    monkeypatch.setattr(
        "app.services.chat_service.logger.info",
        lambda msg, *args: logs.append(msg % args),
    )
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
    result = _result(
        answer="RAG 完整回答",
        references=references,
        tool_trace=[
            {
                "step": "tool_call",
                "tool_name": "search_knowledge_base",
                "tool_status": "success",
                "cost_ms": 15,
            },
            {
                "step": "tool_call",
                "tool_name": "get_chunk_text",
                "tool_status": "success",
                "cost_ms": 2,
            },
        ],
    )
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
    assert final.data["outcome"].result["tool_trace"][1:] == result["tool_trace"]
    assert [line for line in logs if "tool_call_chain=" in line] == [
        "[trace_id=trace-1] [agent] tool_call_chain=search_knowledge_base(success,15ms) -> get_chunk_text(success,2ms)"
    ]


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


def test_cancelled_runtime_stream_propagates_and_finalizes_as_disconnect():
    service, calls = _build_service(
        [],
        stream_error=asyncio.CancelledError(),
    )

    async def consume_until_cancelled():
        seen = []
        with pytest.raises(asyncio.CancelledError):
            async for event in service.stream(_request()):
                seen.append(event)
        return seen

    events = asyncio.run(consume_until_cancelled())

    assert all(event.event not in {"final", "done"} for event in events)
    assert calls["runtime_closed"] is True
    assert calls["messages"] == []
    assert calls["bad_cases"] == []
    assert calls["metrics"][0]["client_disconnected"] is True
    assert calls["metrics"][0]["status"] == "client_disconnected"


@pytest.mark.parametrize("rewrite_llm_used", [True, False])
def test_unconverted_runtime_stream_exception_uses_invocation_fallback(
    rewrite_llm_used,
):
    service, calls = _build_service(
        [],
        stream_error=RuntimeError("stream infrastructure failed"),
        rewrite_llm_used=rewrite_llm_used,
    )

    events = _collect(service)
    final = next(event for event in events if event.event == "final")

    assert final.data["outcome"].http_status == 500
    assert final.data["outcome"].code == "MODEL_ERROR"
    assert calls["runtime_closed"] is True
    assert calls["messages"] == []
    assert len(calls["bad_cases"]) == 1
    assert calls["bad_cases"][0]["error_type"] == "internal_error"
    assert calls["metrics"][0]["client_disconnected"] is False
    outcome = final.data["outcome"]
    response = build_chat_response_from_outcome(outcome, trace_id="trace-1")
    assert outcome.result["llm_call_count"] == int(rewrite_llm_used)
    assert calls["metrics"][0]["llm_call_count"] == int(rewrite_llm_used)
    assert response.data.llm_call_count == int(rewrite_llm_used)


@pytest.mark.parametrize("rewrite_llm_used", [True, False])
@pytest.mark.parametrize("has_rewrite_trace", [True, False])
def test_final_merges_rewrite_metrics_once_without_mutating_runtime(
    rewrite_llm_used, has_rewrite_trace
):
    result = _result(
        answer="一致回答",
        answer_confidence=0.8,
        retrieval_confidence=0.7,
        decompose_llm_used=True,
        answer_llm_used=True,
        llm_call_count=2,
        tool_trace=[
            {
                "step": "llm_usage",
                "router_llm_used": False,
                "decompose_llm_used": True,
                "answer_llm_used": True,
                "llm_call_count": 2,
            }
        ],
        timings={"llm_cost_ms": 7},
    )
    if has_rewrite_trace:
        result["tool_trace"].insert(
            0, {"step": "query_rewrite", "rewrite_llm_used": not rewrite_llm_used}
        )
    original = deepcopy(result)
    service, calls = _build_service(
        [AgentRuntimeStreamEvent("result", {"result": result})],
        rewrite_llm_used=rewrite_llm_used,
    )

    stream_events = _collect(service, _request(session_id=""))
    stream_outcome = next(
        event.data["outcome"] for event in stream_events if event.event == "final"
    )
    response = build_chat_response_from_outcome(stream_outcome, trace_id="trace-1")
    expected_count = 3 if rewrite_llm_used else 2
    assert response.data.llm_call_count == expected_count
    trace = stream_outcome.result["tool_trace"]
    assert sum(step["step"] == "query_rewrite" for step in trace) == 1
    assert [step for step in trace if step["step"] != "query_rewrite"] == [
        step for step in original["tool_trace"] if step["step"] != "query_rewrite"
    ]
    assert len(calls["metrics"]) == 1
    metrics = calls["metrics"][0]
    assert metrics["llm_call_count"] == expected_count
    assert metrics["decompose_llm_used"] is True
    assert metrics["answer_llm_used"] is True
    assert metrics["llm_cost_ms"] == 7
    assert calls["messages"] == []
    assert result == original


@pytest.mark.parametrize(
    ("status", "code", "public_status", "http_status", "metric_status"),
    [
        ("OK", 0, "ok", 200, "success"),
        ("NEED_CLARIFICATION", "NEED_CLARIFICATION", "clarify", 200, "clarify"),
        (
            "NO_RELEVANT_INFO",
            "NO_RELEVANT_INFO",
            "no_evidence",
            200,
            "no_relevant_info",
        ),
        ("TOOL_ERROR", "TOOL_ERROR", "error", 200, "tool_error"),
        ("MODEL_ERROR", "MODEL_ERROR", "error", 200, "model_error"),
        ("INTERNAL_ERROR", "INTERNAL_ERROR", "error", 500, "internal_error"),
        ("UNKNOWN", "INTERNAL_ERROR", "error", 500, "internal_error"),
    ],
)
def test_business_statuses_stay_in_final_and_never_become_stream_error(
    status,
    code,
    public_status,
    http_status,
    metric_status,
):
    result = _result(
        answer="测试回答",
        need_clarification=status == "NEED_CLARIFICATION",
        clarification_question="请补充具体对象",
    )
    result["status"] = status
    service, calls = _build_service(
        [AgentRuntimeStreamEvent("result", {"result": result})]
    )
    events = _collect(service)
    outcome = next(event.data["outcome"] for event in events if event.event == "final")
    assert all(event.event != "error" for event in events)
    assert (outcome.code, outcome.response_status, outcome.http_status) == (
        code,
        public_status,
        http_status,
    )
    assert len(calls["messages"]) == 2
    assert len(calls["metrics"]) == 1
    assert calls["metrics"][0]["status"] == metric_status
    assert events[-1] == ChatStreamEvent("done", {"reason": "completed"})


def test_empty_question_stream_skips_business_and_finalizes_once():
    def forbidden(*args, **kwargs):
        pytest.fail("empty question must skip business dependencies")

    service, calls = _build_service(
        [],
        settings_provider=forbidden,
        get_history=forbidden,
        rewrite_query_with_history=forbidden,
    )
    events = _collect(service, _request(question="   "))
    assert [event.event for event in events] == ["start", "final", "done"]
    outcome = events[1].data["outcome"]
    assert (outcome.http_status, outcome.code, outcome.response_status) == (
        400,
        "EMPTY_QUESTION",
        "clarify",
    )
    assert outcome.result["need_clarification"] is True
    assert events[-1].data == {"reason": "completed"}
    assert calls["stream_kwargs"] == calls["messages"] == calls["bad_cases"] == []
    assert len(calls["metrics"]) == 1
    assert calls["metrics"][0]["status"] == "clarify"


def test_service_failure_finishes_with_internal_error_and_no_runtime():
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected chat failure")

    service, calls = _build_service([], rewrite_query_with_history=fail)
    events = _collect(service)
    assert [event.event for event in events][-2:] == ["final", "done"]
    outcome = events[-2].data["outcome"]
    assert (outcome.http_status, outcome.code, outcome.response_status) == (
        500,
        "INTERNAL_ERROR",
        "error",
    )
    assert calls["stream_kwargs"] == calls["messages"] == []
    assert len(calls["bad_cases"]) == len(calls["metrics"]) == 1
    assert (
        calls["bad_cases"][0]["error_type"]
        == calls["metrics"][0]["status"]
        == "internal_error"
    )


@pytest.mark.parametrize(
    "boundary", ["get_history", "append_message", "log_and_write_metrics"]
)
def test_storage_failures_preserve_successful_stream(boundary):
    seen = []

    def fail(*args, **kwargs):
        seen.append(args)
        raise RuntimeError("storage unavailable")

    def rewrite(question, history, trace_id):
        assert history == []
        return {"rewritten_question": question, "rewrite_llm_used": False}

    service, calls = _build_service(
        [AgentRuntimeStreamEvent("result", {"result": _result()})],
        rewrite_query_with_history=rewrite,
        **{boundary: fail},
    )
    events = _collect(service)
    assert events[-2].data["outcome"].code == 0
    assert events[-1].data == {"reason": "completed"}
    assert len(calls["stream_kwargs"]) == 1
    assert len(seen) == 1
    if boundary != "log_and_write_metrics":
        assert len(calls["metrics"]) == 1


def test_internal_runtime_error_remains_redacted_in_final_response():
    result = build_internal_error_result(
        RuntimeError("/home/test-user/checkpoints.sqlite api_key=sk-test-secret"),
        runtime_label="Agent",
        trace_step="agent_error",
    )
    service, _ = _build_service([AgentRuntimeStreamEvent("result", {"result": result})])
    outcome = _collect(service)[-2].data["outcome"]
    response = build_chat_response_from_outcome(outcome, trace_id="trace-1")
    assert (outcome.http_status, response.code, response.data.status) == (
        500,
        "INTERNAL_ERROR",
        "error",
    )
    assert response.data.reason == "Agent 内部执行异常，请稍后重试"
    assert {
        "step": "agent_error",
        "status": "error",
        "error": "internal_error",
    } in response.data.tool_trace
    for private in (
        "/home/test-user",
        "checkpoints.sqlite",
        "sk-test-secret",
        "RuntimeError",
    ):
        assert private not in response.model_dump_json()
