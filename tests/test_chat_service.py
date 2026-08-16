from dataclasses import fields
from types import SimpleNamespace

import pytest

from app.agent.results import build_internal_error_result
from app.contracts.chat import get_chat_status_policy
from app.observability.metrics import (
    attach_query_rewrite_trace,
    build_query_rewrite_trace,
    collect_trace_metrics,
)
from app.services.chat_service import (
    ChatService,
    ChatServiceDependencies,
)
from app.services.chat_models import ChatServiceRequest


class IncrementingClock:
    """为耗时断言提供无外部依赖的单调时钟。"""

    def __init__(self):
        self.value = 10.0

    def __call__(self):
        self.value += 0.001
        return self.value


def build_runtime_result(status="OK", answer="测试回答", **overrides):
    result = {
        "status": status,
        "answer": answer,
        "references": [{"reference_id": "Reference 1"}],
        "retrieval_confidence": 0.7,
        "answer_confidence": 0.8,
        "need_clarification": False,
        "clarification_question": "",
        "reason": "测试原因",
        "action": "direct_answer",
        "tool_result_summary": "测试摘要",
        "tool_trace": [
            {
                "step": "decide",
                "decision_type": "direct_answer",
                "action": "direct_answer",
                "route_cost_ms": 7,
            },
            {
                "step": "llm_usage",
                "router_llm_used": False,
                "decompose_llm_used": False,
                "answer_llm_used": False,
                "llm_call_count": 0,
            }
        ],
        "router_llm_used": False,
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
        "timings": {"llm_cost_ms": 3},
    }
    result.update(overrides)
    return result


def build_service(
    *,
    history=None,
    rewrite_result=None,
    runtime_result=None,
    append_error=False,
    metrics_error=False,
):
    """构造依赖可控的 ChatService，并返回调用记录容器。"""
    calls = {
        "history": [],
        "rewrite": [],
        "agent": [],
        "bad_cases": [],
        "messages": [],
        "metrics": [],
    }
    settings = SimpleNamespace(top_k=5, decompose_top_k=10)
    history = list(history or [])
    rewrite_result = rewrite_result or {
        "original_question": "",
        "rewritten_question": "",
        "history_used": False,
        "rewritten": False,
        "need_check": False,
        "need_rewrite": False,
        "reason": "no rewrite",
        "rewrite_llm_used": False,
        "error": "",
    }
    runtime_result = (
        build_runtime_result()
        if runtime_result is None
        else runtime_result
    )

    def fake_get_history(session_id):
        calls["history"].append(session_id)
        return list(history)

    def fake_rewrite(question, history, trace_id=""):
        calls["rewrite"].append(
            {"question": question, "history": list(history), "trace_id": trace_id}
        )
        result = dict(rewrite_result)
        result["original_question"] = result.get("original_question") or question
        result["rewritten_question"] = (
            result.get("rewritten_question") or question
        )
        return result

    def run_runtime(label, kwargs):
        calls[label].append(kwargs)
        if isinstance(runtime_result, Exception):
            raise runtime_result
        if callable(runtime_result):
            return runtime_result(**kwargs)
        return dict(runtime_result)

    def fake_append(session_id, role, content):
        calls["messages"].append((session_id, role, content))
        if append_error:
            raise RuntimeError("session unavailable")

    def fake_metrics(record):
        calls["metrics"].append(dict(record))
        if metrics_error:
            raise RuntimeError("metrics unavailable")

    dependencies = ChatServiceDependencies(
        settings_provider=lambda: settings,
        get_history=fake_get_history,
        rewrite_query_with_history=fake_rewrite,
        run_agent=lambda **kwargs: run_runtime("agent", kwargs),
        build_query_rewrite_trace=build_query_rewrite_trace,
        attach_query_rewrite_trace=attach_query_rewrite_trace,
        collect_trace_metrics=collect_trace_metrics,
        log_and_write_metrics=fake_metrics,
        write_bad_cases=lambda record: calls["bad_cases"].append(dict(record)),
        append_message=fake_append,
        now_iso=lambda: "2026-07-14T10:00:00.000",
        perf_counter=IncrementingClock(),
    )
    return ChatService(dependencies), calls


def make_request(**overrides):
    values = {
        "question": "原始问题",
        "top_k": 4,
        "decompose_top_k": 8,
        "session_id": "session-1",
        "trace_id": "trace-1",
        "request_enter_time": "2026-07-14T09:59:59.000",
        "request_enter_perf": 9.5,
    }
    values.update(overrides)
    return ChatServiceRequest(**values)


def test_chat_service_request_contains_only_chat_business_fields():
    assert {field.name for field in fields(ChatServiceRequest)} == {
        "question",
        "top_k",
        "decompose_top_k",
        "session_id",
        "trace_id",
        "request_enter_time",
        "request_enter_perf",
    }


def test_empty_question_skips_business_dependencies_and_writes_metrics_once():
    service, calls = build_service()

    outcome = service.handle(make_request(question="   "))

    assert outcome.http_status == 400
    assert outcome.code == "EMPTY_QUESTION"
    assert outcome.response_status == "clarify"
    assert outcome.result["need_clarification"] is True
    assert calls["history"] == []
    assert calls["rewrite"] == []
    assert calls["agent"] == []
    assert calls["messages"] == []
    assert len(calls["metrics"]) == 1
    assert calls["metrics"][0]["status"] == "clarify"


def test_rewrite_routes_and_runs_once_with_original_question_preserved():
    rewritten_question = "改写后的独立问题"
    service, calls = build_service(
        history=[{"role": "user", "content": "上一轮问题"}],
        rewrite_result={
            "original_question": "原始问题",
            "rewritten_question": rewritten_question,
            "history_used": True,
            "rewritten": True,
            "need_check": True,
            "need_rewrite": True,
            "reason": "依赖历史",
            "rewrite_llm_used": True,
            "error": "",
        },
    )

    outcome = service.handle(make_request())

    assert calls["history"] == ["session-1"]
    assert len(calls["rewrite"]) == 1
    assert "question" not in calls["agent"][0]
    assert calls["agent"][0]["original_question"] == "原始问题"
    assert (
        calls["agent"][0]["effective_question"]
        == rewritten_question
    )
    assert "decision" not in calls["agent"][0]
    assert outcome.question == "原始问题"
    assert outcome.http_status == 200
    assert outcome.code == 0
    assert outcome.response_status == "ok"
    assert [
        item["step"] for item in outcome.result["tool_trace"]
    ].count("query_rewrite") == 1
    assert outcome.result["references"] == [{"reference_id": "Reference 1"}]
    assert calls["messages"] == [
        ("session-1", "user", "原始问题"),
        ("session-1", "assistant", "测试回答"),
    ]
    assert len(calls["metrics"]) == 1
    assert calls["metrics"][0]["llm_call_count"] == 1
    assert calls["metrics"][0]["route_cost_ms"] == 7


def test_question_without_rewrite_is_passed_through_without_extra_llm_count():
    service, calls = build_service()

    service.handle(make_request())

    assert "question" not in calls["agent"][0]
    assert calls["agent"][0]["original_question"] == "原始问题"
    assert calls["agent"][0]["effective_question"] == "原始问题"
    assert calls["metrics"][0]["llm_call_count"] == 0


def test_agent_tool_task_has_no_router_level_rag_preflight():
    service, calls = build_service(
        runtime_result=build_runtime_result(action="agent"),
    )

    outcome = service.handle(make_request())

    assert len(calls["agent"]) == 1
    assert "decision" not in calls["agent"][0]
    assert outcome.http_status == 200


@pytest.mark.parametrize("action", ["agent", "direct_answer", "clarify"])
def test_current_route_actions_do_not_emit_retired_rag_ready_metric(action):
    service, calls = build_service(
        runtime_result=build_runtime_result(action=action),
    )

    outcome = service.handle(make_request())

    assert len(calls["agent"]) == 1
    assert outcome.http_status == 200
    assert "rag_ready_check" + "_cost_ms" not in calls["metrics"][0]


def test_agent_runtime_receives_complete_existing_protocol_once():
    service, calls = build_service()

    service.handle(make_request())

    assert len(calls["agent"]) == 1
    kwargs = calls["agent"][0]
    assert "question" not in kwargs
    assert kwargs["original_question"] == "原始问题"
    assert kwargs["effective_question"] == "原始问题"
    assert kwargs["top_k"] == 4
    assert kwargs["decompose_top_k"] == 8
    assert kwargs["trace_id"] == "trace-1"
    assert kwargs["session_id"] == "session-1"
    assert "decision" not in kwargs
    assert kwargs["bad_case"]["question"] == "原始问题"
    assert callable(kwargs["write_bad_cases"])
    assert {
        "principal_id",
        "principal_type",
        "roles",
        "permissions",
        "tenant_id",
        "request_source",
        "idempotency_key",
    }.isdisjoint(kwargs)


def test_create_agent_runtime_start_is_logged_once_with_trace_id(monkeypatch):
    """每次业务请求只记录一次带 trace_id 的 Runtime 启动日志。"""
    service, _ = build_service()
    messages = []

    def record_log(message, *args, **kwargs):
        rendered = message % args if args else message
        messages.append(rendered)

    monkeypatch.setattr("app.services.chat_service.logger.info", record_log)
    service.handle(make_request())
    runtime_logs = [
        message
        for message in messages
        if "runtime_start agent_runtime=" in message
    ]
    assert runtime_logs == [
        "[trace_id=trace-1] [agent] runtime_start agent_runtime=create_agent"
    ]


def test_tool_call_chain_is_logged_once_in_execution_order(monkeypatch):
    """一次请求把所有实际工具步骤压缩为一条有序终端日志。"""
    runtime_result = build_runtime_result(
        tool_trace=[
            {
                "step": "tool_call",
                "tool_name": "web_search",
                "tool_status": "success",
                "cost_ms": 12,
            },
            {
                "step": "tool_call",
                "tool_name": "web_fetch",
                "tool_status": "success",
                "cost_ms": 34,
            },
        ]
    )
    service, _ = build_service(runtime_result=runtime_result)
    messages = []

    def record_log(message, *args, **kwargs):
        del kwargs
        rendered = message % args if args else message
        messages.append(rendered)

    monkeypatch.setattr("app.services.chat_service.logger.info", record_log)
    service.handle(make_request())

    chain_logs = [message for message in messages if "tool_call_chain=" in message]
    assert chain_logs == [
        "[trace_id=trace-1] [agent] "
        "tool_call_chain=web_search(success,12ms) -> web_fetch(success,34ms)"
    ]


@pytest.mark.parametrize(
    ("status", "http_status", "code", "public_status", "metric_status"),
    [
        ("OK", 200, 0, "ok", "success"),
        (
            "NEED_CLARIFICATION",
            200,
            "NEED_CLARIFICATION",
            "clarify",
            "clarify",
        ),
        (
            "NO_RELEVANT_INFO",
            200,
            "NO_RELEVANT_INFO",
            "no_evidence",
            "no_relevant_info",
        ),
        ("TOOL_ERROR", 200, "TOOL_ERROR", "error", "tool_error"),
        ("MODEL_ERROR", 200, "MODEL_ERROR", "error", "model_error"),
        (
            "INTERNAL_ERROR",
            500,
            "INTERNAL_ERROR",
            "error",
            "internal_error",
        ),
    ],
)
def test_known_runtime_statuses_use_single_status_policy(
    status,
    http_status,
    code,
    public_status,
    metric_status,
):
    result = build_runtime_result(
        status=status,
        need_clarification=status == "NEED_CLARIFICATION",
        clarification_question="请补充信息",
    )
    service, calls = build_service(runtime_result=result)

    outcome = service.handle(make_request())

    assert outcome.http_status == http_status
    assert outcome.code == code
    assert outcome.response_status == public_status
    assert get_chat_status_policy(status).public_status == public_status
    assert len(calls["messages"]) == 2
    assert len(calls["metrics"]) == 1
    assert calls["metrics"][0]["status"] == metric_status


def test_unknown_runtime_status_fails_closed_across_response_and_metrics():
    service, calls = build_service(
        runtime_result=build_runtime_result(status="UNKNOWN"),
    )

    outcome = service.handle(make_request())

    assert outcome.http_status == 500
    assert outcome.code == "INTERNAL_ERROR"
    assert outcome.response_status == "error"
    assert outcome.result["status"] == "INTERNAL_ERROR"
    assert calls["metrics"][0]["status"] == "internal_error"


def test_agent_internal_error_result_maps_to_http_500():
    result = build_internal_error_result(
        RuntimeError("sensitive runtime failure"),
        runtime_label="LangGraph Agent",
        trace_step="agent_langgraph_error",
    )
    service, calls = build_service(
        runtime_result=result,
    )

    outcome = service.handle(make_request())

    assert len(calls["agent"]) == 1
    assert "decision" not in calls["agent"][0]
    assert outcome.http_status == 500
    assert outcome.code == "INTERNAL_ERROR"
    assert outcome.response_status == "error"
    assert outcome.result["reason"] == "Agent 内部执行异常，请稍后重试"
    assert {
        "step": "agent_langgraph_error",
        "status": "error",
        "error": "internal_error",
    } in outcome.result["tool_trace"]


def test_runtime_exception_returns_compatible_error_and_writes_bad_case_once():
    service, calls = build_service(runtime_result=RuntimeError("runtime failed"))

    outcome = service.handle(make_request())

    assert outcome.http_status == 500
    assert outcome.code == "MODEL_ERROR"
    assert outcome.response_status == "error"
    assert outcome.result["reason"] == "工具调用或模型调用异常"
    assert len(calls["bad_cases"]) == 1
    assert calls["bad_cases"][0]["error_type"] == "internal_error"
    assert calls["messages"] == []
    assert len(calls["metrics"]) == 1
    assert calls["metrics"][0]["status"] == "internal_error"


def test_unknown_exception_returns_internal_error_with_error_status(monkeypatch):
    service, calls = build_service()

    def fail_rewrite(_context):
        raise RuntimeError("unexpected chat failure")

    monkeypatch.setattr(service, "_rewrite_question", fail_rewrite)

    outcome = service.handle(make_request())

    assert outcome.http_status == 500
    assert outcome.code == "INTERNAL_ERROR"
    assert outcome.response_status == "error"
    assert calls["agent"] == []
    assert calls["metrics"][0]["status"] == "internal_error"


def test_session_failure_does_not_change_success_outcome():
    service, calls = build_service(append_error=True)

    outcome = service.handle(make_request())

    assert outcome.code == 0
    assert outcome.http_status == 200
    assert len(calls["metrics"]) == 1


def test_metrics_failure_does_not_change_success_outcome():
    service, calls = build_service(metrics_error=True)

    outcome = service.handle(make_request())

    assert outcome.code == 0
    assert outcome.http_status == 200
    assert len(calls["metrics"]) == 1
