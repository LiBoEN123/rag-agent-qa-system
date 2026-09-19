"""聊天 telemetry 的纯数据合并与安全持久化测试。"""

from copy import deepcopy

from app.contracts.chat import get_chat_status_policy
from app.observability.metrics import collect_trace_metrics
from app.services.chat_models import ChatRequestContext, ChatServiceRequest
from app.services.chat_telemetry import (
    apply_result_metrics,
    build_initial_bad_case,
    build_initial_chat_metrics,
    finalize_chat_telemetry,
    format_tool_call_chain,
    initialize_stream_metrics,
    record_stream_completion,
    record_stream_event,
    record_stream_termination,
    safe_write_bad_case,
)


def make_context() -> ChatRequestContext:
    request = ChatServiceRequest(
        question="测试问题",
        session_id="session-1",
        trace_id="trace-1",
        request_enter_time="2026-07-29T10:00:00.000",
        request_enter_perf=9.5,
    )
    return ChatRequestContext(
        request=request,
        process_start_perf=10.0,
        process_start_time="2026-07-29T10:00:00.500",
        request_enter_perf=9.5,
        original_question="测试问题",
        session_id="session-1",
        effective_question="测试问题",
        metrics={},
        bad_case={},
    )


def test_initial_metrics_and_bad_case_keep_current_contract():
    context = make_context()

    metrics = build_initial_chat_metrics(
        context,
        request_enter_time=context.request.request_enter_time,
        queue_wait_ms=500,
    )
    bad_case = build_initial_bad_case(
        context,
        created_at="2026-07-29T10:00:00.500",
    )

    assert metrics["trace_id"] == "trace-1"
    assert metrics["queue_wait_ms"] == 500
    assert metrics["status"] == "internal_error"
    assert metrics["router_llm_used"] is False
    assert metrics["effective_question_len"] == 4
    assert not {
        "retrieve_cost_ms", "decompose_cost_ms", "rerank_cost_ms",
        "retrieve_with_sub_queries_cost_ms", "context_build_cost_ms",
        "llm_cost_ms", "parse_cost_ms", "rag_answer_cost_ms", "decompose_llm_used",
    } & metrics.keys()
    assert "rag_ready_check" + "_cost_ms" not in metrics
    assert bad_case == {
        "created_at": "2026-07-29T10:00:00.500",
        "trace_id": "trace-1",
        "question": "测试问题",
        "answer": "",
        "references": [],
        "route": "",
        "error_type": "",
        "note": "",
    }


def test_format_tool_call_chain_handles_none_and_preserves_order():
    assert format_tool_call_chain([]) == "none"
    assert format_tool_call_chain(
        [
            {"step": "decide", "action": "agent"},
            {
                "step": "tool_call",
                "tool_name": " web_search ",
                "tool_status": "success",
                "cost_ms": 12,
            },
            {
                "step": "tool_call",
                "tool_name": "web_fetch",
                "tool_status": " error ",
                "cost_ms": -5,
            },
        ]
    ) == "web_search(success,12ms) -> web_fetch(error,0ms)"


def test_apply_result_metrics_reads_result_without_trace_fallback():
    context = make_context()
    context.metrics = build_initial_chat_metrics(
        context,
        request_enter_time=context.request.request_enter_time,
        queue_wait_ms=500,
    )
    metric_keys = set(context.metrics)
    result = {
        "status": "OK",
        "references": [{"reference_id": "Reference 1"}],
        "action": "agent",
        "reason": "证据充分",
        "router_llm_used": False,
        "answer_llm_used": True,
        "llm_call_count": 2,
        "tool_trace": [
            {"step": "query_rewrite", "rewrite_llm_used": True},
            {
                "step": "llm_usage",
                "router_llm_used": True,
                "answer_llm_used": False,
                "llm_call_count": 99,
            },
            {
                "step": "compose_final_answer",
                "mode": "agent_loop",
            },
        ],
    }
    original = deepcopy(result)
    trace_metrics = {
        "retrieved_count": 4,
        "tool_called": True,
        "tool_success": True,
        "tool_name": "web_fetch",
        "route_cost_ms": 7,
    }

    apply_result_metrics(
        context.metrics,
        result,
        trace_metrics,
        get_chat_status_policy("OK"),
    )

    assert result == original
    assert set(context.metrics) == metric_keys
    assert context.metrics["status"] == "success"
    assert context.metrics["reference_count"] == 1
    for key, value in trace_metrics.items():
        assert context.metrics[key] == value
    assert context.metrics["router_llm_used"] is False
    assert context.metrics["answer_llm_used"] is True
    assert context.metrics["llm_call_count"] == 2
    assert context.metrics["action"] == "agent"
    assert context.metrics["reason"] == "证据充分"


def test_collect_trace_metrics_keeps_existing_activity_semantics_only():
    result = {
        "llm_call_count": 3,
        "tool_trace": [
            {"step": "decide", "route_cost_ms": 7},
            {"step": "query_rewrite", "rewrite_llm_used": True},
            {"step": "llm_usage", "llm_call_count": 2},
            {
                "step": "compose_final_answer",
                "mode": "agent_loop",
            },
            {
                "tool_name": "search_knowledge_base",
                "final_chunk_count": 4,
                "tool_status": "success",
            },
            {"tool_name": "compare_documents", "tool_status": "success"},
            {"tool_name": "mcp__example__lookup", "tool_status": "success"},
        ],
    }
    original = deepcopy(result)

    assert collect_trace_metrics(result) == {
        "retrieved_count": 4,
        "tool_called": False,
        "tool_success": False,
        "tool_name": "",
        "route_cost_ms": 7,
    }
    assert result == original

    result["tool_trace"].extend([
        {"tool_name": "web_search", "tool_status": "success"},
        {"tool_name": "web_fetch", "tool_status": "error"},
        {"tool_name": "search_knowledge_base", "final_chunk_count": 2},
        {"step": "decide", "route_cost_ms": -1},
    ])

    assert collect_trace_metrics(result) == {
        "retrieved_count": 2,
        "tool_called": True,
        "tool_success": True,
        "tool_name": "web_fetch",
        "route_cost_ms": 0,
    }


def test_stream_events_update_accounting_and_completion_state():
    context = make_context()
    context.metrics = build_initial_chat_metrics(
        context,
        request_enter_time=context.request.request_enter_time,
        queue_wait_ms=500,
    )
    initialize_stream_metrics(context.metrics)
    answer_parts = []

    first = record_stream_event(
        context,
        "start",
        {"session_id_present": True},
        perf_counter=lambda: 10.025,
    )
    record_stream_event(
        context,
        "answer_delta",
        {"text": "完整"},
        perf_counter=lambda: 10.050,
        streamed_answer_parts=answer_parts,
    )
    record_stream_event(
        context,
        "tool",
        {
            "tool_name": "web_search",
            "phase": "completed",
            "status": "success",
        },
        perf_counter=lambda: 10.075,
    )
    record_stream_completion(
        context.metrics,
        streamed_answer_parts=answer_parts,
        final_answer="完整",
    )
    record_stream_termination(context.metrics, completed=True)

    assert first.event == "start"
    assert context.metrics["stream_event_count"] == 3
    assert context.metrics["first_event_ms"] == 25
    assert context.metrics["first_answer_delta_ms"] == 50
    assert context.metrics["answer_delta_count"] == 1
    assert context.metrics["streamed_answer_chars"] == 2
    assert context.metrics["tool_name"] == "web_search"
    assert context.metrics["tool_success"] is True
    assert context.metrics["streamed_answer_matches_final"] is True
    assert context.metrics["client_disconnected"] is False


def test_finalize_calculates_latency_total_cost_and_writes_once():
    context = make_context()
    context.metrics = build_initial_chat_metrics(
        context,
        request_enter_time=context.request.request_enter_time,
        queue_wait_ms=500,
    )
    writes = []

    finalize_chat_telemetry(
        context,
        now_iso=lambda: "2026-07-29T10:00:00.625",
        perf_counter=lambda: 10.125,
        log_and_write_metrics=lambda record: writes.append(dict(record)),
    )

    assert context.metrics["latency_ms"] == 125
    assert context.metrics["total_cost_ms"] == 625
    assert context.metrics["end_time"] == "2026-07-29T10:00:00.625"
    assert writes == [context.metrics]


def test_metrics_and_bad_case_write_failures_do_not_escape():
    context = make_context()
    context.metrics = build_initial_chat_metrics(
        context,
        request_enter_time=context.request.request_enter_time,
        queue_wait_ms=500,
    )

    def fail(_record):
        raise RuntimeError("storage unavailable")

    safe_write_bad_case(
        {"trace_id": "trace-1"},
        trace_id="trace-1",
        write_bad_cases=fail,
    )
    finalize_chat_telemetry(
        context,
        now_iso=lambda: "2026-07-29T10:00:00.625",
        perf_counter=lambda: 10.125,
        log_and_write_metrics=fail,
    )

    assert context.metrics["latency_ms"] == 125
