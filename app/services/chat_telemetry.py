"""聊天流程的 metrics、bad case 和终端调用链辅助逻辑。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from app.common.logging import get_logger
from app.contracts.chat import ChatStatusPolicy
from app.services.chat_models import ChatRequestContext


logger = get_logger("chat_service")

_TIMING_METRIC_KEYS = (
    "retrieve_cost_ms",
    "decompose_cost_ms",
    "rerank_cost_ms",
    "retrieve_with_sub_queries_cost_ms",
    "context_build_cost_ms",
    "llm_cost_ms",
    "parse_cost_ms",
    "rag_answer_cost_ms",
)


def build_initial_chat_metrics(
    context: ChatRequestContext,
    *,
    request_enter_time: str,
    queue_wait_ms: int,
) -> dict[str, Any]:
    """构造与现有 JSONL 字段兼容的初始聊天指标。"""
    question_len = len(context.original_question)
    return {
        "trace_id": context.request.trace_id,
        "request_enter_time": request_enter_time,
        "process_start_time": context.process_start_time,
        "end_time": "",
        "queue_wait_ms": queue_wait_ms,
        "latency_ms": 0,
        "total_cost_ms": 0,
        "route_cost_ms": 0,
        "agent_cost_ms": 0,
        "retrieve_cost_ms": 0,
        "decompose_cost_ms": 0,
        "rerank_cost_ms": 0,
        "retrieve_with_sub_queries_cost_ms": 0,
        "context_build_cost_ms": 0,
        "llm_cost_ms": 0,
        "parse_cost_ms": 0,
        "rag_answer_cost_ms": 0,
        "status": "internal_error",
        "question_len": question_len,
        "reference_count": 0,
        "retrieved_count": 0,
        "tool_called": False,
        "tool_success": False,
        "tool_name": "",
        "router_llm_used": False,
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
        "query_rewrite_used": False,
        "query_rewrite_llm_used": False,
        "effective_question_len": question_len,
        "action": "",
        "reason": "",
    }


def build_initial_bad_case(
    context: ChatRequestContext,
    *,
    created_at: str,
) -> dict[str, Any]:
    """构造供 Runtime 和服务异常路径复用的失败记录。"""
    return {
        "created_at": created_at,
        "trace_id": context.request.trace_id,
        "question": context.original_question,
        "answer": "",
        "references": [],
        "route": "",
        "error_type": "",
        "note": "",
    }


def log_chat_process_start(
    context: ChatRequestContext,
    *,
    queue_wait_ms: int,
) -> None:
    """记录不含业务内容全文的请求处理起点。"""
    logger.info(
        "[trace_id=%s] [chat] process_start queue_wait_ms=%s "
        "question_len=%s session_id=%s preview=%s",
        context.request.trace_id,
        queue_wait_ms,
        len(context.original_question),
        context.session_id or "-",
        context.original_question[:50],
    )


def format_tool_call_chain(tool_trace: list[dict[str, Any]]) -> str:
    """把工具 trace 压缩为适合终端单行展示的调用链。"""
    calls = []
    for step in tool_trace:
        if step.get("step") != "tool_call":
            continue
        tool_name = " ".join(str(step.get("tool_name") or "").split())[:100]
        if not tool_name:
            continue
        tool_status = " ".join(
            str(step.get("tool_status") or "unknown").split()
        )[:40]
        cost_ms = max(0, int(step.get("cost_ms", 0) or 0))
        calls.append(f"{tool_name}({tool_status},{cost_ms}ms)")
    return " -> ".join(calls) or "none"


def log_tool_call_chain(
    trace_id: str,
    tool_trace: list[dict[str, Any]],
) -> None:
    """按实际顺序记录脱敏后的工具调用链。"""
    logger.info(
        "[trace_id=%s] [agent] tool_call_chain=%s",
        trace_id,
        format_tool_call_chain(tool_trace),
    )


def apply_result_metrics(
    metrics: dict[str, Any],
    result: Mapping[str, Any],
    trace_metrics: Mapping[str, Any],
    policy: ChatStatusPolicy,
) -> None:
    """把已校验 Result 的 trace 汇总到 metrics，不修改 Result。"""
    metrics["status"] = policy.metric_status
    metrics["reference_count"] = len(result.get("references") or [])
    for key in (
        "retrieved_count",
        "tool_called",
        "tool_success",
        "tool_name",
        "router_llm_used",
        "decompose_llm_used",
        "answer_llm_used",
        "llm_call_count",
        "route_cost_ms",
    ):
        metrics[key] = trace_metrics[key]
    timings = trace_metrics["timings"]
    for timing_key in _TIMING_METRIC_KEYS:
        metrics[timing_key] = int((timings.get(timing_key, 0) or 0))
    metrics["action"] = result.get("action", "")
    metrics["reason"] = result.get("reason") or policy.default_reason


def safe_write_bad_case(
    record: dict[str, Any],
    *,
    trace_id: str,
    write_bad_cases: Callable[[dict], None],
) -> None:
    """安全写入失败案例；持久化失败不覆盖主业务结果。"""
    try:
        write_bad_cases(record)
    except Exception as exc:
        logger.warning(
            "[trace_id=%s] [bad_case] write_failed error=%s",
            trace_id,
            str(exc)[:200],
        )


def finalize_chat_telemetry(
    context: ChatRequestContext,
    *,
    now_iso: Callable[[], str],
    perf_counter: Callable[[], float],
    log_and_write_metrics: Callable[[dict], None],
) -> None:
    """计算最终耗时并安全持久化 metrics。"""
    process_end_perf = perf_counter()
    context.metrics["end_time"] = now_iso()
    context.metrics["latency_ms"] = int(
        (process_end_perf - context.process_start_perf) * 1000
    )
    context.metrics["total_cost_ms"] = int(
        (process_end_perf - context.request_enter_perf) * 1000
    )

    logger.info(
        "[trace_id=%s] [chat] process_end status=%s queue_wait_ms=%s "
        "latency_ms=%s total_cost_ms=%s route_cost_ms=%s "
        "agent_cost_ms=%s retrieve_cost_ms=%s decompose_cost_ms=%s "
        "rerank_cost_ms=%s llm_cost_ms=%s llm_call_count=%s "
        "query_rewrite_used=%s query_rewrite_llm_used=%s "
        "effective_question_len=%s",
        context.request.trace_id,
        context.metrics.get("status"),
        context.metrics.get("queue_wait_ms"),
        context.metrics.get("latency_ms"),
        context.metrics.get("total_cost_ms"),
        context.metrics.get("route_cost_ms"),
        context.metrics.get("agent_cost_ms"),
        context.metrics.get("retrieve_cost_ms"),
        context.metrics.get("decompose_cost_ms"),
        context.metrics.get("rerank_cost_ms"),
        context.metrics.get("llm_cost_ms"),
        context.metrics.get("llm_call_count"),
        context.metrics.get("query_rewrite_used"),
        context.metrics.get("query_rewrite_llm_used"),
        context.metrics.get("effective_question_len"),
    )
    try:
        log_and_write_metrics(context.metrics)
    except Exception as exc:
        logger.exception(
            "[trace_id=%s] [metrics] write_failed error=%s",
            context.request.trace_id,
            exc,
        )

