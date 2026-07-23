"""指标与失败案例模块。

负责汇总调用链指标、记录 LLM 使用情况，并持久化 metrics 与 bad case 日志。
"""

import json
from datetime import datetime
from pathlib import Path

from app.common.logging import get_logger


logger = get_logger("chat_service")
METRICS_PATH = Path("logs") / "metrics.jsonl"
BAD_CASES_PATH = Path("logs") / "bad_cases.jsonl"

REAL_BAD_CASE = {
    "tool_error",
    "parse_error",
    "reference_error",
    "model_error",
    "internal_error",
    "agent_decision_error",
    "index_not_ready",
}

EXPECTED_NON_BAD_CASE_TYPES = {
    "",
    "empty_question",
    "invalid_input",
    "clarify_needed",
    "ambiguous_error",
    "expected_refusal",
    "out_of_scope",
    "out_of_scope_error",
    "out_of_scope_refused",
    "no_relevant_info",
    "insufficient",
    "retrieval_error",
}


def now_iso() -> str:
    """返回带毫秒精度的当前 ISO 时间字符串。"""
    return datetime.now().isoformat(timespec="milliseconds")


def should_write_bad_case(record: dict) -> bool:
    """判断当前错误记录是否值得写入 bad case 日志。"""
    error_type = str(record.get("error_type", "") or "").strip()
    if error_type in REAL_BAD_CASE:
        return True
    if error_type in EXPECTED_NON_BAD_CASE_TYPES:
        return False
    logger.info(
        "[trace_id=%s] [bad_case] skip unknown error_type=%s",
        record.get("trace_id", ""),
        error_type,
    )
    return False


def normalize_metrics_status(result_status: str) -> str:
    """把内部业务状态映射为指标统计使用的状态名称。"""
    return {
        "OK": "success",
        "NO_RELEVANT_INFO": "no_relevant_info",
        "TOOL_ERROR": "tool_error",
        "MODEL_BAD_OUTPUT": "model_bad_output",
        "NEED_CLARIFICATION": "clarify",
        "OUT_OF_SCOPE": "out_of_scope",
        "MODEL_ERROR": "model_error",
        "INTERNAL_ERROR": "internal_error",
    }.get(result_status, "internal_error")


def extract_llm_usage(
    tool_trace: list | None = None,
    router_llm_used: bool | None = None,
    decompose_llm_used: bool | None = None,
    answer_llm_used: bool | None = None,
    llm_call_count: int | None = None,
) -> dict:
    """从调用链追踪中提取各阶段的 LLM 使用情况。"""
    query_rewrite_llm_used = False
    for step in tool_trace or []:
        if step.get("step") == "query_rewrite":
            query_rewrite_llm_used = bool(step.get("rewrite_llm_used", False))
            continue
        if step.get("step") != "llm_usage":
            continue
        router_llm_used = bool(step.get("router_llm_used", router_llm_used or False))
        decompose_llm_used = bool(
            step.get("decompose_llm_used", decompose_llm_used or False)
        )
        answer_llm_used = bool(step.get("answer_llm_used", answer_llm_used or False))
        llm_call_count = step.get("llm_call_count", llm_call_count)
        break

    router_llm_used = bool(router_llm_used)
    decompose_llm_used = bool(decompose_llm_used)
    answer_llm_used = bool(answer_llm_used)
    if llm_call_count is None:
        llm_call_count = sum(
            (router_llm_used, decompose_llm_used, answer_llm_used)
        )
    llm_call_count = int(llm_call_count or 0) + int(query_rewrite_llm_used)
    return {
        "router_llm_used": router_llm_used,
        "decompose_llm_used": decompose_llm_used,
        "answer_llm_used": answer_llm_used,
        "llm_call_count": max(0, llm_call_count),
    }


def collect_trace_metrics(result: dict) -> dict:
    """从 Agent 结果中汇总检索、工具、耗时和模型指标。"""
    tool_trace = result.get("tool_trace") or []
    retrieved_count = 0
    tool_called = False
    tool_success = False
    tool_name = ""
    real_tools = {
        "simple_calculator",
        "list_documents",
        "get_chunk_text",
        "web_search",
        "web_fetch",
        "web_pdf_fetch",
    }
    rag_timings = result.get("timings")
    if not isinstance(rag_timings, dict):
        rag_timings = {}

    for step in tool_trace:
        current_tool_name = step.get("tool_name", "")
        if current_tool_name == "search_knowledge_base":
            retrieved_count = step.get("final_chunk_count", retrieved_count)
        if current_tool_name in real_tools:
            tool_name = current_tool_name
            tool_called = True
            tool_success = tool_success or step.get("tool_status") == "success"
        if (
            not rag_timings
            and step.get("step") == "compose_final_answer"
            and step.get("mode") == "rag"
            and isinstance(step.get("timings"), dict)
        ):
            rag_timings = step["timings"]

    return {
        "retrieved_count": retrieved_count,
        "tool_called": tool_called,
        "tool_success": tool_success,
        "tool_name": tool_name,
        **extract_llm_usage(
            tool_trace,
            router_llm_used=bool(result.get("router_llm_used", False)),
            decompose_llm_used=bool(result.get("decompose_llm_used", False)),
            answer_llm_used=bool(result.get("answer_llm_used", False)),
            llm_call_count=result.get("llm_call_count"),
        ),
        "timings": rag_timings,
    }


def build_query_rewrite_trace(rewrite_result: dict) -> dict:
    """把问题改写结果转换为标准追踪步骤。"""
    return {
        "step": "query_rewrite",
        "original_question": rewrite_result.get("original_question", ""),
        "rewritten_question": rewrite_result.get("rewritten_question", ""),
        "history_used": bool(rewrite_result.get("history_used", False)),
        "rewritten": bool(rewrite_result.get("rewritten", False)),
        "need_check": bool(rewrite_result.get("need_check", False)),
        "need_rewrite": bool(rewrite_result.get("need_rewrite", False)),
        "rewrite_llm_used": bool(rewrite_result.get("rewrite_llm_used", False)),
        "reason": rewrite_result.get("reason", ""),
        "error": rewrite_result.get("error", ""),
    }


def attach_query_rewrite_trace(result: dict, rewrite_trace: dict) -> dict:
    """将问题改写追踪插入 Agent 返回的调用链首部。"""
    result = dict(result or {})
    tool_trace = list(result.get("tool_trace") or [])
    if not any(step.get("step") == "query_rewrite" for step in tool_trace):
        tool_trace.insert(0, rewrite_trace)
    result["tool_trace"] = tool_trace
    result["query_rewrite_used"] = bool(rewrite_trace.get("rewritten", False))
    result["query_rewrite_llm_used"] = bool(
        rewrite_trace.get("rewrite_llm_used", False)
    )
    result["effective_question_len"] = len(
        rewrite_trace.get("rewritten_question", "") or ""
    )
    return result


def write_metrics(record: dict) -> None:
    """将单条请求指标追加写入 JSONL 文件。"""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_bad_cases(record: dict) -> None:
    """筛选并持久化真正需要排查的失败案例。"""
    if not should_write_bad_case(record):
        logger.info(
            "[trace_id=%s] [bad_case] skip error_type=%s note=%s",
            record.get("trace_id", ""),
            record.get("error_type", ""),
            record.get("note", ""),
        )
        return

    BAD_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized_record = {
        "created_at": now_iso(),
        "trace_id": record.get("trace_id", ""),
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "references": record.get("references", []),
        "route": record.get("route", ""),
        "error_type": record.get("error_type", ""),
        "note": record.get("note", ""),
    }
    with BAD_CASES_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(normalized_record, ensure_ascii=False) + "\n")


def log_and_write_metrics(record: dict) -> None:
    """记录指标摘要日志，并将完整指标持久化。"""
    logger.info(
        "[trace_id=%s] [metrics] status=%s latency_ms=%s total_cost_ms=%s "
        "route_cost_ms=%s agent_cost_ms=%s retrieved_count=%s tool_name=%s "
        "llm_call_count=%s action=%s reason=%s",
        record.get("trace_id"),
        record.get("status"),
        record.get("latency_ms"),
        record.get("total_cost_ms"),
        record.get("route_cost_ms"),
        record.get("agent_cost_ms"),
        record.get("retrieved_count"),
        record.get("tool_name"),
        record.get("llm_call_count"),
        record.get("action"),
        record.get("reason"),
    )
    write_metrics(record)


def read_recent_metrics(limit: int = 20) -> list[dict]:
    """读取最近指定数量的有效指标记录。"""
    if not METRICS_PATH.exists():
        return []
    records = []
    for line in METRICS_PATH.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
