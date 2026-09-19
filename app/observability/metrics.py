"""指标与失败案例模块。

负责汇总调用链指标、记录 LLM 使用情况，并持久化 metrics 与 bad case 日志。
"""

import json
from datetime import datetime

from app.common.logging import get_logger
from app.config import get_settings


logger = get_logger("chat_service")

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


def collect_trace_metrics(result: dict) -> dict:
    """仅汇总 trace 独有的检索数量、工具活动和路由耗时。"""
    tool_trace = result.get("tool_trace") or []
    retrieved_count = 0
    tool_called = False
    tool_success = False
    tool_name = ""
    route_cost_ms = 0
    real_tools = {
        "list_documents",
        "get_chunk_text",
        "web_search",
        "web_fetch",
        "web_pdf_fetch",
    }
    for step in tool_trace:
        if step.get("step") == "decide":
            route_cost_ms = max(
                0,
                int(step.get("route_cost_ms", 0) or 0),
            )
        current_tool_name = step.get("tool_name", "")
        if current_tool_name == "search_knowledge_base":
            retrieved_count = step.get("final_chunk_count", retrieved_count)
        if current_tool_name in real_tools:
            tool_name = current_tool_name
            tool_called = True
            tool_success = tool_success or step.get("tool_status") == "success"

    return {
        "retrieved_count": retrieved_count,
        "tool_called": tool_called,
        "tool_success": tool_success,
        "tool_name": tool_name,
        "route_cost_ms": route_cost_ms,
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
    return result


def write_metrics(record: dict) -> None:
    """将单条请求指标追加写入 JSONL 文件。"""
    path = get_settings().metrics_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
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

    path = get_settings().bad_cases_path
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("a", encoding="utf-8") as file:
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
    path = get_settings().metrics_path
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
