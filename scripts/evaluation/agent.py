"""Agent Evaluation v2 的执行、判分、失败分类和指标汇总。

用途：逐条运行单轮或多轮 Agent case，并核对路由、工具、状态和答案合同。
输入：已校验的 Agent 数据集、评测专用 Settings，以及 Core/Online suite 选择。
输出：逐 case 明细和按类别、工具、状态等维度聚合的确定性指标。
边界：通过正式 ``ChatService`` 调用链执行；case 间隔离运行态，不写生产持久化数据。
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import AppSettings
from scripts.evaluation.contracts import (
    AgentEvaluationCase,
    AgentEvaluationDataset,
    ExpectedOutcome,
)


def safe_lower(text: Any) -> str:
    """安全地把任意值转换成 casefold 后的小写文本。"""
    return "" if text is None else str(text).casefold()


def init_rag_for_eval(force_rebuild: bool = False) -> None:
    """保留 Agent-only baseline 原有的 RAG 初始化语义。"""
    from app.config import get_settings, validate_startup_environment
    from app.rag import (
        get_rag_status,
        index_cache_exists,
        load_index_cache,
        needs_rebuild,
        rebuild_and_persist_index,
    )

    settings = get_settings()
    validate_startup_environment(settings)
    data_dir = str(settings.rag_file_path)
    if force_rebuild:
        ok, message = rebuild_and_persist_index(data_dir)
        print(f"[init] force_rebuild=True, ok={ok}, msg={message}")
    else:
        need, reason = needs_rebuild(data_dir)
        if not need and index_cache_exists():
            ok, message = load_index_cache()
            print(f"[init] load_index_cache, ok={ok}, msg={message}")
            if not ok:
                ok, message = rebuild_and_persist_index(data_dir)
                print(f"[init] cache failed, rebuild, ok={ok}, msg={message}")
        else:
            ok, message = rebuild_and_persist_index(data_dir)
            print(
                f"[init] rebuild, need={need}, reason={reason}, "
                f"ok={ok}, msg={message}"
            )
    status = get_rag_status()
    print(
        "[init] RAG status:",
        f"ready={status.get('ready')},",
        f"message={status.get('message')},",
        f"doc_count={status.get('doc_count')},",
        f"chunk_count={status.get('chunk_count')}",
    )
    if not status.get("ready"):
        raise RuntimeError(f"RAG 未准备好：{status}")


def extract_actual_fields(result: dict[str, Any]) -> dict[str, Any]:
    """从 decide 与真实 tool_call trace 中提取路由和工具行为。"""
    trace = [
        step
        for step in (result.get("tool_trace") or [])
        if isinstance(step, dict)
    ]
    decide_trace = next(
        (step for step in trace if step.get("step") == "decide"),
        {},
    )
    actual_tools = [
        str(step.get("tool_name") or "")
        for step in trace
        if step.get("step") == "tool_call" and step.get("tool_name")
    ]
    return {
        "actual_decision": decide_trace.get("decision_type", ""),
        "actual_action": result.get("action")
        or decide_trace.get("action", ""),
        "actual_tool": actual_tools[0] if actual_tools else "",
        "actual_tools": actual_tools,
        "actual_status": result.get("status", ""),
    }


def _build_search_text(result: dict[str, Any]) -> str:
    """汇总可用于确定性关键词检查的结果字段。"""
    import json

    parts = [
        result.get("answer", ""),
        result.get("reason", ""),
        result.get("tool_result_summary", ""),
        json.dumps(result.get("references", []), ensure_ascii=False),
        json.dumps(result.get("tool_trace", []), ensure_ascii=False),
    ]
    return "\n".join(str(value) for value in parts if value is not None)


def check_keywords(
    case: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """检查结果文本是否命中评测用例要求的关键词。"""
    expected = case.get("expected_keywords", []) or []
    if not expected:
        return {
            "keyword_pass": True,
            "keyword_hits": [],
            "keyword_missing": [],
            "keyword_hit_count": 0,
            "keyword_required": 0,
        }
    search_text = safe_lower(_build_search_text(result))
    hits = [item for item in expected if safe_lower(item) in search_text]
    missing = [item for item in expected if item not in hits]
    default_required = max(1, math.ceil(len(expected) * 0.6))
    required = int(case.get("min_keyword_hits", default_required))
    return {
        "keyword_pass": len(hits) >= required,
        "keyword_hits": hits,
        "keyword_missing": missing,
        "keyword_hit_count": len(hits),
        "keyword_required": required,
    }


def extract_reference_sources(result: dict[str, Any]) -> list[str]:
    """从引用列表中提取可用于来源匹配的文件或路径。"""
    sources: list[str] = []
    for reference in result.get("references", []) or []:
        if not isinstance(reference, dict):
            continue
        for key in ("source_file", "file_name", "source_path"):
            value = reference.get(key)
            if value:
                sources.append(str(value))
    return sources


def _extract_tool_output(result: dict[str, Any]) -> dict[str, Any] | None:
    """从结果或 trace 中提取最近一次完整工具输出。"""
    tool_result = result.get("tool_result")
    if isinstance(tool_result, dict):
        output = tool_result.get("output")
        if isinstance(output, dict):
            return output
    direct_output = result.get("tool_output")
    if isinstance(direct_output, dict):
        return direct_output
    trace = result.get("tool_trace") or []
    if not isinstance(trace, list):
        return None
    for step in reversed(trace):
        if not isinstance(step, dict):
            continue
        output = step.get("tool_output")
        if isinstance(output, dict):
            return output
        nested = step.get("tool_result")
        if isinstance(nested, dict) and isinstance(nested.get("output"), dict):
            return nested["output"]
    return None


def check_tool_output_keys(
    case: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """按 case 要求检查工具输出中是否包含指定字段。"""
    expected = case.get("expected_tool_output_keys", []) or []
    if not expected:
        return {
            "tool_output_keys_pass": True,
            "tool_output_keys_expected": [],
            "tool_output_keys_missing": [],
        }
    output = _extract_tool_output(result)
    missing = (
        [key for key in expected if key not in output]
        if isinstance(output, dict)
        else list(expected)
    )
    return {
        "tool_output_keys_pass": not missing,
        "tool_output_keys_expected": expected,
        "tool_output_keys_missing": missing,
    }


class InMemorySessionAdapter:
    """评测专用 session 边界，不触碰 SQLite。"""

    def __init__(self) -> None:
        """初始化按 session_id 隔离的进程内消息表。"""
        self._messages: dict[str, list[dict[str, str]]] = {}

    def get_history(self, session_id: str | None) -> list[dict[str, str]]:
        """返回指定会话的消息副本，避免调用方修改内部状态。"""
        return [
            dict(item)
            for item in self._messages.get(str(session_id or ""), [])
        ]

    def append_message(
        self,
        session_id: str | None,
        role: str,
        content: str,
    ) -> None:
        """按正式会话接口追加一条角色消息。"""
        key = str(session_id or "")
        self._messages.setdefault(key, []).append(
            {"role": role, "content": content}
        )

    def clear(self) -> None:
        """清空本 case 使用的全部内存会话。"""
        self._messages.clear()


@contextmanager
def capture_tool_output_keys() -> Iterator[list[dict[str, Any]]]:
    """仅为评测捕获真实工具输出的字段名，不保留字段值。"""
    from app.tools.registry import ToolRegistry

    records: list[dict[str, Any]] = []
    original_dispatch = ToolRegistry.dispatch

    def dispatch_with_schema_capture(
        registry: ToolRegistry,
        name: str,
        args: dict | None = None,
        *,
        context: Any = None,
    ) -> dict:
        """调用真实工具分发，并只记录输出结构而不保存字段值。"""
        result = original_dispatch(registry, name, args, context=context)
        output = result.get("output") if isinstance(result, dict) else None
        records.append(
            {
                "tool_name": name,
                "status": result.get("status", "")
                if isinstance(result, dict)
                else "",
                "ok": result.get("ok") is True
                if isinstance(result, dict)
                else False,
                "output_keys": sorted(str(key) for key in output)
                if isinstance(output, dict)
                else [],
            }
        )
        return result

    ToolRegistry.dispatch = dispatch_with_schema_capture
    try:
        yield records
    finally:
        ToolRegistry.dispatch = original_dispatch


def _rate(numerator: int, denominator: int) -> float:
    """安全计算四位小数比率。"""
    return round(numerator / denominator, 4) if denominator else 0.0


def _expected_as_legacy(expected: ExpectedOutcome) -> dict[str, Any]:
    """把 v2 字段映射为旧 evaluator 的纯函数输入。"""
    return {
        "expected_keywords": expected.keywords,
        "min_keyword_hits": expected.min_keyword_hits,
        "expected_tool_output_keys": expected.tool_output_keys,
    }


def _check_sources(
    expected: ExpectedOutcome,
    result: dict[str, Any],
) -> dict[str, Any]:
    """按 any/all/none 口径检查引用来源。"""
    actual = extract_reference_sources(result)
    actual_text = safe_lower(" ".join(actual))
    hits = [
        source
        for source in expected.sources
        if safe_lower(source) in actual_text
    ]
    missing = [
        source for source in expected.sources if source not in hits
    ]
    if not expected.sources:
        passed = True
    elif expected.source_match == "all":
        passed = not missing
    elif expected.source_match == "none":
        passed = not hits
    else:
        passed = bool(hits)
    return {
        "source_pass": passed,
        "source_hits": hits,
        "source_missing": missing,
        "actual_sources": actual,
    }


def _check_forbidden(
    expected: ExpectedOutcome,
    result: dict[str, Any],
) -> dict[str, Any]:
    """确定性检查禁止关键词。"""
    search_text = safe_lower(
        "\n".join(
            [
                str(result.get("answer") or ""),
                str(result.get("reason") or ""),
                str(result.get("tool_result_summary") or ""),
            ]
        )
    )
    hits = [
        keyword
        for keyword in expected.forbidden_keywords
        if safe_lower(keyword) in search_text
    ]
    return {
        "forbidden_pass": not hits,
        "forbidden_keyword_hits": hits,
    }


def extract_query_rewrite(result: dict[str, Any]) -> dict[str, Any]:
    """从最终 tool_trace 提取唯一 Query Rewrite 追踪。"""
    trace = result.get("tool_trace") or []
    item = next(
        (
            step
            for step in trace
            if isinstance(step, dict)
            and step.get("step") == "query_rewrite"
        ),
        {},
    )
    return {
        "original_question": str(item.get("original_question") or ""),
        "rewritten_question": str(item.get("rewritten_question") or ""),
        "rewritten": bool(item.get("rewritten", False)),
        "rewrite_llm_used": bool(item.get("rewrite_llm_used", False)),
        "reason": str(item.get("reason") or ""),
    }


def _check_rewrite(
    expected: ExpectedOutcome,
    result: dict[str, Any],
) -> dict[str, Any]:
    """检查 Query Rewrite 布尔期望与改写关键词。"""
    actual = extract_query_rewrite(result)
    text = safe_lower(actual["rewritten_question"])
    hits = [
        keyword
        for keyword in expected.rewrite_keywords
        if safe_lower(keyword) in text
    ]
    missing = [
        keyword
        for keyword in expected.rewrite_keywords
        if keyword not in hits
    ]
    required = expected.min_rewrite_keyword_hits
    if expected.rewrite_keywords and required == 0:
        required = max(1, math.ceil(len(expected.rewrite_keywords) * 0.6))
    checks = [len(hits) >= required]
    if expected.rewritten is not None:
        checks.append(actual["rewritten"] is expected.rewritten)
    if expected.rewrite_llm_used is not None:
        checks.append(
            actual["rewrite_llm_used"] is expected.rewrite_llm_used
        )
    return {
        "rewrite_pass": all(checks),
        "rewrite_keyword_hits": hits,
        "rewrite_keyword_missing": missing,
        "rewrite_keyword_required": required,
        "query_rewrite": actual,
    }


def _tool_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    """返回按 trace 顺序排列的真实工具调用。"""
    return [
        step
        for step in (result.get("tool_trace") or [])
        if isinstance(step, dict) and step.get("step") == "tool_call"
    ]


def _check_tool_output_schema(
    expected: ExpectedOutcome,
    result: dict[str, Any],
    tool_output_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """优先检查评测边界捕获的真实字段名，兼容旧完整输出。"""
    legacy = check_tool_output_keys(_expected_as_legacy(expected), result)
    if not expected.tool_output_keys or tool_output_records is None:
        return legacy

    expected_tools = expected.expected_tools()
    actual_keys = {
        str(key)
        for record in tool_output_records
        if isinstance(record, dict)
        and record.get("ok") is True
        and (
            not expected_tools
            or str(record.get("tool_name") or "") in expected_tools
        )
        for key in (record.get("output_keys") or [])
    }
    missing = [
        key for key in expected.tool_output_keys if key not in actual_keys
    ]
    return {
        "tool_output_keys_pass": not missing,
        "tool_output_keys_expected": expected.tool_output_keys,
        "tool_output_keys_missing": missing,
    }


def score_agent_turn(
    *,
    case_id: str,
    category: str,
    suite: str,
    tags: list[str],
    turn_index: int,
    question: str,
    expected: ExpectedOutcome,
    result: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    latency_ms: float = 0.0,
    runtime_error: str = "",
    tool_output_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """复用旧纯函数并补齐 v2 的多原因判分。"""
    metrics = dict(metrics or {})
    actual = extract_actual_fields(result)
    calls = _tool_calls(result)
    actual_tools = actual["actual_tools"]
    actual_tool_statuses = [
        str(step.get("tool_status") or "") for step in calls
    ]
    legacy_case = _expected_as_legacy(expected)
    keyword = check_keywords(legacy_case, result)
    source = _check_sources(expected, result)
    forbidden = _check_forbidden(expected, result)
    tool_output = _check_tool_output_schema(
        expected,
        result,
        tool_output_records,
    )
    rewrite = _check_rewrite(expected, result)

    checks = {
        "decision": (
            not expected.decision
            or actual["actual_decision"] == expected.decision
        ),
        "action": (
            not expected.action
            or actual["actual_action"] == expected.action
        ),
        "status": (
            expected.status is None
            or actual["actual_status"] == expected.status.value
        ),
        "first_tool": (
            not expected.first_tool
            or actual["actual_tool"] == expected.first_tool
        ),
        "tools_contains": all(
            tool in actual_tools for tool in expected.tools_contains
        ),
        "tool_sequence": (
            not expected.tool_sequence
            or actual_tools == expected.tool_sequence
        ),
        "tool_status_sequence": (
            not expected.tool_status_sequence
            or actual_tool_statuses == expected.tool_status_sequence
        ),
        "keyword": keyword["keyword_pass"],
        "source": source["source_pass"],
        "forbidden": forbidden["forbidden_pass"],
        "tool_output_schema": tool_output["tool_output_keys_pass"],
        "query_rewrite": rewrite["rewrite_pass"],
    }
    failure_reasons: list[str] = []
    if runtime_error:
        failure_reasons.append(
            "timeout"
            if "timeout" in runtime_error.casefold()
            else "runtime_error"
        )
    if any(
        isinstance(step, dict)
        and step.get("step") == "chat_result_contract_error"
        for step in (result.get("tool_trace") or [])
    ):
        failure_reasons.append("invalid_result_contract")
    if not checks["decision"]:
        failure_reasons.append("decision_mismatch")
    if not checks["action"]:
        failure_reasons.append("action_mismatch")
    if not checks["status"]:
        failure_reasons.append("status_mismatch")
        if actual["actual_status"] == "NEED_CLARIFICATION":
            failure_reasons.append("unexpected_clarification")
        elif actual["actual_status"] == "NO_RELEVANT_INFO":
            failure_reasons.append("unexpected_no_evidence")
    if not checks["first_tool"]:
        failure_reasons.append("first_tool_mismatch")
    if not checks["tools_contains"] or not checks["tool_sequence"]:
        failure_reasons.append("tool_sequence_mismatch")
    if not checks["tool_status_sequence"]:
        failure_reasons.append("tool_status_sequence_mismatch")
    if (
        not expected.tool_status_sequence
        and any(step.get("tool_status") != "success" for step in calls)
    ):
        failure_reasons.append("tool_call_failed")
    if not checks["keyword"]:
        failure_reasons.append("keyword_missing")
    if not checks["forbidden"]:
        failure_reasons.append("forbidden_keyword_hit")
    if not checks["source"]:
        failure_reasons.append("source_missing")
    if not checks["tool_output_schema"]:
        failure_reasons.append("tool_output_schema_missing")
    if not checks["query_rewrite"]:
        failure_reasons.append("query_rewrite_mismatch")
    failure_reasons = list(dict.fromkeys(failure_reasons))

    expected_payload = expected.model_dump(mode="json")
    return {
        "case_id": case_id,
        "turn_index": turn_index,
        "suite": suite,
        "category": category,
        "tags": tags,
        "question": question,
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "expected": expected_payload,
        "actual_decision": actual["actual_decision"],
        "actual_action": actual["actual_action"],
        "actual_status": actual["actual_status"],
        "actual_first_tool": actual["actual_tool"],
        "actual_tools": actual_tools,
        "actual_tool_statuses": actual_tool_statuses,
        "tool_call_count": len(calls),
        "successful_tool_call_count": sum(
            step.get("tool_status") == "success" for step in calls
        ),
        "llm_call_count": int(
            metrics.get("llm_call_count", result.get("llm_call_count", 0))
            or 0
        ),
        "latency_ms": round(float(latency_ms), 2),
        "keyword_pass": checks["keyword"],
        "keyword_hits": keyword["keyword_hits"],
        "keyword_missing": keyword["keyword_missing"],
        "source_pass": checks["source"],
        "source_hits": source["source_hits"],
        "source_missing": source["source_missing"],
        "forbidden_pass": checks["forbidden"],
        "forbidden_keyword_hits": forbidden["forbidden_keyword_hits"],
        "tool_output_schema_pass": checks["tool_output_schema"],
        "tool_output_keys_missing": tool_output[
            "tool_output_keys_missing"
        ],
        "query_rewrite_pass": checks["query_rewrite"],
        **rewrite,
        "answer": str(result.get("answer") or ""),
        "references": list(result.get("references") or []),
        "tool_trace": list(result.get("tool_trace") or []),
        "runtime_error": runtime_error,
    }


def combine_case_turns(
    case_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """把多轮 case 汇总成用于 pass rate 的单个 case 结果。"""
    last = rows[-1]
    reasons = list(
        dict.fromkeys(
            reason
            for row in rows
            for reason in row["failure_reasons"]
        )
    )
    return {
        **last,
        "case_id": case_id,
        "turn_count": len(rows),
        "passed": all(row["passed"] for row in rows),
        "failure_reasons": reasons,
        "latency_ms": round(
            sum(float(row["latency_ms"]) for row in rows),
            2,
        ),
        "tool_call_count": sum(
            int(row["tool_call_count"]) for row in rows
        ),
        "successful_tool_call_count": sum(
            int(row["successful_tool_call_count"]) for row in rows
        ),
        "llm_call_count": sum(
            int(row["llm_call_count"]) for row in rows
        ),
        "turns": rows,
    }


def _percentile(values: list[float], percentile: float) -> float:
    """使用线性插值计算确定性的百分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    weight = position - lower
    return round(
        ordered[lower] * (1 - weight) + ordered[upper] * weight,
        2,
    )


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """计算一组 case 的通过率。"""
    return {
        "total_cases": len(rows),
        "passed_cases": sum(row["passed"] for row in rows),
        "pass_rate": _rate(
            sum(row["passed"] for row in rows),
            len(rows),
        ),
    }


def calculate_agent_metrics(
    rows: list[dict[str, Any]],
    *,
    total_cases: int | None = None,
    skipped_cases: int = 0,
) -> dict[str, Any]:
    """汇总 Agent case 指标，并把 Core/Online 保持为独立分组。"""
    executed = len(rows)
    total = executed + skipped_cases if total_cases is None else total_cases
    passed = sum(row["passed"] for row in rows)
    failed = executed - passed

    def accuracy(key: str) -> float:
        """计算包含指定期望字段的 case 准确率。"""
        applicable = [
            row
            for row in rows
            if row["expected"].get(key) not in ("", None, [])
        ]
        actual_key = {
            "decision": "actual_decision",
            "action": "actual_action",
            "status": "actual_status",
            "first_tool": "actual_first_tool",
            "tool_sequence": "actual_tools",
        }[key]
        return _rate(
            sum(
                row[actual_key] == row["expected"][key]
                for row in applicable
            ),
            len(applicable),
        )

    tool_calls = sum(int(row["tool_call_count"]) for row in rows)
    successful_tool_calls = sum(
        int(row["successful_tool_call_count"]) for row in rows
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    by_dimension: dict[str, dict[str, Any]] = {}
    for dimension, getter in {
        "suite": lambda row: row["suite"],
        "category": lambda row: row["category"],
        "expected_status": lambda row: (
            row["expected"].get("status") or "<none>"
        ),
        "expected_first_tool": lambda row: (
            row["expected"].get("first_tool") or "<none>"
        ),
        "tags": lambda row: row["tags"] or ["<none>"],
    }.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            values = getter(row)
            if not isinstance(values, list):
                values = [values]
            for value in values:
                grouped[str(value)].append(row)
        by_dimension[dimension] = {
            name: _group_metrics(items)
            for name, items in sorted(grouped.items())
        }

    def category_accuracy(predicate: Any) -> float:
        """计算满足给定筛选条件的 case 通过率。"""
        applicable = [row for row in rows if predicate(row)]
        return _rate(sum(row["passed"] for row in applicable), len(applicable))

    return {
        "total_cases": total,
        "executed_cases": executed,
        "skipped_cases": skipped_cases,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": _rate(passed, executed),
        "decision_accuracy": accuracy("decision"),
        "action_accuracy": accuracy("action"),
        "status_accuracy": accuracy("status"),
        "first_tool_accuracy": accuracy("first_tool"),
        "tool_sequence_accuracy": accuracy("tool_sequence"),
        "keyword_pass_rate": _rate(
            sum(row["keyword_pass"] for row in rows),
            len(rows),
        ),
        "source_pass_rate": _rate(
            sum(row["source_pass"] for row in rows),
            len(rows),
        ),
        "tool_output_schema_pass_rate": _rate(
            sum(row["tool_output_schema_pass"] for row in rows),
            len(rows),
        ),
        "clarify_accuracy": category_accuracy(
            lambda row: row["expected"].get("status")
            == "NEED_CLARIFICATION"
        ),
        "no_evidence_accuracy": category_accuracy(
            lambda row: row["expected"].get("status")
            == "NO_RELEVANT_INFO"
        ),
        "direct_answer_accuracy": category_accuracy(
            lambda row: row["expected"].get("decision")
            == "direct_answer"
        ),
        "multi_turn_pass_rate": category_accuracy(
            lambda row: int(row.get("turn_count", 1)) > 1
        ),
        "tool_call_success_rate": _rate(
            successful_tool_calls,
            tool_calls,
        ),
        "average_tool_call_count": round(
            statistics.mean(
                [int(row["tool_call_count"]) for row in rows]
            ),
            4,
        )
        if rows
        else 0.0,
        "average_llm_call_count": round(
            statistics.mean(
                [int(row["llm_call_count"]) for row in rows]
            ),
            4,
        )
        if rows
        else 0.0,
        "average_latency_ms": round(statistics.mean(latencies), 2)
        if latencies
        else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        "failure_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in rows
                    for reason in row["failure_reasons"]
                ).items()
            )
        ),
        "groups": by_dimension,
    }


def _permissions_for_suite(suite: str) -> tuple[str, ...]:
    """Core 禁用在线权限，Online 才显式开放。"""
    if suite == "online":
        return ("knowledge_base:read", "web:search", "web:fetch")
    return ("knowledge_base:read",)


def execute_agent_case(
    case: AgentEvaluationCase,
    *,
    settings: AppSettings,
    top_k: int,
    decompose_top_k: int,
) -> dict[str, Any]:
    """通过真实 ChatService 执行一个完全隔离的 Agent case。"""
    from dataclasses import replace

    from app.agent.runtime import run_agent
    from app.observability.metrics import (
        attach_query_rewrite_trace,
        build_query_rewrite_trace,
        collect_trace_metrics,
    )
    from app.services.chat_models import ChatServiceRequest
    from app.services.chat_service import (
        ChatService,
        get_default_chat_service_dependencies,
    )
    from app.tools.context import PrincipalType, ToolInvocationContext

    session = InMemorySessionAdapter()
    captured_metrics: list[dict[str, Any]] = []
    base = get_default_chat_service_dependencies()
    permissions = _permissions_for_suite(case.suite)

    def run_runtime(**kwargs: Any) -> dict[str, Any]:
        """为真实 Agent 运行时注入评测专用身份与权限上下文。"""
        kwargs["tool_context"] = ToolInvocationContext(
            trace_id=str(kwargs.get("trace_id") or ""),
            session_id=str(kwargs.get("session_id") or ""),
            principal_id="evaluation_baseline_v2",
            principal_type=PrincipalType.SERVICE,
            permissions=permissions,
            source="evaluation_baseline_v2",
        )
        return run_agent(**kwargs)

    dependencies = replace(
        base,
        settings_provider=lambda: settings,
        get_history=session.get_history,
        run_agent=run_runtime,
        build_query_rewrite_trace=build_query_rewrite_trace,
        attach_query_rewrite_trace=attach_query_rewrite_trace,
        collect_trace_metrics=collect_trace_metrics,
        log_and_write_metrics=lambda item: captured_metrics.append(dict(item)),
        write_bad_cases=lambda item: None,
        append_message=session.append_message,
    )
    service = ChatService(dependencies)
    session_id = case.session_id.strip() or f"evaluation-{case.id}"
    turn_rows: list[dict[str, Any]] = []
    for turn_index, (question, expected) in enumerate(
        zip(case.iter_questions(), case.iter_expected(), strict=True),
        start=1,
    ):
        trace_id = f"evaluation-{case.id}-turn-{turn_index}"
        started = time.perf_counter()
        with capture_tool_output_keys() as tool_output_records:
            outcome = service.handle(
                ChatServiceRequest(
                    question=question,
                    top_k=top_k,
                    decompose_top_k=decompose_top_k,
                    session_id=session_id,
                    trace_id=trace_id,
                )
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = dict(outcome.result)
        latest_metrics = captured_metrics[-1] if captured_metrics else {}
        runtime_error = (
            str(result.get("reason") or "internal error")
            if result.get("status") == "INTERNAL_ERROR"
            else ""
        )
        turn_rows.append(
            score_agent_turn(
                case_id=case.id,
                category=case.category,
                suite=case.suite,
                tags=case.tags,
                turn_index=turn_index,
                question=question,
                expected=expected,
                result=result,
                metrics=latest_metrics,
                latency_ms=elapsed_ms,
                runtime_error=runtime_error,
                tool_output_records=tool_output_records,
            )
        )
    session.clear()
    return combine_case_turns(case.id, turn_rows)


def run_agent_baseline(
    dataset: AgentEvaluationDataset,
    *,
    settings: AppSettings,
    include_online: bool,
    top_k: int,
    decompose_top_k: int,
    rag_already_initialized: bool,
    force_rebuild: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """运行 Core 以及显式启用的 Online Agent cases。"""
    selected = [
        case
        for case in dataset.cases
        if not case.disabled and (case.suite == "core" or include_online)
    ]
    skipped_online = sum(
        not case.disabled and case.suite == "online"
        for case in dataset.cases
    )
    if include_online:
        skipped_online = 0
    if not rag_already_initialized:
        init_rag_for_eval(force_rebuild=force_rebuild)
    rows = [
        execute_agent_case(
            case,
            settings=settings,
            top_k=top_k,
            decompose_top_k=decompose_top_k,
        )
        for case in selected
    ]
    metrics = calculate_agent_metrics(
        rows,
        total_cases=len(selected) + skipped_online,
        skipped_cases=skipped_online,
    )
    return rows, metrics


__all__ = [
    "calculate_agent_metrics",
    "capture_tool_output_keys",
    "check_keywords",
    "check_tool_output_keys",
    "combine_case_turns",
    "execute_agent_case",
    "extract_actual_fields",
    "extract_query_rewrite",
    "extract_reference_sources",
    "init_rag_for_eval",
    "run_agent_baseline",
    "score_agent_turn",
]
