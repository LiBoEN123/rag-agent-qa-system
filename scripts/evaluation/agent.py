"""Agent Evaluation v2 的执行、最终结果判分和指标汇总。

用途：逐条运行单轮或多轮 Agent case，并核对 action、工具、状态和答案合同。
输入：已校验的 Agent 数据集、评测专用 Settings，以及 Core/Online suite 选择。
输出：逐 case 明细和按 suite、category 聚合的确定性指标。
边界：通过正式 ``ChatService`` 调用链执行；case 间隔离运行态，不写生产持久化数据。
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import AppSettings
from app.contracts.chat import RAG_EVIDENCE_TOOL_NAMES
from app.services.chat_models import ChatServiceOutcome, ChatServiceRequest
from scripts.evaluation.contracts import (
    AgentEvaluationCase,
    AgentEvaluationDataset,
    ExpectedOutcome,
)
from scripts.evaluation.deepeval_adapter import tool_correctness


def safe_lower(text: Any) -> str:
    """安全地把任意值转换成 casefold 后的小写文本。"""
    return "" if text is None else str(text).casefold()


def init_rag_for_eval(force_rebuild: bool = False) -> None:
    """读取已有索引；仅显式 force 参数允许写入。"""
    from app.config import get_settings, validate_startup_environment
    from app.rag.runtime import get_rag_status, load_index_cache

    settings = get_settings()
    validate_startup_environment(settings)
    if force_rebuild:
        from app.indexing.service import IndexingService

        IndexingService(settings=settings).sync_local_corpus(force_rebuild=True)
    ok, message = load_index_cache()
    print(f"[init] load index, ok={ok}, msg={message}")
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


def check_keywords(
    expected: ExpectedOutcome,
    result: dict[str, Any],
) -> dict[str, Any]:
    """检查最终答案是否命中评测用例要求的关键词。"""
    keywords = expected.keywords
    search_text = safe_lower(result.get("answer"))
    hits = [item for item in keywords if safe_lower(item) in search_text]
    missing = [item for item in keywords if item not in hits]
    required = expected.min_keyword_hits
    return {
        "keyword_pass": len(hits) >= required,
        "keyword_hits": hits,
        "keyword_missing": missing,
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
    """捕获工具字段名与 KB 标识；不保存正文、参数或外部来源内容。"""
    from app.agent import framework_tools

    records: list[dict[str, Any]] = []
    original_execute = framework_tools.execute_project_tool

    async def execute_with_schema_capture(
        *,
        spec,
        tool_args: dict,
        **kwargs,
    ) -> dict:
        """调用异步本地执行入口，只记录输出结构。"""
        result = await original_execute(
            spec=spec,
            tool_args=tool_args,
            **kwargs,
        )
        output = result.get("output") if isinstance(result, dict) else None
        records.append(
            {
                "tool_name": spec.name,
                "tool_call_id": str(result.get("tool_call_id") or ""),
                "has_evidence": isinstance(output, dict) and "evidence" in output,
                "kb_evidence": [
                    {
                        "reference_id": document["id"],
                        "file_name": document["metadata"].get("file_name", ""),
                        "source_path": document["metadata"].get("source_path", ""),
                    }
                    for document in (output.get("evidence") or [])
                    if isinstance(document, dict)
                    and document.get("id")
                    and str(document.get("page_content") or "").strip()
                    and isinstance(document.get("metadata"), dict)
                    and document["metadata"].get("source_type") == "knowledge_base"
                ] if isinstance(output, dict) and result.get("ok") is True
                and spec.name in RAG_EVIDENCE_TOOL_NAMES else [],
                "kb_chunk_ids": [
                    int(value) if not isinstance(value := document["metadata"].get("chunk_id"), bool)
                    and str(value).isascii() and str(value).isdigit() else None
                    for document in (output.get("evidence") or [])
                    if isinstance(document, dict)
                    and isinstance(document.get("metadata"), dict)
                    and document["metadata"].get("source_type") == "knowledge_base"
                ] if isinstance(output, dict) and result.get("ok") is True else [],
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

    framework_tools.execute_project_tool = execute_with_schema_capture
    try:
        yield records
    finally:
        framework_tools.execute_project_tool = original_execute


def _rate(numerator: int, denominator: int) -> float:
    """安全计算四位小数比率。"""
    return round(numerator / denominator, 4) if denominator else 0.0


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


def _tool_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    """返回按 trace 顺序排列的真实工具调用。"""
    return [
        step
        for step in (result.get("tool_trace") or [])
        if isinstance(step, dict) and step.get("step") == "tool_call"
    ]


def _check_evidence_grounding(
    expected: ExpectedOutcome,
    result: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """引用须来自本轮成功 KB 正文输出，且真实证据来源满足期望。"""
    if not expected.sources:
        return {"evidence_grounding_pass": True}
    successful_calls = {
        (call.get("tool_call_id"), call.get("tool_name"))
        for call in _tool_calls(result)
        if call.get("tool_call_id") and call.get("tool_status") == "success"
    }
    successful = [
        record
        for record in records
        if record.get("ok") is True and record.get("status") == "success"
        and record.get("tool_name") in RAG_EVIDENCE_TOOL_NAMES
        and (record.get("tool_call_id"), record.get("tool_name")) in successful_calls
    ]
    evidence = {
        item["reference_id"]: item
        for record in successful
        for item in record.get("kb_evidence", [])
    }
    references = result.get("references") or []
    cited_ids = [item.get("reference_id") for item in references if isinstance(item, dict)]
    grounded = bool(
        cited_ids and len(cited_ids) == len(references)
        and all(isinstance(value, str) and value in evidence for value in cited_ids)
        and _check_sources(expected, {"references": [evidence[value] for value in cited_ids]})["source_pass"]
    )
    return {
        "evidence_grounding_pass": grounded,
        "evidence_observations": [
            {"tool_name": record["tool_name"], "tool_call_id": record["tool_call_id"],
             "evidence": record["kb_evidence"]}
            for record in successful if record.get("kb_evidence")
        ],
    }


def _check_tool_output_schema(
    expected: ExpectedOutcome,
    tool_output_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """只检查评测边界捕获的成功工具输出字段名。"""
    expected_tools = expected.expected_tools()
    actual_keys = {
        str(key)
        for record in tool_output_records
        if isinstance(record, dict)
        and record.get("ok") is True
        and record.get("status") == "success"
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
    """结合 DeepEval 工具判分与项目确定性合同，保留 v2 多原因诊断。"""
    metrics = dict(metrics or {})
    calls = _tool_calls(result)
    actual_tools = [str(call["tool_name"]) for call in calls if call.get("tool_name")]
    actual_tool_statuses = [
        str(step.get("tool_status") or "") for step in calls
    ]
    keyword = check_keywords(expected, result)
    source = _check_sources(expected, result)
    forbidden = _check_forbidden(expected, result)
    tool_output = _check_tool_output_schema(expected, tool_output_records or [])
    grounding = _check_evidence_grounding(expected, result, tool_output_records or [])

    checks = {
        "action": (
            not expected.action
            or result.get("action", "") == expected.action
        ),
        "status": (
            expected.status is None
            or result.get("status", "") == expected.status.value
        ),
        "evidence_grounding": grounding["evidence_grounding_pass"],
        "tools_contains": tool_correctness(
            expected.tools_contains, actual_tools,
        ),
        "tool_sequence": tool_correctness(
            expected.tool_sequence, actual_tools, exact=True,
        ),
        "tool_status_sequence": (
            not expected.tool_status_sequence
            or actual_tool_statuses == expected.tool_status_sequence
        ),
        "keyword": keyword["keyword_pass"],
        "source": source["source_pass"],
        "forbidden": forbidden["forbidden_pass"],
        "tool_output_schema": tool_output["tool_output_keys_pass"],
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
    if not checks["action"]:
        failure_reasons.append("action_mismatch")
    if not checks["status"]:
        failure_reasons.append("status_mismatch")
        if result.get("status", "") == "NEED_CLARIFICATION":
            failure_reasons.append("unexpected_clarification")
        elif result.get("status", "") == "NO_RELEVANT_INFO":
            failure_reasons.append("unexpected_no_evidence")
    if not checks["evidence_grounding"]:
        failure_reasons.append("evidence_grounding_mismatch")
    if not checks["tools_contains"]:
        failure_reasons.append("required_tool_missing")
    if not checks["tool_sequence"]:
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
        "actual_action": result.get("action", ""),
        "actual_status": result.get("status", ""),
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
        **grounding,
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
    """通过率按 case；状态、来源和 grounding 按适用 turn；成本按 case。"""
    executed = len(rows)
    passed = sum(row["passed"] for row in rows)
    turns = [turn for row in rows for turn in row.get("turns", [row])]
    status_turns = [turn for turn in turns if turn["expected"].get("status")]
    source_turns = [turn for turn in turns if turn["expected"].get("sources")]
    tool_calls = sum(int(row["tool_call_count"]) for row in rows)
    successful_tool_calls = sum(int(row["successful_tool_call_count"]) for row in rows)
    by_dimension: dict[str, dict[str, Any]] = {}
    for dimension in ("suite", "category"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row[dimension]].append(row)
        by_dimension[dimension] = {
            name: _group_metrics(items)
            for name, items in sorted(grouped.items())
        }
    return {
        "total_cases": executed + skipped_cases if total_cases is None else total_cases,
        "executed_cases": executed,
        "skipped_cases": skipped_cases,
        "passed_cases": passed,
        "failed_cases": executed - passed,
        "pass_rate": _rate(passed, executed),
        "status_accuracy": _rate(
            sum(turn["actual_status"] == turn["expected"]["status"] for turn in status_turns),
            len(status_turns),
        ),
        "source_pass_rate": _rate(
            sum(turn["source_pass"] for turn in source_turns), len(source_turns),
        ),
        "evidence_grounding_pass_rate": _rate(
            sum(turn["evidence_grounding_pass"] for turn in source_turns), len(source_turns),
        ),
        "tool_call_success_rate": _rate(successful_tool_calls, tool_calls),
        "average_tool_call_count": round(tool_calls / executed, 4) if executed else 0.0,
        "average_llm_call_count": round(
            statistics.mean(int(row["llm_call_count"]) for row in rows), 4,
        ) if rows else 0.0,
        "p95_latency_ms": _percentile([float(row["latency_ms"]) for row in rows], 0.95),
        "failure_reason_counts": dict(sorted(Counter(
            reason for row in rows for reason in row["failure_reasons"]
        ).items())),
        "groups": by_dimension,
    }


def _permissions_for_suite(
    suite: str,
) -> tuple[str, ...]:
    """Core 只读知识库，Online 额外开放 Web 权限。"""
    permissions = ["knowledge_base:read"]
    if suite == "online":
        permissions.extend(["web:search", "web:fetch"])
    return tuple(permissions)


def run_evaluation_chat_turn(
    request: ChatServiceRequest,
    *,
    settings: AppSettings,
    permissions: tuple[str, ...],
    session: InMemorySessionAdapter | None = None,
    captured_metrics: list[dict[str, Any]] | None = None,
) -> ChatServiceOutcome:
    """复用正式 ChatService，隔离会话、指标写入与评测权限。"""
    from dataclasses import replace

    from app.agent.streaming import stream_agent
    from app.services.chat_service import (
        ChatService,
        get_default_chat_service_dependencies,
    )
    from app.tools.context import PrincipalType, ToolInvocationContext

    session = session or InMemorySessionAdapter()
    captured_metrics = captured_metrics if captured_metrics is not None else []
    base = get_default_chat_service_dependencies()

    async def run_runtime(**kwargs: Any):
        """为真实 Agent 运行时注入评测专用身份与权限上下文。"""
        kwargs["tool_context"] = ToolInvocationContext(
            trace_id=str(kwargs.get("trace_id") or ""),
            session_id=str(kwargs.get("session_id") or ""),
            principal_id="evaluation_baseline_v2",
            principal_type=PrincipalType.SERVICE,
            permissions=permissions,
            tool_mode=request.tool_mode,
            source="evaluation_baseline_v2",
        )
        async for event in stream_agent(**kwargs):
            yield event

    dependencies = replace(
        base,
        settings_provider=lambda: settings,
        get_history=session.get_history,
        stream_agent=run_runtime,
        log_and_write_metrics=lambda item: captured_metrics.append(dict(item)),
        write_bad_cases=lambda item: None,
        append_message=session.append_message,
    )
    service = ChatService(dependencies)

    async def collect_turn():
        outcome = None
        async for event in service.stream(request):
            if event.event == "final":
                outcome = event.data["outcome"]
        if outcome is None:
            raise RuntimeError("chat stream ended without final")
        return outcome

    return asyncio.run(collect_turn())


def execute_agent_case(
    case: AgentEvaluationCase,
    *,
    settings: AppSettings,
) -> dict[str, Any]:
    """通过真实 ChatService 执行一个完全隔离的 Agent case。"""
    session = InMemorySessionAdapter()
    captured_metrics: list[dict[str, Any]] = []
    permissions = _permissions_for_suite(case.suite)
    session_id = case.session_id.strip() or f"evaluation-{case.id}"
    turn_rows: list[dict[str, Any]] = []
    for turn_index, (question, expected) in enumerate(
        zip(case.iter_questions(), case.iter_expected(), strict=True),
        start=1,
    ):
        trace_id = f"evaluation-{case.id}-turn-{turn_index}"
        started = time.perf_counter()
        with capture_tool_output_keys() as tool_output_records:
            outcome = run_evaluation_chat_turn(
                ChatServiceRequest(
                    question=question, session_id=session_id, trace_id=trace_id,
                ),
                settings=settings,
                permissions=permissions,
                session=session,
                captured_metrics=captured_metrics,
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
    rag_already_initialized: bool,
    force_rebuild: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """运行 Core 以及由独立 CLI 开关显式启用的外部 suite。"""
    selected_suites = {"core"}
    if include_online:
        selected_suites.add("online")
    selected = [
        case
        for case in dataset.cases
        if not case.disabled and case.suite in selected_suites
    ]
    skipped_cases = sum(
        not case.disabled and case.suite not in selected_suites
        for case in dataset.cases
    )
    if not rag_already_initialized:
        init_rag_for_eval(force_rebuild=force_rebuild)
    rows = [
        execute_agent_case(
            case,
            settings=settings,
        )
        for case in selected
    ]
    metrics = calculate_agent_metrics(
        rows,
        total_cases=len(selected) + skipped_cases,
        skipped_cases=skipped_cases,
    )
    return rows, metrics


__all__ = [
    "calculate_agent_metrics",
    "capture_tool_output_keys",
    "run_evaluation_chat_turn",
    "check_keywords",
    "combine_case_turns",
    "execute_agent_case",
    "extract_reference_sources",
    "init_rag_for_eval",
    "run_agent_baseline",
    "score_agent_turn",
]
