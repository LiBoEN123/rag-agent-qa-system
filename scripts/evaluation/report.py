"""Evaluation Baseline v2 的统一文件与 Markdown 报告输出。

用途：把执行器已经计算完成的结果稳定序列化，供机器比较和人工复核。
输入：逐题 Agent/RAG 记录、汇总指标、运行清单与失败分类。
输出：UTF-8 JSON/JSONL、带 BOM 的 CSV 和统一 Markdown 报告。
边界：本模块不重新判分、不调用模型或网络；字段名和数值按上游结果原样保留。
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    """写入稳定、可读的 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入 UTF-8 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def write_agent_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """写入适合人工筛选的 Agent case 摘要 CSV。"""
    fieldnames = [
        "case_id",
        "suite",
        "category",
        "passed",
        "failure_reasons",
        "expected_status",
        "actual_status",
        "expected_first_tool",
        "actual_first_tool",
        "expected_tool_origin",
        "actual_tool_origin",
        "expected_mcp_server",
        "actual_mcp_server",
        "classification",
        "unnecessary_tool_call_count",
        "actual_tools",
        "tool_call_count",
        "llm_call_count",
        "latency_ms",
        "question",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            expected = row.get("expected") or {}
            writer.writerow(
                {
                    "case_id": row.get("case_id", ""),
                    "suite": row.get("suite", ""),
                    "category": row.get("category", ""),
                    "passed": row.get("passed", False),
                    "failure_reasons": " | ".join(
                        row.get("failure_reasons") or []
                    ),
                    "expected_status": expected.get("status") or "",
                    "actual_status": row.get("actual_status", ""),
                    "expected_first_tool": (
                        expected.get("first_tool") or ""
                    ),
                    "actual_first_tool": row.get(
                        "actual_first_tool",
                        "",
                    ),
                    "expected_tool_origin": row.get(
                        "expected_tool_origin",
                        "",
                    ),
                    "actual_tool_origin": row.get(
                        "actual_tool_origin",
                        "",
                    ),
                    "expected_mcp_server": row.get(
                        "expected_mcp_server",
                        "",
                    ),
                    "actual_mcp_server": row.get(
                        "actual_mcp_server",
                        "",
                    ),
                    "classification": row.get("classification", ""),
                    "unnecessary_tool_call_count": row.get(
                        "unnecessary_tool_call_count",
                        0,
                    ),
                    "actual_tools": " -> ".join(
                        row.get("actual_tools") or []
                    ),
                    "tool_call_count": row.get("tool_call_count", 0),
                    "llm_call_count": row.get("llm_call_count", 0),
                    "latency_ms": row.get("latency_ms", 0),
                    "question": str(row.get("question") or "")[:300],
                }
            )


def classify_rag_failures(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按现有 RAG evaluator 字段生成诊断性失败分类。"""
    failures: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        if not row.get("has_prediction"):
            reasons.append("runtime_error")
        if (
            row.get("expected_behavior") != "abstain"
            and float(row.get("retrieval_doc_hit_at_k", 0) or 0) == 0
        ):
            reasons.append("rag_document_miss")
        if (
            row.get("gold_chunk_ids") not in ("", "[]", None)
            and float(
                row.get("retrieval_gold_chunk_hit_at_k", 0) or 0
            )
            == 0
        ):
            reasons.append("rag_chunk_miss")
        if row.get(
            "gold_present_in_final_but_missing_in_context"
        ) is True:
            reasons.append("rag_context_gold_miss")
        if row.get(
            "gold_present_in_context_but_not_referenced"
        ) is True or row.get("gold_observed_but_not_referenced") is True:
            reasons.append("rag_reference_gold_miss")
        if (
            row.get("key_point_judge_used")
            and int(row.get("missing_count", 0) or 0) > 0
        ):
            reasons.append("rag_answer_keypoint_miss")
        behavior = str(row.get("behavior_label") or "")
        if behavior == "should_answer_but_refused":
            reasons.append("rag_unexpected_refusal")
        if behavior == "should_abstain_but_answered":
            reasons.append("rag_missing_refusal")
        if reasons:
            failures.append(
                {
                    "kind": "rag",
                    "case_id": str(row.get("qid") or ""),
                    "category": str(row.get("category") or ""),
                    "question": str(row.get("question") or ""),
                    "failure_reasons": list(dict.fromkeys(reasons)),
                    "metrics": {
                        "doc_hit_at_k": row.get(
                            "retrieval_doc_hit_at_k",
                            0,
                        ),
                        "gold_chunk_hit_at_k": row.get(
                            "retrieval_gold_chunk_hit_at_k",
                            0,
                        ),
                        "context_gold_chunk_hit": row.get(
                            "context_gold_chunk_hit"
                        ),
                        "reference_gold_chunk_hit": row.get(
                            "reference_gold_chunk_hit"
                        ),
                        "key_point_recall": row.get(
                            "key_point_recall",
                            0,
                        ),
                    },
                }
            )
    return failures


def _format_rate(value: Any) -> str:
    """把 0..1 指标渲染成百分比。"""
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _format_number(value: Any) -> str:
    """渲染计数或均值；不可用指标不伪造为零。"""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return str(int(number)) if number.is_integer() else f"{number:.2f}"


def build_markdown_report(
    *,
    manifest: dict[str, Any],
    coverage: dict[str, Any],
    summary: dict[str, Any],
    failures: list[dict[str, Any]],
    reproduce_command: str,
) -> str:
    """生成统一、脱敏的 Markdown 报告。"""
    code = manifest["code"]
    data = manifest["data"]
    agent = summary.get("agent")
    rag = summary.get("rag")
    failure_counts = Counter(
        reason
        for item in failures
        for reason in item.get("failure_reasons", [])
    )
    lines = [
        "# Evaluation Baseline v2",
        "",
        "## 运行摘要",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Git commit: `{code['git_commit']}`",
        f"- Git branch: `{code['git_branch']}`",
        f"- Dirty worktree: `{str(code['git_dirty']).lower()}`",
        f"- Evaluation type: "
        f"`{manifest.get('evaluation_type', 'core_baseline')}`",
        f"- Selected suites: "
        f"`{json.dumps(manifest.get('selected_suites', []), ensure_ascii=False)}`",
        f"- External network enabled: "
        f"`{str(manifest.get('external_network_enabled', False)).lower()}`",
        "",
        "## 代码与环境",
        "",
        f"- Python: `{code['python_version']}`",
        f"- Platform: `{code['platform']}`",
        "",
        "## 数据集",
        "",
        f"- Agent cases: `{data['agent_cases_path']}`",
        f"- Agent SHA-256: `{data['agent_cases_sha256']}`",
        f"- RAG eval: `{data['rag_eval_path']}`",
        f"- RAG SHA-256: `{data['rag_eval_sha256']}`",
        "",
        "## 索引与模型",
        "",
        f"- Index source: `{manifest['rag']['index_source']}`",
        f"- Documents/chunks: `{manifest['rag']['document_count']}` / "
        f"`{manifest['rag']['chunk_count']}`",
        f"- Embedding: `{manifest['rag']['embedding_model']}`",
        f"- Rerank: `{manifest['rag']['rerank_model']}`",
        f"- Answer model: `{manifest['agent']['answer_model']}`",
        "",
        "## Coverage Matrix",
        "",
        f"- Suite: `{json.dumps(coverage['suite'], ensure_ascii=False)}`",
        f"- Category: `{json.dumps(coverage['category'], ensure_ascii=False)}`",
        f"- Expected tools: "
        f"`{json.dumps(coverage['expected_tools'], ensure_ascii=False)}`",
        f"- Gaps: `{', '.join(coverage['gaps']) or 'none'}`",
        "",
        "## Core Agent 指标",
        "",
    ]
    if agent and not agent.get("skipped"):
        core = (
            agent.get("groups", {})
            .get("suite", {})
            .get("core", {})
        )
        if core:
            lines.extend(
                [
                    f"- Pass rate: {_format_rate(core.get('pass_rate', 0))}",
                    f"- First tool accuracy: "
                    f"{_format_rate(agent.get('first_tool_accuracy', 0))}",
                    f"- Status accuracy: "
                    f"{_format_rate(agent.get('status_accuracy', 0))}",
                    f"- Source pass rate: "
                    f"{_format_rate(agent.get('source_pass_rate', 0))}",
                ]
            )
        else:
            lines.append("- skipped: 当前数据集未选择 Core case")
    else:
        lines.append(f"- skipped: {agent.get('reason', '') if agent else 'not run'}")
    lines.extend(["", "## Online Agent 指标", ""])
    online = (
        (agent or {}).get("groups", {})
        .get("suite", {})
        .get("online")
    )
    if online:
        lines.append(
            f"- Pass rate: {_format_rate(online.get('pass_rate', 0))}"
        )
    else:
        lines.append("- skipped: `--include-online` 未启用")
    lines.extend(["", "## Extended Tool 指标", ""])
    extended_suites = {
        name: (
            (agent or {}).get("groups", {})
            .get("suite", {})
            .get(name)
        )
        for name in ("mcp", "mixed")
    }
    if any(extended_suites.values()):
        for name, values in extended_suites.items():
            if values:
                lines.append(
                    f"- {name} pass rate: "
                    f"{_format_rate(values.get('pass_rate', 0))}"
                )
        for name in (
            "tool_origin_accuracy",
            "mcp_server_accuracy",
            "external_tool_success_rate",
            "mcp_tool_success_rate",
            "web_tool_success_rate",
            "permission_boundary_pass_rate",
            "approval_gate_pass_rate",
            "recovery_pass_rate",
            "cross_tool_selection_accuracy",
        ):
            lines.append(
                f"- {name}: {_format_rate((agent or {}).get(name, 0))}"
            )
        lines.extend(["", "### Cross-Tool Confusion", ""])
        confusion = (agent or {}).get("cross_tool_confusion", {})
        lines.extend(
            f"- `{route}`: {count}"
            for route, count in confusion.items()
        )
        if not confusion:
            lines.append("- 无可统计路由")
    else:
        lines.append("- skipped: MCP/Mixed 显式开关未启用")
    lines.extend(["", "## RAG 指标", ""])
    if rag and not rag.get("skipped"):
        overall = rag.get("overall", {})
        for name in (
            "prediction_coverage",
            "avg_final_score",
            "avg_behavior_score",
            "avg_key_point_recall",
            "gold_chunk_hit_at_k",
            "gold_chunk_recall_at_k",
            "all_gold_hit_at_k",
            "doc_hit_at_k",
            "forbidden_rate",
            "llm_judge_coverage",
        ):
            if name in overall:
                lines.append(f"- {name}: {_format_rate(overall[name])}")
        lines.append(
            "- 口径来源：`scripts/evaluation/rag/evaluation.py`；"
            "答案关键点与 forbidden 语义判断仅在 Judge 启用时有效。"
        )
        if summary.get("skipped", {}).get("llm_judge"):
            lines.append(
                "- LLM Judge skipped: `--use-llm-judge` 未启用"
            )
        if manifest.get("rag", {}).get("prediction_schema_version", 1) >= 3:
            lines.extend([
                "",
                "Prediction Schema v3 is a new baseline generation contract.",
                "Raw scores are not strictly comparable to historical v2 workflow scores.",
                "",
                "| Evidence | Gold hit | Gold recall | Doc hit | Average chunks |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| Observed KB evidence @K | {_format_rate(overall.get('gold_chunk_hit_at_k'))} | "
                f"{_format_rate(overall.get('gold_chunk_recall_at_k'))} | "
                f"{_format_rate(overall.get('doc_hit_at_k'))} | "
                f"{_format_number(overall.get('avg_retrieved_chunk_count'))} |",
                f"| Grounded references | {_format_rate(overall.get('reference_gold_chunk_hit'))} | "
                f"{_format_rate(overall.get('reference_gold_chunk_recall'))} | "
                f"{_format_rate(overall.get('reference_doc_hit'))} | "
                f"{_format_number(overall.get('avg_reference_chunk_count'))} |",
                "",
                "- retrieved_chunk_ids：成功 KB 工具输出按观察顺序稳定去重；平均数量为完整观察数量。",
                "- retrieval-k 仅对观察到的前 K 项评分，不控制 Agent 工具 top_k。",
                "- Context metrics: unavailable / not applicable；不计为零。",
                "- Final Score 权重不变；Reference 指标仅用于诊断。",
            ])
        else:
            lines.extend(
                [
                    "",
                    "## RAG Evidence Funnel",
                    "",
                    "Final-ranked evidence → Prompt context evidence → "
                    "Final cited evidence",
                    "",
                    "| Evidence stage | Gold hit | Gold recall | Doc hit | "
                    "Average chunks | Lost cases |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                    "| Final-ranked Recall@k | "
                    f"{_format_rate(overall.get('gold_chunk_hit_at_k'))} | "
                    f"{_format_rate(overall.get('gold_chunk_recall_at_k'))} | "
                    f"{_format_rate(overall.get('doc_hit_at_k'))} | "
                    f"{_format_number(overall.get('avg_final_chunk_count'))} | "
                    "n/a |",
                    "| Prompt context | "
                    f"{_format_rate(overall.get('context_gold_chunk_hit'))} | "
                    f"{_format_rate(overall.get('context_gold_chunk_recall'))} | "
                    f"{_format_rate(overall.get('context_doc_hit'))} | "
                    f"{_format_number(overall.get('avg_context_chunk_count'))} | "
                    f"{_format_number(overall.get('gold_lost_by_context_count'))} |",
                    "| Final cited | "
                    f"{_format_rate(overall.get('reference_gold_chunk_hit'))} | "
                    f"{_format_rate(overall.get('reference_gold_chunk_recall'))} | "
                    f"{_format_rate(overall.get('reference_doc_hit'))} | "
                    f"{_format_number(overall.get('avg_reference_chunk_count'))} | "
                    f"{_format_number(overall.get('gold_lost_by_reference_count'))} |",
                    "",
                    "- Retrieval Recall@k 只评价最终排序 chunks 的前 k 个，"
                    "不等于 Prompt Context Recall。",
                    "- Prompt Context Recall 评价字符预算后进入回答上下文的 chunks，"
                    "不等于最终 Reference Recall。",
                    "- Context 与 Reference 指标是诊断指标，"
                    "不参与 Evaluation Baseline v2 Final Score。",
                ]
            )
    else:
        lines.append(f"- skipped: {rag.get('reason', '') if rag else 'not run'}")
    lines.extend(
        [
            "",
            "## 多轮 Query Rewrite 指标",
            "",
            f"- Multi-turn pass rate: "
            f"{_format_rate((agent or {}).get('multi_turn_pass_rate', 0))}",
            "",
            "## 延迟与调用成本",
            "",
            f"- P50/P95 latency: "
            f"`{(agent or {}).get('p50_latency_ms', 0)}` / "
            f"`{(agent or {}).get('p95_latency_ms', 0)}` ms",
            f"- Average LLM calls: "
            f"`{(agent or {}).get('average_llm_call_count', 0)}`",
            f"- Average tool calls: "
            f"`{(agent or {}).get('average_tool_call_count', 0)}`",
            "",
            "## 失败类型分布",
            "",
        ]
    )
    if failure_counts:
        lines.extend(
            f"- `{name}`: {count}"
            for name, count in failure_counts.most_common()
        )
    else:
        lines.append("- 无失败")
    lines.extend(["", "## 失败 case 列表", ""])
    if failures:
        lines.extend(
            [
                "| case_id | category | question | expected | actual | class | reasons | latency |",
                "| --- | --- | --- | --- | --- | --- | --- | ---: |",
            ]
        )
        for item in failures[:100]:
            expected = item.get("expected") or {}
            lines.append(
                "| {case} | {category} | {question} | {expected} | "
                "{actual} | {classification} | {reasons} | {latency} |".format(
                    case=str(item.get("case_id") or "").replace("|", "\\|"),
                    category=str(item.get("category") or "").replace("|", "\\|"),
                    question=str(item.get("question") or "")[:100].replace(
                        "|",
                        "\\|",
                    ),
                    expected=(
                        f"{expected.get('status') or ''}/"
                        f"{expected.get('first_tool') or ''}"
                    ),
                    actual=(
                        f"{item.get('actual_status') or ''}/"
                        f"{','.join(item.get('actual_tools') or [])}"
                    ),
                    classification=item.get("classification", ""),
                    reasons=",".join(item.get("failure_reasons") or []),
                    latency=item.get("latency_ms", ""),
                )
            )
    else:
        lines.append("- 无失败 case")
    lines.extend(
        [
            "",
            "## 已知限制",
            "",
            "- Online 工具结果随网络和外部内容变化，不纳入 Core pass rate。",
            "- LLM Judge 只辅助答案语义质量，不替代确定性合同检查。",
            "- 工作区为 dirty 时只能通过 `--allow-dirty` 生成实验结果。",
            "",
            "## 复现命令",
            "",
            "```bash",
            reproduce_command,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "build_markdown_report",
    "classify_rag_failures",
    "write_agent_csv",
    "write_json",
    "write_jsonl",
]
