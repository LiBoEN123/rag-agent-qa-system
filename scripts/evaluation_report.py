"""Evaluation Baseline v2 的统一文件与 Markdown 报告输出。"""

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
            "- 口径来源：`scripts/rag_eval/evaluation.py`；"
            "答案关键点与 forbidden 语义判断仅在 Judge 启用时有效。"
        )
        if summary.get("skipped", {}).get("llm_judge"):
            lines.append(
                "- LLM Judge skipped: `--use-llm-judge` 未启用"
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
                "| case_id | category | question | expected | actual | reasons | latency |",
                "| --- | --- | --- | --- | --- | --- | ---: |",
            ]
        )
        for item in failures[:100]:
            expected = item.get("expected") or {}
            lines.append(
                "| {case} | {category} | {question} | {expected} | "
                "{actual} | {reasons} | {latency} |".format(
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
