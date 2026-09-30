"""比较两次 Evaluation Baseline v2 运行。

用途：定位候选运行相对基线运行的整体回归、分组变化与逐题状态变化。
输入：两份 ``manifest.json``、``baseline_summary.json``、Agent 逐题结果和可选阈值。
输出：``comparison.json``、``comparison.md``；不兼容默认拒绝，也可显式检查回归阈值。
边界：只读取既有评测产物，不重新调用 Agent、RAG、模型或网络服务。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluation.report import write_json


KNOWN_THRESHOLDS = {
    "core_agent_pass_rate_drop",
    "source_pass_rate_drop",
    "p95_latency_increase_ratio",
}


def _load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象并给出明确错误。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到比较输入：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"比较输入必须是 JSON object：{path}")
    return value


def _load_jsonl_by_case(path: Path) -> dict[str, dict[str, Any]]:
    """按 case_id 读取 Agent 结果；未运行 Agent 时返回空字典。"""
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        case_id = str(value.get("case_id") or "")
        if not case_id:
            raise ValueError(
                f"{path}:{line_number} 缺少 case_id"
            )
        rows[case_id] = value
    return rows


def _metric(
    summary: dict[str, Any],
    path: tuple[str, ...],
) -> float | None:
    """读取嵌套数值指标；未运行的评测层或可选指标返回 None。"""
    value: Any = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _category_rates(summary: dict[str, Any]) -> dict[str, float]:
    """读取 Agent category pass rate。"""
    groups = (
        summary.get("agent", {})
        .get("groups", {})
        .get("category", {})
    )
    return {
        str(name): float(metrics.get("pass_rate", 0) or 0)
        for name, metrics in groups.items()
        if isinstance(metrics, dict)
    }


def _metric_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    path: tuple[str, ...],
) -> dict[str, float | None]:
    """返回基线、候选和候选减基线。"""
    before = _metric(baseline, path)
    after = _metric(candidate, path)
    return {
        "baseline": before,
        "candidate": after,
        "delta": (
            round(after - before, 6)
            if before is not None and after is not None
            else None
        ),
    }


def find_manifest_mismatches(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """仅检查实际运行层的实验控制条件，不限制被测代码、Prompt 或工具合同。"""
    if baseline is None or candidate is None:
        return [{"field": "manifest.json", "baseline": "missing" if baseline is None else "present",
                 "candidate": "missing" if candidate is None else "present"}]
    mismatches = []

    def check(field, before, after):
        missing = any(value is None or value == "" for value in (before, after))
        if before != after or missing:
            mismatches.append({"field": field, "baseline": before, "candidate": after})

    def value(manifest, field):
        result = manifest
        for key in field.split("."):
            result = result.get(key) if isinstance(result, dict) else None
        return result

    fields = ["evaluation_type"]
    check("selected_suites", *[
        sorted(set(m["selected_suites"])) if m.get("selected_suites") is not None else None
        for m in (baseline, candidate)
    ])
    check("data.selected_case_ids", *[
        sorted(set(value(m, "data.selected_case_ids") or []))
        or (None if m.get("evaluation_type") == "targeted_agent_check" else [])
        for m in (baseline, candidate)
    ])
    active = {}
    for layer, dataset in (("agent", "agent_cases_sha256"), ("rag", "rag_eval_sha256")):
        before, after = [bool(s.get(layer)) and not s[layer].get("skipped", False)
                         for s in (baseline_summary, candidate_summary)]
        check(f"{layer}.skipped", not before, not after)
        active[layer] = before and after
        if active[layer]:
            fields.append(f"data.{dataset}")
    if active["rag"]:
        fields.extend(["rag.prediction_schema_version", "rag.retrieval_metric_k"])
        if all(_metric(s, ("rag", "overall", "avg_answer_correctness")) is not None
               for s in (baseline_summary, candidate_summary)):
            fields.extend(["judge.enabled", "judge.model"])
    if any(active.values()):
        fields.extend(f"agent.{key}" for key in (
            "agent_model", "agent_temperature", "agent_max_tokens",
            "agent_enable_thinking",
            "answer_model", "answer_temperature", "answer_max_tokens", "budget",
            "query_rewrite_enabled", "query_rewrite_model",
        ))
        # Agent-only 也会检索。无完整 hash 时只比较稳定元数据，不比较存储路径。
        if all(value(m, "rag.index_manifest_sha256") for m in (baseline, candidate)):
            fields.append("rag.index_manifest_sha256")
        else:
            fields.extend(f"rag.{key}" for key in (
                "document_count", "chunk_count", "embedding_model", "rerank_model",
                "chunk_size", "chunk_overlap",
            ))
    for field in fields:
        check(field, value(baseline, field), value(candidate, field))
    return mismatches


def compare_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    allow_incompatible: bool = False,
) -> dict[str, Any]:
    """先验证运行控制条件，再构造差异；强制比较必须在产物中标记。"""
    baseline = _load_json(baseline_dir / "baseline_summary.json")
    candidate = _load_json(candidate_dir / "baseline_summary.json")
    manifests = [
        _load_json(path) if path.is_file() else None
        for path in (baseline_dir / "manifest.json", candidate_dir / "manifest.json")
    ]
    mismatches = find_manifest_mismatches(*manifests, baseline, candidate)
    if mismatches and not allow_incompatible:
        raise ValueError(
            "Incompatible run controls; use --allow-incompatible to force comparison:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )
    before_cases = _load_jsonl_by_case(
        baseline_dir / "agent" / "per_case.jsonl"
    )
    after_cases = _load_jsonl_by_case(
        candidate_dir / "agent" / "per_case.jsonl"
    )
    before_ids = set(before_cases)
    after_ids = set(after_cases)
    common = sorted(before_ids & after_ids)
    new_failures = [
        case_id
        for case_id in common
        if before_cases[case_id].get("passed")
        and not after_cases[case_id].get("passed")
    ]
    fixed_failures = [
        case_id
        for case_id in common
        if not before_cases[case_id].get("passed")
        and after_cases[case_id].get("passed")
    ]
    changed_cases = [
        case_id
        for case_id in common
        if any(
            before_cases[case_id].get(key)
            != after_cases[case_id].get(key)
            for key in (
                "passed",
                "actual_status",
                "actual_action",
                "actual_tools",
                "failure_reasons",
            )
        )
    ]
    metric_paths = {
        "overall_agent_pass_rate": ("agent", "pass_rate"),
        "core_agent_pass_rate": (
            "agent",
            "groups",
            "suite",
            "core",
            "pass_rate",
        ),
        "status_accuracy": ("agent", "status_accuracy"),
        "source_pass_rate": ("agent", "source_pass_rate"),
        "evidence_grounding_pass_rate": ("agent", "evidence_grounding_pass_rate"),
        "p95_latency_ms": ("agent", "p95_latency_ms"),
        "average_llm_call_count": (
            "agent",
            "average_llm_call_count",
        ),
        "average_tool_call_count": (
            "agent",
            "average_tool_call_count",
        ),
        "rag_prediction_coverage": ("rag", "overall", "prediction_coverage"),
        "rag_retrieval_gold_chunk_recall_at_k": ("rag", "overall", "retrieval_gold_chunk_recall_at_k"),
        "reference_gold_chunk_recall": ("rag", "overall", "reference_gold_chunk_recall"),
        "rag_llm_judge_coverage": ("rag", "overall", "llm_judge_coverage"),
        "rag_avg_answer_correctness": (
            "rag",
            "overall",
            "avg_answer_correctness",
        ),
    }
    before_categories = _category_rates(baseline)
    after_categories = _category_rates(candidate)
    category_names = sorted(set(before_categories) | set(after_categories))
    metrics = {name: _metric_delta(baseline, candidate, path) for name, path in metric_paths.items()}
    coverage = metrics["rag_llm_judge_coverage"]
    warnings = []
    if coverage["baseline"] is None or coverage["baseline"] != coverage["candidate"]:
        metrics["rag_avg_answer_correctness"]["delta"] = None
        warnings.append("Judge coverage differs or is unavailable; correctness delta is unavailable.")
    return {
        "schema_version": 2,
        "compatibility": {
            "compatible": not mismatches,
            "overridden": bool(mismatches and allow_incompatible),
            "mismatches": mismatches,
        },
        "baseline_run_id": baseline.get("run_id", baseline_dir.name),
        "candidate_run_id": candidate.get(
            "run_id",
            candidate_dir.name,
        ),
        "case_changes": {
            "added": sorted(after_ids - before_ids),
            "removed": sorted(before_ids - after_ids),
            "new_failures": new_failures,
            "fixed_failures": fixed_failures,
            "changed_cases": changed_cases,
        },
        "metrics": metrics,
        "warnings": warnings,
        "category_pass_rate": {
            name: {
                "baseline": before_categories.get(name),
                "candidate": after_categories.get(name),
                "delta": (
                    round(
                        after_categories[name] - before_categories[name],
                        6,
                    )
                    if name in before_categories
                    and name in after_categories
                    else None
                ),
            }
            for name in category_names
        },
    }


def load_thresholds(path: Path) -> dict[str, float]:
    """加载显式阈值并拒绝未知或负数值。"""
    raw = _load_json(path)
    unknown = sorted(set(raw) - KNOWN_THRESHOLDS)
    if unknown:
        raise ValueError(f"未知阈值：{', '.join(unknown)}")
    thresholds: dict[str, float] = {}
    for name, value in raw.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"阈值 {name} 必须是数字") from exc
        if number < 0:
            raise ValueError(f"阈值 {name} 不能为负数")
        thresholds[name] = number
    return thresholds


def evaluate_regressions(
    comparison: dict[str, Any],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """仅按调用者提供的阈值判断回退。"""
    metrics = comparison["metrics"]
    regressions: list[dict[str, Any]] = []
    mappings = {
        "core_agent_pass_rate_drop": "core_agent_pass_rate",
        "source_pass_rate_drop": "source_pass_rate",
    }
    for threshold_name, metric_name in mappings.items():
        if threshold_name not in thresholds:
            continue
        delta = metrics[metric_name]["delta"]
        if delta is not None and delta < -thresholds[threshold_name]:
            regressions.append(
                {
                    "threshold": threshold_name,
                    "metric": metric_name,
                    "allowed": thresholds[threshold_name],
                    "actual_drop": round(-delta, 6),
                }
            )
    latency_name = "p95_latency_increase_ratio"
    if latency_name in thresholds:
        before = metrics["p95_latency_ms"]["baseline"]
        after = metrics["p95_latency_ms"]["candidate"]
        if before is not None and after is not None and before > 0:
            ratio = (after - before) / before
            if ratio > thresholds[latency_name]:
                regressions.append(
                    {
                        "threshold": latency_name,
                        "metric": "p95_latency_ms",
                        "allowed": thresholds[latency_name],
                        "actual_increase_ratio": round(ratio, 6),
                    }
                )
    return regressions


def render_markdown(comparison: dict[str, Any]) -> str:
    """渲染紧凑的比较报告。"""
    changes = comparison["case_changes"]
    compatibility = comparison["compatibility"]
    lines = [
        "# Evaluation Run Comparison",
        "",
        f"- Baseline: `{comparison['baseline_run_id']}`",
        f"- Candidate: `{comparison['candidate_run_id']}`",
        "",
        "## Compatibility",
        "",
        ("**Comparison was forced across incompatible run controls.**"
         if compatibility["overridden"] else "Run controls are compatible."),
        *[f"- `{item['field']}`: baseline={json.dumps(item['baseline'], ensure_ascii=False)}, "
          f"candidate={json.dumps(item['candidate'], ensure_ascii=False)}"
          for item in compatibility["mismatches"]],
        "",
        "## Case changes",
        "",
        f"- Added: `{', '.join(changes['added']) or 'none'}`",
        f"- Removed: `{', '.join(changes['removed']) or 'none'}`",
        f"- New failures: "
        f"`{', '.join(changes['new_failures']) or 'none'}`",
        f"- Fixed failures: "
        f"`{', '.join(changes['fixed_failures']) or 'none'}`",
        f"- Changed cases: "
        f"`{', '.join(changes['changed_cases']) or 'none'}`",
        "",
        "## Metric deltas",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in comparison["metrics"].items():
        before = (
            values["baseline"]
            if values["baseline"] is not None
            else "unavailable"
        )
        after = (
            values["candidate"]
            if values["candidate"] is not None
            else "unavailable"
        )
        delta = (
            values["delta"]
            if values["delta"] is not None
            else "unavailable"
        )
        lines.append(
            f"| {name} | {before} | {after} | {delta} |"
        )
    lines.append("")
    lines.extend(f"- {warning}" for warning in comparison.get("warnings", []))
    lines.extend(["", "## Threshold regressions", ""])
    regressions = comparison.get("regressions") or []
    if regressions:
        lines.extend(
            f"- `{item['metric']}` exceeded `{item['threshold']}`"
            for item in regressions
        )
    else:
        lines.append("- none (or thresholds not supplied)")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """构造比较 CLI。"""
    parser = argparse.ArgumentParser(
        description="Compare Evaluation Baseline v2 runs",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--allow-incompatible", action="store_true",
                        help="Force comparison across incompatible or missing run controls")
    return parser


def main(argv: list[str] | None = None) -> int:
    """默认拒绝不兼容输入；兼容时仅按显式阈值判断回归。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fail_on_regression and args.thresholds is None:
        parser.error(
            "--fail-on-regression 必须与显式 --thresholds 一起使用"
        )
    try:
        comparison = compare_runs(args.baseline, args.candidate, allow_incompatible=args.allow_incompatible)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    thresholds = (
        load_thresholds(args.thresholds)
        if args.thresholds is not None
        else {}
    )
    regressions = evaluate_regressions(comparison, thresholds)
    comparison["thresholds"] = thresholds
    comparison["regressions"] = regressions
    output_dir = args.output_dir or args.candidate
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", comparison)
    (output_dir / "comparison.md").write_text(
        render_markdown(comparison),
        encoding="utf-8",
    )
    print(f"Comparison written to: {output_dir}")
    if args.fail_on_regression and regressions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
