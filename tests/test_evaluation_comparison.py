"""Evaluation baseline 比较工具测试。"""

import json
from pathlib import Path

import pytest

from scripts.evaluation.comparison import (
    compare_runs,
    evaluate_regressions,
    load_thresholds,
    main,
)


def _write_run(
    path: Path,
    *,
    cases: list[dict],
    pass_rate: float,
    first_tool: float,
    source: float,
    p95: float,
):
    path.mkdir(parents=True)
    summary = {
        "run_id": path.name,
        "agent": {
            "pass_rate": pass_rate,
            "first_tool_accuracy": first_tool,
            "status_accuracy": pass_rate,
            "source_pass_rate": source,
            "p50_latency_ms": p95 / 2,
            "p95_latency_ms": p95,
            "average_llm_call_count": 2,
            "average_tool_call_count": 1,
            "groups": {
                "suite": {
                    "core": {
                        "pass_rate": pass_rate,
                    }
                },
                "category": {
                    "tool": {
                        "pass_rate": pass_rate,
                    }
                },
            },
        },
        "rag": {
            "overall": {
                "gold_chunk_hit_at_k": 0.8,
                "gold_chunk_recall_at_k": 0.7,
                "doc_hit_at_k": 0.9,
                "avg_final_score": 0.75,
            }
        },
    }
    (path / "baseline_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    agent_dir = path / "agent"
    agent_dir.mkdir()
    with (agent_dir / "per_case.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        for case in cases:
            file.write(json.dumps(case) + "\n")


def test_compare_detects_added_removed_new_and_fixed_failures(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        cases=[
            {"case_id": "fixed", "passed": False},
            {"case_id": "regressed", "passed": True},
            {"case_id": "removed", "passed": True},
        ],
        pass_rate=0.8,
        first_tool=0.9,
        source=0.9,
        p95=100,
    )
    _write_run(
        candidate,
        cases=[
            {"case_id": "fixed", "passed": True},
            {"case_id": "regressed", "passed": False},
            {"case_id": "added", "passed": True},
        ],
        pass_rate=0.7,
        first_tool=0.8,
        source=0.85,
        p95=140,
    )

    comparison = compare_runs(baseline, candidate)

    assert comparison["case_changes"]["added"] == ["added"]
    assert comparison["case_changes"]["removed"] == ["removed"]
    assert comparison["case_changes"]["new_failures"] == ["regressed"]
    assert comparison["case_changes"]["fixed_failures"] == ["fixed"]
    assert comparison["metrics"]["core_agent_pass_rate"]["delta"] == -0.1
    assert comparison["category_pass_rate"]["tool"]["delta"] == -0.1


def test_explicit_thresholds_control_regression_exit(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        cases=[{"case_id": "one", "passed": True}],
        pass_rate=0.9,
        first_tool=0.9,
        source=0.9,
        p95=100,
    )
    _write_run(
        candidate,
        cases=[{"case_id": "one", "passed": False}],
        pass_rate=0.8,
        first_tool=0.8,
        source=0.8,
        p95=150,
    )
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "core_agent_pass_rate_drop": 0.05,
                "p95_latency_increase_ratio": 0.2,
            }
        ),
        encoding="utf-8",
    )
    comparison = compare_runs(baseline, candidate)
    thresholds = load_thresholds(thresholds_path)

    regressions = evaluate_regressions(comparison, thresholds)

    assert {item["metric"] for item in regressions} == {
        "core_agent_pass_rate",
        "p95_latency_ms",
    }
    assert (
        main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--thresholds",
                str(thresholds_path),
                "--fail-on-regression",
            ]
        )
        == 1
    )
    assert (candidate / "comparison.json").is_file()
    assert (candidate / "comparison.md").is_file()


def test_default_comparison_reports_without_failure_exit(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        cases=[],
        pass_rate=1,
        first_tool=1,
        source=1,
        p95=10,
    )
    _write_run(
        candidate,
        cases=[],
        pass_rate=0,
        first_tool=0,
        source=0,
        p95=1000,
    )

    assert (
        main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
            ]
        )
        == 0
    )


def test_unknown_or_negative_threshold_is_rejected(tmp_path):
    unknown = tmp_path / "unknown.json"
    unknown.write_text('{"unknown": 1}', encoding="utf-8")
    negative = tmp_path / "negative.json"
    negative.write_text(
        '{"core_agent_pass_rate_drop": -1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="未知阈值"):
        load_thresholds(unknown)
    with pytest.raises(ValueError, match="不能为负数"):
        load_thresholds(negative)


def test_fail_on_regression_requires_threshold_file(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        cases=[],
        pass_rate=1,
        first_tool=1,
        source=1,
        p95=10,
    )
    _write_run(
        candidate,
        cases=[],
        pass_rate=1,
        first_tool=1,
        source=1,
        p95=10,
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--fail-on-regression",
            ]
        )

    assert exc.value.code == 2
