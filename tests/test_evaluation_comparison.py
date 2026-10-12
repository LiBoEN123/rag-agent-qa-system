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
    source: float,
    p95: float,
):
    path.mkdir(parents=True)
    summary = {
        "run_id": path.name,
        "agent": {
            "pass_rate": pass_rate,
            "status_accuracy": pass_rate,
            "source_pass_rate": source,
            "evidence_grounding_pass_rate": source,
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
                "prediction_coverage": 1.0,
                "retrieval_gold_chunk_recall_at_k": 0.7,
                "reference_gold_chunk_recall": 0.6,
                "avg_answer_correctness": 0.75,
                "llm_judge_coverage": 1.0,
            }
        },
    }
    (path / "baseline_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    manifest = {
        "evaluation_type": "core_baseline", "selected_suites": ["core"],
        "code": {"git_commit": "A", "git_branch": "branch", "git_dirty": False},
        "data": {"agent_cases_sha256": "agent-data", "rag_eval_sha256": "rag-data"},
        "rag": {
            "prediction_schema_version": 3, "retrieval_metric_k": 5,
            "index_manifest_sha256": "index", "index_manifest_path": "index/a.json",
            "document_count": 2, "chunk_count": 10, "embedding_model": "embedding",
            "rerank_model": "reranker", "chunk_size": 500, "chunk_overlap": 50,
        },
        "agent": {
            "agent_model": "agent", "agent_temperature": 0, "agent_max_tokens": 1024,
            "agent_enable_thinking": "provider_default",
            "answer_model": "answer", "answer_temperature": 0, "answer_max_tokens": 1024,
            "budget": {"max_iterations": 6, "max_tool_calls": 10},
            "query_rewrite_enabled": True, "query_rewrite_model": "rewrite",
            "tool_contract_sha256": "tools", "registered_tool_names": ["search_document"],
        },
        "prompts": {"agent": {"combined_sha256": "prompt"}},
        "judge": {"enabled": True, "model": "judge"},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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
            {"case_id": "action", "passed": True, "actual_action": "agent"},
        ],
        pass_rate=0.8,
        source=0.9,
        p95=100,
    )
    _write_run(
        candidate,
        cases=[
            {"case_id": "fixed", "passed": True},
            {"case_id": "regressed", "passed": False},
            {"case_id": "added", "passed": True},
            {"case_id": "action", "passed": True, "actual_action": "direct_answer"},
        ],
        pass_rate=0.7,
        source=0.85,
        p95=140,
    )

    comparison = compare_runs(baseline, candidate)

    assert comparison["case_changes"]["changed_cases"] == ["action", "fixed", "regressed"]
    assert comparison["case_changes"]["added"] == ["added"]
    assert comparison["case_changes"]["removed"] == ["removed"]
    assert comparison["case_changes"]["new_failures"] == ["regressed"]
    assert comparison["case_changes"]["fixed_failures"] == ["fixed"]
    assert comparison["metrics"]["core_agent_pass_rate"]["delta"] == -0.1
    assert comparison["category_pass_rate"]["tool"]["delta"] == -0.1
    assert comparison["metrics"]["evidence_grounding_pass_rate"]["delta"] == -0.05


def test_explicit_thresholds_control_regression_exit(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        cases=[{"case_id": "one", "passed": True}],
        pass_rate=0.9,
        source=0.9,
        p95=100,
    )
    _write_run(
        candidate,
        cases=[{"case_id": "one", "passed": False}],
        pass_rate=0.8,
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
        source=1,
        p95=10,
    )
    _write_run(
        candidate,
        cases=[],
        pass_rate=0,
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
    comparison = json.loads((candidate / "comparison.json").read_text())
    assert comparison["compatibility"] == {"compatible": True, "overridden": False, "mismatches": []}
    assert "Run controls are compatible." in (candidate / "comparison.md").read_text()


def test_comparison_calculates_reference_recall_delta(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for path, value in ((baseline, 0.7), (candidate, 0.8)):
        _write_run(
            path,
            cases=[],
            pass_rate=1,
            source=1,
            p95=10,
        )
        summary_path = path / "baseline_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["rag"]["overall"][
            "reference_gold_chunk_recall"
        ] = value
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    comparison = compare_runs(baseline, candidate)

    assert comparison["metrics"]["reference_gold_chunk_recall"] == {
        "baseline": 0.7,
        "candidate": 0.8,
        "delta": 0.1,
    }


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
        source=1,
        p95=10,
    )
    _write_run(
        candidate,
        cases=[],
        pass_rate=1,
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


@pytest.mark.parametrize("before,after,coverage,delta", [
    (0.7, 0.8, 1.0, 0.1), (None, 0.8, 1.0, None),
    (0.7, None, 1.0, None), (None, None, 0.0, None), (0.7, 0.8, 0.5, None),
])
def test_correctness_delta_requires_scores_and_equal_coverage(tmp_path, before, after, coverage, delta):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for path, score, used in ((baseline, before, 1.0), (candidate, after, coverage)):
        _write_run(path, cases=[], pass_rate=1, source=1, p95=10)
        report = path / "baseline_summary.json"
        summary = json.loads(report.read_text())
        summary["rag"]["overall"].update(avg_answer_correctness=score, llm_judge_coverage=used)
        report.write_text(json.dumps(summary))
    comparison = compare_runs(baseline, candidate)
    assert comparison["metrics"]["rag_avg_answer_correctness"] == {
        "baseline": before, "candidate": after, "delta": delta,
    }
    assert bool(comparison["warnings"]) is (coverage != 1.0)
    assert evaluate_regressions(comparison, {}) == []


@pytest.fixture
def run_pair(tmp_path):
    paths = tmp_path / "baseline", tmp_path / "candidate"
    for path in paths:
        _write_run(path, cases=[], pass_rate=1, source=1, p95=10)
    return paths


def _change(path, field, value):
    document = json.loads(path.read_text())
    parent = document
    *keys, name = field.split(".")
    for key in keys:
        parent = parent[key]
    parent[name] = value
    path.write_text(json.dumps(document))


@pytest.mark.parametrize("field,value", [
    ("evaluation_type", "targeted_agent_check"), ("selected_suites", ["core", "online"]),
    ("data.agent_cases_sha256", "other"), ("data.rag_eval_sha256", "other"),
    ("rag.retrieval_metric_k", 10), ("rag.prediction_schema_version", 2),
    ("rag.index_manifest_sha256", "other"), ("judge.model", "other"), ("judge.enabled", False),
    ("agent.agent_model", "other"), ("agent.agent_temperature", 0.5), ("agent.agent_max_tokens", 2048),
    ("agent.agent_enable_thinking", False), ("agent.agent_enable_thinking", True),
    ("agent.agent_enable_thinking", None),
    ("agent.answer_model", "other"), ("agent.answer_temperature", 0.5), ("agent.answer_max_tokens", 2048),
    ("agent.budget.max_tool_calls", 20), ("agent.query_rewrite_enabled", False),
    ("agent.query_rewrite_model", "other"), ("data.agent_cases_sha256", ""),
])
def test_incompatible_controls_reject_before_thresholds_or_output(run_pair, tmp_path, capsys, field, value):
    baseline, candidate = run_pair
    _change(candidate / "manifest.json", field, value)
    output = tmp_path / "comparison"
    with pytest.raises(SystemExit) as error:
        main(["--baseline", str(baseline), "--candidate", str(candidate), "--output-dir", str(output),
              "--thresholds", str(tmp_path / "not-read.json"), "--fail-on-regression"])
    assert error.value.code == 2
    assert field.rsplit(".max_tool_calls", 1)[0] in capsys.readouterr().err
    assert not output.exists()


def test_system_under_test_changes_and_index_path_are_allowed(run_pair):
    baseline, candidate = run_pair
    for field, value in {
        "code.git_commit": "B", "code.git_branch": "candidate", "code.git_dirty": True,
        "code.sha256": "other-code", "prompts.agent.combined_sha256": "other-prompt",
        "agent.tool_contract_sha256": "other-tools", "agent.registered_tool_names": ["find_documents"],
        "rag.index_manifest_path": "/other/index.json",
    }.items():
        _change(candidate / "manifest.json", field, value)
    assert compare_runs(baseline, candidate)["compatibility"]["compatible"]


@pytest.mark.parametrize("missing", [[], [0], [1], [0, 1]])
@pytest.mark.parametrize("field,value", [("rag.retrieval_metric_k", 10), ("agent.agent_model", "other")])
def test_explicit_override_marks_incompatible_and_missing_manifests(run_pair, missing, field, value):
    baseline, candidate = run_pair
    _change(candidate / "manifest.json", field, value)
    for index in missing:
        (run_pair[index] / "manifest.json").unlink()
    with pytest.raises(ValueError, match="Incompatible run controls"):
        compare_runs(baseline, candidate)
    args = ["--baseline", str(baseline), "--candidate", str(candidate)]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert not (candidate / "comparison.json").exists()
    assert main([*args, "--allow-incompatible"]) == 0
    comparison = json.loads((candidate / "comparison.json").read_text())
    compatibility = comparison["compatibility"]
    assert not compatibility["compatible"] and compatibility["overridden"]
    assert compatibility["mismatches"][0]["field"] == ("manifest.json" if missing else field)
    assert comparison["metrics"]["rag_retrieval_gold_chunk_recall_at_k"]["delta"] == 0
    assert "**Comparison was forced across incompatible run controls.**" in (candidate / "comparison.md").read_text()


def test_legacy_manifest_without_separate_model_roles_is_incompatible(run_pair):
    baseline, candidate = run_pair
    path = baseline / "manifest.json"
    manifest = json.loads(path.read_text())
    agent = manifest["agent"]
    for field in ("agent_model", "agent_temperature", "agent_max_tokens"):
        del agent[field]
    agent["temperature"] = agent.pop("answer_temperature")
    agent["max_tokens"] = agent.pop("answer_max_tokens")
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="agent.agent_model"):
        compare_runs(baseline, candidate)


@pytest.mark.parametrize("layer,unused_dataset", [("rag", "rag_eval_sha256"), ("agent", "agent_cases_sha256")])
def test_single_layer_runs_ignore_unused_controls_but_keep_runtime_controls(run_pair, layer, unused_dataset):
    baseline, candidate = run_pair
    for path in run_pair:
        _change(path / "baseline_summary.json", layer, {"skipped": True})
        _change(path / "manifest.json", f"data.{unused_dataset}", None)
    _change(candidate / "manifest.json", f"data.{unused_dataset}", "unused")
    if layer == "rag":
        for field in ("rag.prediction_schema_version", "rag.retrieval_metric_k", "judge.model"):
            _change(candidate / "manifest.json", field, None)
    assert compare_runs(baseline, candidate)["compatibility"]["compatible"]
    _change(candidate / "manifest.json", "agent.answer_model", "other")
    with pytest.raises(ValueError, match="agent.answer_model"):
        compare_runs(baseline, candidate)


@pytest.mark.parametrize("layer", ["agent", "rag"])
def test_run_shape_mismatch_rejected(run_pair, layer):
    baseline, candidate = run_pair
    _change(candidate / "baseline_summary.json", layer, {"skipped": True})
    with pytest.raises(ValueError, match=rf"{layer}\.skipped"):
        compare_runs(baseline, candidate)


def test_targeted_cases_and_suites_compare_as_sets(run_pair):
    baseline, candidate = run_pair
    for path, ids, suites in ((baseline, ["A", "B"], ["core", "online"]),
                             (candidate, ["B", "A"], ["online", "core"])):
        _change(path / "baseline_summary.json", "rag", {"skipped": True})
        _change(path / "manifest.json", "evaluation_type", "targeted_agent_check")
        _change(path / "manifest.json", "data.selected_case_ids", ids)
        _change(path / "manifest.json", "selected_suites", suites)
    assert compare_runs(baseline, candidate)["compatibility"]["compatible"]
    _change(candidate / "manifest.json", "data.selected_case_ids", ["A"])
    with pytest.raises(ValueError, match="data.selected_case_ids"):
        compare_runs(baseline, candidate)
    _change(candidate / "manifest.json", "data.selected_case_ids", ["A", "B"])
    _change(candidate / "manifest.json", "evaluation_type", "core_baseline")
    with pytest.raises(ValueError, match="evaluation_type"):
        compare_runs(baseline, candidate)


@pytest.mark.parametrize("candidate_hash", ["", "index"])
def test_index_fallback_uses_stable_metadata_when_either_hash_missing(run_pair, candidate_hash):
    baseline, candidate = run_pair
    _change(baseline / "manifest.json", "rag.index_manifest_sha256", "")
    _change(candidate / "manifest.json", "rag.index_manifest_sha256", candidate_hash)
    _change(candidate / "manifest.json", "rag.index_manifest_path", "/other/index.json")
    assert compare_runs(baseline, candidate)["compatibility"]["compatible"]
    _change(candidate / "manifest.json", "rag.chunk_count", 11)
    with pytest.raises(ValueError, match="rag.chunk_count"):
        compare_runs(baseline, candidate)


def test_judge_config_is_ignored_when_correctness_is_unavailable(run_pair):
    baseline, candidate = run_pair
    _change(candidate / "baseline_summary.json", "rag.overall.avg_answer_correctness", None)
    _change(candidate / "manifest.json", "judge", {"enabled": False, "model": ""})
    comparison = compare_runs(baseline, candidate)
    assert comparison["compatibility"]["compatible"]
    assert comparison["metrics"]["rag_avg_answer_correctness"]["delta"] is None
