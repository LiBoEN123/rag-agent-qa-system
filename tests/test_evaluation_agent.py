"""Agent Evaluation helper、判分、失败原因和指标测试。"""

import pytest

from scripts.evaluation.agent import (
    calculate_agent_metrics,
    check_keywords,
    check_tool_output_keys,
    combine_case_turns,
    extract_actual_fields,
    extract_reference_sources,
    score_agent_turn,
)
from scripts.evaluation.contracts import ExpectedOutcome


def test_check_tool_output_keys_reads_tool_result_output():
    case = {
        "tool_output_keys": [
            "doc_a_evidence",
            "doc_b_evidence",
            "comparison",
        ]
    }
    result = {
        "tool_result": {
            "output": {
                "doc_a_evidence": [],
                "doc_b_evidence": [],
                "comparison": "compared",
            }
        }
    }

    check = check_tool_output_keys(ExpectedOutcome.model_validate(case), result)

    assert check["tool_output_keys_pass"] is True
    assert check["tool_output_keys_missing"] == []


def test_check_tool_output_keys_reports_missing_from_trace_output():
    case = {"tool_output_keys": ["comparison", "conflict_detected"]}
    result = {
        "tool_trace": [
            {
                "step": "tool_call",
                "tool_name": "search_document",
                "tool_output": {"comparison": "compared"},
            }
        ]
    }

    check = check_tool_output_keys(ExpectedOutcome.model_validate(case), result)

    assert check["tool_output_keys_pass"] is False
    assert check["tool_output_keys_missing"] == ["conflict_detected"]


def test_agent_helpers_extract_real_tool_trace_keywords_and_sources():
    result = {
        "status": "OK",
        "action": "agent",
        "answer": "最长学习年限存在差异",
        "references": [{"source_file": "rules.md"}],
        "tool_trace": [
            {
                "step": "decide",
                "decision_type": "use_agent",
                "action": "agent",
            },
            {
                "step": "tool_call",
                "tool_name": "search_document",
                "tool_status": "success",
            },
        ],
    }

    actual = extract_actual_fields(result)
    keyword = check_keywords(
        ExpectedOutcome(keywords=["学习年限", "差异"], min_keyword_hits=2),
        result,
    )

    assert actual == {
        "actual_decision": "use_agent",
        "actual_action": "agent",
        "actual_tool": "search_document",
        "actual_tools": ["search_document"],
        "actual_status": "OK",
    }
    assert keyword["keyword_pass"] is True
    assert extract_reference_sources(result) == ["rules.md"]


def _result(
    *,
    status="OK",
    tool="find_documents",
    tool_status="success",
    answer="共有 7 份文档",
    rewrite=None,
):
    trace = []
    if rewrite is not None:
        trace.append(
            {
                "step": "query_rewrite",
                "original_question": "那结果呢",
                "rewritten_question": rewrite,
                "rewritten": True,
                "rewrite_llm_used": True,
                "reason": "follow-up",
            }
        )
    trace.extend(
        [
            {
                "step": "decide",
                "decision_type": "use_agent",
                "action": "agent",
            },
            {
                "step": "tool_call",
                "tool_name": tool,
                "tool_status": tool_status,
                "tool_output": {
                    "documents": [{}] * 7,
                    "count": 7,
                },
            },
        ]
    )
    return {
        "status": status,
        "action": "agent",
        "answer": answer,
        "reason": "",
        "tool_result_summary": "",
        "references": [{"source_file": "rules.md"}],
        "tool_trace": trace,
        "llm_call_count": 2,
    }


def _expected(**changes):
    values = {
        "decision": "use_agent",
        "action": "agent",
        "status": "OK",
        "first_tool": "find_documents",
        "tools_contains": ["find_documents"],
        "keywords": ["7"],
        "min_keyword_hits": 1,
        "sources": ["rules.md"],
        "source_match": "any",
        "tool_output_keys": ["count", "documents"],
    }
    values.update(changes)
    return ExpectedOutcome.model_validate(values)


def _score(expected=None, result=None, **changes):
    return score_agent_turn(
        case_id="case-1",
        category="tool_use",
        suite="core",
        tags=["fixture"],
        turn_index=1,
        question="请列出知识库文档",
        expected=expected or _expected(),
        result=result or _result(),
        metrics={"llm_call_count": 3},
        latency_ms=changes.pop("latency_ms", 10),
        runtime_error=changes.pop("runtime_error", ""),
        **changes,
    )


def test_score_agent_turn_all_checks_pass():
    row = _score()

    assert row["passed"] is True
    assert row["failure_reasons"] == []
    assert row["tool_call_count"] == 1
    assert row["llm_call_count"] == 3


@pytest.mark.parametrize("include_online", [False, True])
def test_current_web_cases_remain_opt_in_and_use_core_scoring(include_online, monkeypatch, settings_factory):
    from pathlib import Path

    from scripts.evaluation import agent
    from scripts.evaluation.contracts import load_agent_evaluation_dataset

    dataset = load_agent_evaluation_dataset(Path(__file__).parents[1] / "data/eval/agent_eval_cases_v2.json")

    def fake_execute(case, **kwargs):
        expected = case.iter_expected()[0]
        return score_agent_turn(
            case_id=case.id, category=case.category, suite=case.suite, tags=case.tags,
            turn_index=1, question=case.iter_questions()[0], expected=expected,
            result=_result(tool=expected.first_tool),
        )

    monkeypatch.setattr(agent, "execute_agent_case", fake_execute)
    rows, summary = agent.run_agent_baseline(
        dataset, settings=settings_factory(), include_online=include_online,
        rag_already_initialized=True, force_rebuild=False,
    )
    online = [row for row in rows if row["suite"] == "online"]
    assert len(rows) == (28 if include_online else 25)
    assert summary["skipped_cases"] == (0 if include_online else 3)
    assert {row["actual_first_tool"] for row in online} == (
        {"web_search", "web_fetch", "web_pdf_fetch"} if include_online else set()
    )
    assert all(row["passed"] for row in online)


def test_score_agent_turn_uses_captured_keys_without_persisting_values():
    result = _result()
    result["tool_trace"][-1].pop("tool_output")

    row = _score(
        result=result,
        tool_output_records=[
            {
                "tool_name": "find_documents",
                "status": "success",
                "ok": True,
                "output_keys": ["count", "documents"],
            }
        ],
    )

    assert row["passed"] is True
    assert row["tool_output_schema_pass"] is True
    assert "tool_output_records" not in row


def test_score_agent_turn_accepts_explicit_error_then_success_recovery():
    result = _result(tool_status="error")
    result["tool_trace"].append(
        {
            "step": "tool_call",
            "tool_name": "find_documents",
            "tool_status": "success",
            "tool_output": {
                "documents": [{}] * 7,
                "count": 7,
            },
        }
    )
    expected = _expected(
        tool_sequence=["find_documents", "find_documents"],
        tool_status_sequence=["error", "success"],
    )

    row = _score(expected=expected, result=result)

    assert row["passed"] is True
    assert row["actual_tool_statuses"] == ["error", "success"]
    assert row["successful_tool_call_count"] == 1
    assert "tool_call_failed" not in row["failure_reasons"]


def test_score_agent_turn_collects_multiple_failure_reasons():
    expected = _expected(
        status="NO_RELEVANT_INFO",
        first_tool="search_knowledge_base",
        tools_contains=["search_knowledge_base"],
        keywords=["missing"],
        sources=["other.md"],
        forbidden_keywords=["7"],
        tool_output_keys=["missing_key"],
    )

    row = _score(
        expected=expected,
        result=_result(tool_status="error"),
        runtime_error="timeout from fake runtime",
    )

    assert row["passed"] is False
    assert {
        "timeout",
        "status_mismatch",
        "first_tool_mismatch",
        "tool_sequence_mismatch",
        "tool_call_failed",
        "keyword_missing",
        "forbidden_keyword_hit",
        "source_missing",
        "tool_output_schema_missing",
    } <= set(row["failure_reasons"])


def test_query_rewrite_trace_is_checked():
    expected = _expected(
        rewrite_keywords=["休学创业", "学习年限"],
        min_rewrite_keyword_hits=2,
        rewritten=True,
        rewrite_llm_used=True,
    )

    row = _score(
        expected=expected,
        result=_result(rewrite="休学创业学生的学习年限"),
    )

    assert row["query_rewrite_pass"] is True
    assert row["query_rewrite"]["original_question"] == "那结果呢"


def test_invalid_result_contract_trace_has_dedicated_failure_reason():
    result = _result()
    result["tool_trace"].append(
        {
            "step": "chat_result_contract_error",
            "status": "error",
        }
    )

    row = _score(result=result)

    assert row["passed"] is False
    assert "invalid_result_contract" in row["failure_reasons"]


def test_combine_multi_turn_case_accumulates_counts_and_failures():
    first = _score(latency_ms=10)
    second = _score(
        expected=_expected(status="NO_RELEVANT_INFO"),
        latency_ms=20,
    )

    combined = combine_case_turns("multi-1", [first, second])

    assert combined["passed"] is False
    assert combined["turn_count"] == 2
    assert combined["latency_ms"] == 30
    assert combined["tool_call_count"] == 2
    assert "status_mismatch" in combined["failure_reasons"]


def test_metrics_handle_partial_failures_online_skip_percentiles_and_groups():
    first = _score(latency_ms=10)
    second = _score(
        expected=_expected(status="NO_RELEVANT_INFO"),
        latency_ms=110,
    )
    second["category"] = "no_evidence"

    metrics = calculate_agent_metrics(
        [first, second],
        total_cases=3,
        skipped_cases=1,
    )

    assert metrics["executed_cases"] == 2
    assert metrics["skipped_cases"] == 1
    assert metrics["pass_rate"] == 0.5
    assert metrics["average_tool_call_count"] == 1
    assert metrics["average_llm_call_count"] == 3
    assert metrics["p50_latency_ms"] == 60
    assert metrics["p95_latency_ms"] == 105
    assert metrics["groups"]["category"]["tool_use"]["pass_rate"] == 1
    assert metrics["groups"]["category"]["no_evidence"]["pass_rate"] == 0


def test_first_tool_any_of_accepts_a_reasonable_alternative_route():
    expected = _expected(
        first_tool="",
        first_tool_any_of=["find_documents", "search_knowledge_base"],
    )

    row = _score(expected=expected)

    assert row["passed"] is True
    assert "first_tool_mismatch" not in row["failure_reasons"]


def test_first_tool_any_of_rejects_empty_trace():
    row = _score(
        expected=ExpectedOutcome(first_tool_any_of=["a", "b"]),
        result={"status": "OK", "answer": "fixture", "tool_trace": []},
    )
    assert "first_tool_mismatch" in row["failure_reasons"]


def _kb_run(tools, sources=None):
    sources = sources or ["rules.md"] * len(tools)
    calls, records, references = [], [], []
    for index, (tool, source) in enumerate(zip(tools, sources, strict=True)):
        call_id = f"call-{index}"
        evidence = [] if tool == "find_documents" else [{
            "reference_id": f"evidence-{index}", "file_name": source,
        }]
        calls.append({"step": "tool_call", "tool_name": tool,
                      "tool_call_id": call_id, "tool_status": "success"})
        records.append({"tool_name": tool, "tool_call_id": call_id,
                        "ok": True, "status": "success", "kb_evidence": evidence})
        references.extend(evidence)
    return {"status": "OK", "answer": "有依据的回答", "tool_trace": calls,
            "references": references}, records


@pytest.mark.parametrize("tools,passes", [
    (["search_knowledge_base"], True),
    (["find_documents", "search_document"], True),
    (["find_documents", "search_document", "get_chunk_context"], True),
    (["find_documents", "search_knowledge_base"], True),
    (["find_documents"], False),
    (["search_document", "find_documents"], False),
    ([], False),
])
def test_valid_kb_paths_require_successful_body_evidence(tools, passes):
    result, records = _kb_run(tools)
    expected = ExpectedOutcome(valid_tool_paths=[
        ["search_knowledge_base"], ["find_documents", "search_document"],
    ], sources=["rules.md"])
    row = _score(expected=expected, result=result, tool_output_records=records)
    assert row["passed"] is passes


@pytest.mark.parametrize("fault", [
    "tool_error", "captured_error", "missing_capture", "wrong_call_id",
    "no_body", "no_references", "unknown_evidence", "forged_source",
])
def test_path_cannot_pass_with_failed_empty_or_unavailable_evidence(fault):
    result, records = _kb_run(["search_knowledge_base"])
    if fault == "tool_error":
        result["tool_trace"][0]["tool_status"] = "error"
    elif fault == "captured_error":
        records[0]["ok"] = False
    elif fault == "missing_capture":
        records.clear()
    elif fault == "wrong_call_id":
        records[0]["tool_call_id"] = "other-run"
    elif fault == "no_body":
        records[0]["kb_evidence"] = []
    elif fault == "no_references":
        result["references"] = []
    elif fault == "unknown_evidence":
        result["references"] = [{"reference_id": "invented", "file_name": "rules.md"}]
    elif fault == "forged_source":
        result["references"] = [{"reference_id": "evidence-0", "file_name": "rules.md"}]
        records[0]["kb_evidence"][0]["file_name"] = "wrong.md"
    row = _score(expected=ExpectedOutcome(
        valid_tool_paths=[["search_knowledge_base"]], sources=["rules.md"],
    ), result=result, tool_output_records=records)
    assert row["passed"] is False
    assert row["evidence_grounding_pass"] is False


@pytest.mark.parametrize("fault", ["none", "same_document", "missing_citation", "wrong_tool"])
def test_compare_requires_both_target_bodies_and_final_citations(fault):
    tools = ["search_document", "search_document"]
    if fault == "wrong_tool":
        tools = ["search_document", "search_document", "get_chunk_context"]
    sources = ["a.md", "a.md"] if fault == "same_document" else ["a.md", "b.md"]
    if fault == "wrong_tool":
        sources = ["a.md", "a.md", "b.md"]
    result, records = _kb_run(tools, sources)
    if fault == "missing_citation":
        result["references"].pop()
    row = _score(expected=ExpectedOutcome(
        valid_tool_paths=[["search_document", "search_document"]],
        sources=["a.md", "b.md"], source_match="all",
    ), result=result, tool_output_records=records)
    assert row["passed"] is (fault == "none")


def test_multi_turn_failure_classification_is_not_taken_from_passing_last_turn():
    first = _score(expected=_expected(first_tool="search_document"))
    last = _score()
    row = combine_case_turns("multi", [first, last])
    assert not row["passed"]
    assert row["classification"] == first["classification"] != "PASS"
