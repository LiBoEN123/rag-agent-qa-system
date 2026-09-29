"""Agent 最终结果、必要工具、真实证据和适用轮次指标测试。"""

from pathlib import Path

import pytest

from scripts.evaluation.agent import calculate_agent_metrics, combine_case_turns, score_agent_turn
from scripts.evaluation.contracts import ExpectedOutcome, load_agent_evaluation_dataset


DATASET = Path(__file__).parents[1] / "data/eval/agent_eval_cases_v2.json"


def _result(*, status="OK", tool="find_documents", tool_status="success", answer="共有 7 份文档"):
    return {
        "status": status,
        "action": "agent",
        "answer": answer,
        "references": [],
        "tool_trace": [{
            "step": "tool_call",
            "tool_name": tool,
            "tool_status": tool_status,
        }],
        "llm_call_count": 2,
    }


def _expected(**changes):
    return ExpectedOutcome.model_validate({
        "action": "agent",
        "status": "OK",
        "tools_contains": ["find_documents"],
        "keywords": ["7"],
        "min_keyword_hits": 1,
        **changes,
    })


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
        **changes,
    )


def test_score_agent_turn_all_checks_pass():
    row = _score()
    assert row["passed"] is True
    assert row["failure_reasons"] == []
    assert row["tool_call_count"] == 1
    assert row["llm_call_count"] == 3
    assert row["evidence_grounding_pass"]  # 没有来源期望时不适用。


@pytest.mark.parametrize("include_online", [False, True])
def test_current_web_cases_remain_opt_in_and_use_core_scoring(include_online, monkeypatch, settings_factory):
    from scripts.evaluation import agent

    dataset = load_agent_evaluation_dataset(DATASET)

    def fake_execute(case, **kwargs):
        expected = case.iter_expected()[0]
        return score_agent_turn(
            case_id=case.id, category=case.category, suite=case.suite, tags=case.tags,
            turn_index=1, question=case.iter_questions()[0], expected=expected,
            result=_result(tool=next(iter(expected.tools_contains), "find_documents")),
        )

    monkeypatch.setattr(agent, "execute_agent_case", fake_execute)
    rows, summary = agent.run_agent_baseline(
        dataset, settings=settings_factory(), include_online=include_online,
        rag_already_initialized=True, force_rebuild=False,
    )
    online = [row for row in rows if row["suite"] == "online"]
    assert len(rows) == (28 if include_online else 25)
    assert summary["skipped_cases"] == (0 if include_online else 3)
    assert {tool for row in online for tool in row["actual_tools"]} == (
        {"web_search", "web_fetch", "web_pdf_fetch"} if include_online else set()
    )
    assert all(row["passed"] for row in online)


@pytest.mark.parametrize("tool,status,keys,passes", [
    ("find_documents", "success", ["count", "documents"], True),
    ("find_documents", "success", ["count"], False),
    ("find_documents", "error", ["count", "documents"], False),
    ("search_document", "success", ["count", "documents"], False),
])
def test_tool_schema_uses_successful_capture_from_required_tool(tool, status, keys, passes):
    row = _score(
        expected=_expected(tool_output_keys=["count", "documents"]),
        tool_output_records=[{
            "tool_name": tool, "status": status, "ok": status == "success", "output_keys": keys,
        }],
    )
    assert row["passed"] is passes
    assert row["tool_output_schema_pass"] is passes
    assert "tool_output_records" not in row


@pytest.mark.parametrize("statuses", [["error", "success"], ["success", "error"]])
def test_score_agent_turn_explicit_error_then_success_recovery(statuses):
    result = _result(tool_status=statuses[0])
    result["tool_trace"].append({
        "step": "tool_call", "tool_name": "find_documents", "tool_status": statuses[1],
    })
    row = _score(result=result, expected=_expected(
        tool_sequence=["find_documents", "find_documents"],
        tool_status_sequence=["error", "success"],
    ))
    assert row["passed"] is (statuses == ["error", "success"])
    assert row["successful_tool_call_count"] == 1
    assert "tool_call_failed" not in row["failure_reasons"]


def test_score_agent_turn_collects_multiple_failure_reasons():
    row = _score(
        expected=_expected(
            action="clarify", status="NO_RELEVANT_INFO", tools_contains=["search_knowledge_base"],
            keywords=["missing"], sources=["other.md"], forbidden_keywords=["7"],
            tool_output_keys=["missing_key"],
        ),
        result=_result(tool_status="error"), runtime_error="timeout from fake runtime",
    )
    assert not row["passed"]
    assert {
        "timeout", "action_mismatch", "status_mismatch", "required_tool_missing", "tool_call_failed",
        "keyword_missing", "forbidden_keyword_hit", "source_missing", "evidence_grounding_mismatch",
        "tool_output_schema_missing",
    } <= set(row["failure_reasons"])


def test_keywords_must_appear_in_final_answer():
    result = _result(answer="没有给出数量")
    result["tool_trace"][0]["summary"] = "共有 7 份文档"
    result["references"] = [{"file_name": "7.md"}]
    assert _score(result=result)["failure_reasons"] == ["keyword_missing"]


def test_invalid_result_contract_trace_has_dedicated_failure_reason():
    result = _result()
    result["tool_trace"].append({"step": "chat_result_contract_error", "status": "error"})
    assert "invalid_result_contract" in _score(result=result)["failure_reasons"]


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
        references.extend(dict(item) for item in evidence)
    return {"action": "agent", "status": "OK", "answer": "有依据的回答", "tool_trace": calls,
            "references": references}, records


@pytest.mark.parametrize("tools,passes", [
    (["search_knowledge_base"], True),
    (["find_documents", "search_document"], True),
    (["search_document", "get_chunk_context"], True),
    (["search_document", "find_documents"], True),
    (["find_documents"], False),
    ([], False),
])
def test_grounding_accepts_any_successful_body_tool_order(tools, passes):
    result, records = _kb_run(tools)
    expected = ExpectedOutcome(
        sources=["rules.md"], tools_contains=["find_documents"] if "find_documents" in tools else [],
    )
    row = _score(expected=expected, result=result, tool_output_records=records)
    assert row["passed"] is passes


@pytest.mark.parametrize("fault", [
    "tool_error", "captured_error", "captured_status_error", "missing_capture", "wrong_call_id",
    "no_body", "no_references", "unknown_evidence", "forged_source", "wrong_source",
])
def test_grounding_rejects_failed_empty_unavailable_or_wrong_source_evidence(fault):
    result, records = _kb_run(["search_knowledge_base"])
    if fault == "tool_error":
        result["tool_trace"][0]["tool_status"] = "error"
    elif fault == "captured_error":
        records[0]["ok"] = False
    elif fault == "captured_status_error":
        records[0]["status"] = "error"
    elif fault == "missing_capture":
        records.clear()
    elif fault == "wrong_call_id":
        records[0]["tool_call_id"] = "other-run"
    elif fault == "no_body":
        records[0]["kb_evidence"] = []
    elif fault == "no_references":
        result["references"] = []
    elif fault == "unknown_evidence":
        result["references"][0]["reference_id"] = "invented"
    elif fault == "forged_source":
        records[0]["kb_evidence"][0]["file_name"] = "wrong.md"
    elif fault == "wrong_source":
        result["references"][0]["file_name"] = records[0]["kb_evidence"][0]["file_name"] = "wrong.md"
    row = _score(expected=ExpectedOutcome(sources=["rules.md"]), result=result, tool_output_records=records)
    assert not row["passed"] and not row["evidence_grounding_pass"]


@pytest.mark.parametrize("fault", ["none", "same_document", "missing_citation", "alternative_tool"])
def test_compare_requires_both_target_bodies_and_final_citations(fault):
    tools = ["search_document", "get_chunk_context" if fault == "alternative_tool" else "search_document"]
    sources = ["a.md", "a.md"] if fault == "same_document" else ["a.md", "b.md"]
    result, records = _kb_run(tools, sources)
    if fault == "missing_citation":
        result["references"].pop()
    row = _score(expected=ExpectedOutcome(sources=["a.md", "b.md"], source_match="all"),
                 result=result, tool_output_records=records)
    assert row["passed"] is (fault in {"none", "alternative_tool"})


@pytest.mark.parametrize("tools,passes", [
    (["find_documents", "get_chunk_context"], True),
    (["get_chunk_context", "find_documents"], False),
    (["find_documents", "get_chunk_context", "search_document"], False),
])
def test_multi_tool_case_keeps_exact_sequence(tools, passes):
    result, _ = _kb_run(tools)
    expected = ExpectedOutcome(tool_sequence=["find_documents", "get_chunk_context"])
    assert _score(expected=expected, result=result)["passed"] is passes


def test_followup_case_scores_final_outcomes_without_internal_flags():
    case = next(case for case in load_agent_evaluation_dataset(DATASET).cases if case.id == "agent-followup-001")
    rows = []
    for expected in case.iter_expected():
        result, records = _kb_run(["get_chunk_context"], expected.sources)
        result["answer"] = "，".join(expected.keywords)
        rows.append(_score(expected=expected, result=result, tool_output_records=records))
    assert len(rows) == 2 and combine_case_turns(case.id, rows)["passed"]


def test_metrics_include_failing_earlier_turn_and_only_applicable_source_turns():
    result, records = _kb_run(["search_document"])
    first = _score(expected=ExpectedOutcome(status="NO_RELEVANT_INFO", sources=["other.md"]),
                   result=result, tool_output_records=records, latency_ms=10)
    last = _score(latency_ms=20)
    combined = combine_case_turns("multi", [first, last])
    passing = _score(latency_ms=110)
    passing["category"] = "direct_answer"
    metrics = calculate_agent_metrics([combined, passing], total_cases=3, skipped_cases=1)
    assert not combined["passed"] and combined["turn_count"] == 2
    assert combined["latency_ms"] == 30 and combined["tool_call_count"] == 2
    assert "status_mismatch" in combined["failure_reasons"]
    assert metrics["executed_cases"] == 2 and metrics["skipped_cases"] == 1
    assert metrics["pass_rate"] == 0.5 and metrics["status_accuracy"] == 0.6667
    assert metrics["source_pass_rate"] == metrics["evidence_grounding_pass_rate"] == 0
    assert metrics["average_tool_call_count"] == 1.5 and metrics["average_llm_call_count"] == 4.5
    assert metrics["p95_latency_ms"] == 106
    assert set(metrics["groups"]) == {"suite", "category"}
    assert metrics["groups"]["category"]["tool_use"]["pass_rate"] == 0
    assert metrics["groups"]["category"]["direct_answer"]["pass_rate"] == 1
