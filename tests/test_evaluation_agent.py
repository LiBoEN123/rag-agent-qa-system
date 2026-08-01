"""Agent v2 判分、失败原因和指标测试。"""

from scripts.evaluation.agent import (
    calculate_agent_metrics,
    combine_case_turns,
    score_agent_turn,
)
from scripts.evaluation.contracts import ExpectedOutcome


def _result(
    *,
    status="OK",
    tool="simple_calculator",
    tool_status="success",
    answer="结果是 7",
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
                    "expression": "1+2*3",
                    "result": 7,
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
        "first_tool": "simple_calculator",
        "tools_contains": ["simple_calculator"],
        "keywords": ["7"],
        "min_keyword_hits": 1,
        "sources": ["rules.md"],
        "source_match": "any",
        "tool_output_keys": ["expression", "result"],
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
        question="计算",
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


def test_score_agent_turn_uses_captured_keys_without_persisting_values():
    result = _result()
    result["tool_trace"][-1].pop("tool_output")

    row = _score(
        result=result,
        tool_output_records=[
            {
                "tool_name": "simple_calculator",
                "status": "success",
                "ok": True,
                "output_keys": ["expression", "result"],
            }
        ],
    )

    assert row["passed"] is True
    assert row["tool_output_schema_pass"] is True
    assert "tool_output_records" not in row


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
