"""Agent evaluation internal helper contracts."""

from scripts.evaluation.agent import (
    check_keywords,
    check_tool_output_keys,
    extract_actual_fields,
    extract_reference_sources,
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
