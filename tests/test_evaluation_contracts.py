"""Evaluation Baseline v2 数据合同测试。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.evaluation.contracts import (
    ExpectedOutcome,
    AgentEvaluationDataset,
    build_coverage_report,
    load_agent_evaluation_dataset,
    validate_tool_names,
)
from scripts.evaluation.external import (
    McpPreflightError,
    validate_mcp_runtime_catalog,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "evaluation"
    / "minimal_agent_cases_v2.json"
)
KNOWN_TOOLS = {
    "find_documents",
    "search_knowledge_base",
}


@pytest.mark.parametrize("paths", [[[]], [[""]], [["find_documents"]], [["bad-name", "search_document"]]])
def test_invalid_body_paths_are_rejected(paths):
    with pytest.raises(ValueError):
        ExpectedOutcome(valid_tool_paths=paths)


def test_path_tools_participate_in_registration_validation():
    dataset = load_agent_evaluation_dataset(FIXTURE)
    dataset.cases[0].expected = ExpectedOutcome(
        valid_tool_paths=[["unknown_tool", "search_document"]],
    )
    with pytest.raises(ValueError, match="unknown_tool"):
        validate_tool_names(dataset, {"search_document"})


def test_core_semantic_routes_and_strict_business_invariants():
    dataset = load_agent_evaluation_dataset(
        Path(__file__).parents[1] / "data/eval/agent_eval_cases_v2.json",
    )
    cases = {case.id: case for case in dataset.cases}
    for case_id in [f"D{i:03d}" for i in range(1, 14)]:
        expected = cases[case_id].expected
        assert not expected.first_tool
        assert expected.sources and expected.keywords
        assert expected.valid_tool_paths == (
            [["search_knowledge_base"], ["search_document"]]
            if case_id == "D005" else
            [["search_knowledge_base"], ["find_documents", "search_document"]]
        )
    for case_id, tool in [("T001", "find_documents"), ("T002", "find_documents"), ("T003", "get_chunk_context")]:
        assert cases[case_id].expected.first_tool == tool
    assert cases["D008"].expected.sources == ["内蒙古工业大学学籍管理办法（2018年修订）.md"]
    compare = cases["compare_documents_explicit_001"].expected
    assert compare.source_match == "all" and len(compare.sources) == 2
    assert compare.valid_tool_paths == [["search_document", "search_document"]]
    assert cases["agent-tool-recovery-001"].expected.tool_status_sequence == ["error", "success"]


def _single_case(**changes):
    case = {
        "id": "case-1",
        "version": 2,
        "suite": "core",
        "category": "test",
        "question": "问题",
        "expected": {
            "status": "OK",
            "source_match": "any",
        },
    }
    case.update(changes)
    return {
        "dataset_version": 2,
        "name": "test",
        "cases": [case],
    }


def test_load_valid_single_turn_fixture():
    dataset = load_agent_evaluation_dataset(FIXTURE)

    assert len(dataset.cases) == 1
    assert dataset.cases[0].question == "请列出知识库文档"


def test_valid_multi_turn_case():
    raw = _single_case(
        question=None,
        session_id="session-1",
        turns=[
            {"question": "第一轮", "expected": {"status": "OK"}},
            {
                "question": "第二轮",
                "expected": {
                    "status": "OK",
                    "rewrite_keywords": ["第一轮"],
                    "min_rewrite_keyword_hits": 1,
                },
            },
        ],
    )

    dataset = AgentEvaluationDataset.model_validate(raw)

    assert len(dataset.cases[0].turns) == 2


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda raw: raw["cases"].append(dict(raw["cases"][0])),
            "重复 case ID",
        ),
        (
            lambda raw: raw["cases"][0].update(
                {"turns": [{"question": "x"}, {"question": "y"}]}
            ),
            "只能提供一个",
        ),
        (
            lambda raw: raw["cases"][0]["expected"].update(
                {"status": "NOT_A_STATUS"}
            ),
            "validation error",
        ),
        (
            lambda raw: raw["cases"][0]["expected"].update(
                {"source_match": "some"}
            ),
            "validation error",
        ),
        (
            lambda raw: raw["cases"][0]["expected"].update(
                {"keywords": ["a"], "min_keyword_hits": 2}
            ),
            "不能超过",
        ),
        (
            lambda raw: raw["cases"][0]["expected"].update(
                {
                    "tool_sequence": ["search_knowledge_base"],
                    "tool_status_sequence": ["error", "success"],
                }
            ),
            "长度必须与 tool_sequence 一致",
        ),
    ],
)
def test_invalid_schema_cases(mutation, match):
    raw = _single_case()
    mutation(raw)

    with pytest.raises(ValidationError, match=match):
        AgentEvaluationDataset.model_validate(raw)


def test_unknown_tool_is_rejected():
    dataset = AgentEvaluationDataset.model_validate(
        _single_case(
            expected={
                "status": "OK",
                "first_tool": "unknown_tool",
            }
        )
    )

    with pytest.raises(ValueError, match="未知工具"):
        validate_tool_names(dataset, KNOWN_TOOLS)


def test_question_and_turns_cannot_both_be_missing():
    raw = _single_case(question=None)

    with pytest.raises(ValidationError, match="必须且只能提供一个"):
        AgentEvaluationDataset.model_validate(raw)


def test_coverage_separates_suites_and_case_shapes():
    dataset = load_agent_evaluation_dataset(FIXTURE)
    report = build_coverage_report(dataset, KNOWN_TOOLS)

    assert report["suite"] == {"core": 1}
    assert report["case_shape"] == {
        "single_turn": 1,
        "multi_turn": 0,
    }
    assert "web_search" in report["gaps"]


def test_dataset_is_json_serializable_without_nan():
    dataset = load_agent_evaluation_dataset(FIXTURE)

    json.dumps(
        dataset.model_dump(mode="json"),
        allow_nan=False,
    )


def test_first_tool_any_of_is_exclusive_with_first_tool():
    valid = AgentEvaluationDataset.model_validate(
        _single_case(
            expected={
                "status": "OK",
                "first_tool_any_of": [
                    "find_documents",
                    "search_knowledge_base",
                ],
            }
        )
    )
    validate_tool_names(valid, KNOWN_TOOLS)

    raw = _single_case(
        expected={
            "status": "OK",
            "first_tool": "find_documents",
            "first_tool_any_of": ["search_knowledge_base"],
        }
    )
    with pytest.raises(ValidationError, match="只能使用一个"):
        AgentEvaluationDataset.model_validate(raw)


@pytest.mark.parametrize(
    ("suite", "servers", "match"),
    [
        ("core", ["github"], "必须为空"),
        ("online", ["firecrawl"], "必须为空"),
        ("mcp", [], "必须非空"),
    ],
)
def test_suite_mcp_server_contract_is_fail_closed(suite, servers, match):
    with pytest.raises(ValidationError, match=match):
        AgentEvaluationDataset.model_validate(
            _single_case(suite=suite, mcp_servers=servers)
        )


def test_mcp_tool_name_is_deferred_to_runtime_catalog_validation():
    dataset = AgentEvaluationDataset.model_validate(
        _single_case(
            suite="mcp",
            mcp_servers=["github"],
            expected={
                "status": "OK",
                "first_tool": "github_get_file_contents",
                "tool_origin": "mcp",
                "mcp_server": "github",
            },
        )
    )
    validate_tool_names(dataset, KNOWN_TOOLS)

    with pytest.raises(
        McpPreflightError,
        match="REQUIRED_MCP_TOOL_NOT_AVAILABLE",
    ):
        validate_mcp_runtime_catalog(
            dataset,
            [],
            selected_suites={"mcp"},
        )

    validate_mcp_runtime_catalog(
        dataset,
        [
            {
                "server": "github",
                "tool_name": "github_get_file_contents",
                "remote_tool": "get_file_contents",
                "risk_level": "read_only",
                "approval_required": False,
                "enabled": True,
                "enabled_for_default_principal": False,
            }
        ],
        selected_suites={"mcp"},
    )
