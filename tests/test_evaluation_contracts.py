"""Evaluation Baseline v2 数据合同测试。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.evaluation.contracts import (
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
