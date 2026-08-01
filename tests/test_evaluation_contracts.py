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


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "evaluation"
    / "minimal_agent_cases_v2.json"
)
KNOWN_TOOLS = {
    "simple_calculator",
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
    assert dataset.cases[0].question == "1+1 等于多少？"


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
