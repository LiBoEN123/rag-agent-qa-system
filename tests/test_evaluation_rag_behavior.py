"""无 Judge 时 RAG 行为标签的高精度拒答回归测试。"""

from unittest.mock import Mock

import pytest

from scripts.evaluation.rag.common import ChunkIndex
from scripts.evaluation.rag.evaluation import behavior_score, evaluate
from scripts.evaluation.rag.judge import (
    LLMJudgeConfig,
    unavailable_forbidden_result,
    unavailable_key_point_result,
)


def test_answer_with_local_limitation_is_not_a_full_refusal():
    answer = (
        "本科课程不及格可参加一次补考，也可以申请重修；考试作弊不得补考。"
        "当前资料中未说明作弊后的其他处理细节。"
    )

    score, label = behavior_score(
        answer,
        expected="answer",
        key_recall=0.0,
        forbidden_hit=False,
    )

    assert score == 1.0
    assert label == "answered"


def test_abstain_with_concrete_unsupported_details_is_not_a_refusal():
    answer = (
        "一等奖每学期800元，二等奖每学期500元，三等奖每学期350元。"
        "其他研究生奖助项目未说明具体比例。"
    )

    score, label = behavior_score(
        answer,
        expected="abstain",
        key_recall=0.0,
        forbidden_hit=False,
    )

    assert score == 0.0
    assert label == "should_abstain_but_answered"


def test_canonical_no_evidence_answer_remains_a_correct_abstain():
    score, label = behavior_score(
        "我无法根据提供的文档确定答案",
        expected="abstain",
        key_recall=0.0,
        forbidden_hit=False,
    )

    assert score == 1.0
    assert label == "correct_abstain"


@pytest.mark.parametrize("expected", ["answer", "abstain", "partial_answer"])
@pytest.mark.parametrize("with_gold", [False, True])
@pytest.mark.parametrize("judge_enabled", [False, True])
def test_missing_prediction_scores_zero_and_never_calls_judge(expected, with_gold, judge_enabled):
    questions = [{
        "qid": qid, "question": "政策是什么", "category": "test", "difficulty": "easy",
        "expected_behavior": expected, "gold_key_points": ["政策"], "forbidden_points": ["臆造"],
        "gold_chunk_ids": [1] if with_gold else [], "gold_doc_ids": ["gold.md"] if with_gold else [],
    } for qid in ("q1", "q2")]
    key_point = Mock(return_value=unavailable_key_point_result(["政策"], "fake judge"))
    forbidden = Mock(return_value=unavailable_forbidden_result(["臆造"], "fake judge"))
    rows, summary, details = evaluate(
        questions,
        {"q1": {"prediction_schema_version": 3, "answer": "政策", "retrieved_chunk_ids": "1", "reference_chunk_ids": "1"}},
        ChunkIndex({1: "gold.md"}, {}, "test"), k=5,
        judge_config=LLMJudgeConfig(judge_enabled, "fake", "", "", 0, 0, 0),
        key_point_evaluator=key_point, forbidden_evaluator=forbidden,
    )
    missing = rows[1]
    assert missing["has_prediction"] is False
    assert missing["behavior_label"] == "runtime_error"
    for field in (
        "final_score", "behavior_score", "key_point_recall",
        "retrieval_gold_chunk_recall_at_k", "reference_gold_chunk_recall",
    ):
        assert missing[field] == 0
    assert missing["retrieved_chunk_ids"] == "[]"
    assert missing["key_point_judge_used"] is missing["forbidden_judge_used"] is False
    assert summary["overall"]["prediction_coverage"] == 0.5
    assert summary["category:test"]["prediction_coverage"] == 0.5
    assert key_point.call_count == forbidden.call_count == int(judge_enabled)
    for evaluator in (key_point, forbidden):
        assert all(call.kwargs["qid"] == "q1" for call in evaluator.call_args_list)
    for field in ("key_point_judgments", "forbidden_judgments"):
        assert details[1][field][0]["reason"] == "prediction unavailable due runtime failure"


def test_missing_prediction_without_key_points_still_has_zero_recall():
    rows, _, _ = evaluate([{"qid": "q", "expected_behavior": "abstain"}], {}, ChunkIndex({}, {}, "test"), k=5)
    assert rows[0]["key_point_recall"] == rows[0]["final_score"] == 0
