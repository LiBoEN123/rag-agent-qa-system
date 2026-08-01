"""无 Judge 时 RAG 行为标签的高精度拒答回归测试。"""

from scripts.rag_eval.evaluation import behavior_score


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
