"""RAG 检索指标、证据漏斗、报告合同与失败分类测试。"""

import pytest

from scripts.evaluation.rag.common import ChunkIndex
from scripts.evaluation.rag.evaluation import evaluate
from scripts.evaluation.rag.retrieval import score_retrieval
from scripts.evaluation.report import classify_rag_failures


def test_evidence_hits_do_not_pollute_matched_gold_chunk_ids():
    question = {
        "gold_doc_ids": ["gold.md"],
        "gold_chunk_ids": [1, 2],
        "gold_evidence": [
            {"chunk_id": 1, "text": "alpha policy"},
            {"chunk_id": 2, "text": "beta policy"},
        ],
    }
    pred = {"retrieved_chunk_ids": "101,2,3"}
    chunk_doc_map = {
        1: "gold.md",
        2: "gold.md",
        3: "gold.md",
        101: "faq.md",
    }
    chunk_text_map = {
        1: "alpha policy",
        2: "beta policy",
        3: "unrelated",
        101: "alpha policy",
    }

    result = score_retrieval(pred, question, chunk_doc_map, chunk_text_map, k=5)

    assert result["evidence_hit_chunk_ids"] == [2, 101]
    assert result["hit_chunk_ids"] == [2]
    assert result["matched_gold_chunk_ids"] == [2]
    assert result["missing_gold_chunk_ids"] == [1]
    assert result["evidence_recall_at_k"] == 1.0
    assert result["gold_chunk_recall_at_k"] == 0.5
    assert result["all_evidence_hit_at_k"] == 1.0
    assert result["all_gold_hit_at_k"] == 0.0


def test_gold_chunks_are_scored_against_current_index_when_evidence_resolves():
    question = {
        "gold_doc_ids": ["gold.md"],
        "gold_chunk_ids": [99],
        "gold_evidence": [
            {"chunk_id": 99, "text": "moved evidence"},
        ],
    }
    pred = {"retrieved_chunk_ids": [4]}
    chunk_doc_map = {
        4: "gold.md",
        99: "gold.md",
    }
    chunk_text_map = {
        4: "moved evidence",
        99: "old chunk without the evidence",
    }

    result = score_retrieval(pred, question, chunk_doc_map, chunk_text_map, k=5)

    assert result["scored_gold_chunk_ids"] == [4]
    assert result["matched_gold_chunk_ids"] == [4]
    assert result["missing_gold_chunk_ids"] == []
    assert result["gold_chunk_recall_at_k"] == 1.0


def test_report_keeps_simple_chunk_id_columns():
    question = {
        "qid": "q-window",
        "question": "Which policy?",
        "category": "single_doc",
        "expected_behavior": "answer",
        "gold_answer": "gold policy",
        "gold_key_points": [],
        "forbidden_points": [],
        "gold_doc_ids": ["gold.md"],
        "gold_chunk_ids": [3],
    }
    preds = {
        "q-window": {
            "answer": "gold policy",
            "retrieved_chunk_ids": "1,2,3",
        }
    }

    rows, _, _ = evaluate(
        [question],
        preds,
        ChunkIndex({1: "other.md", 2: "other.md", 3: "gold.md"}, {}, "test"),
        k=3,
    )

    row = rows[0]
    assert row["retrieved_chunk_ids"] == "[1, 2, 3]"
    assert row["hit_chunk_ids"] == "[3]"
    assert row["missing_gold_chunk_ids"] == "[]"
    assert "matched_gold_chunk_ids" not in row
    assert "retrieval_evidence_recall_at_k" not in row
    assert "retrieved_chunk_ids_at_k" not in row
    assert "missing_gold_chunk_ids_at_k" not in row


def _question(gold_chunk_ids=None) -> dict:
    return {
        "qid": "q1",
        "question": "政策是什么？",
        "category": "single_doc",
        "difficulty": "easy",
        "expected_behavior": "answer",
        "gold_answer": "政策",
        "gold_key_points": [],
        "forbidden_points": [],
        "gold_doc_ids": ["gold.md"],
        "gold_chunk_ids": gold_chunk_ids or [1],
    }


def _prediction(
    final_ids: str,
    context_ids: str,
    reference_ids: str,
) -> dict:
    return {
        "prediction_schema_version": 2,
        "answer": "政策",
        "final_chunk_ids": final_ids,
        "retrieved_chunk_ids": final_ids,
        "context_chunk_ids": context_ids,
        "reference_chunk_ids": reference_ids,
        "context_char_count": 100,
        "top_k": 5,
        "decompose_top_k": 7,
        "decompose_llm_used": False,
        "status": "OK",
    }


def _evaluate(pred: dict, question: dict | None = None):
    rows, summary, _ = evaluate(
        [question or _question()],
        {"q1": pred},
        ChunkIndex(
            {1: "gold.md", 2: "other.md", 3: "gold.md"},
            {},
            "test",
        ),
        k=5,
    )
    return rows[0], summary["overall"]


@pytest.mark.parametrize(
    ("final_ids", "context_ids", "reference_ids", "expected"),
    [
        ("2", "2", "", (0.0, 0.0, 0.0)),
        ("1,2", "2", "", (1.0, 0.0, 0.0)),
        ("1", "1", "", (1.0, 1.0, 0.0)),
        ("1", "1", "1", (1.0, 1.0, 1.0)),
    ],
)
def test_evidence_funnel_scenarios(
    final_ids,
    context_ids,
    reference_ids,
    expected,
):
    row, _ = _evaluate(
        _prediction(final_ids, context_ids, reference_ids)
    )

    assert (
        row["retrieval_gold_chunk_hit_at_k"],
        row["context_gold_chunk_hit"],
        row["reference_gold_chunk_hit"],
    ) == expected


def test_context_truncation_and_reference_loss_have_distinct_taxonomy():
    context_row, _ = _evaluate(_prediction("1,2", "2", ""))
    reference_row, _ = _evaluate(_prediction("1", "1", ""))

    context_failure = classify_rag_failures([context_row])[0]
    reference_failure = classify_rag_failures([reference_row])[0]

    assert context_row["context_truncated"] is True
    assert context_row["final_to_context_drop_count"] == 1
    assert context_row[
        "gold_present_in_final_but_missing_in_context"
    ] is True
    assert "rag_context_gold_miss" in context_failure["failure_reasons"]
    assert "rag_reference_gold_miss" not in context_failure[
        "failure_reasons"
    ]
    assert reference_row[
        "gold_present_in_context_but_not_referenced"
    ] is True
    assert "rag_reference_gold_miss" in reference_failure[
        "failure_reasons"
    ]


def test_equivalent_gold_hit_is_not_classified_as_complete_loss():
    row, _ = _evaluate(
        _prediction("3", "3", "3"),
        question=_question([1, 3]),
    )

    assert row["retrieval_gold_chunk_hit_at_k"] == 1.0
    assert row["context_gold_chunk_hit"] == 1.0
    assert row["reference_gold_chunk_hit"] == 1.0
    assert row["gold_present_in_final_but_missing_in_context"] is False
    assert row["gold_present_in_context_but_not_referenced"] is False
    assert classify_rag_failures([row]) == []


def test_v1_prediction_keeps_retrieval_metrics_and_marks_new_metrics_unavailable():
    row, overall = _evaluate(
        {
            "answer": "政策",
            "retrieved_chunk_ids": "1",
        }
    )

    assert row["prediction_schema_version"] == 1
    assert row["retrieval_gold_chunk_recall_at_k"] == 1.0
    assert row["context_gold_chunk_recall"] is None
    assert row["reference_gold_chunk_recall"] is None
    assert row["context_chunk_ids"] is None
    assert row["reference_chunk_ids"] is None
    assert overall["context_gold_chunk_recall"] is None
    assert overall["reference_gold_chunk_recall"] is None
    assert overall["context_truncation_rate"] is None
    assert overall["gold_lost_by_context_count"] is None


def test_evidence_funnel_summary_includes_counts_and_diagnostics():
    row, overall = _evaluate(_prediction("1,2", "2", ""))

    assert row["final_chunk_count"] == 2
    assert row["context_chunk_count"] == 1
    assert row["reference_chunk_count"] == 0
    assert overall["avg_final_chunk_count"] == 2.0
    assert overall["avg_context_chunk_count"] == 1.0
    assert overall["avg_reference_chunk_count"] == 0.0
    assert overall["context_truncation_rate"] == 1.0
    assert overall["gold_lost_by_context_count"] == 1
    assert overall["gold_lost_by_reference_count"] == 0


def test_v3_observation_cutoff_and_reference_metrics_without_context():
    pred = {"prediction_schema_version": 3, "answer": "政策", "status": "OK",
            "retrieved_chunk_ids": "2,1,2,3", "reference_chunk_ids": "3",
            "final_chunk_ids": "1"}  # 历史字段不能覆盖 v3 实际观察。
    rows, summary, _ = evaluate([_question([3])], {"q1": pred},
                                ChunkIndex({1: "gold.md", 2: "other.md", 3: "gold.md"}, {}, "test"), k=2)
    row = rows[0]
    assert row["retrieval_gold_chunk_recall_at_k"] == 0.0
    assert row["retrieved_chunk_ids"] == "[2, 1, 3]"
    assert row["retrieved_chunk_count"] == 3
    assert row["reference_gold_chunk_recall"] == 1.0
    assert row["context_gold_chunk_recall"] is None
    assert row["context_chunk_ids"] is None
    assert row["final_chunk_ids"] is None
    assert summary["overall"]["context_gold_chunk_recall"] is None


def test_runtime_failure_has_no_secondary_retrieval_or_answer_classification():
    row, _ = _evaluate({})
    row.update({
        "gold_present_in_final_but_missing_in_context": True,
        "gold_present_in_context_but_not_referenced": True,
        "gold_observed_but_not_referenced": True,
        "key_point_judge_used": True, "missing_count": 1,
        "behavior_label": "should_answer_but_refused",
    })
    failure, = classify_rag_failures([row])
    assert failure["failure_reasons"] == ["runtime_error"]
