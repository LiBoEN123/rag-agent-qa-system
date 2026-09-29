"""Prediction v3 的严格 Gold Recall、引用覆盖与离线输入合同。"""

import json

import pytest

from scripts.evaluation.rag.common import ChunkIndex, load_predictions
from scripts.evaluation.rag.evaluation import evaluate
from scripts.evaluation.rag.retrieval import score_retrieval
from scripts.evaluation.report import classify_rag_failures


def test_evidence_in_another_document_does_not_inflate_gold_recall():
    question = {
        "gold_doc_ids": ["gold.md"], "gold_chunk_ids": [1, 2],
        "gold_evidence": [{"text": "alpha policy"}, {"text": "beta policy"}],
    }
    result = score_retrieval(
        {"retrieved_chunk_ids": "101,2,3"}, question,
        {1: "gold.md", 2: "gold.md", 3: "gold.md", 101: "faq.md"},
        {1: "alpha policy", 2: "beta policy", 3: "unrelated", 101: "alpha policy"}, k=5,
    )
    assert result["hit_chunk_ids"] == [2]
    assert result["missing_gold_chunk_ids"] == [1]
    assert result["gold_chunk_recall_at_k"] == 0.5


def test_gold_chunks_are_scored_against_current_index_when_evidence_resolves():
    result = score_retrieval(
        {"retrieved_chunk_ids": [4]},
        {"gold_doc_ids": ["gold.md"], "gold_chunk_ids": [99],
         "gold_evidence": [{"chunk_id": 99, "text": "moved evidence"}]},
        {4: "gold.md", 99: "gold.md"},
        {4: "moved evidence", 99: "old chunk without the evidence"}, k=5,
    )
    assert result["scored_gold_chunk_ids"] == result["hit_chunk_ids"] == [4]
    assert result["missing_gold_chunk_ids"] == []
    assert result["gold_chunk_recall_at_k"] == 1.0


def _prediction(retrieved="1", referenced="1"):
    return {"prediction_schema_version": 3, "answer": "政策", "status": "OK",
            "retrieved_chunk_ids": retrieved, "reference_chunk_ids": referenced}


def _evaluate(pred, gold=(1,), k=5):
    return evaluate(
        [{"qid": "q1", "question": "政策是什么？", "category": "single_doc",
          "expected_behavior": "answer", "gold_chunk_ids": list(gold)}],
        {"q1": pred} if pred is not None else {},
        ChunkIndex({1: "gold.md", 2: "other.md", 3: "gold.md"}, {}, "test"), k=k,
    )


def test_v3_observation_cutoff_and_reference_recall_are_independent():
    rows, summary = _evaluate(_prediction("2,1,2,3", "3"), gold=(3,), k=2)
    row = rows[0]
    assert row["retrieval_gold_chunk_recall_at_k"] == 0.0
    assert row["retrieved_chunk_ids"] == "[2, 1, 3]"
    assert row["gold_chunk_ids"] == "[3]"
    assert row["reference_gold_chunk_recall"] == 1.0
    assert summary["overall"]["reference_gold_chunk_recall"] == 1.0
    assert not any("context" in key or "final_chunk" in key for key in row)
    assert set(summary["overall"]) == {
        "num_questions", "prediction_coverage", "retrieval_gold_chunk_recall_at_k",
        "reference_gold_chunk_recall", "avg_answer_correctness", "llm_judge_coverage",
    }


@pytest.mark.parametrize("retrieved,referenced,gold,recalls,reasons", [
    ("2", "", (1,), (0.0, 0.0), ["rag_chunk_miss"]),
    ("1,2", "", (1,), (1.0, 0.0), ["rag_reference_gold_miss"]),
    ("3", "3", (1, 3), (0.5, 0.5), []),
])
def test_recall_and_failure_classification(retrieved, referenced, gold, recalls, reasons):
    rows, _ = _evaluate(_prediction(retrieved, referenced), gold)
    row = rows[0]
    assert (row["retrieval_gold_chunk_recall_at_k"], row["reference_gold_chunk_recall"]) == recalls
    failures = classify_rag_failures(rows)
    assert (failures[0]["failure_reasons"] if failures else []) == reasons


@pytest.mark.parametrize("version", [None, 1, 2, 4, "invalid"])
def test_non_v3_predictions_are_rejected_before_judge(version, tmp_path):
    pred = {**_prediction(), "qid": "q1", "prediction_schema_version": version}
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps([pred]))
    with pytest.raises(ValueError, match="Schema v3"):
        load_predictions(path)
    with pytest.raises(ValueError, match="Schema v3"):
        _evaluate(pred)


@pytest.mark.parametrize("field,value", [("reference_chunk_ids", None), ("retrieved_chunk_ids", "chunk_1")])
def test_v3_rejects_incomplete_or_legacy_evidence_fields(field, value):
    with pytest.raises(ValueError, match=field):
        _evaluate({**_prediction(), field: value})


def test_runtime_failure_has_no_secondary_retrieval_or_answer_classification():
    rows, _ = _evaluate(None)
    rows[0]["gold_observed_but_not_referenced"] = True
    failure, = classify_rag_failures(rows)
    assert failure["failure_reasons"] == ["runtime_error"]
