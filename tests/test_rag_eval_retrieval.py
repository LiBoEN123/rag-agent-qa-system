import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_eval.retrieval import score_retrieval
from scripts.rag_eval.evaluation import evaluate
from scripts.rag_eval.common import ChunkIndex


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
