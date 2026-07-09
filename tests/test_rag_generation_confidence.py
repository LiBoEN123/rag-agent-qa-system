from app.rag.generation import (
    estimate_answer_confidence,
    estimate_retrieval_confidence,
)


def test_retrieval_confidence_is_low_for_weak_evidence():
    chunks = [
        {
            "vector_score": 0.12,
            "rerank_score": 0.10,
            "hybrid_score": 0.10,
            "bm25_score": 0.0,
        }
    ]

    assert estimate_retrieval_confidence(chunks) < 0.2


def test_retrieval_confidence_uses_rerank_and_hybrid_signals():
    chunks = [
        {
            "reference_id": "Reference 1",
            "vector_score": 0.28,
            "rerank_score": 0.50,
            "hybrid_score": 0.38,
            "bm25_score": 0.0,
        },
        {
            "reference_id": "Reference 2",
            "vector_score": 0.12,
            "rerank_score": 0.10,
            "hybrid_score": 0.10,
            "bm25_score": 0.0,
        },
    ]

    assert estimate_retrieval_confidence(chunks) >= 0.3


def test_answer_confidence_requires_bound_references():
    chunks = [
        {
            "reference_id": "Reference 1",
            "vector_score": 0.74,
            "rerank_score": 0.98,
            "hybrid_score": 0.96,
            "bm25_score": 10.0,
        }
    ]
    retrieval_confidence = estimate_retrieval_confidence(chunks)

    assert estimate_answer_confidence(chunks, [], "OK", retrieval_confidence) == 0.0


def test_answer_confidence_combines_retrieval_and_citation_quality():
    chunks = [
        {
            "reference_id": "Reference 1",
            "vector_score": 0.74,
            "rerank_score": 0.98,
            "hybrid_score": 0.96,
            "bm25_score": 10.0,
        },
        {
            "reference_id": "Reference 2",
            "vector_score": 0.70,
            "rerank_score": 0.97,
            "hybrid_score": 0.94,
            "bm25_score": 8.0,
        },
    ]
    references = [
        {"reference_id": "Reference 1"},
        {"reference_id": "Reference 2"},
    ]
    retrieval_confidence = estimate_retrieval_confidence(chunks)

    answer_confidence = estimate_answer_confidence(
        chunks,
        references,
        "OK",
        retrieval_confidence
    )

    assert answer_confidence > 0.7
    assert answer_confidence <= retrieval_confidence + 0.1
