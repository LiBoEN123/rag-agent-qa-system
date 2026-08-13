"""独立 RAG 参数链与 Prediction Schema v2 测试。"""

import csv
import json

import pytest

from scripts.evaluation import rag_predictions, runner


def _production_chunk(chunk_id: int, text: str = "证据") -> dict:
    return {
        "reference_id": f"Reference {chunk_id}",
        "file_name": "policy.md",
        "source_path": "/tmp/policy.md",
        "chunk_id": chunk_id,
        "text": text,
        "hybrid_score": 0.9,
        "metadata": {"chunk_id": chunk_id},
    }


def _result(chunks: list[dict], references: list[dict]) -> dict:
    return {
        "status": "OK",
        "answer": "回答",
        "final_chunks": chunks,
        "references": references,
        "decompose_llm_used": True,
    }


def test_run_rag_once_passes_both_parameters_to_subgraph(monkeypatch):
    captured = {}
    chunks = [_production_chunk(1)]
    monkeypatch.setattr(
        rag_predictions,
        "retrieve",
        lambda question, top_k: chunks,
    )

    def fake_subgraph(**kwargs):
        captured.update(kwargs)
        return _result(chunks, [{"chunk_id": 1}])

    monkeypatch.setattr(
        rag_predictions,
        "run_rag_subgraph",
        fake_subgraph,
    )

    _, trace = rag_predictions.run_rag_once_with_result(
        "问题",
        top_k=3,
        decompose_top_k=7,
    )

    assert captured["top_k"] == 3
    assert captured["decompose_top_k"] == 7
    assert captured["retrieved_chunks"] is chunks
    assert trace.top_k == 3
    assert trace.decompose_top_k == 7


@pytest.mark.parametrize("value", [0, -1])
def test_decompose_top_k_must_be_positive(value, tmp_path, monkeypatch):
    monkeypatch.setattr(
        rag_predictions,
        "initialize_rag_for_predictions",
        lambda force_rebuild=False: None,
    )
    eval_file = tmp_path / "eval.json"
    eval_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="decompose_top_k"):
        rag_predictions.run_predictions(
            eval_file=eval_file,
            out_file=tmp_path / "predictions.csv",
            decompose_top_k=value,
        )


@pytest.mark.parametrize("decompose_top_k", [1, 7])
def test_prediction_v2_writes_safe_evidence_metadata(
    decompose_top_k,
    tmp_path,
    monkeypatch,
):
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps([{"qid": "q1", "question": "问题"}]),
        encoding="utf-8",
    )
    output = tmp_path / "predictions.csv"
    trace = rag_predictions.RagPredictionTrace(
        final_chunk_ids=(1, 2),
        context_chunk_ids=(1,),
        reference_chunk_ids=(1,),
        context_char_count=123,
        top_k=5,
        decompose_top_k=decompose_top_k,
        decompose_llm_used=True,
        status="OK",
    )
    monkeypatch.setattr(
        rag_predictions,
        "initialize_rag_for_predictions",
        lambda force_rebuild=False: None,
    )
    monkeypatch.setattr(
        rag_predictions,
        "run_rag_once_with_result",
        lambda question, top_k, decompose_top_k: (
            {"status": "OK", "answer": "回答"},
            trace,
        ),
    )

    rows = rag_predictions.run_predictions(
        eval_file=eval_file,
        out_file=output,
        top_k=5,
        decompose_top_k=decompose_top_k,
    )
    persisted = list(
        csv.DictReader(
            output.open(encoding="utf-8-sig", newline="")
        )
    )[0]

    assert rows[0]["prediction_schema_version"] == 2
    assert persisted["final_chunk_ids"] == "1,2"
    assert persisted["retrieved_chunk_ids"] == "1,2"
    assert persisted["context_chunk_ids"] == "1"
    assert persisted["reference_chunk_ids"] == "1"
    assert persisted["final_chunk_count"] == "2"
    assert persisted["context_chunk_count"] == "1"
    assert persisted["reference_chunk_count"] == "1"
    assert persisted["context_char_count"] == "123"
    assert persisted["decompose_top_k"] == str(decompose_top_k)
    assert persisted["status"] == "OK"
    assert {
        "context",
        "chunk_text",
        "full_text",
        "api_key",
        "authorization",
        "secret",
    }.isdisjoint(persisted)
    assert "证据" not in output.read_text(encoding="utf-8-sig")


def test_run_rag_baseline_passes_parameters_and_validates_metadata(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_predictions(**kwargs):
        captured.update(kwargs)
        return [
            {
                "prediction_schema_version": 2,
                "top_k": kwargs["top_k"],
                "decompose_top_k": kwargs["decompose_top_k"],
            }
        ]

    monkeypatch.setattr(
        rag_predictions,
        "run_predictions",
        fake_predictions,
    )
    monkeypatch.setattr(
        "scripts.evaluation.rag_evaluator.run_rag_evaluation",
        lambda **kwargs: {
            "rows": [],
            "summary": {"overall": {}},
            "judge_enabled": False,
        },
    )

    result = runner.run_rag_baseline(
        rag_eval_path=tmp_path / "eval.json",
        output_dir=tmp_path / "out",
        top_k=3,
        decompose_top_k=7,
        force_rebuild=False,
        use_llm_judge=False,
        judge_model=None,
    )

    assert captured["top_k"] == 3
    assert captured["decompose_top_k"] == 7
    assert result["prediction_metadata"] == {
        "prediction_schema_version": 2,
        "effective_top_k": 3,
        "effective_decompose_top_k": 7,
    }


def test_effective_parameter_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="decompose_top_k"):
        runner.validate_effective_rag_parameters(
            requested_top_k=5,
            requested_decompose_top_k=5,
            prediction_metadata={
                "effective_top_k": 5,
                "effective_decompose_top_k": 10,
            },
        )
