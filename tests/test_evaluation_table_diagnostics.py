"""仅校验本次题集合同和跨块/跨来源假命中的防护，不调用模型。"""

import json
from copy import deepcopy
from pathlib import Path

from scripts.evaluation.rag.common import load_eval
from scripts.evaluation.rag.retrieval import chunks_to_index, resolve_gold_chunk_ids
from scripts.evaluation.table_diagnostics import relation_chunk_ids


DATASET = Path(__file__).resolve().parents[1] / "data/eval/rag_eval_table_v1.json"


def test_table_dataset_reuses_baseline_contract_and_has_fact_gold():
    baseline = json.loads(DATASET.with_name("rag_eval_questions_v2_semantic_fixed.json").read_text())
    payload = json.loads(DATASET.read_text())
    questions = load_eval(DATASET)
    assert payload["schema"] == baseline["schema"]
    assert 12 <= len(questions) <= 16
    assert len({q["qid"] for q in questions}) == len(questions)
    for question in questions:
        assert set(question) == set(baseline["items"][0])
        assert question["category"] in {"single_doc", "cross_doc", "partial_answer", "not_in_doc", "trap"}
        assert question["expected_behavior"] in {"answer", "partial_answer", "abstain"}
        assert question["notes"].startswith(("CONTROLLED_PARSE:", "CONTROLLED_STRUCTURE:"))
        if question["expected_behavior"] == "answer":
            assert question["gold_doc_ids"] and question["gold_evidence"] and question["gold_key_points"]
            assert all("[表格]" not in e["text"] for e in question["gold_evidence"])
            assert all(p["aliases"] and all(p["aliases"]) for p in question["gold_key_points"])


def test_relation_requires_same_table_chunk_and_source():
    question = next(q for q in load_eval(DATASET) if q["qid"] == "t007")

    def chunk(cid, text, *, source="registration-html", kind="table"):
        return {"text": text, "metadata": {"chunk_id": cid, "file_name": source, "element_type": kind}}

    split = [chunk(1, "企业注册 身份证"), chunk(2, "申请表")]
    assert relation_chunk_ids(question, split) == []
    complete = "企业注册 身份证 申请表"
    assert relation_chunk_ids(question, [chunk(3, complete, source="wrong-source")]) == []
    assert relation_chunk_ids(question, [chunk(4, complete, kind="text")]) == []
    assert relation_chunk_ids(question, [*split, chunk(5, complete)]) == [5]


def test_gold_realigns_after_snapshot_ids_change():
    question = deepcopy(next(q for q in load_eval(DATASET) if q["qid"] == "t009"))
    question["gold_chunk_ids"] = [-1]
    chunks = [{
        "text": "事项：企业变更；条件：企业已注册；材料：营业执照、变更表。",
        "metadata": {"chunk_id": 9001, "file_name": "registration-structure", "element_type": "table"},
    }]
    index = chunks_to_index(chunks, "test")
    assert resolve_gold_chunk_ids(question, index.chunk_text_map, index.chunk_doc_map) == [9001]
    assert relation_chunk_ids(question, chunks) == [9001]


def test_abstain_is_not_counted_as_relation_success():
    question = next(q for q in load_eval(DATASET) if q["expected_behavior"] == "abstain")
    assert relation_chunk_ids(question, []) == []


def test_diagnostic_relation_measures_final_selection_without_context(monkeypatch):
    from app.rag import runtime as pipeline
    from scripts.evaluation.table_diagnostics import diagnose_questions
    question = next(q for q in load_eval(DATASET) if q['qid'] == 't007')
    evidence = {
        "chunk_id": 1, "text": "企业注册 身份证 申请表", "file_name": "registration-html",
        "source_path": "fixture", "hybrid_score": 0.9, "vector_score": 0.9,
        "metadata": {"chunk_id": 1, "file_name": "registration-html", "element_type": "table"},
    }
    monkeypatch.setattr(pipeline, "chunks", [evidence])
    monkeypatch.setattr(pipeline, "retrieve", lambda *args: [evidence])
    monkeypatch.setattr(pipeline, "rerank", lambda chunks, query: chunks)
    row = diagnose_questions([question], top_k=1)[0]
    assert row["relation_chunk_survives_diagnostic_selection"] is True
    assert row["diagnostic_selected_chunk_ids"] == [1]
    assert not any("context" in key for key in row)
    monkeypatch.setattr("app.rag.policy.should_refuse", lambda chunks: True)
    row = diagnose_questions([question], top_k=1)[0]
    assert row["relation_chunk_survives_diagnostic_selection"] is False
    assert row["diagnostic_selected_chunk_ids"] == []
