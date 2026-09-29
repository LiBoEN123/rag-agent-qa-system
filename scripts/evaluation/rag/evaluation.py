"""独立汇总 Prediction coverage、检索/引用召回与可选答案正确性。"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Dict, List

from scripts.evaluation.rag.common import ChunkIndex, mean, parse_chunk_ids_field, validate_prediction
from scripts.evaluation.rag.retrieval import score_evidence_chunks, score_retrieval


def _optional_int(value: Any) -> int | None:
    """解析可选整数预测元数据，缺失或非法时返回 None。"""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evaluate(
    eval_items: List[Dict[str, Any]],
    preds: Dict[str, Dict[str, Any]],
    chunk_index: ChunkIndex,
    k: int,
    answer_evaluator: Callable[[dict, str], tuple[float, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Judge 缺席或失败不产生语义分；运行失败独立计入 coverage。"""
    for prediction in preds.values():
        validate_prediction(prediction)
    rows = []
    for q in eval_items:
        qid = str(q["qid"])
        pred = preds.get(qid, {})
        answer = str(pred.get("answer") or "")
        retrieval = score_retrieval(pred, q, chunk_index.chunk_doc_map, chunk_index.chunk_text_map, k)
        retrieved_chunk_ids = retrieval["retrieved_chunk_ids"]
        reference_chunk_ids = parse_chunk_ids_field(pred, "reference_chunk_ids")
        reference_scored = score_evidence_chunks(reference_chunk_ids, retrieval["scored_gold_chunk_ids"])
        if not pred:
            retrieval["gold_chunk_recall_at_k"] = reference_scored["gold_chunk_recall"] = 0.0
        correctness, reason, judge_error = None, "", ""
        if pred and answer_evaluator is not None:
            try:
                correctness, reason = answer_evaluator(q, answer)
            except Exception as exc:
                judge_error = str(exc)
        row = {
            "qid": qid,
            "question": q.get("question", ""),
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "expected_behavior": q.get("expected_behavior", ""),
            "has_prediction": bool(pred),
            "retrieval_gold_chunk_recall_at_k": round(retrieval["gold_chunk_recall_at_k"], 4),
            "reference_gold_chunk_recall": round(reference_scored["gold_chunk_recall"], 4),
            "answer_correctness": correctness,
            "answer_correctness_reason": reason,
            "judge_used": correctness is not None,
            "judge_error": judge_error,
            "gold_chunk_ids": json.dumps(retrieval["scored_gold_chunk_ids"], ensure_ascii=False),
            "retrieved_chunk_ids": json.dumps(retrieved_chunk_ids, ensure_ascii=False),
            "reference_chunk_ids": json.dumps(reference_chunk_ids, ensure_ascii=False),
            "retrieval_call_count": _optional_int(pred.get("retrieval_call_count")),
            "tool_call_count": _optional_int(pred.get("tool_call_count")),
            "llm_call_count": _optional_int(pred.get("llm_call_count")),
            "status": pred.get("status"),
            "gold_observed_but_not_referenced": (
                bool(set(retrieval["scored_gold_chunk_ids"]) & set(retrieved_chunk_ids))
                and not reference_scored["hit_chunk_ids"]
            ),
            "gold_answer": q.get("gold_answer", ""),
            "answer": answer,
        }
        rows.append(row)
    return rows, summarize(rows)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按整体、类别、难度和期望行为汇总评测指标。"""
    groups = {
        "overall": rows,
    }
    for key in ("category", "difficulty", "expected_behavior"):
        by = defaultdict(list)
        for r in rows:
            by[r[key]].append(r)
        for name, vals in by.items():
            groups[f"{key}:{name}"] = vals

    out = {}
    for name, vals in groups.items():
        judged = [r["answer_correctness"] for r in vals if r["judge_used"]]
        out[name] = {
            "num_questions": len(vals),
            "avg_answer_correctness": round(mean(judged), 4) if judged else None,
            "llm_judge_coverage": round(len(judged) / len(vals), 4) if vals else 0.0,
            "prediction_coverage": round(mean([1.0 if r["has_prediction"] else 0.0 for r in vals]), 4),
            "retrieval_gold_chunk_recall_at_k": round(mean([float(r["retrieval_gold_chunk_recall_at_k"]) for r in vals]), 4),
            "reference_gold_chunk_recall": round(mean([float(r["reference_gold_chunk_recall"]) for r in vals]), 4),
        }
    return out
