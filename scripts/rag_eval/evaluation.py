from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from app.common.logging import get_logger
from scripts.rag_eval.common import ChunkIndex, get_answer, is_refusal, json_dumps, mean
from scripts.rag_eval.judge import (
    LLMJudgeConfig,
    evaluate_forbidden_with_llm,
    evaluate_key_points_with_llm,
    unavailable_forbidden_result,
    unavailable_key_point_result,
)
from scripts.rag_eval.retrieval import collect_retrieved_evidence, score_retrieval


logger = get_logger("rag_evaluator")


def behavior_score(answer: str, expected: str, key_recall: float, forbidden_hit: bool) -> Tuple[float, str]:
    """按期望行为、关键点覆盖和禁用信息给答案行为打分。"""
    refused = is_refusal(answer)
    if not answer.strip():
        return 0.0, "empty_answer"

    if expected == "abstain":
        if refused and not forbidden_hit:
            return 1.0, "correct_abstain"
        if refused and forbidden_hit:
            return 0.5, "abstain_with_forbidden_detail"
        return 0.0, "should_abstain_but_answered"

    if expected == "partial_answer":
        # A good partial answer should answer the known part and explicitly refuse/limit the unknown part.
        if refused and key_recall >= 0.45 and not forbidden_hit:
            return 1.0, "correct_partial"
        if refused and key_recall >= 0.25:
            return 0.7, "partial_but_incomplete"
        if not refused and key_recall >= 0.60:
            return 0.45, "answered_known_part_but_no_refusal"
        if refused:
            return 0.35, "refused_too_much"
        return 0.0, "poor_partial"

    # expected == answer
    # 注意：trap 题里“文档没有说明 X / 不能确认 X”经常是正确答案的一部分，
    # 不能把所有 refusal pattern 都视为拒答。关键点覆盖足够高时，视为“有依据的限制性回答”。
    if refused:
        if key_recall >= 0.70 and not forbidden_hit:
            return 1.0, "answered_with_supported_limitation"
        if key_recall >= 0.45 and not forbidden_hit:
            return 0.6, "partly_answered_with_limitation"
        return 0.0, "should_answer_but_refused"
    if forbidden_hit:
        return 0.5, "answered_with_forbidden_detail"
    return 1.0, "answered"


def final_score(expected: str, behavior: float, key_recall: float, retrieval_hit: float, forbidden_hit: bool) -> float:
    """综合行为、关键点、检索和禁用命中计算最终分数。"""
    clean = 0.0 if forbidden_hit else 1.0
    if expected == "abstain":
        return 0.70 * behavior + 0.20 * key_recall + 0.10 * clean
    if expected == "partial_answer":
        return 0.25 * behavior + 0.40 * key_recall + 0.20 * retrieval_hit + 0.15 * clean
    return 0.20 * behavior + 0.50 * key_recall + 0.20 * retrieval_hit + 0.10 * clean


def evaluate(
    eval_items: List[Dict[str, Any]],
    preds: Dict[str, Dict[str, Any]],
    chunk_index: ChunkIndex,
    k: int,
    judge_config: LLMJudgeConfig | None = None,
    key_point_evaluator: Any = None,
    forbidden_evaluator: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """对所有评测题逐条判分并生成汇总。"""
    rows = []
    judge_details = []
    use_llm_judge = bool(judge_config and judge_config.enabled)
    key_point_evaluator = key_point_evaluator or evaluate_key_points_with_llm
    forbidden_evaluator = forbidden_evaluator or evaluate_forbidden_with_llm
    for q in eval_items:
        qid = str(q["qid"])
        pred = preds.get(qid, {})
        answer = get_answer(pred)
        retrieval = score_retrieval(pred, q, chunk_index.chunk_doc_map, chunk_index.chunk_text_map, k)
        judge_evidence = collect_retrieved_evidence(retrieval, chunk_index)

        key_point_result = unavailable_key_point_result(q.get("gold_key_points", []), "LLM judge 未启用")
        forbidden_result = unavailable_forbidden_result(q.get("forbidden_points", []), "LLM judge 未启用")
        key_point_judge_raw = ""
        forbidden_judge_raw = ""
        key_point_judge_used = False
        forbidden_judge_used = False
        judge_errors: List[str] = []

        if use_llm_judge:
            try:
                key_point_result = key_point_evaluator(
                    question=q.get("question", ""),
                    gold_answer=q.get("gold_answer", ""),
                    gold_key_points=q.get("gold_key_points", []),
                    model_answer=answer,
                    evidence=judge_evidence,
                    judge_config=judge_config,
                    category=q.get("category", ""),
                    expected_behavior=q.get("expected_behavior", ""),
                    qid=qid,
                )
                key_point_judge_used = True
                key_point_judge_raw = key_point_result.get("judge_raw_response", "")
            except Exception as exc:
                message = f"key_point_judge_error: {exc}"
                judge_errors.append(message)
                logger.warning(f"[judge] qid={qid} {message}")
                key_point_result = unavailable_key_point_result(q.get("gold_key_points", []), message)

            try:
                forbidden_result = forbidden_evaluator(
                    question=q.get("question", ""),
                    model_answer=answer,
                    forbidden_points=q.get("forbidden_points", []),
                    evidence=judge_evidence,
                    judge_config=judge_config,
                    gold_answer=q.get("gold_answer", ""),
                    qid=qid,
                )
                forbidden_judge_used = True
                forbidden_judge_raw = forbidden_result.get("judge_raw_response", "")
            except Exception as exc:
                message = f"forbidden_judge_error: {exc}"
                judge_errors.append(message)
                logger.warning(f"[judge] qid={qid} {message}")
                forbidden_result = unavailable_forbidden_result(q.get("forbidden_points", []), message)

        key_point_judgments = key_point_result["key_point_judgments"]
        forbidden_judgments = forbidden_result["forbidden_judgments"]
        key_point_recall = float(key_point_result["key_point_recall"])
        forbidden_hit = bool(forbidden_result["forbidden_hit"])
        missing_key_points = key_point_result["missing_key_points"]
        partial_key_points = key_point_result["partial_key_points"]
        contradicted_key_points = key_point_result["contradicted_key_points"]
        matched_key_points = [
            item["key_point"]
            for item in key_point_judgments
            if item.get("status") == "covered"
        ]
        forbidden_items = [
            item["forbidden_point"]
            for item in forbidden_judgments
            if item.get("hit")
        ]

        behavior, behavior_label = behavior_score(
            answer,
            q.get("expected_behavior", "answer"),
            key_point_recall,
            forbidden_hit,
        )
        # final_score 使用严格 gold chunk-id recall，而不是 loose 的 evidence recall。
        # 否则检索到非 gold 文档中的重复片段时，chunk-id 指标和综合分会被高估。
        fscore = final_score(
            q.get("expected_behavior", "answer"),
            behavior,
            key_point_recall,
            retrieval["gold_chunk_recall_at_k"],
            forbidden_hit,
        )

        judge_error = " | ".join(judge_errors)
        row = {
            "qid": qid,
            "question": q.get("question", ""),
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "expected_behavior": q.get("expected_behavior", ""),
            "has_prediction": bool(pred),
            "final_score": round(fscore, 4),
            "behavior_score": round(behavior, 4),
            "behavior_label": behavior_label,
            "key_point_recall": round(key_point_recall, 4),
            "retrieval_gold_chunk_hit_at_k": round(retrieval["gold_chunk_hit_at_k"], 4),
            "retrieval_gold_chunk_recall_at_k": round(retrieval["gold_chunk_recall_at_k"], 4),
            "retrieval_all_gold_hit_at_k": round(retrieval["all_gold_hit_at_k"], 4),
            "retrieval_doc_hit_at_k": round(retrieval["doc_hit_at_k"], 4),
            "forbidden_hit": forbidden_hit,
            "covered_count": key_point_result["covered_count"],
            "partial_count": key_point_result["partial_count"],
            "missing_count": key_point_result["missing_count"],
            "contradicted_count": key_point_result["contradicted_count"],
            "missing_key_points": " | ".join(missing_key_points),
            "partial_key_points": " | ".join(partial_key_points),
            "contradicted_key_points": " | ".join(contradicted_key_points),
            "matched_key_points": " | ".join(matched_key_points),
            "forbidden_points_hit": " | ".join(forbidden_items),
            "key_point_judgments_json": json_dumps(key_point_judgments),
            "forbidden_judgments_json": json_dumps(forbidden_judgments),
            "judge_error": judge_error,
            "key_point_judge_used": key_point_judge_used,
            "forbidden_judge_used": forbidden_judge_used,
            "gold_chunk_ids": json.dumps(retrieval["scored_gold_chunk_ids"], ensure_ascii=False),
            "hit_chunk_ids": json.dumps(retrieval["hit_chunk_ids"], ensure_ascii=False),
            "missing_gold_chunk_ids": json.dumps(retrieval["missing_gold_chunk_ids"], ensure_ascii=False),
            "retrieved_chunk_ids": json.dumps(retrieval["retrieved_chunk_ids"], ensure_ascii=False),
            "gold_answer": q.get("gold_answer", ""),
            "answer": answer,
        }
        rows.append(row)
        judge_details.append({
            "qid": qid,
            "question": q.get("question", ""),
            "category": q.get("category", ""),
            "expected_behavior": q.get("expected_behavior", ""),
            "answer": answer,
            "key_point_recall": round(key_point_recall, 4),
            "forbidden_hit": forbidden_hit,
            "key_point_judge_used": key_point_judge_used,
            "forbidden_judge_used": forbidden_judge_used,
            "key_point_judgments": key_point_judgments,
            "forbidden_judgments": forbidden_judgments,
            "missing_key_points": missing_key_points,
            "partial_key_points": partial_key_points,
            "contradicted_key_points": contradicted_key_points,
            "covered_count": key_point_result["covered_count"],
            "partial_count": key_point_result["partial_count"],
            "missing_count": key_point_result["missing_count"],
            "contradicted_count": key_point_result["contradicted_count"],
            "judge_error": judge_error,
            "key_point_judge_raw": key_point_judge_raw,
            "forbidden_judge_raw": forbidden_judge_raw,
        })
    summary = summarize(rows)
    return rows, summary, judge_details


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
        out[name] = {
            "num_questions": len(vals),
            "prediction_coverage": round(mean([1.0 if r["has_prediction"] else 0.0 for r in vals]), 4),
            "avg_final_score": round(mean([float(r["final_score"]) for r in vals]), 4),
            "avg_behavior_score": round(mean([float(r["behavior_score"]) for r in vals]), 4),
            "avg_key_point_recall": round(mean([float(r["key_point_recall"]) for r in vals]), 4),
            "gold_chunk_hit_at_k": round(mean([float(r["retrieval_gold_chunk_hit_at_k"]) for r in vals]), 4),
            "gold_chunk_recall_at_k": round(mean([float(r["retrieval_gold_chunk_recall_at_k"]) for r in vals]), 4),
            "all_gold_hit_at_k": round(mean([float(r["retrieval_all_gold_hit_at_k"]) for r in vals]), 4),
            "doc_hit_at_k": round(mean([float(r["retrieval_doc_hit_at_k"]) for r in vals]), 4),
            "forbidden_rate": round(mean([1.0 if r["forbidden_hit"] else 0.0 for r in vals]), 4),
            "llm_judge_coverage": round(mean([1.0 if r["key_point_judge_used"] else 0.0 for r in vals]), 4),
            "avg_contradicted_count": round(mean([float(r["contradicted_count"]) for r in vals]), 4),
        }
    return out
