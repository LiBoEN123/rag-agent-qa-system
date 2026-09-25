"""表格评估的少量胶水：复用生产检索、重排、诊断分数门及现有 evidence scorer。

调用前由评估方加载隔离索引。本模块不初始化、不写索引、不调用回答模型。
单查询诊断选择不代表 Agent 的 ToolMessage 上下文。
"""

from copy import deepcopy
from typing import Any

from scripts.evaluation.rag.common import normalize_text
from scripts.evaluation.rag.retrieval import (
    chunks_to_index,
    doc_id_matches,
    doc_label_from_metadata,
    resolve_gold_chunk_ids,
    score_retrieval,
)


def relation_chunk_ids(question: dict, chunks: list[dict]) -> list[int]:
    """所有 gold 关键点须出现在同一来源的同一个 table chunk；不跨块拼接。"""
    points = question.get("gold_key_points") or []
    if not points or question["expected_behavior"] == "abstain":
        return []
    matches = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        if metadata.get("element_type") != "table" or not doc_id_matches(
            doc_label_from_metadata(metadata), question["gold_doc_ids"],
        ):
            continue
        text = normalize_text(chunk["text"])
        if all(any(
            normalize_text(alias) and normalize_text(alias) in text
            for alias in (point.get("aliases") or [point["point"]])
        ) for point in points):
            matches.append(int(metadata["chunk_id"]))
    return matches


def diagnose_questions(questions: list[dict], *, top_k: int = 5) -> list[dict[str, Any]]:
    """只运行生产 retrieve/rerank/diagnostic gate，并复用 scorer 计算固定 K 指标。"""
    from app.rag import runtime
    from app.rag.policy import should_refuse

    index = chunks_to_index(runtime.chunks, source="current-chroma")
    results = []
    for question in questions:
        answerable = question["expected_behavior"] != "abstain"
        gold = resolve_gold_chunk_ids(question, index.chunk_text_map, index.chunk_doc_map)
        relation_gold = relation_chunk_ids(question, runtime.chunks)
        retrieved = runtime.retrieve(question["question"], top_k)
        # 生产 rerank 会在候选字典上写 score；保留独立的原始召回快照。
        reranked = runtime.rerank(deepcopy(retrieved), question["question"])
        final = reranked[:top_k]
        gate_passed = bool(final) and not should_refuse(final)
        selected = final if gate_passed else []

        def ids(items):
            return [int(item["chunk_id"]) for item in items]

        def metrics(items):
            return {str(k): score_retrieval(
                {"retrieved_chunk_ids": ids(items)}, question,
                index.chunk_doc_map, index.chunk_text_map, k,
            ) for k in (1, 3, 5)} if answerable else None

        def brief(items):
            return [{
                "rank": rank, "chunk_id": item["chunk_id"],
                "source": item["source_path"], "file_name": item["file_name"],
                **{key: item["metadata"].get(key) for key in (
                    "source_type", "element_type", "table_id", "row_start", "row_end", "page_number",
                )},
                "hybrid_score": item["hybrid_score"],
                "rerank_score": item.get("rerank_score"),
            } for rank, item in enumerate(items, 1)]

        def relation_hit(items):
            return bool(set(ids(items)) & set(relation_gold)) if answerable else None

        results.append({
            "qid": question["qid"], "question": question["question"],
            "expected_behavior": question["expected_behavior"], "notes": question["notes"],
            "gold_source": question["gold_doc_ids"], "gold_evidence": question["gold_evidence"],
            "resolved_gold_chunk_ids": gold, "relation_gold_chunk_ids": relation_gold,
            "gold_chunks": [c for c in runtime.chunks if c["metadata"]["chunk_id"] in gold],
            "retrieve_top5": brief(retrieved[:5]), "rerank_top5": brief(final),
            "retrieve_candidate_ids": ids(retrieved), "candidate_count": len(retrieved),
            "rerank_all_ids": ids(reranked), "final_chunk_ids": ids(final),
            "diagnostic_selected_chunk_ids": ids(selected),
            "evidence_gate_passed": gate_passed,
            "retrieval_metrics": metrics(retrieved), "rerank_metrics": metrics(reranked),
            "correct_chunk_retrieved": relation_hit(retrieved),
            "correct_chunk_retrieved_top5": relation_hit(retrieved[:5]),
            "correct_chunk_reranked": relation_hit(reranked),
            "correct_chunk_entered_final_chunks": relation_hit(final),
            "relation_chunk_survives_diagnostic_selection": relation_hit(selected),
        })
        print(f"Diagnosed {question['qid']}", flush=True)
    return results
