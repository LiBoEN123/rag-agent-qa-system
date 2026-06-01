# 启动命令：python scripts/evaluate_rag_results.py --eval-file data/eval/rag_eval_questions_v2.json --pred-file outputs/predictions.csv --chunks-file data/manifest/chunks.json --out-dir outputs/rag_baseline --k 5

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ANSWER_FIELDS = ["answer", "generated_answer", "actual_answer", "response", "model_answer"]
CHUNK_FIELDS = ["retrieved_chunk_ids", "retrieved_chunks", "chunk_ids", "top_chunk_ids"]
DOC_FIELDS = ["retrieved_doc_ids", "retrieved_docs", "doc_ids", "top_doc_ids"]

REFUSAL_PATTERNS = [
    "无法确认", "不能确认", "无法从文档", "文档没有", "文档未", "未提到",
    "没有提供", "不知道", "不能回答", "无法回答", "没有说明", "无相关信息",
    "根据现有文档无法", "从给定文档无法", "不在文档中"
]


def normalize_text(s: Any) -> str:
    """Normalize Chinese/English text for loose substring matching."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", "", s)
    # remove common punctuation while keeping code-ish chars useful for Vue terms
    s = re.sub(r"[，。！？、；：,.!?;:（）()【】\[\]{}<>《》“”\"'`·\-_—|/\\]", "", s)
    return s


def has_any(text: str, aliases: Iterable[str]) -> bool:
    norm = normalize_text(text)
    for a in aliases:
        if normalize_text(a) and normalize_text(a) in norm:
            return True
    return False


def is_refusal(answer: str) -> bool:
    return has_any(answer, REFUSAL_PATTERNS)


def parse_listish(value: Any) -> List[Any]:
    """Parse JSON list, pipe/comma/space separated strings, or list of dicts."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            if isinstance(x, dict):
                # common retrieval result shapes
                for key in ("chunk_id", "id", "doc_id", "file_name"):
                    if key in x:
                        out.append(x[key])
                        break
            else:
                out.append(x)
        return out
    if isinstance(value, (int, float)):
        return [int(value)]
    s = str(value).strip()
    if not s:
        return []
    # try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parse_listish(parsed)
        if isinstance(parsed, dict):
            return parse_listish([parsed])
    except Exception:
        pass
    # extract ids from common separators
    parts = re.split(r"[,\|;，；\s]+", s)
    return [p for p in parts if p != ""]


def parse_chunk_ids(pred: Dict[str, Any]) -> List[int]:
    value = None
    for k in CHUNK_FIELDS:
        if k in pred:
            value = pred[k]
            break
    raw = parse_listish(value)
    ids = []
    for x in raw:
        if isinstance(x, int):
            ids.append(x)
            continue
        m = re.search(r"\d+", str(x))
        if m:
            ids.append(int(m.group(0)))
    return ids


def parse_doc_ids(pred: Dict[str, Any]) -> List[str]:
    value = None
    for k in DOC_FIELDS:
        if k in pred:
            value = pred[k]
            break
    return [str(x) for x in parse_listish(value)]


def get_answer(pred: Dict[str, Any]) -> str:
    for k in ANSWER_FIELDS:
        if k in pred:
            return "" if pred[k] is None else str(pred[k])
    return ""


def load_eval(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "items" in obj:
        return obj["items"]
    if isinstance(obj, list):
        return obj
    raise ValueError("eval file must be a list or a JSON object with an 'items' field")


def load_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    suffix = path.suffix.lower()
    rows: List[Dict[str, Any]] = []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "items" in obj:
            rows = obj["items"]
        elif isinstance(obj, dict):
            # allow {"q001": {...}}
            rows = []
            for qid, val in obj.items():
                if isinstance(val, dict):
                    val = {"qid": qid, **val}
                else:
                    val = {"qid": qid, "answer": val}
                rows.append(val)
        elif isinstance(obj, list):
            rows = obj
        else:
            raise ValueError("Unsupported JSON prediction format")
    else:
        raise ValueError("Prediction file must be .csv, .json, or .jsonl")

    out = {}
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        if qid:
            out[qid] = row
    return out


def load_chunk_doc_map(chunks_file: Path | None) -> Dict[int, str]:
    if not chunks_file:
        return {}
    data = json.loads(chunks_file.read_text(encoding="utf-8"))
    mp = {}
    for item in data:
        md = item.get("metadata", {})
        if "chunk_id" in md:
            mp[int(md["chunk_id"])] = str(md.get("file_name") or md.get("doc_id") or "")
    return mp


def score_key_points(answer: str, question: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    points = question.get("gold_key_points", [])
    if not points:
        return 1.0, [], []
    total = 0.0
    hit = 0.0
    missing = []
    matched = []
    for p in points:
        weight = float(p.get("weight", 1.0))
        total += weight
        aliases = p.get("aliases") or [p.get("point", "")]
        if has_any(answer, aliases):
            hit += weight
            matched.append(p.get("point", ""))
        else:
            missing.append(p.get("point", ""))
    return (hit / total if total else 1.0), missing, matched


def score_forbidden(answer: str, question: Dict[str, Any]) -> Tuple[bool, List[str]]:
    hits = []
    for p in question.get("forbidden_points", []):
        aliases = p.get("aliases") or [p.get("point", "")]
        if has_any(answer, aliases):
            hits.append(p.get("point", ""))
    return bool(hits), hits


def score_retrieval(pred: Dict[str, Any], question: Dict[str, Any], chunk_doc_map: Dict[int, str], k: int) -> Dict[str, Any]:
    retrieved = parse_chunk_ids(pred)
    retrieved_at_k = retrieved[:k] if k and k > 0 else retrieved
    gold_chunks = [int(x) for x in question.get("gold_chunk_ids", [])]
    gold_set = set(gold_chunks)
    hit_chunks = [cid for cid in retrieved_at_k if cid in gold_set]

    if gold_chunks and retrieved_at_k:
        chunk_hit = 1.0 if hit_chunks else 0.0
        chunk_recall = len(set(hit_chunks)) / len(gold_set)
    elif gold_chunks:
        chunk_hit = 0.0
        chunk_recall = 0.0
    else:
        chunk_hit = 1.0
        chunk_recall = 1.0

    retrieved_docs = parse_doc_ids(pred)
    if not retrieved_docs and chunk_doc_map:
        retrieved_docs = [chunk_doc_map.get(cid, "") for cid in retrieved_at_k if cid in chunk_doc_map]
    gold_docs = set(str(x) for x in question.get("gold_doc_ids", []))
    doc_hit = 1.0 if (gold_docs and set(retrieved_docs) & gold_docs) else 0.0
    if not gold_docs:
        doc_hit = 1.0

    return {
        "retrieved_chunk_ids": retrieved,
        "retrieved_chunk_ids_at_k": retrieved_at_k,
        "retrieved_doc_ids": retrieved_docs,
        "chunk_hit_at_k": chunk_hit,
        "chunk_recall_at_k": chunk_recall,
        "doc_hit_at_k": doc_hit,
        "hit_chunk_ids": sorted(set(hit_chunks)),
    }


def behavior_score(answer: str, expected: str, key_recall: float, forbidden_hit: bool) -> Tuple[float, str]:
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
    if refused:
        if key_recall >= 0.70:
            return 0.6, "answered_but_over_refused"
        return 0.0, "should_answer_but_refused"
    if forbidden_hit:
        return 0.5, "answered_with_forbidden_detail"
    return 1.0, "answered"


def final_score(expected: str, behavior: float, key_recall: float, retrieval_hit: float, forbidden_hit: bool) -> float:
    clean = 0.0 if forbidden_hit else 1.0
    if expected == "abstain":
        return 0.70 * behavior + 0.20 * key_recall + 0.10 * clean
    if expected == "partial_answer":
        return 0.25 * behavior + 0.40 * key_recall + 0.20 * retrieval_hit + 0.15 * clean
    return 0.20 * behavior + 0.50 * key_recall + 0.20 * retrieval_hit + 0.10 * clean


def evaluate(eval_items: List[Dict[str, Any]], preds: Dict[str, Dict[str, Any]], chunk_doc_map: Dict[int, str], k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for q in eval_items:
        qid = str(q["qid"])
        pred = preds.get(qid, {})
        answer = get_answer(pred)
        key_recall, missing, matched = score_key_points(answer, q)
        forbidden_hit, forbidden_items = score_forbidden(answer, q)
        retrieval = score_retrieval(pred, q, chunk_doc_map, k)
        behavior, behavior_label = behavior_score(answer, q.get("expected_behavior", "answer"), key_recall, forbidden_hit)
        fscore = final_score(q.get("expected_behavior", "answer"), behavior, key_recall, retrieval["chunk_hit_at_k"], forbidden_hit)

        rows.append({
            "qid": qid,
            "category": q.get("category", ""),
            "skill_type": q.get("skill_type", ""),
            "difficulty": q.get("difficulty", ""),
            "expected_behavior": q.get("expected_behavior", ""),
            "has_prediction": bool(pred),
            "behavior_score": round(behavior, 4),
            "behavior_label": behavior_label,
            "key_point_recall": round(key_recall, 4),
            "retrieval_chunk_hit_at_k": round(retrieval["chunk_hit_at_k"], 4),
            "retrieval_chunk_recall_at_k": round(retrieval["chunk_recall_at_k"], 4),
            "retrieval_doc_hit_at_k": round(retrieval["doc_hit_at_k"], 4),
            "forbidden_hit": forbidden_hit,
            "final_score": round(fscore, 4),
            "missing_key_points": " | ".join(missing),
            "matched_key_points": " | ".join(matched),
            "forbidden_points_hit": " | ".join(forbidden_items),
            "gold_chunk_ids": json.dumps(q.get("gold_chunk_ids", []), ensure_ascii=False),
            "hit_chunk_ids": json.dumps(retrieval["hit_chunk_ids"], ensure_ascii=False),
            "retrieved_chunk_ids": json.dumps(retrieval["retrieved_chunk_ids"], ensure_ascii=False),
            "question": q.get("question", ""),
            "gold_answer": q.get("gold_answer", ""),
            "answer": answer,
        })
    summary = summarize(rows)
    return rows, summary


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            "chunk_hit_at_k": round(mean([float(r["retrieval_chunk_hit_at_k"]) for r in vals]), 4),
            "doc_hit_at_k": round(mean([float(r["retrieval_doc_hit_at_k"]) for r in vals]), 4),
            "forbidden_rate": round(mean([1.0 if r["forbidden_hit"] else 0.0 for r in vals]), 4),
        }
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    rows = []
    for group, metrics in summary.items():
        rows.append({"group": group, **metrics})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", required=True, type=Path)
    ap.add_argument("--pred-file", required=True, type=Path)
    ap.add_argument("--chunks-file", type=Path, default=None, help="Optional chunks.json, used to infer retrieved doc ids from chunk ids")
    ap.add_argument("--out-dir", type=Path, default=Path("eval_report"))
    ap.add_argument("--k", type=int, default=5, help="Use top-k retrieved chunks for retrieval hit metrics")
    args = ap.parse_args()

    eval_items = load_eval(args.eval_file)
    preds = load_predictions(args.pred_file)
    chunk_doc_map = load_chunk_doc_map(args.chunks_file)

    rows, summary = evaluate(eval_items, preds, chunk_doc_map, args.k)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.out_dir / "per_question_report.csv", rows)
    write_summary_csv(args.out_dir / "summary_report.csv", summary)
    (args.out_dir / "summary_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = summary["overall"]
    print("=== RAG Evaluation Summary ===")
    print(f"Questions: {overall['num_questions']}")
    print(f"Prediction coverage: {overall['prediction_coverage']:.2%}")
    print(f"Average final score: {overall['avg_final_score']:.2%}")
    print(f"Average behavior score: {overall['avg_behavior_score']:.2%}")
    print(f"Average key-point recall: {overall['avg_key_point_recall']:.2%}")
    print(f"Chunk hit@{args.k}: {overall['chunk_hit_at_k']:.2%}")
    print(f"Doc hit@{args.k}: {overall['doc_hit_at_k']:.2%}")
    print(f"Forbidden/hallucination rate: {overall['forbidden_rate']:.2%}")
    print(f"\nReports written to: {args.out_dir}")


if __name__ == "__main__":
    main()
