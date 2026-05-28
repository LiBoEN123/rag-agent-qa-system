from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

from app.config import Settings
from app.rag import ask, get_rag_status, init_rag, search_relevant_chunks


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "generated_20_questions_key_points_v2.json"
KB_DIR = PROJECT_ROOT / "data" / "kb"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NO_INFO = "NO_RELEVANT_INFO"


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["items"] if isinstance(data, dict) and "items" in data else data


def dump_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def norm(text) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def compact(text) -> str:
    return re.sub(r"[\s'\"“”‘’]+", "", norm(text).lower())


def hit_regex(text: str, patterns: list[str]) -> list[str]:
    hits = []
    for p in patterns or []:
        try:
            if re.search(p, text, re.I):
                hits.append(p)
        except re.error:
            pass
    return hits


def hit_alias(text: str, aliases: list[str]) -> list[str]:
    c = compact(text)
    return [a for a in aliases or [] if compact(a) and compact(a) in c]


def hit_point(answer: str, point: dict) -> dict:
    text = norm(answer)
    alias_hits = hit_alias(text, point.get("aliases", []))
    regex_hits = hit_regex(text, point.get("regex_patterns", []))
    return {
        "id": point.get("id", ""),
        "source_doc": point.get("source_doc", ""),
        "required": bool(point.get("required", False)),
        "hit": bool(alias_hits or regex_hits),
        "hit_by": "alias" if alias_hits else ("regex" if regex_hits else ""),
    }


def init_system(kb_dir: Path) -> None:
    Settings.RAG_FILE_PATH = kb_dir
    Settings.validate()
    init_rag(str(kb_dir), chunk_size=Settings.CHUNK_SIZE, chunk_overlap=Settings.CHUNK_OVERLAP)


def call_rag(question: str, top_k: int) -> dict:
    status = get_rag_status()
    if not status.get("ready"):
        raise RuntimeError(status.get("message", "RAG 未初始化"))

    chunks = search_relevant_chunks(question, top_k)
    result = ask(question, top_k=top_k, retrieved_chunks=chunks)
    refs = result.get("references", []) or []
    docs = [c.get("file_name", "") for c in chunks]

    return {
        "pred_status": result.get("status", ""),
        "pred_answer": result.get("answer", ""),
        "retrieved_count": len(chunks),
        "retrieved_doc_ids": docs,
        "top3_doc_ids": docs[:3],
        "retrieved_chunk_ids": [c.get("chunk_id", "") for c in chunks],
        "reference_chunk_ids": [r.get("chunk_id", "") for r in refs],
    }


def first_rank(retrieved_docs: list[str], gold_docs: list[str]) -> int | None:
    gold = set(gold_docs or [])
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in gold:
            return i
    return None


def eval_retrieval(resp: dict, gold_docs: list[str]) -> dict:
    rank = first_rank(resp.get("retrieved_doc_ids", []), gold_docs)
    return {
        "gold_rank": rank,
        "hit_at_1": rank == 1,
        "hit_at_3": rank is not None and rank <= 3,
        "hit_at_5": rank is not None and rank <= 5,
    }


def eval_answer(answer: str, item: dict) -> dict:
    ev = item.get("eval_v2", {})
    criteria = ev.get("pass_criteria", {})
    points = [hit_point(answer, p) for p in ev.get("answer_points", [])]

    hit_points = [p for p in points if p["hit"]]
    hit_ids = {p["id"] for p in hit_points}
    required_ids = set(criteria.get("required_point_ids", []))
    missed_required = sorted(required_ids - hit_ids)
    hit_docs = sorted({p["source_doc"] for p in hit_points if p.get("source_doc") not in {"", "derived"}})

    min_points = int(criteria.get("min_points", 1))
    min_doc_coverage = int(criteria.get("min_doc_coverage", 1))
    passed = len(hit_points) >= min_points and not missed_required and len(hit_docs) >= min_doc_coverage

    return {
        "answer_passed": passed,
        "point_hit_count": len(hit_points),
        "point_total": len(points),
        "point_recall": round(len(hit_points) / len(points), 4) if points else 0.0,
        "hit_point_ids": sorted(hit_ids),
        "miss_point_ids": [p["id"] for p in points if not p["hit"]],
        "missed_required_point_ids": missed_required,
        "hit_source_docs": hit_docs,
        "doc_coverage": len(hit_docs),
        "min_points": min_points,
        "min_doc_coverage": min_doc_coverage,
    }


def eval_abstain(answer: str, status: str, item: dict) -> dict:
    ae = item.get("eval_v2", {}).get("abstain_evaluation", {})
    signal_patterns = ae.get("abstain_signal", {}).get("regex_patterns", [])
    forbidden_patterns = ae.get("forbidden_claim_patterns", [])

    abstain_signal_hit = status == NO_INFO or bool(hit_regex(norm(answer), signal_patterns))
    forbidden_hits = hit_regex(norm(answer), forbidden_patterns)
    passed = abstain_signal_hit and not forbidden_hits

    support = [hit_point(answer, p) for p in ae.get("supporting_evidence_points", [])]
    missing = [hit_point(answer, p) for p in ae.get("missing_information_points", [])]

    return {
        "answer_passed": passed,
        "abstain_signal_hit": abstain_signal_hit,
        "forbidden_claim_hits": forbidden_hits,
        "support_hit_ids": [p["id"] for p in support if p["hit"]],
        "missing_info_hit_ids": [p["id"] for p in missing if p["hit"]],
    }


def behavior_of(item: dict) -> str:
    value = str(item.get("expected_behavior", "")).strip().lower()
    if value in {"answer", "abstain"}:
        return value
    return "abstain" if item.get("category") == "not_in_doc" else "answer"


def classify(row: dict) -> tuple[str, str]:
    if row["pred_status"] == "EXCEPTION":
        return "exception", row.get("failure_reason", "程序异常")

    if row["expected_behavior"] == "abstain":
        if not row.get("abstain_signal_hit"):
            return "bad_abstain", "应拒答，但缺少明确拒答信号"
        if row.get("forbidden_claim_hits"):
            return "fabrication", f"拒答题疑似编造: {row['forbidden_claim_hits']}"
        return "success", "拒答正确"

    if not row.get("hit_at_3"):
        return "miss_retrieval", "正确文档没有进入 top-3"
    if row["pred_status"] == NO_INFO:
        return "false_abstain", "正确文档已召回，但系统仍拒答"
    if row.get("missed_required_point_ids"):
        return "missing_required_points", f"缺少必需关键点: {row['missed_required_point_ids']}"
    if row.get("point_hit_count", 0) < row.get("min_points", 1):
        return "low_point_coverage", "关键点命中数量不足"
    if row.get("doc_coverage", 0) < row.get("min_doc_coverage", 1):
        return "low_doc_coverage", "跨文档来源覆盖不足"
    return "success", "通过"


def run_one(item: dict, top_k: int) -> dict:
    start = time.time()
    behavior = behavior_of(item)
    row = {
        "qid": item.get("qid", ""),
        "question": item.get("question", ""),
        "category": item.get("category", ""),
        "expected_behavior": behavior,
        "gold_answer": item.get("gold_answer", ""),
        "gold_doc_ids": item.get("gold_doc_ids", []),
    }

    try:
        resp = call_rag(row["question"], top_k)
        row.update(resp)
        row.update(eval_retrieval(resp, row["gold_doc_ids"]))
        if behavior == "abstain":
            row.update(eval_abstain(row["pred_answer"], row["pred_status"], item))
        else:
            row.update(eval_answer(row["pred_answer"], item))
    except Exception as e:
        row.update({
            "pred_status": "EXCEPTION", "pred_answer": "", "retrieved_count": 0,
            "retrieved_doc_ids": [], "top3_doc_ids": [], "retrieved_chunk_ids": [],
            "reference_chunk_ids": [], "gold_rank": None, "hit_at_1": False,
            "hit_at_3": False, "hit_at_5": False, "answer_passed": False,
            "failure_reason": f"程序异常: {e}",
        })

    row["latency_sec"] = round(time.time() - start, 3)
    row["error_type"], row["failure_reason"] = classify(row)
    return row


def rate(rows: list[dict], key: str) -> float:
    return round(sum(bool(r.get(key)) for r in rows) / len(rows), 4) if rows else 0.0


def count_by(rows: list[dict], key: str) -> dict:
    out = {}
    for r in rows:
        v = r.get(key, "UNKNOWN")
        out[v] = out.get(v, 0) + 1
    return out


def summary(results: list[dict]) -> dict:
    answer_rows = [r for r in results if r["expected_behavior"] == "answer"]
    abstain_rows = [r for r in results if r["expected_behavior"] == "abstain"]
    avg_recall = round(sum(r.get("point_recall", 0) for r in answer_rows) / len(answer_rows), 4) if answer_rows else 0.0
    return {
        "total": len(results),
        "answer_questions": len(answer_rows),
        "abstain_questions": len(abstain_rows),
        "overall_pass_rate": rate(results, "answer_passed"),
        "answer_pass_rate": rate(answer_rows, "answer_passed"),
        "abstain_pass_rate": rate(abstain_rows, "answer_passed"),
        "hit_at_1": rate(answer_rows, "hit_at_1"),
        "hit_at_3": rate(answer_rows, "hit_at_3"),
        "hit_at_5": rate(answer_rows, "hit_at_5"),
        "avg_answer_point_recall": avg_recall,
        "status_count": count_by(results, "pred_status"),
        "category_count": count_by(results, "category"),
        "error_type_count": count_by(results, "error_type"),
    }


def flat(v) -> str:
    if isinstance(v, (list, tuple, set)):
        return "|".join(map(str, v))
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v)


def save_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "qid", "category", "expected_behavior", "answer_passed", "error_type", "failure_reason",
        "question", "pred_status", "pred_answer", "gold_answer", "gold_doc_ids", "top3_doc_ids",
        "gold_rank", "hit_at_1", "hit_at_3", "hit_at_5", "point_hit_count", "point_total",
        "point_recall", "hit_point_ids", "miss_point_ids", "missed_required_point_ids",
        "hit_source_docs", "doc_coverage", "abstain_signal_hit", "forbidden_claim_hits",
        "support_hit_ids", "missing_info_hit_ids", "retrieved_doc_ids", "retrieved_chunk_ids",
        "reference_chunk_ids", "latency_sec",
    ]
    fields = [f for f in fields if any(f in r for r in rows)]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({f: flat(r.get(f)) for f in fields})


def save_bad_cases(rows: list[dict], path: Path, top_n: int = 5) -> None:
    bad = [r for r in rows if r.get("error_type") != "success"][:top_n]
    lines = ["# Bad Cases", ""]
    if not bad:
        lines.append("没有失败案例。")
    for i, r in enumerate(bad, 1):
        lines += [
            f"## Case {i}: {r['qid']}",
            f"- 问题: {r['question']}",
            f"- 错误: {r['error_type']}，{r['failure_reason']}",
            f"- 标准文档: {', '.join(r.get('gold_doc_ids', [])) or '无'}",
            f"- Top3 文档: {', '.join(r.get('top3_doc_ids', [])) or '无'}",
            f"- 命中点: {', '.join(r.get('hit_point_ids', [])) or '无'}",
            f"- 缺失点: {', '.join(r.get('miss_point_ids', [])) or '无'}",
            f"- 回答: {r.get('pred_answer') or '空'}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    p.add_argument("--kb-dir", type=Path, default=KB_DIR)
    p.add_argument("--top-k", type=int, default=7)
    p.add_argument("--run-name", default="v2_eval_clean_topk7")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_system(args.kb_dir)
    results = [run_one(item, args.top_k) for item in load_items(args.questions)]
    stat = summary(results)

    out_dir = OUTPUTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(results, out_dir / "results.json")
    dump_json(stat, out_dir / "summary.json")
    save_csv(results, out_dir / "results.csv")
    save_bad_cases(results, out_dir / "bad_cases.md")

    print(f"[eval] 完成 {len(results)} 道题")
    print(f"[eval] overall={stat['overall_pass_rate']} answer={stat['answer_pass_rate']} abstain={stat['abstain_pass_rate']}")
    print(f"[eval] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
