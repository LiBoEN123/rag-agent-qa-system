# 启动命令：python scripts/evaluate_rag_results_semantic_fixed.py --eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json --pred-file outputs/predictions.csv --chunks-file data/manifest/chunks.json --out-dir outputs/rag_baseline1 --k 5

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ANSWER_FIELDS = ["answer", "generated_answer", "actual_answer", "response", "model_answer"]
CHUNK_FIELDS = ["retrieved_chunk_ids", "retrieved_chunks", "chunk_ids", "top_chunk_ids"]
DOC_FIELDS = ["retrieved_doc_ids", "retrieved_docs", "doc_ids", "top_doc_ids"]

REFUSAL_PATTERNS = [
    # 强拒答
    "无法确认", "不能确认", "无法确定", "无法回答", "不能回答", "不知道",
    # 文档范围限制
    "无法根据提供的文档确定", "无法根据当前文档确定", "无法根据提供的资料确定",
    "根据提供的文档无法", "根据当前文档无法", "根据现有文档无法", "从给定文档无法",
    "参考资料中未提供", "当前资料中未提供", "提供的资料中未提供",
    "文档中未说明", "文档未说明", "文档没有说明", "文档没有提供",
    "资料中未说明", "资料未说明", "没有相关信息", "无相关信息",
    "未提到", "未提及", "没有提到", "没有提及", "不在文档中",
]

NEGATION_CUES = [
    "未提供", "没有提供", "未说明", "没有说明", "未提到", "没有提到", "未提及", "没有提及",
    "无法确认", "不能确认", "无法确定", "不能确定", "不包含", "不在文档", "文档中没有",
    "参考资料中没有", "参考资料中未", "当前资料中没有", "当前资料中未",
]

CANONICAL_REPLACEMENTS = [
    ("无需", "不需要"),
    ("不用", "不需要"),
    ("不必", "不需要"),
    ("未提及", "未说明"),
    ("未提到", "未说明"),
    ("没有提及", "未说明"),
    ("没有提到", "未说明"),
    ("参考资料", "文档"),
    ("当前资料", "文档"),
    ("提供的资料", "文档"),
]


def normalize_text(s: Any) -> str:
    """Normalize Chinese/English text for loose matching."""
    if s is None:
        return ""
    s = str(s).lower()
    for old, new in CANONICAL_REPLACEMENTS:
        s = s.replace(old.lower(), new.lower())
    s = re.sub(r"\s+", "", s)
    # remove common punctuation while keeping code-ish chars useful for Vue terms
    s = re.sub(r"[，。！？、；：,.!?;:（）()【】\[\]{}<>《》“”\"'`·\-_—|/\\]", "", s)
    return s


def is_subsequence(short: str, long: str) -> bool:
    """Allow small insertions between Chinese key-point words, e.g. 可以多个应用实例 -> 可以创建多个Vue应用实例."""
    if not short or len(short) < 4:
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def has_any(text: str, aliases: Iterable[str]) -> bool:
    norm = normalize_text(text)
    for a in aliases:
        na = normalize_text(a)
        if not na:
            continue
        if na in norm:
            return True
        # 对中文短语做宽松匹配，避免“可以多个应用实例”与“可以创建多个 Vue 应用实例”这种同义表述漏判
        if len(na) >= 5 and is_subsequence(na, norm):
            return True
    return False


def has_all_terms(text: str, terms: Iterable[str]) -> bool:
    """一个语义关键点可由若干词共同表达，例如：导入 + 模板 + 直接 + 使用。"""
    norm = normalize_text(text)
    for term in terms:
        nt = normalize_text(term)
        if not nt:
            continue
        if nt not in norm and not (len(nt) >= 5 and is_subsequence(nt, norm)):
            return False
    return True


def match_point(answer: str, point: Dict[str, Any]) -> bool:
    """
    判断一个 gold_key_point 是否被答案覆盖。

    支持三层匹配：
    1. aliases：原有的短语/同义词包含匹配；
    2. match_any：若干“词组共同出现”规则，适合自然表达、代码示例、换说法；
       例：["导入", "模板", "直接", "使用"]；
    3. regex：少量需要正则的情况。
    """
    aliases = point.get("aliases") or [point.get("point", "")]
    if has_any(answer, aliases):
        return True

    for group in point.get("match_any", []) or []:
        if has_all_terms(answer, group):
            return True

    for pattern in point.get("regex", []) or []:
        if re.search(pattern, answer, flags=re.I):
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
        if match_point(answer, p):
            hit += weight
            matched.append(p.get("point", ""))
        else:
            missing.append(p.get("point", ""))
    return (hit / total if total else 1.0), missing, matched


def has_negation_near(answer: str, start: int, end: int, window: int = 36) -> bool:
    """If the model says '文档没有提供 X', X should not be counted as hallucination."""
    left = max(0, start - window)
    right = min(len(answer), end + window)
    ctx = answer[left:right]
    return any(cue in ctx for cue in NEGATION_CUES)


def regex_forbidden_hit(answer: str, point: str) -> bool:
    """Precise rules for common forbidden types to avoid false positives."""
    # 只有真的给出百分比数值，才算“百分比数值”幻觉；“没有提供百分比”不算。
    if "百分比" in point:
        return bool(re.search(r"\d+(?:\.\d+)?\s*%|百分之[一二三四五六七八九十百千万零点\d]+", answer))

    # 只有真的给出日期，才算发布日期幻觉；“没有发布日期”不算。
    if "发布日期" in point:
        return bool(re.search(r"\d{4}[-年/]\d{1,2}(?:[-月/]\d{1,2}日?)?", answer))

    # 只有给出具体 star 数，才算；“无法确认 star 数量”不算。
    if "star" in point.lower() or "星标" in point:
        return bool(re.search(r"\d+(?:\.\d+)?\s*(?:k|m|万)?\s*(?:star|stars|星标)", answer, flags=re.I))

    # 只有给出具体源码路径/行号，才算；“没有行号”不算。
    if "源码" in point or "行号" in point:
        return bool(re.search(r"packages/|runtime-core|apiCreateApp|第?\d+\s*行", answer, flags=re.I))

    return False


def alias_forbidden_hit(answer: str, alias: str) -> bool:
    """Forbidden matching with context negation and safer English token boundaries."""
    if not alias:
        return False
    # 单字符 forbidden 太容易误报，比如 k、行，直接跳过。
    if len(alias.strip()) <= 1:
        return False

    # 英文内部实现词用单词边界，避免 Set 命中 setter。
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", alias):
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])"
        for m in re.finditer(pattern, answer, flags=re.I):
            if not has_negation_near(answer, m.start(), m.end()):
                return True
        return False

    # 中文/混合短语：找到原文位置后看附近是否是否定语境。
    for m in re.finditer(re.escape(alias), answer, flags=re.I):
        if not has_negation_near(answer, m.start(), m.end()):
            return True
    return False


def is_regex_only_forbidden(point: str) -> bool:
    """这些 forbidden 类型只能按精确正则判定，不能再用 alias 模糊匹配，否则会误报。"""
    p = point.lower()
    return (
        "百分比" in p
        or "发布日期" in p
        or "star" in p
        or "星标" in p
        or "源码" in p
        or "行号" in p
    )


def score_forbidden(answer: str, question: Dict[str, Any]) -> Tuple[bool, List[str]]:
    hits = []
    for p in question.get("forbidden_points", []):
        point = p.get("point", "")

        # 数值/日期/star/源码等类型必须用精确规则，不能因为答案里出现“star/发布日期”等词就判幻觉。
        if regex_forbidden_hit(answer, point):
            hits.append(point)
            continue
        if is_regex_only_forbidden(point):
            continue

        aliases = p.get("aliases") or [point]
        if any(alias_forbidden_hit(answer, str(a)) for a in aliases):
            hits.append(point)
    return bool(hits), hits


def score_retrieval(pred: Dict[str, Any], question: Dict[str, Any], chunk_doc_map: Dict[int, str], k: int) -> Dict[str, Any]:
    """
    Retrieval metrics at top-k.

    chunk_hit_at_k:
        Whether at least one gold chunk is included in top-k retrieved chunks.
        This is a loose metric.

    gold_chunk_recall_at_k:
        matched gold chunks / all gold chunks.
        This measures how complete the retrieved evidence is.

    all_gold_hit_at_k:
        Whether all gold chunks are included in top-k retrieved chunks.
        This is strict and especially useful for cross-doc / multi-evidence questions.
    """
    retrieved = parse_chunk_ids(pred)
    retrieved_at_k = retrieved[:k] if k and k > 0 else retrieved

    gold_chunks = [int(x) for x in question.get("gold_chunk_ids", [])]
    gold_set = set(gold_chunks)
    retrieved_set_at_k = set(retrieved_at_k)

    matched_gold_chunks = sorted(gold_set & retrieved_set_at_k)
    missing_gold_chunks = sorted(gold_set - retrieved_set_at_k)

    if gold_set:
        chunk_hit = 1.0 if matched_gold_chunks else 0.0
        gold_chunk_recall = len(matched_gold_chunks) / len(gold_set)
        all_gold_hit = 1.0 if gold_set.issubset(retrieved_set_at_k) else 0.0
    else:
        # 如果评测题没有 gold_chunk_ids，就不惩罚检索。
        chunk_hit = 1.0
        gold_chunk_recall = 1.0
        all_gold_hit = 1.0

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

        # 旧字段保留，兼容已有输出；它和 gold_chunk_recall_at_k 含义一致。
        "chunk_recall_at_k": gold_chunk_recall,

        # 新增两个更有解释力的指标。
        "gold_chunk_recall_at_k": gold_chunk_recall,
        "all_gold_hit_at_k": all_gold_hit,

        "doc_hit_at_k": doc_hit,
        "hit_chunk_ids": matched_gold_chunks,
        "matched_gold_chunk_ids": matched_gold_chunks,
        "missing_gold_chunk_ids": missing_gold_chunks,
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
        # final_score 使用 gold_chunk_recall_at_k，而不是 loose 的 chunk_hit_at_k。
        # 否则只命中一个证据 chunk 的 cross-doc 问题会被高估。
        fscore = final_score(q.get("expected_behavior", "answer"), behavior, key_recall, retrieval["gold_chunk_recall_at_k"], forbidden_hit)

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
            # 旧字段保留：等价于 gold_chunk_recall_at_k。
            "retrieval_chunk_recall_at_k": round(retrieval["chunk_recall_at_k"], 4),
            # 新增：标准证据 chunk 召回比例。
            "retrieval_gold_chunk_recall_at_k": round(retrieval["gold_chunk_recall_at_k"], 4),
            # 新增：是否召回全部标准证据 chunk。
            "retrieval_all_gold_hit_at_k": round(retrieval["all_gold_hit_at_k"], 4),
            "retrieval_doc_hit_at_k": round(retrieval["doc_hit_at_k"], 4),
            "forbidden_hit": forbidden_hit,
            "final_score": round(fscore, 4),
            "missing_key_points": " | ".join(missing),
            "matched_key_points": " | ".join(matched),
            "forbidden_points_hit": " | ".join(forbidden_items),
            "gold_chunk_ids": json.dumps(q.get("gold_chunk_ids", []), ensure_ascii=False),
            "hit_chunk_ids": json.dumps(retrieval["hit_chunk_ids"], ensure_ascii=False),
            "matched_gold_chunk_ids": json.dumps(retrieval["matched_gold_chunk_ids"], ensure_ascii=False),
            "missing_gold_chunk_ids": json.dumps(retrieval["missing_gold_chunk_ids"], ensure_ascii=False),
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
            "gold_chunk_recall_at_k": round(mean([float(r["retrieval_gold_chunk_recall_at_k"]) for r in vals]), 4),
            "all_gold_hit_at_k": round(mean([float(r["retrieval_all_gold_hit_at_k"]) for r in vals]), 4),
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
    print(f"Gold chunk recall@{args.k}: {overall['gold_chunk_recall_at_k']:.2%}")
    print(f"All gold hit@{args.k}: {overall['all_gold_hit_at_k']:.2%}")
    print(f"Doc hit@{args.k}: {overall['doc_hit_at_k']:.2%}")
    print(f"Forbidden/hallucination rate: {overall['forbidden_rate']:.2%}")
    print(f"\nReports written to: {args.out_dir}")


if __name__ == "__main__":
    main()
