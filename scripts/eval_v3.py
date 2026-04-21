import json
import csv
import time
from pathlib import Path

from app.rag import init_rag, search_relevant_chunks, get_rag_status, ask
from app.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "generated_20_questions_latest_chunks_v2.json"


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_rag_system(question, top_k=None):
    final_top_k = top_k if top_k is not None else Settings.TOP_K

    rag_status = get_rag_status()
    if not rag_status["ready"]:
        raise RuntimeError(rag_status["message"])

    retrieved_chunks = search_relevant_chunks(question, final_top_k)
    origin_refs = [retrieved_chunks[i].get("chunk_id", []) for i in range(len(retrieved_chunks))]
    result = ask(question, top_k=final_top_k, retrieved_chunks=retrieved_chunks)
    refs = result.get("references", [])

    retrieved_doc_ids = [c.get("file_name", "") for c in retrieved_chunks]

    return {
        "status": result.get("status", ""),
        "answer": result.get("answer", ""),
        "retrieved_count": len(retrieved_chunks),
        "retrieved_doc_ids": retrieved_doc_ids,
        "top3_doc_ids": retrieved_doc_ids[:3],
        "reference_chunk_ids": [r.get("chunk_id", "") for r in refs],
        "retrieved_chunk_ids": origin_refs
    }


def find_first_hit_rank(retrieved_doc_ids, gold_doc_ids):
    if not gold_doc_ids:
        return None

    gold_set = set(gold_doc_ids)
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_set:
            return i
    return None


def infer_expected_behavior(item):
    value = str(item.get("expected_behavior", "")).strip().lower()
    if value in {"answer", "abstain"}:
        return value

    # 兜底逻辑：如果没写 expected_behavior，再按 category 推断
    if item.get("category") == "not_in_doc":
        return "abstain"
    return "answer"


def classify_simple_error(row):
    """
    今天只做最小分类，不做复杂评测框架。
    """
    expected_behavior = row["expected_behavior"]
    pred_status = row["pred_status"]
    hit_at_3 = row["hit_at_3"]
    reference_chunk_ids = row["reference_chunk_ids"]
    gold_chunk_ids = row["gold_chunk_ids"]

    if pred_status == "EXCEPTION":
        return "exception", "程序异常"

    if expected_behavior == "abstain":
        if pred_status == "NO_RELEVANT_INFO":
            return "success", "拒答正确"
        return "should_abstain_but_answered", "本题应拒答，但系统给了具体回答"

    # expected_behavior == "answer"
    if not hit_at_3:
        return "miss_retrieval", "正确文档没有进入 top-3，属于漏召回"

    if pred_status == "NO_RELEVANT_INFO":
        return "false_abstain", "正确文档已进入 top-3，但系统仍然拒答"

    if set(reference_chunk_ids) != set(gold_chunk_ids):
        return "discrepant_chunk_ids", "最终引用的chunk与实际用到的chunk不一致"

    return "success", "检索和回答基本正常"


def run_one_question(item, top_k=5):
    start = time.time()

    expected_behavior = infer_expected_behavior(item)
    gold_doc_ids = item.get("gold_doc_ids", [])

    try:
        resp = call_rag_system(item["question"], top_k=top_k)

        gold_rank = find_first_hit_rank(
            resp.get("retrieved_doc_ids", []),
            gold_doc_ids
        )

        hit_at_1 = gold_rank == 1
        hit_at_3 = gold_rank is not None and gold_rank <= 3
        hit_at_5 = gold_rank is not None and gold_rank <= 5

        is_correct_abstain = (
            expected_behavior == "abstain"
            and resp.get("status", "") == "NO_RELEVANT_INFO"
        )

        result = {
            "qid": item["qid"],
            "question": item["question"],
            "category": item.get("category", ""),
            "gold_answer": item.get("gold_answer", ""),
            "gold_doc_ids": gold_doc_ids,
            "gold_chunk_ids":item.get("gold_chunk_ids", []),
            "expected_behavior": expected_behavior,

            "pred_answer": resp.get("answer", ""),
            "pred_status": resp.get("status", ""),

            "retrieved_count": resp.get("retrieved_count", 0),
            "retrieved_doc_ids": resp.get("retrieved_doc_ids", []),
            "top3_doc_ids": resp.get("top3_doc_ids", []),
            "retrieved_chunk_ids": resp.get("retrieved_chunk_ids", []),
            "reference_chunk_ids": resp.get("reference_chunk_ids", []),

            "gold_rank": gold_rank,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "is_correct_abstain": is_correct_abstain,

            "latency_sec": round(time.time() - start, 3),
        }

        error_type, failure_reason = classify_simple_error(result)
        result["error_type"] = error_type
        result["failure_reason"] = failure_reason

        return result

    except Exception as e:
        return {
            "qid": item["qid"],
            "question": item["question"],
            "category": item.get("category", ""),
            "gold_answer": item.get("gold_answer", ""),
            "gold_doc_ids": gold_doc_ids,
            "gold_chunk_ids": item.get("gold_chunk_ids", []),
            "expected_behavior": expected_behavior,

            "pred_answer": "",
            "pred_status": "EXCEPTION",

            "retrieved_count": 0,
            "retrieved_doc_ids": [],
            "top3_doc_ids": [],
            "retrieved_chunk_ids": [],
            "reference_chunk_ids": [],

            "gold_rank": None,
            "hit_at_1": False,
            "hit_at_3": False,
            "hit_at_5": False,
            "is_correct_abstain": False,

            "latency_sec": round(time.time() - start, 3),
            "error_type": "exception",
            "failure_reason": f"程序异常: {e}",
        }


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(results, path):
    rows = []
    for r in results:
        rows.append({
            "qid": r["qid"],
            "question": r["question"],
            "category": r["category"],
            "expected_behavior": r["expected_behavior"],
            "gold_answer": r["gold_answer"],
            "gold_doc_ids": "|".join(map(str, r.get("gold_doc_ids", []))),
            "gold_chunk_ids": "|".join(map(str, r.get("gold_chunk_ids", []))),

            "pred_status": r["pred_status"],
            "pred_answer": r["pred_answer"],

            "gold_rank": r["gold_rank"],
            "hit_at_1": r["hit_at_1"],
            "hit_at_3": r["hit_at_3"],
            "hit_at_5": r["hit_at_5"],
            "is_correct_abstain": r["is_correct_abstain"],

            "retrieved_count": r["retrieved_count"],
            "retrieved_doc_ids": "|".join(map(str, r.get("retrieved_doc_ids", []))),
            "top3_doc_ids": "|".join(map(str, r.get("top3_doc_ids", []))),
            "retrieved_chunk_ids": "|".join(map(str, r.get("retrieved_chunk_ids", []))),
            "reference_chunk_ids": "|".join(map(str, r.get("reference_chunk_ids", []))),

            "error_type": r["error_type"],
            "failure_reason": r["failure_reason"],
            "latency_sec": r["latency_sec"],
        })

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_summary(results):
    total = len(results)

    status_count = {}
    category_count = {}
    error_type_count = {}

    answer_questions = [r for r in results if r["expected_behavior"] == "answer"]
    abstain_questions = [r for r in results if r["expected_behavior"] == "abstain"]

    for r in results:
        s = r.get("pred_status", "UNKNOWN")
        status_count[s] = status_count.get(s, 0) + 1

        c = r.get("category", "UNKNOWN")
        category_count[c] = category_count.get(c, 0) + 1

        e = r.get("error_type", "UNKNOWN")
        error_type_count[e] = error_type_count.get(e, 0) + 1

    def safe_rate(n, d):
        return round(n / d, 4) if d else 0.0

    hit_at_1 = safe_rate(sum(r["hit_at_1"] for r in answer_questions), len(answer_questions))
    hit_at_3 = safe_rate(sum(r["hit_at_3"] for r in answer_questions), len(answer_questions))
    hit_at_5 = safe_rate(sum(r["hit_at_5"] for r in answer_questions), len(answer_questions))

    abstain_accuracy = safe_rate(
        sum(r["is_correct_abstain"] for r in abstain_questions),
        len(abstain_questions)
    )

    return {
        "total_questions": total,
        "status_count": status_count,
        "category_count": category_count,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "abstain_accuracy": abstain_accuracy,
        "error_type_count": error_type_count,
    }


def build_bad_cases_md(results, path, top_n=3):
    """
    只挑最差的 3 个案例。
    优先级：
    1. exception
    2. should_abstain_but_answered
    3. miss_retrieval
    4. false_abstain
    """
    priority = {
        "exception": 100,
        "should_abstain_but_answered": 90,
        "miss_retrieval": 80,
        "false_abstain": 70,
        "discrepant_chunk_ids":60,
        "success": 0,
    }

    bad_cases = [r for r in results if r["error_type"] != "success"]
    bad_cases.sort(key=lambda x: priority.get(x["error_type"], 10), reverse=True)
    bad_cases = bad_cases[:top_n]

    lines = ["# Bad Cases\n"]

    if not bad_cases:
        lines.append("没有失败案例。")
    else:
        for i, r in enumerate(bad_cases, start=1):
            lines.append(f"## Case {i}: {r['qid']}")
            lines.append(f"- 问题: {r['question']}")
            lines.append(f"- 类别: {r['category']}")
            lines.append(f"- 期望行为: {r['expected_behavior']}")
            lines.append(f"- 期望来源: {', '.join(r['gold_doc_ids']) if r['gold_doc_ids'] else '无'}")
            lines.append(f"- 实际 top-3: {', '.join(r['top3_doc_ids']) if r['top3_doc_ids'] else '无'}")
            lines.append(f"- gold_rank: {r['gold_rank']}")
            lines.append(f"- pred_status: {r['pred_status']}")
            lines.append(f"- 最终回答: {r['pred_answer'] or '空'}")
            lines.append(f"- 错误类型: {r['error_type']}")
            lines.append(f"- 失败原因: {r['failure_reason']}")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    run_name = "baseline_20q_topk5_cs300_ol60_0.4b0.6v_min_eval"
    top_k = 5

    Settings.RAG_FILE_PATH = PROJECT_ROOT / "data" / "kb"
    Settings.validate()

    init_rag(
        str(Settings.RAG_FILE_PATH),
        chunk_size=Settings.CHUNK_SIZE,
        chunk_overlap=Settings.CHUNK_OVERLAP
    )

    questions = load_questions(QUESTIONS_PATH)
    results = [run_one_question(q, top_k=top_k) for q in questions]

    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(results, out_dir / "results.json")
    save_csv(results, out_dir / "results.csv")
    save_json(build_summary(results), out_dir / "summary.json")
    build_bad_cases_md(results, out_dir / "bad_cases.md", top_n=3)


if __name__ == "__main__":
    main()
