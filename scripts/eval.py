import json
import csv
import time
from pathlib import Path
from app.rag import init_rag, search_relevant_chunks, get_rag_status, ask
from app.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "kb"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def load_question(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def call_rag_system(question, top_k=None):
    final_top_k = top_k if top_k is not None else Settings.TOP_K

    rag_status = get_rag_status()
    if not rag_status["ready"]:
        raise RuntimeError(rag_status["message"])

    retrieved_chunks = search_relevant_chunks(question, final_top_k)
    result = ask(question, top_k=final_top_k, retrieved_chunks=retrieved_chunks)

    refs = result.get("references", [])

    return {
        "status": result.get("status", ""),
        "reason": result.get("reason", ""),
        "confidence": result.get("confidence", 0.0),
        "need_clarification": result.get("need_clarification", False),
        "clarification_question": result.get("clarification_question", ""),
        "answer": result.get("answer", ""),
        "retrieved_count": len(retrieved_chunks),
        "retrieved_doc_ids": [p.get("file_name", "") for p in retrieved_chunks],
        "retrieved_file_names": [p.get("file_name", "") for p in retrieved_chunks],
        "reference_chunk_ids": [p.get("chunk_id", "") for p in refs],
        "reference_snippets": [p.get("snippet") or p.get("text", "") for p in refs],
    }
def run_one_question(item, run_name):
    start = time.time()
    try:
        resp = call_rag_system(
            item["question"],
            top_k=item.get("top_k")
        )

        result = {
            "qid": item["qid"],
            "question": item["question"],
            "category": item["category"],
            "gold_answer": item.get("gold_answer", ""),
            "gold_doc_ids": item.get("gold_doc_ids", []),

            # 评测预测结果
            "pred_answer": resp.get("answer", ""),
            "pred_status": resp.get("status", ""),

            # 引用信息
            "retrieved_doc_ids": resp.get("retrieved_doc_ids", []),
            "retrieved_file_names": resp.get("retrieved_file_names", []),
            "reference_chunk_ids": resp.get("reference_chunk_ids", []),
            "reference_snippets": resp.get("reference_snippets", []),
            "retrieved_count": resp.get("retrieved_count", 0),
        }

    except Exception as e:
        result = {
            "qid": item["qid"],
            "question": item["question"],
            "category": item["category"],
            "gold_answer": item.get("gold_answer", ""),
            "gold_doc_ids": item.get("gold_doc_ids", []),

            "pred_answer": "",
            "pred_status": "EXCEPTION",

            "retrieved_doc_ids": [],
            "retrieved_file_names": [],
            "reference_chunk_ids": [],
            "reference_snippets": [],
            "retrieved_count": 0,
        }

    return result

def save_json(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_csv(results, path):
    rows = []
    for p in results:
        rows.append({
            "qid": p["qid"],
            "question": p["question"],
            "category": p["category"],
            "gold_answer": p["gold_answer"],
            "gold_doc_ids": "|".join(map(str, p.get("gold_doc_ids", []))),

            "pred_answer": p.get("pred_answer", ""),
            "pred_status": p.get("pred_status", ""),

            "retrieved_count": p.get("retrieved_count", 0),
            "retrieved_doc_ids": "|".join(map(str, p.get("retrieved_doc_ids", []))),
            "retrieved_file_names": "|".join(map(str, p.get("retrieved_file_names", []))),
            "reference_chunk_ids": "|".join(map(str, p.get("reference_chunk_ids", []))),
        })

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def build_summarys(results):
    total = len(results)

    status_count = {}
    for r in results:
        s = r.get("pred_status", "UNKNOWN")
        status_count[s] = status_count.get(s, 0) + 1

    return {
        "total_questions": total,
        "status_count": status_count
    }
def main():
    run_name = "baseline_20q_topk5_cs300_ol60_A"
    Settings.RAG_FILE_PATH = PROJECT_ROOT / "data" / "kb"

    Settings.validate()
    init_rag(str(Settings.RAG_FILE_PATH),
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP)

    questions = load_question(PROJECT_ROOT / "data" / "eval" / "eval_questions_utf8_updated.json")
    results = [run_one_question({**p, "top_k": 3}, run_name) for p in questions]

    out_dir = PROJECT_ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(results, out_dir / 'results.json')
    save_csv(results, out_dir/ 'results.csv')

    summary = build_summarys(results)
    save_json(summary, out_dir / 'summary.json')

if __name__ == "__main__":
    main()



































