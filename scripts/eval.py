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

def call_rag_system(question):
    final_top_k = Settings.TOP_K

    rag_status = get_rag_status()
    if not rag_status['ready']:
        raise RuntimeError(rag_status["message"])

    retrieved_chunks = search_relevant_chunks(question, final_top_k)
    result = ask(question, top_k=final_top_k, retrieved_chunks=retrieved_chunks)

    answer = result["answer"]
    retrieved_doc_ids = [p['source_file'] for p in result['references']]
    reference_snippets = [p ['text'] for p in result['references']]

    return {
        "answer":answer,
        "retrieved_doc_ids":retrieved_doc_ids,
        "reference_snippets":reference_snippets
    }

def run_one_question(item, run_name):
    start = time.time()
    try:
        resp = call_rag_system(item['question'])
        result = {
            "qid":item["qid"],
            "question":item["question"],
            "category":item["category"],
            "gold_answer":item.get("gold_answer", ""),
            "gold_doc_ids":item.get("gold_doc_ids", []),

            "pred_answer":resp.get("answer", ""),
            "retrieved_doc_ids":resp.get("retrieved_doc_ids", []),
            "reference_snippets":resp.get("reference_snippets", []),

            "error":False,
            "error_message":"",
            "latency_ms":int((time.time() - start) * 1000),
            "run_name":run_name
        }

    except Exception as e:
        result = {
            "qid":item["qid"],
            "question":item["question"],
            "category":item["category"],
            "gold_answer":item.get("gold_answer", ""),
            "gold_doc_ids":item.get("gold_doc_ids", []),

            "pred_answer":"",
            "retrieved_doc_ids":[],
            "reference_snippets":[],

            "error":True,
            "error_message":str(e),
            "latency_ms":int((time.time() - start) * 1000),
            "run_name":run_name
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
            "gold_doc_ids": p["gold_doc_ids"],
            "pred_answer": p["pred_answer"],
            "error": p["error"],
            "error_message":p["error_message"],
            "retrieved_doc_ids":"|".join(p["retrieved_doc_ids"]),
            "reference_snippets":"||".join(p["reference_snippets"]),
            "latency_ms": p["latency_ms"],
            "run_name": p["run_name"]
        })
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def build_summarys(result):
    total = len(result)
    errors = sum(1 for r in result if r["error"])
    return {
        "total_questions":total,
        "error_count":errors,
        "success_count":total - errors
    }

def main():
    run_name = "baseline_topk5"
    Settings.RAG_FILE_PATH = PROJECT_ROOT / "data" / "kb"

    Settings.validate()
    init_rag(str(Settings.RAG_FILE_PATH),
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP)

    questions = load_question(PROJECT_ROOT / "data" / "eval" / "eval_questions_utf8_updated.json")
    results = [run_one_question(p, run_name) for p in questions]

    out_dir = PROJECT_ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(results, out_dir / 'results.json')
    save_csv(results, out_dir/ 'results.csv')

    summary = build_summarys(results)
    save_json(summary, out_dir / 'summary.json')

if __name__ == "__main__":
    main()



































