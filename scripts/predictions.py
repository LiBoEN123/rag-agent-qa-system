# # 启动命令：python scripts/predictions.py --eval-file data/eval/rag_eval_questions_nmgut_graduate_v1.json --out-file outputs/predictions.csv

import json
import csv
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import retrieve, ask, init_rag, retrieve_with_sub_queries
from app.config import Settings

init_rag(data_dir=str(Settings.RAG_FILE_PATH),
         chunk_size=Settings.CHUNK_SIZE,
         chunk_overlap=Settings.CHUNK_OVERLAP)

def load_eval_questions(eval_file):
    """读取评测题列表，兼容 list 和带 items 的 JSON 对象。"""
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "items" in data:
        return data["items"]

    raise ValueError("评测文件格式不正确：既不是题目列表，也没有 items 字段")


def run_rag_once(question: str, top_k: int):
    """对单个问题运行 RAG 并返回答案及召回 chunk_id。"""
    retrieved_chunks = retrieve(question, top_k)
    result = ask(question, top_k, retrieved_chunks)
    final_chunks = result.get("final_chunks")
    answer = result["answer"]

    retrieved_chunk_ids = [
        chunk["metadata"]["chunk_id"]
        for chunk in final_chunks
    ]

    return answer, retrieved_chunk_ids


def main():
    """批量生成预测结果并写入 CSV。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--out-file", required=True)
    args = parser.parse_args()

    eval_items = load_eval_questions(args.eval_file)

    rows = []

    top_k = 5

    for item in eval_items:
        qid = item["qid"]
        question = item["question"]

        print(f"Running {qid}: {question}")

        answer, retrieved_chunk_ids = run_rag_once(question, top_k)

        rows.append({
            "qid": qid,
            "question": question,
            "answer": answer,
            "retrieved_chunk_ids": ",".join(str(x) for x in retrieved_chunk_ids),
        })

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["qid", "question", "answer", "retrieved_chunk_ids"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"predictions saved to: {out_path}")


if __name__ == "__main__":
    main()