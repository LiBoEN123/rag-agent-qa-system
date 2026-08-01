"""Evaluation Baseline v2 的 RAG 预测生成实现。

用途：用当前生产 RAG 生命周期逐题检索和生成答案，为语义评测准备预测文件。
输入：RAG 评测题、top-k、输出路径，以及是否强制重建索引等运行选项。
输出：包含答案、召回 chunk_id 和运行状态的预测 CSV 与内存记录。
边界：真实运行可能加载或重建本地索引并调用模型；离线 ``--validate-only`` 不会进入这里。
"""

import csv
import json
from pathlib import Path

from app.api.lifecycle import StartupDependencies, initialize_rag
from app.config import get_settings
from app.rag import (
    get_rag_status,
    index_cache_exists,
    load_index_cache,
    needs_rebuild,
    rebuild_and_persist_index,
    retrieve,
)
from app.rag.graph import run_rag_subgraph


def initialize_rag_for_predictions(force_rebuild: bool = False) -> None:
    """使用与 API 启动流程相同的索引生命周期初始化 RAG。"""
    settings = get_settings()
    if force_rebuild:
        ok, message = rebuild_and_persist_index(str(settings.rag_file_path))
        if not ok:
            raise RuntimeError(f"RAG 强制重建失败：{message}")
    else:
        initialize_rag(
            StartupDependencies(
                settings=settings,
                needs_rebuild=needs_rebuild,
                index_cache_exists=index_cache_exists,
                load_index_cache=load_index_cache,
                rebuild_and_persist_index=rebuild_and_persist_index,
            )
        )
    status = get_rag_status()
    if not status.get("ready"):
        raise RuntimeError(f"RAG 未准备好：{status}")

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
    result, retrieved_chunk_ids = run_rag_once_with_result(question, top_k)
    return result["answer"], retrieved_chunk_ids


def run_rag_once_with_result(
    question: str,
    top_k: int,
) -> tuple[dict, list]:
    """运行单题 RAG，并保留统一 Runner 所需的状态合同。"""
    retrieved_chunks = retrieve(question, top_k)
    result = run_rag_subgraph(
        question=question,
        top_k=top_k,
        retrieved_chunks=retrieved_chunks,
    )
    final_chunks = result.get("final_chunks")
    retrieved_chunk_ids = [
        chunk["metadata"]["chunk_id"]
        for chunk in final_chunks
    ]

    return result, retrieved_chunk_ids


def run_predictions(
    *,
    eval_file: str | Path,
    out_file: str | Path,
    top_k: int = 5,
    force_rebuild: bool = False,
    fail_on_model_error: bool = False,
) -> list[dict]:
    """运行现有 RAG 预测流程并返回写入的行。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    initialize_rag_for_predictions(force_rebuild=force_rebuild)
    eval_items = load_eval_questions(eval_file)

    rows = []

    for item in eval_items:
        qid = item["qid"]
        question = item["question"]

        print(f"Running {qid}: {question}")

        result, retrieved_chunk_ids = run_rag_once_with_result(
            question,
            top_k,
        )
        status = str(result.get("status") or "")
        if fail_on_model_error and status in {
            "MODEL_ERROR",
            "MODEL_BAD_OUTPUT",
        }:
            raise RuntimeError(
                f"RAG prediction failed for qid={qid}: {status}"
            )
        answer = result["answer"]

        rows.append({
            "qid": qid,
            "question": question,
            "answer": answer,
            "retrieved_chunk_ids": ",".join(str(x) for x in retrieved_chunk_ids),
        })

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["qid", "question", "answer", "retrieved_chunk_ids"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"predictions saved to: {out_path}")
    return rows


__all__ = [
    "initialize_rag_for_predictions",
    "load_eval_questions",
    "run_predictions",
    "run_rag_once",
    "run_rag_once_with_result",
]
