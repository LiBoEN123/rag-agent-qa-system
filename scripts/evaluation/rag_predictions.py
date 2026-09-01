"""Evaluation Baseline v2 的 RAG 预测生成实现。

用途：用当前生产 RAG 生命周期逐题检索和生成答案，为语义评测准备预测文件。
输入：RAG 评测题、top-k、输出路径，以及是否强制重建索引等运行选项。
输出：包含答案、召回 chunk_id 和运行状态的预测 CSV 与内存记录。
边界：真实运行可能加载或重建本地索引并调用模型；离线 ``--validate-only`` 不会进入这里。
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from app.rag.generation import build_context_with_selection


PREDICTION_SCHEMA_VERSION = 2
PREDICTION_FIELDNAMES = [
    "prediction_schema_version",
    "qid",
    "question",
    "answer",
    "retrieved_chunk_ids",
    "final_chunk_ids",
    "context_chunk_ids",
    "reference_chunk_ids",
    "final_chunk_count",
    "context_chunk_count",
    "reference_chunk_count",
    "context_char_count",
    "top_k",
    "decompose_top_k",
    "decompose_llm_used",
    "status",
]


@dataclass(frozen=True)
class RagPredictionTrace:
    """单题预测中允许持久化的 RAG 证据与有效参数元数据。"""

    final_chunk_ids: tuple[int, ...]
    context_chunk_ids: tuple[int, ...]
    reference_chunk_ids: tuple[int, ...]
    context_char_count: int
    top_k: int
    decompose_top_k: int
    decompose_llm_used: bool
    status: str


def _chunk_ids(items: Any, *, field: str) -> tuple[int, ...]:
    """从生产 chunk/reference 字典提取稳定整数 ID。"""
    if items is None:
        return ()
    if not isinstance(items, list):
        raise ValueError(f"RAG result {field} 必须是 list")
    chunk_ids: list[int] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"RAG result {field}[{index}] 必须是 object")
        value = item.get("chunk_id")
        if value is None:
            metadata = item.get("metadata") or {}
            if isinstance(metadata, dict):
                value = metadata.get("chunk_id")
        try:
            chunk_ids.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"RAG result {field}[{index}] 缺少整数 chunk_id"
            ) from exc
    return tuple(chunk_ids)


def _serialize_chunk_ids(chunk_ids: tuple[int, ...]) -> str:
    """使用与 v1 兼容的逗号分隔形式写入 CSV。"""
    return ",".join(str(chunk_id) for chunk_id in chunk_ids)


def initialize_rag_for_predictions(force_rebuild: bool = False) -> None:
    """使用与 API 启动流程相同的索引生命周期初始化 RAG。"""
    settings = get_settings()
    if force_rebuild:
        ok, message = rebuild_and_persist_index(
            str(settings.rag_file_path),
            force_rebuild=True,
        )
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


def run_rag_once(
    question: str,
    top_k: int,
    decompose_top_k: int,
) -> tuple[str, list[int]]:
    """对单个问题运行 RAG 并返回答案及最终排序 chunk_id。"""
    result, trace = run_rag_once_with_result(
        question,
        top_k,
        decompose_top_k,
    )
    return result["answer"], list(trace.final_chunk_ids)


def run_rag_once_with_result(
    question: str,
    top_k: int,
    decompose_top_k: int,
) -> tuple[dict[str, Any], RagPredictionTrace]:
    """运行单题 RAG，并返回不含正文的结构化预测追踪。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    if decompose_top_k < 1:
        raise ValueError("decompose_top_k 必须大于 0")
    retrieved_chunks = retrieve(question, top_k)
    result = run_rag_subgraph(
        question=question,
        top_k=top_k,
        decompose_top_k=decompose_top_k,
        retrieved_chunks=retrieved_chunks,
    )
    final_chunks = result.get("final_chunks") or []
    context, context_chunks = build_context_with_selection(final_chunks)
    trace = RagPredictionTrace(
        final_chunk_ids=_chunk_ids(final_chunks, field="final_chunks"),
        context_chunk_ids=_chunk_ids(
            context_chunks,
            field="context_chunks",
        ),
        reference_chunk_ids=_chunk_ids(
            result.get("references") or [],
            field="references",
        ),
        context_char_count=len(context),
        top_k=top_k,
        decompose_top_k=decompose_top_k,
        decompose_llm_used=bool(result.get("decompose_llm_used", False)),
        status=str(result.get("status") or ""),
    )

    return result, trace


def run_predictions(
    *,
    eval_file: str | Path,
    out_file: str | Path,
    top_k: int = 5,
    decompose_top_k: int = 5,
    force_rebuild: bool = False,
    fail_on_model_error: bool = False,
) -> list[dict[str, Any]]:
    """运行现有 RAG 预测流程并返回写入的行。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    if decompose_top_k < 1:
        raise ValueError("decompose_top_k 必须大于 0")
    initialize_rag_for_predictions(force_rebuild=force_rebuild)
    eval_items = load_eval_questions(eval_file)

    rows = []

    for item in eval_items:
        qid = item["qid"]
        question = item["question"]

        print(f"Running {qid}: {question}")

        result, trace = run_rag_once_with_result(
            question,
            top_k,
            decompose_top_k,
        )
        status = trace.status
        if fail_on_model_error and status in {
            "MODEL_ERROR",
            "MODEL_BAD_OUTPUT",
        }:
            raise RuntimeError(
                f"RAG prediction failed for qid={qid}: {status}"
            )
        answer = result["answer"]

        final_chunk_ids = _serialize_chunk_ids(trace.final_chunk_ids)
        rows.append(
            {
                "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
                "qid": qid,
                "question": question,
                "answer": answer,
                # v1 兼容别名；语义固定为最终排序 chunks。
                "retrieved_chunk_ids": final_chunk_ids,
                "final_chunk_ids": final_chunk_ids,
                "context_chunk_ids": _serialize_chunk_ids(
                    trace.context_chunk_ids
                ),
                "reference_chunk_ids": _serialize_chunk_ids(
                    trace.reference_chunk_ids
                ),
                "final_chunk_count": len(trace.final_chunk_ids),
                "context_chunk_count": len(trace.context_chunk_ids),
                "reference_chunk_count": len(trace.reference_chunk_ids),
                "context_char_count": trace.context_char_count,
                "top_k": trace.top_k,
                "decompose_top_k": trace.decompose_top_k,
                "decompose_llm_used": trace.decompose_llm_used,
                "status": trace.status,
            }
        )

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=PREDICTION_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"predictions saved to: {out_path}")
    return rows


__all__ = [
    "PREDICTION_FIELDNAMES",
    "PREDICTION_SCHEMA_VERSION",
    "RagPredictionTrace",
    "initialize_rag_for_predictions",
    "load_eval_questions",
    "run_predictions",
    "run_rag_once",
    "run_rag_once_with_result",
]
