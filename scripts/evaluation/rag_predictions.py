"""通过正式 ChatService 生成 Agentic RAG Prediction v3，仅捕获安全证据元数据。"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.api.lifecycle import StartupDependencies, initialize_rag
from app.config import AppSettings
from app.contracts.chat import ToolMode
from app.rag import pipeline
from app.rag.pipeline import get_rag_status, rebuild_and_persist_index
from app.rag.store import needs_rebuild
from app.services.chat_models import ChatServiceRequest
from scripts.evaluation.agent import capture_tool_output_keys, run_evaluation_chat_turn

PREDICTION_SCHEMA_VERSION = 3
PREDICTION_FIELDNAMES = [
    "prediction_schema_version", "qid", "question", "answer", "status",
    "retrieved_chunk_ids", "reference_chunk_ids",
    "retrieved_chunk_count", "reference_chunk_count",
    "retrieval_call_count", "tool_call_count", "llm_call_count",
]


@dataclass(frozen=True)
class RagPredictionTrace:
    """跨工具观察的有序证据与 grounded 引用，均不包含正文。"""

    retrieved_chunk_ids: tuple[int, ...]
    reference_chunk_ids: tuple[int, ...]
    retrieval_call_count: int
    tool_call_count: int
    llm_call_count: int
    status: str


def _evaluable_id(value: Any, valid_ids: set[int]) -> int:
    if isinstance(value, bool) or not str(value).isascii() or not str(value).isdigit():
        raise ValueError("Knowledge-base evidence is missing evaluable chunk_id")
    chunk_id = int(value)
    if chunk_id not in valid_ids:
        raise ValueError("Knowledge-base evidence chunk_id is not in the evaluation index")
    return chunk_id


def initialize_rag_for_predictions(force_rebuild: bool = False) -> None:
    """使用与 API 启动流程相同的索引生命周期初始化 RAG。"""
    from app.config import get_settings

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


def run_rag_once_with_result(
    question: str,
    *,
    settings: AppSettings | None = None,
) -> tuple[dict[str, Any], RagPredictionTrace]:
    """通过唯一生产 Agent 链取得结果，再投影实际观察与实际引用。"""
    from app.config import get_settings

    with capture_tool_output_keys() as records:
        outcome = run_evaluation_chat_turn(
            ChatServiceRequest(
                question=question, session_id="", trace_id="evaluation-rag",
                tool_mode=ToolMode.RAG_ONLY,
            ),
            settings=settings or get_settings(),
            permissions=("knowledge_base:read",),
        )
    result = dict(outcome.result)
    status = str(result.get("status") or "")
    if status in {"MODEL_ERROR", "MODEL_BAD_OUTPUT", "TOOL_ERROR", "INTERNAL_ERROR"}:
        raise RuntimeError(f"RAG prediction failed: {status}")
    valid_ids = {
        int(value)
        for chunk in pipeline.chunks
        if (value := (chunk.get("metadata") or {}).get("chunk_id", chunk.get("chunk_id"))) is not None
        and not isinstance(value, bool) and str(value).isascii() and str(value).isdigit()
    }
    # ToolNode 的 state reducer 顺序对应 ToolMessages；线程完成先后不决定观察顺序。
    by_call = {record["tool_call_id"]: record for record in records}
    if len(by_call) != len(records):
        raise RuntimeError("Tool output capture has ambiguous duplicate call IDs")
    ordered = [
        by_call[step["tool_call_id"]]
        for step in result.get("tool_trace", [])
        if step.get("step") == "tool_call" and step.get("tool_call_id") in by_call
    ]
    if len(ordered) != len(records):
        raise RuntimeError("Tool output capture cannot be aligned with Agent observations")
    retrieved = tuple(dict.fromkeys(
        _evaluable_id(value, valid_ids)
        for record in ordered if record["ok"]
        for value in record["kb_chunk_ids"]
    ))
    referenced = tuple(dict.fromkeys(
        _evaluable_id(reference.get("chunk_id"), valid_ids)
        for reference in result.get("references", [])
    ))
    trace = RagPredictionTrace(
        retrieved_chunk_ids=retrieved,
        reference_chunk_ids=referenced,
        retrieval_call_count=sum(record["ok"] and record["has_evidence"] for record in ordered),
        tool_call_count=sum(step.get("step") == "tool_call" for step in result.get("tool_trace", [])),
        llm_call_count=int(result.get("llm_call_count", 0)),
        status=status,
    )
    return result, trace


def run_predictions(
    *,
    eval_file: str | Path,
    out_file: str | Path,
    force_rebuild: bool = False,
) -> list[dict[str, Any]]:
    """每题使用空 session 的正式 ChatService；错误状态使整次生成失败。"""
    initialize_rag_for_predictions(force_rebuild=force_rebuild)
    rows = []
    for item in load_eval_questions(eval_file):
        print(f"Running {item['qid']}: {item['question']}")
        try:
            result, trace = run_rag_once_with_result(item["question"])
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"RAG prediction failed for qid={item['qid']}: {exc}") from exc
        rows.append({
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "qid": item["qid"], "question": item["question"],
            "answer": result["answer"], "status": trace.status,
            "retrieved_chunk_ids": ",".join(map(str, trace.retrieved_chunk_ids)),
            "reference_chunk_ids": ",".join(map(str, trace.reference_chunk_ids)),
            "retrieved_chunk_count": len(trace.retrieved_chunk_ids),
            "reference_chunk_count": len(trace.reference_chunk_ids),
            "retrieval_call_count": trace.retrieval_call_count,
            "tool_call_count": trace.tool_call_count,
            "llm_call_count": trace.llm_call_count,
        })
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"predictions saved to: {out_path}")
    return rows
