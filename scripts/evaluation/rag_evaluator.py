"""装配 Prediction v3、召回指标、可选 GEval/只读缓存与本地报告。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.evaluation.deepeval_adapter import ProjectJudge, answer_correctness
from scripts.evaluation.rag.common import load_eval, load_predictions, warn
from scripts.evaluation.rag.evaluation import evaluate
from scripts.evaluation.rag.report import write_csv, write_summary_csv
from scripts.evaluation.rag.retrieval import load_chunk_index


def load_judge_cache_report(path: Path) -> dict[str, dict[str, str]]:
    """只读取当前分数合同；缓存不完整时绝不回退调用模型。"""
    fields = {"qid", "answer", "answer_correctness", "answer_correctness_reason"}
    cache = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError("judge cache requires qid, answer, answer_correctness, answer_correctness_reason")
        for row in reader:
            qid = row["qid"]
            if not qid or qid in cache:
                raise ValueError("judge cache qid must be nonempty and unique")
            cache[qid] = {key: row[key] for key in fields}
    return cache


def run_rag_evaluation(
    *,
    eval_file: Path,
    pred_file: Path,
    out_dir: Path,
    k: int = 10,
    chunks_file: Path | None = None,
    index_source: str = "auto",
    chroma_persist_dir: Path | None = None,
    chroma_collection_name: str | None = None,
    judge_cache_report: Path | None = None,
    use_llm_judge: bool | None = None,
    judge_model: str | None = None,
    judge_api_url: str | None = None,
    judge_api_key: str | None = None,
) -> dict[str, Any]:
    """运行离线评分；只有显式启用 Judge 才可能创建模型客户端。"""
    from app.config import get_settings

    eval_items, preds = load_eval(eval_file), load_predictions(pred_file)
    chunk_index = load_chunk_index(
        chunks_file=chunks_file, index_source=index_source,
        chroma_persist_dir=chroma_persist_dir, chroma_collection_name=chroma_collection_name,
    )
    if index_source != "none" and not chunk_index.chunk_text_map:
        warn("no chunk index loaded; gold alignment uses dataset chunk IDs")

    settings = get_settings()
    enabled = settings.use_llm_judge if use_llm_judge is None else use_llm_judge
    model_name = (settings.judge_model if judge_model is None else judge_model).strip()
    api_url = settings.judge_api_url_resolved if judge_api_url is None else judge_api_url
    answer_evaluator = None
    if judge_cache_report is not None:
        cache = load_judge_cache_report(judge_cache_report)
        enabled, model_name = True, f"cached:{judge_cache_report.name}"

        def answer_evaluator(question, answer):
            row = cache.get(str(question["qid"]))
            if row is None:
                raise ValueError("qid missing from judge cache")
            if row["answer"] != answer:
                raise ValueError("answer changed; judge cache cannot be reused")
            if not row["answer_correctness"]:
                raise ValueError("judge cache has no answer correctness score")
            score = float(row["answer_correctness"])
            if not 0 <= score <= 1:
                raise ValueError("judge cache score must be in [0, 1]")
            return score, row["answer_correctness_reason"] or ""

    elif enabled:
        if not model_name or not api_url.strip():
            warn("Judge model or URL is empty; answer_correctness remains unavailable")
            enabled = False
        else:
            judge = None

            def answer_evaluator(question, answer):
                nonlocal judge
                if judge is None:
                    judge = ProjectJudge(model_name=model_name, api_url=api_url, api_key=judge_api_key)
                return answer_correctness(question, answer, model=judge)

    rows, summary = evaluate(eval_items, preds, chunk_index, k, answer_evaluator=answer_evaluator)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_question_report.csv", rows)
    write_summary_csv(out_dir / "summary_report.csv", summary)
    (out_dir / "summary_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print("=== RAG Evaluation Summary ===")
    print(f"Chunk index source: {chunk_index.source} ({len(chunk_index.chunk_text_map)} chunks)")
    print(f"LLM judge: {model_name if enabled else 'disabled'}")
    for name, value in summary["overall"].items():
        print(f"{name}: {value if value is not None else 'n/a'}")
    print(f"Reports written to: {out_dir}")
    return {
        "rows": rows, "summary": summary,
        "chunk_index_source": chunk_index.source,
        "chunk_index_count": len(chunk_index.chunk_text_map),
        "judge_enabled": enabled,
        "judge_model": model_name if enabled else "",
    }
