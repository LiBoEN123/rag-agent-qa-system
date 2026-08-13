"""Evaluation Baseline v2 的 RAG 语义评测编排层。

用途：装配预测、当前 chunk 索引、确定性检索指标、可选 LLM judge 和报告写入。
输入：RAG 评测集、预测文件、top-k、索引位置、judge 配置与可选只读缓存。
输出：逐题指标、分组汇总、judge 明细，以及 CSV/JSONL 评测产物。
边界：判分公式分别由 ``rag.evaluation`` 与 ``rag.retrieval`` 维护；本模块只负责编排兼容接口。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.evaluation.rag import evaluation as _evaluation
from scripts.evaluation.rag import judge as _judge
from scripts.evaluation.rag.common import (
    ChunkIndex,
    load_eval,
    load_predictions,
    point_texts,
    warn,
)
from scripts.evaluation.rag.judge import (
    FORBIDDEN_JUDGE_SYSTEM_PROMPT,
    KEY_POINT_JUDGE_SYSTEM_PROMPT,
    LLMJudgeConfig,
    LLMJudgeError,
    make_judge_config,
    normalize_forbidden_judge_result,
    normalize_key_point_judge_result,
)
from scripts.evaluation.rag.report import write_csv, write_jsonl, write_summary_csv
from scripts.evaluation.rag.retrieval import load_chunk_index


def call_judge_llm(
    system_prompt: str,
    user_prompt: str,
    judge_config: LLMJudgeConfig,
    trace_id: str = "",
) -> str:
    """转交到底层 judge 网关，保留便于测试替换的兼容调用点。"""
    return _judge.call_judge_llm(system_prompt, user_prompt, judge_config, trace_id=trace_id)


def call_judge_json(
    system_prompt: str,
    user_prompt: str,
    judge_config: LLMJudgeConfig,
    trace_id: str = "",
) -> Tuple[Dict[str, Any], str]:
    """调用并解析 judge JSON，同时允许测试替换本模块的 LLM 调用函数。"""
    return _judge.call_judge_json(
        system_prompt,
        user_prompt,
        judge_config,
        trace_id=trace_id,
        call_judge_llm_fn=call_judge_llm,
    )


def evaluate_key_points_with_llm(
    question: str,
    gold_answer: str,
    gold_key_points: List[Any],
    model_answer: str,
    evidence: List[str] | None = None,
    judge_config: LLMJudgeConfig | None = None,
    category: str = "",
    expected_behavior: str = "",
    qid: str = "",
) -> Dict[str, Any]:
    """按标准关键点执行语义覆盖判断，并返回规范化结果。"""
    return _judge.evaluate_key_points_with_llm(
        question=question,
        gold_answer=gold_answer,
        gold_key_points=gold_key_points,
        model_answer=model_answer,
        evidence=evidence,
        judge_config=judge_config,
        category=category,
        expected_behavior=expected_behavior,
        qid=qid,
        call_judge_json_fn=call_judge_json,
    )


def evaluate_forbidden_with_llm(
    question: str,
    model_answer: str,
    forbidden_points: List[Any],
    evidence: List[str] | None = None,
    judge_config: LLMJudgeConfig | None = None,
    gold_answer: str = "",
    qid: str = "",
) -> Dict[str, Any]:
    """按禁用信息点执行语义命中判断，并返回规范化结果。"""
    return _judge.evaluate_forbidden_with_llm(
        question=question,
        model_answer=model_answer,
        forbidden_points=forbidden_points,
        evidence=evidence,
        judge_config=judge_config,
        gold_answer=gold_answer,
        qid=qid,
        call_judge_json_fn=call_judge_json,
    )


def evaluate(
    eval_items: List[Dict[str, Any]],
    preds: Dict[str, Dict[str, Any]],
    chunk_index: ChunkIndex,
    k: int,
    judge_config: LLMJudgeConfig | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """转交逐题评分实现，并注入本模块可替换的 judge 适配函数。"""
    return _evaluation.evaluate(
        eval_items,
        preds,
        chunk_index,
        k,
        judge_config=judge_config,
        key_point_evaluator=evaluate_key_points_with_llm,
        forbidden_evaluator=evaluate_forbidden_with_llm,
    )


def load_judge_cache_report(path: Path) -> Dict[str, Dict[str, Any]]:
    """从旧 per_question_report.csv 读取 judge 结果，避免重复消耗 LLM judge token。"""
    if not path.exists():
        raise FileNotFoundError(f"judge cache report not found: {path}")

    cache: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("qid", "")).strip()
            if not qid:
                continue
            if "key_point_judgments_json" not in row or "forbidden_judgments_json" not in row:
                raise ValueError("judge cache report must contain key_point_judgments_json and forbidden_judgments_json")
            try:
                json.loads(row.get("key_point_judgments_json") or "[]")
                json.loads(row.get("forbidden_judgments_json") or "[]")
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid judge JSON in cache for qid={qid}: {exc}") from exc
            cache[qid] = row

    if not cache:
        raise ValueError(f"judge cache report has no rows: {path}")
    return cache


def make_cached_judge_evaluators(cache: Dict[str, Dict[str, Any]]) -> Tuple[Any, Any]:
    """构造只读缓存的 evaluator；缓存缺失时失败，绝不回退调用 LLM。"""

    def cached_key_point_evaluator(
        question: str,
        gold_answer: str,
        gold_key_points: List[Any],
        model_answer: str,
        evidence: List[str] | None = None,
        judge_config: LLMJudgeConfig | None = None,
        category: str = "",
        expected_behavior: str = "",
        qid: str = "",
    ) -> Dict[str, Any]:
        """只从缓存恢复当前 qid 的关键点判断。"""
        if qid not in cache:
            raise LLMJudgeError(f"qid={qid} missing from judge cache")
        row = cache[qid]
        raw = {"key_point_judgments": json.loads(row.get("key_point_judgments_json") or "[]")}
        result = normalize_key_point_judge_result(raw, point_texts(gold_key_points))
        result["judge_raw_response"] = ""
        return result

    def cached_forbidden_evaluator(
        question: str,
        model_answer: str,
        forbidden_points: List[Any],
        evidence: List[str] | None = None,
        judge_config: LLMJudgeConfig | None = None,
        gold_answer: str = "",
        qid: str = "",
    ) -> Dict[str, Any]:
        """只从缓存恢复当前 qid 的禁用信息判断。"""
        if qid not in cache:
            raise LLMJudgeError(f"qid={qid} missing from judge cache")
        row = cache[qid]
        raw = {"forbidden_judgments": json.loads(row.get("forbidden_judgments_json") or "[]")}
        result = normalize_forbidden_judge_result(raw, point_texts(forbidden_points))
        result["judge_raw_response"] = ""
        return result

    return cached_key_point_evaluator, cached_forbidden_evaluator


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
    """运行现有 RAG evaluator 并返回结构化结果，供统一 Runner 复用。"""
    eval_items = load_eval(eval_file)
    preds = load_predictions(pred_file)
    chunk_index = load_chunk_index(
        chunks_file=chunks_file,
        index_source=index_source,
        chroma_persist_dir=chroma_persist_dir,
        chroma_collection_name=chroma_collection_name,
    )
    if index_source != "none" and not chunk_index.chunk_text_map:
        warn("no chunk index loaded; doc-hit and evidence-based retrieval metrics may fall back or be unavailable")

    judge_config = make_judge_config(
        use_llm_judge=use_llm_judge,
        judge_model=judge_model,
        judge_api_url=judge_api_url,
        judge_api_key=judge_api_key,
    )

    if judge_cache_report:
        judge_cache = load_judge_cache_report(judge_cache_report)
        missing_qids = sorted(str(q["qid"]) for q in eval_items if str(q["qid"]) not in judge_cache)
        if missing_qids:
            preview = ", ".join(missing_qids[:10])
            more = "..." if len(missing_qids) > 10 else ""
            raise ValueError(f"judge cache missing {len(missing_qids)} qid(s): {preview}{more}")
        key_point_evaluator, forbidden_evaluator = make_cached_judge_evaluators(judge_cache)
        judge_config = LLMJudgeConfig(
            enabled=True,
            model=f"cached:{judge_cache_report.name}",
            api_url="cache://per_question_report",
            api_key="",
            timeout=0,
            max_retries=0,
            max_tokens=0,
        )
        rows, summary, judge_details = _evaluation.evaluate(
            eval_items,
            preds,
            chunk_index,
            k,
            judge_config=judge_config,
            key_point_evaluator=key_point_evaluator,
            forbidden_evaluator=forbidden_evaluator,
        )
    else:
        rows, summary, judge_details = evaluate(
            eval_items,
            preds,
            chunk_index,
            k,
            judge_config=judge_config,
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "per_question_report.csv", rows)
    write_summary_csv(out_dir / "summary_report.csv", summary)
    write_jsonl(out_dir / "llm_judge_details.jsonl", judge_details)
    (out_dir / "summary_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    overall = summary["overall"]
    print("=== RAG Evaluation Summary ===")
    print(f"Chunk index source: {chunk_index.source} ({len(chunk_index.chunk_text_map)} chunks)")
    if judge_cache_report:
        print(f"LLM judge: cached ({judge_cache_report})")
    else:
        print(f"LLM judge: {'enabled' if judge_config.enabled else 'disabled'}")
    if judge_config.enabled:
        print(f"Judge model: {judge_config.model}")
    print(f"Questions: {overall['num_questions']}")
    print(f"Prediction coverage: {overall['prediction_coverage']:.2%}")
    print(f"Average final score: {overall['avg_final_score']:.2%}")
    print(f"Average behavior score: {overall['avg_behavior_score']:.2%}")
    print(f"Average key-point recall: {overall['avg_key_point_recall']:.2%}")
    print(f"Gold chunk hit@{k}: {overall['gold_chunk_hit_at_k']:.2%}")
    print(f"Gold chunk recall@{k}: {overall['gold_chunk_recall_at_k']:.2%}")
    print(f"All gold hit@{k}: {overall['all_gold_hit_at_k']:.2%}")
    print(f"Doc hit@{k}: {overall['doc_hit_at_k']:.2%}")
    if overall.get("context_gold_chunk_recall") is not None:
        print(
            "Context gold chunk recall: "
            f"{overall['context_gold_chunk_recall']:.2%}"
        )
        print(
            "Reference gold chunk recall: "
            f"{overall['reference_gold_chunk_recall']:.2%}"
        )
        print(
            "Context truncation rate: "
            f"{overall['context_truncation_rate']:.2%}"
        )
        print(
            "Gold lost by context/reference: "
            f"{overall['gold_lost_by_context_count']} / "
            f"{overall['gold_lost_by_reference_count']}"
        )
    print(f"Forbidden/hallucination rate: {overall['forbidden_rate']:.2%}")
    print(f"\nReports written to: {out_dir}")
    return {
        "rows": rows,
        "summary": summary,
        "judge_details": judge_details,
        "chunk_index_source": chunk_index.source,
        "chunk_index_count": len(chunk_index.chunk_text_map),
        "judge_enabled": judge_config.enabled,
        "judge_model": judge_config.model if judge_config.enabled else "",
    }


__all__ = [
    "ChunkIndex",
    "FORBIDDEN_JUDGE_SYSTEM_PROMPT",
    "KEY_POINT_JUDGE_SYSTEM_PROMPT",
    "LLMJudgeConfig",
    "LLMJudgeError",
    "call_judge_json",
    "call_judge_llm",
    "evaluate",
    "evaluate_forbidden_with_llm",
    "evaluate_key_points_with_llm",
    "load_judge_cache_report",
    "make_cached_judge_evaluators",
    "run_rag_evaluation",
]
