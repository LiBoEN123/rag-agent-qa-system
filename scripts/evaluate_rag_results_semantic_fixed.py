# 启动命令：python scripts/evaluate_rag_results_semantic_fixed.py --eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json --pred-file outputs/predictions.csv --out-dir outputs/rag_baseline1 --k 10

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_eval import evaluation as _evaluation
from scripts.rag_eval import judge as _judge
from scripts.rag_eval.common import (
    ANSWER_FIELDS,
    CANONICAL_REPLACEMENTS,
    CHUNK_FIELDS,
    DOC_FIELDS,
    REFUSAL_PATTERNS,
    ChunkIndex,
    get_answer,
    has_any,
    is_refusal,
    is_subsequence,
    json_dumps,
    load_eval,
    load_predictions,
    mean,
    normalize_text,
    parse_chunk_ids,
    parse_doc_ids,
    parse_listish,
    point_text,
    point_texts,
    warn,
)
from scripts.rag_eval.evaluation import behavior_score, final_score, summarize
from scripts.rag_eval.judge import (
    ALLOWED_KEY_POINT_STATUSES,
    FORBIDDEN_JUDGE_SYSTEM_PROMPT,
    KEY_POINT_JUDGE_SYSTEM_PROMPT,
    LLMJudgeConfig,
    LLMJudgeError,
    build_forbidden_judge_prompt,
    build_key_point_judge_prompt,
    compact_error_preview,
    compute_key_point_score,
    make_judge_config,
    normalize_forbidden_judge_result,
    normalize_key_point_judge_result,
    normalize_status,
    parse_bool,
    parse_llm_json_response,
    strip_json_code_fence,
    extract_first_json_object,
    unavailable_forbidden_result,
    unavailable_key_point_result,
)
from scripts.rag_eval.report import write_csv, write_jsonl, write_summary_csv
from scripts.rag_eval.retrieval import (
    chunks_to_index,
    coerce_int,
    collect_retrieved_evidence,
    doc_id_matches,
    doc_label_from_metadata,
    evidence_matches_text,
    evidence_texts,
    find_gold_chunks_for_evidence,
    load_chroma_index,
    load_chunk_index,
    load_chunks_file_index,
    normalize_doc_id,
    resolve_gold_chunk_ids,
    score_retrieval,
)


def call_judge_llm(
    system_prompt: str,
    user_prompt: str,
    judge_config: LLMJudgeConfig,
    trace_id: str = "",
) -> str:
    return _judge.call_judge_llm(system_prompt, user_prompt, judge_config, trace_id=trace_id)


def call_judge_json(
    system_prompt: str,
    user_prompt: str,
    judge_config: LLMJudgeConfig,
    trace_id: str = "",
) -> Tuple[Dict[str, Any], str]:
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
        if qid not in cache:
            raise LLMJudgeError(f"qid={qid} missing from judge cache")
        row = cache[qid]
        raw = {"forbidden_judgments": json.loads(row.get("forbidden_judgments_json") or "[]")}
        result = normalize_forbidden_judge_result(raw, point_texts(forbidden_points))
        result["judge_raw_response"] = ""
        return result

    return cached_key_point_evaluator, cached_forbidden_evaluator


def main() -> None:
    """解析参数、运行语义评测并输出报告文件。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", required=True, type=Path)
    ap.add_argument("--pred-file", required=True, type=Path)
    ap.add_argument(
        "--chunks-file",
        type=Path,
        default=None,
        help="Legacy chunks.json fallback. Auto mode prefers ChromaDB because manifest no longer stores chunks.",
    )
    ap.add_argument(
        "--index-source",
        choices=["auto", "chroma", "chunks-file", "none"],
        default="auto",
        help="Where to load current chunk text/metadata for retrieval scoring. Default: auto (ChromaDB first).",
    )
    ap.add_argument(
        "--chroma-persist-dir",
        type=Path,
        default=None,
        help="Override Settings.CHROMA_PERSIST_DIR for evaluation.",
    )
    ap.add_argument(
        "--chroma-collection-name",
        default=None,
        help="Override Settings.CHROMA_COLLECTION_NAME for evaluation.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_report"))
    ap.add_argument("--k", type=int, default=10, help="Use top-k retrieved chunks for retrieval hit metrics")
    ap.add_argument(
        "--judge-cache-report",
        type=Path,
        default=None,
        help="Reuse judge results from a previous per_question_report.csv. This never calls the LLM judge.",
    )
    judge_group = ap.add_mutually_exclusive_group()
    judge_group.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable LLM semantic judge for key points and forbidden hits. Also configurable via USE_LLM_JUDGE=true.",
    )
    judge_group.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM judge even when USE_LLM_JUDGE=true.",
    )
    ap.add_argument("--judge-model", default=None, help="Override JUDGE_MODEL for evaluation judge")
    ap.add_argument("--judge-api-url", default=None, help="Override JUDGE_API_URL for evaluation judge")
    ap.add_argument("--judge-api-key", default=None, help="Override JUDGE_API_KEY for evaluation judge")
    args = ap.parse_args()

    eval_items = load_eval(args.eval_file)
    preds = load_predictions(args.pred_file)
    chunk_index = load_chunk_index(
        chunks_file=args.chunks_file,
        index_source=args.index_source,
        chroma_persist_dir=args.chroma_persist_dir,
        chroma_collection_name=args.chroma_collection_name,
    )
    if args.index_source != "none" and not chunk_index.chunk_text_map:
        warn("no chunk index loaded; doc-hit and evidence-based retrieval metrics may fall back or be unavailable")

    judge_override = None
    if args.use_llm_judge:
        judge_override = True
    if args.no_llm_judge:
        judge_override = False
    judge_config = make_judge_config(
        use_llm_judge=judge_override,
        judge_model=args.judge_model,
        judge_api_url=args.judge_api_url,
        judge_api_key=args.judge_api_key,
    )

    if args.judge_cache_report:
        judge_cache = load_judge_cache_report(args.judge_cache_report)
        missing_qids = sorted(str(q["qid"]) for q in eval_items if str(q["qid"]) not in judge_cache)
        if missing_qids:
            preview = ", ".join(missing_qids[:10])
            more = "..." if len(missing_qids) > 10 else ""
            raise ValueError(f"judge cache missing {len(missing_qids)} qid(s): {preview}{more}")
        key_point_evaluator, forbidden_evaluator = make_cached_judge_evaluators(judge_cache)
        judge_config = LLMJudgeConfig(
            enabled=True,
            model=f"cached:{args.judge_cache_report.name}",
            api_url="cache://per_question_report",
            api_key="",
            timeout=0,
            max_retries=0,
            retry_sleep=0,
            max_tokens=0,
        )
        rows, summary, judge_details = _evaluation.evaluate(
            eval_items,
            preds,
            chunk_index,
            args.k,
            judge_config=judge_config,
            key_point_evaluator=key_point_evaluator,
            forbidden_evaluator=forbidden_evaluator,
        )
    else:
        rows, summary, judge_details = evaluate(eval_items, preds, chunk_index, args.k, judge_config=judge_config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.out_dir / "per_question_report.csv", rows)
    write_summary_csv(args.out_dir / "summary_report.csv", summary)
    write_jsonl(args.out_dir / "llm_judge_details.jsonl", judge_details)
    (args.out_dir / "summary_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = summary["overall"]
    print("=== RAG Evaluation Summary ===")
    print(f"Chunk index source: {chunk_index.source} ({len(chunk_index.chunk_text_map)} chunks)")
    if args.judge_cache_report:
        print(f"LLM judge: cached ({args.judge_cache_report})")
    else:
        print(f"LLM judge: {'enabled' if judge_config.enabled else 'disabled'}")
    if judge_config.enabled:
        print(f"Judge model: {judge_config.model}")
    print(f"Questions: {overall['num_questions']}")
    print(f"Prediction coverage: {overall['prediction_coverage']:.2%}")
    print(f"Average final score: {overall['avg_final_score']:.2%}")
    print(f"Average behavior score: {overall['avg_behavior_score']:.2%}")
    print(f"Average key-point recall: {overall['avg_key_point_recall']:.2%}")
    print(f"Gold chunk hit@{args.k}: {overall['gold_chunk_hit_at_k']:.2%}")
    print(f"Gold chunk recall@{args.k}: {overall['gold_chunk_recall_at_k']:.2%}")
    print(f"All gold hit@{args.k}: {overall['all_gold_hit_at_k']:.2%}")
    print(f"Doc hit@{args.k}: {overall['doc_hit_at_k']:.2%}")
    print(f"Forbidden/hallucination rate: {overall['forbidden_rate']:.2%}")
    print(f"\nReports written to: {args.out_dir}")


if __name__ == "__main__":
    main()
