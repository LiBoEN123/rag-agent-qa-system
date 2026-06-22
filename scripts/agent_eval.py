import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any
import os
from app.main import write_bad_cases


# =========================
# 1. 让脚本能从项目根目录导入 app
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.config import Settings
from app.agent import run_agent_once
from app.rag import (
    get_rag_status,
    needs_rebuild,
    index_cache_exists,
    load_index_cache,
    rebuild_and_persist_index,
)


# =========================
# 2. 路径配置
# =========================
DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "agent_eval_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "agent_eval"


# =========================
# 3. 通用小工具函数
# =========================
def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"找不到评测集文件：{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def shorten(text: Any, max_len: int = 120) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def safe_lower(text: Any) -> str:
    return "" if text is None else str(text).casefold()


# =========================
# 4. 初始化 RAG
# =========================
def init_rag_for_eval(force_rebuild: bool = False) -> None:
    """
    单独跑 eval_agent.py 时，不会经过 FastAPI 的 startup，
    所以这里需要手动初始化 RAG。

    逻辑：
    1. 如果需要强制重建，就直接 rebuild_and_persist_index。
    2. 如果不需要重建，且存在缓存，就优先 load_index_cache。
    3. 如果缓存不可用，再重建索引。
    """

    if hasattr(Settings, "validate"):
        Settings.validate()

    data_dir = str(Settings.RAG_FILE_PATH)

    if force_rebuild:
        ok, msg = rebuild_and_persist_index(data_dir)
        print(f"[init] force_rebuild=True, ok={ok}, msg={msg}")
    else:
        need, reason = needs_rebuild(data_dir)

        if not need and index_cache_exists():
            ok, msg = load_index_cache()
            print(f"[init] load_index_cache, ok={ok}, msg={msg}")

            if not ok:
                ok, msg = rebuild_and_persist_index(data_dir)
                print(f"[init] cache failed, rebuild, ok={ok}, msg={msg}")
        else:
            ok, msg = rebuild_and_persist_index(data_dir)
            print(f"[init] rebuild, need={need}, reason={reason}, ok={ok}, msg={msg}")

    status = get_rag_status()
    print(
        "[init] RAG status:",
        f"ready={status.get('ready')},",
        f"message={status.get('message')},",
        f"doc_count={status.get('doc_count')},",
        f"chunk_count={status.get('chunk_count')}"
    )

    if not status.get("ready"):
        raise RuntimeError(f"RAG 未准备好：{status}")


# =========================
# 5. 从 Agent 结果中提取实际行为
# =========================
def get_first_trace(result: dict) -> dict:
    trace = result.get("tool_trace") or []
    if isinstance(trace, list) and trace:
        first = trace[0]
        if isinstance(first, dict):
            return first
    return {}


def extract_actual_fields(result: dict) -> dict:
    """
    你的 run_agent_once 返回里：
    - result["status"] 是最终状态
    - result["action"] 是最终动作
    - result["tool_trace"][0]["decision_type"] 是路由决策
    - result["tool_trace"][0]["tool_name"] 是工具名
    """

    first_trace = get_first_trace(result)

    actual_decision = first_trace.get("decision_type", "")
    actual_action = result.get("action") or first_trace.get("action", "")
    actual_tool = first_trace.get("tool_name", "")
    actual_status = result.get("status", "")

    return {
        "actual_decision": actual_decision,
        "actual_action": actual_action,
        "actual_tool": actual_tool,
        "actual_status": actual_status,
    }


# =========================
# 6. 关键词与引用检查
# =========================
def build_search_text(result: dict) -> str:
    """
    关键词不只在 answer 里找，也在 reason、tool_result_summary、references 里找。
    这样工具题、引用题都能检查。
    """

    parts = [
        result.get("answer", ""),
        result.get("reason", ""),
        result.get("tool_result_summary", ""),
        json.dumps(result.get("references", []), ensure_ascii=False),
        json.dumps(result.get("tool_trace", []), ensure_ascii=False),
    ]
    return "\n".join(str(x) for x in parts if x is not None)


def check_keywords(case: dict, result: dict) -> dict:
    expected_keywords = case.get("expected_keywords", []) or []

    if not expected_keywords:
        return {
            "keyword_pass": True,
            "keyword_hits": [],
            "keyword_missing": [],
            "keyword_hit_count": 0,
            "keyword_required": 0,
        }

    search_text = safe_lower(build_search_text(result))

    hits = []
    missing = []

    for kw in expected_keywords:
        kw_text = safe_lower(kw)
        if kw_text and kw_text in search_text:
            hits.append(kw)
        else:
            missing.append(kw)

    # 默认要求命中 60% 的关键词。
    # 如果某个 case 里写了 min_keyword_hits，就按 case 自己的要求来。
    default_required = max(1, math.ceil(len(expected_keywords) * 0.6))
    required = case.get("min_keyword_hits", default_required)

    keyword_pass = len(hits) >= required

    return {
        "keyword_pass": keyword_pass,
        "keyword_hits": hits,
        "keyword_missing": missing,
        "keyword_hit_count": len(hits),
        "keyword_required": required,
    }


def extract_reference_sources(result: dict) -> list[str]:
    references = result.get("references", []) or []

    sources = []

    for ref in references:
        if not isinstance(ref, dict):
            continue

        for key in ["source_file", "file_name", "source_path"]:
            value = ref.get(key)
            if value:
                sources.append(str(value))

    return sources


def check_sources(case: dict, result: dict) -> dict:
    expected_sources = case.get("expected_sources", []) or []

    if not expected_sources:
        return {
            "source_pass": True,
            "source_hits": [],
            "source_missing": [],
        }

    actual_sources = extract_reference_sources(result)
    actual_text = safe_lower(" ".join(actual_sources))

    hits = []
    missing = []

    for src in expected_sources:
        if safe_lower(src) in actual_text:
            hits.append(src)
        else:
            missing.append(src)

    # 这里不要求所有来源都命中。
    # 因为有些问题可能一个 chunk 就足够回答。
    source_pass = len(hits) >= 1

    return {
        "source_pass": source_pass,
        "source_hits": hits,
        "source_missing": missing,
    }


# =========================
# 7. 单条 case 判分
# =========================
def evaluate_one_case(case: dict, top_k: int) -> dict:
    case_id = case.get("id", "")
    question = case.get("question", "")

    started_at = time.time()

    try:
        result = run_agent_once(
            write_bad_cases=write_bad_cases,
            question=question,
            top_k=top_k,
            decompose_top_k=Settings.DECOMPOSE_TOP_K,
        )
        elapsed_ms = round((time.time() - started_at) * 1000, 2)

        actual = extract_actual_fields(result)

        expected_decision = case.get("expected_decision", "")
        expected_action = case.get("expected_action", "")
        expected_tool = case.get("expected_tool", "")
        expected_status = case.get("expected_status", "")

        decision_pass = (
            not expected_decision
            or actual["actual_decision"] == expected_decision
        )

        action_pass = (
            not expected_action
            or actual["actual_action"] == expected_action
        )

        tool_pass = (
            not expected_tool
            or actual["actual_tool"] == expected_tool
        )

        status_pass = (
            not expected_status
            or actual["actual_status"] == expected_status
        )

        keyword_check = check_keywords(case, result)
        source_check = check_sources(case, result)

        passed = all([
            decision_pass,
            action_pass,
            tool_pass,
            status_pass,
            keyword_check["keyword_pass"],
            source_check["source_pass"],
        ])

        error_type, error_reason = diagnose_error(
            decision_pass=decision_pass,
            action_pass=action_pass,
            tool_pass=tool_pass,
            status_pass=status_pass,
            keyword_pass=keyword_check["keyword_pass"],
            source_pass=source_check["source_pass"],
            expected_decision=expected_decision,
            expected_action=expected_action,
            expected_tool=expected_tool,
            expected_status=expected_status,
            actual=actual,
            keyword_check=keyword_check,
            source_check=source_check,
        )

        return {
            "id": case_id,
            "category": case.get("category", ""),
            "question": question,

            "passed": passed,
            "error_type": error_type,
            "error_reason": error_reason,

            "expected_decision": expected_decision,
            "actual_decision": actual["actual_decision"],

            "expected_action": expected_action,
            "actual_action": actual["actual_action"],

            "expected_tool": expected_tool,
            "actual_tool": actual["actual_tool"],

            "expected_status": expected_status,
            "actual_status": actual["actual_status"],

            "keyword_pass": keyword_check["keyword_pass"],
            "keyword_hits": keyword_check["keyword_hits"],
            "keyword_missing": keyword_check["keyword_missing"],
            "keyword_hit_count": keyword_check["keyword_hit_count"],
            "keyword_required": keyword_check["keyword_required"],

            "source_pass": source_check["source_pass"],
            "source_hits": source_check["source_hits"],
            "source_missing": source_check["source_missing"],

            "answer": result.get("answer", ""),
            "reason": result.get("reason", ""),
            "tool_result_summary": result.get("tool_result_summary", ""),
            "confidence": result.get("confidence", 0.0),
            "need_clarification": result.get("need_clarification", False),
            "references": result.get("references", []),
            "tool_trace": result.get("tool_trace", []),
            "elapsed_ms": elapsed_ms,
        }

    except Exception as e:
        elapsed_ms = round((time.time() - started_at) * 1000, 2)

        return {
            "id": case_id,
            "category": case.get("category", ""),
            "question": question,

            "passed": False,
            "error_type": "exception",
            "error_reason": str(e),

            "expected_decision": case.get("expected_decision", ""),
            "actual_decision": "",

            "expected_action": case.get("expected_action", ""),
            "actual_action": "",

            "expected_tool": case.get("expected_tool", ""),
            "actual_tool": "",

            "expected_status": case.get("expected_status", ""),
            "actual_status": "",

            "keyword_pass": False,
            "keyword_hits": [],
            "keyword_missing": case.get("expected_keywords", []) or [],
            "keyword_hit_count": 0,
            "keyword_required": 0,

            "source_pass": False,
            "source_hits": [],
            "source_missing": case.get("expected_sources", []) or [],

            "answer": "",
            "reason": "",
            "tool_result_summary": "",
            "confidence": 0.0,
            "need_clarification": False,
            "references": [],
            "tool_trace": [],
            "elapsed_ms": elapsed_ms,
            "traceback": traceback.format_exc(),
        }


def diagnose_error(
    decision_pass: bool,
    action_pass: bool,
    tool_pass: bool,
    status_pass: bool,
    keyword_pass: bool,
    source_pass: bool,
    expected_decision: str,
    expected_action: str,
    expected_tool: str,
    expected_status: str,
    actual: dict,
    keyword_check: dict,
    source_check: dict,
) -> tuple[str, str]:
    if not decision_pass:
        return (
            "decision_error",
            f"期望 decision={expected_decision}，实际 decision={actual['actual_decision']}"
        )

    if not action_pass:
        return (
            "action_error",
            f"期望 action={expected_action}，实际 action={actual['actual_action']}"
        )

    if not tool_pass:
        return (
            "tool_error",
            f"期望 tool={expected_tool}，实际 tool={actual['actual_tool']}"
        )

    if not status_pass:
        return (
            "status_error",
            f"期望 status={expected_status}，实际 status={actual['actual_status']}"
        )

    if not keyword_pass:
        return (
            "keyword_evidence_error",
            f"关键词命中不足：命中 {keyword_check['keyword_hit_count']} 个，要求至少 {keyword_check['keyword_required']} 个；缺失={keyword_check['keyword_missing']}"
        )

    if not source_pass:
        return (
            "source_evidence_error",
            f"引用来源未命中：缺失={source_check['source_missing']}"
        )

    return "ok", ""


# =========================
# 8. 汇总指标
# =========================
def calc_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])

    decision_ok = sum(
        1 for r in rows
        if not r["expected_decision"] or r["actual_decision"] == r["expected_decision"]
    )

    tool_cases = [r for r in rows if r["category"] == "tool_use"]
    tool_ok = sum(
        1 for r in tool_cases
        if r["actual_decision"] == "use_tool"
        and r["actual_tool"] == r["expected_tool"]
        and r["actual_status"] == "OK"
    )

    refuse_cases = [r for r in rows if r["category"] == "refuse"]
    refuse_ok = sum(
        1 for r in refuse_cases
        if r["actual_decision"] == "refuse"
        and r["actual_action"] == "refuse"
        and r["actual_status"] == "OUT_OF_SCOPE"
    )

    direct_cases = [r for r in rows if r["category"] == "direct"]
    direct_ok = sum(
        1 for r in direct_cases
        if r["actual_decision"] == "use_knowledge_base"
        and r["actual_action"] == "search_knowledge_base"
        and r["actual_status"] == "OK"
        and r["keyword_pass"]
        and r["source_pass"]
    )

    def rate(n: int, d: int) -> float:
        if d == 0:
            return 0.0
        return round(n / d, 4)

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": rate(passed, total),

        "decision_accuracy": rate(decision_ok, total),

        "tool_case_count": len(tool_cases),
        "tool_success_count": tool_ok,
        "tool_success_rate": rate(tool_ok, len(tool_cases)),

        "refuse_case_count": len(refuse_cases),
        "refuse_success_count": refuse_ok,
        "refuse_success_rate": rate(refuse_ok, len(refuse_cases)),

        "direct_case_count": len(direct_cases),
        "direct_success_count": direct_ok,
        "direct_success_rate": rate(direct_ok, len(direct_cases)),
    }


# =========================
# 9. 输出 CSV
# =========================
def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "id",
        "category",
        "passed",
        "error_type",
        "error_reason",

        "expected_decision",
        "actual_decision",
        "expected_action",
        "actual_action",
        "expected_tool",
        "actual_tool",
        "expected_status",
        "actual_status",

        "keyword_hit_count",
        "keyword_required",
        "keyword_hits",
        "keyword_missing",
        "source_hits",
        "source_missing",

        "confidence",
        "need_clarification",
        "elapsed_ms",

        "question",
        "answer",
        "reason",
        "tool_result_summary",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow({
                "id": r.get("id", ""),
                "category": r.get("category", ""),
                "passed": r.get("passed", False),
                "error_type": r.get("error_type", ""),
                "error_reason": shorten(r.get("error_reason", ""), 300),

                "expected_decision": r.get("expected_decision", ""),
                "actual_decision": r.get("actual_decision", ""),
                "expected_action": r.get("expected_action", ""),
                "actual_action": r.get("actual_action", ""),
                "expected_tool": r.get("expected_tool", ""),
                "actual_tool": r.get("actual_tool", ""),
                "expected_status": r.get("expected_status", ""),
                "actual_status": r.get("actual_status", ""),

                "keyword_hit_count": r.get("keyword_hit_count", 0),
                "keyword_required": r.get("keyword_required", 0),
                "keyword_hits": " | ".join(map(str, r.get("keyword_hits", []))),
                "keyword_missing": " | ".join(map(str, r.get("keyword_missing", []))),
                "source_hits": " | ".join(map(str, r.get("source_hits", []))),
                "source_missing": " | ".join(map(str, r.get("source_missing", []))),

                "confidence": r.get("confidence", 0.0),
                "need_clarification": r.get("need_clarification", False),
                "elapsed_ms": r.get("elapsed_ms", 0),

                "question": r.get("question", ""),
                "answer": shorten(r.get("answer", ""), 500),
                "reason": shorten(r.get("reason", ""), 300),
                "tool_result_summary": shorten(r.get("tool_result_summary", ""), 300),
            })


# =========================
# 10. 控制台打印
# =========================
def print_summary(metrics: dict, rows: list[dict]) -> None:
    print("\n==============================")
    print("Agent 评测结果")
    print("==============================")

    print(f"总数：{metrics['total']}")
    print(f"通过：{metrics['passed']}")
    print(f"失败：{metrics['failed']}")
    print(f"通过率：{metrics['pass_rate']}")

    print("\n核心指标：")
    print(f"- 决策正确率：{metrics['decision_accuracy']}")
    print(f"- 工具调用成功率：{metrics['tool_success_rate']} "
          f"({metrics['tool_success_count']}/{metrics['tool_case_count']})")
    print(f"- 拒答合理率：{metrics['refuse_success_rate']} "
          f"({metrics['refuse_success_count']}/{metrics['refuse_case_count']})")
    print(f"- 知识库问答成功率：{metrics['direct_success_rate']} "
          f"({metrics['direct_success_count']}/{metrics['direct_case_count']})")

    failed_rows = [r for r in rows if not r["passed"]]

    if not failed_rows:
        print("\n没有失败样例。")
        return

    print("\n失败样例：")
    for r in failed_rows:
        print("-" * 80)
        print(f"id：{r['id']}")
        print(f"category：{r['category']}")
        print(f"question：{r['question']}")
        print(f"error_type：{r['error_type']}")
        print(f"error_reason：{r['error_reason']}")
        print(
            f"expected：decision={r['expected_decision']}, "
            f"action={r['expected_action']}, "
            f"tool={r['expected_tool']}, "
            f"status={r['expected_status']}"
        )
        print(
            f"actual：decision={r['actual_decision']}, "
            f"action={r['actual_action']}, "
            f"tool={r['actual_tool']}, "
            f"status={r['actual_status']}"
        )
        print(f"answer：{shorten(r.get('answer', ''), 160)}")


# =========================
# 11. main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="最小 Agent 评测脚本")

    parser.add_argument(
        "--cases",
        type=str,
        default=str(DEFAULT_CASES_PATH),
        help="评测集 JSON 文件路径"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="评测结果输出目录"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="传给 run_agent_once 的 top_k"
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="是否强制重建 RAG 索引"
    )

    args = parser.parse_args()

    cases_path = Path(args.cases)
    output_dir = Path(args.output_dir)

    ensure_output_dir(output_dir)

    cases = read_json(cases_path)

    if not isinstance(cases, list):
        raise ValueError("评测集格式错误：最外层必须是 list")

    print(f"[eval] 读取评测集：{cases_path}")
    print(f"[eval] case 数量：{len(cases)}")

    init_rag_for_eval(force_rebuild=args.force_rebuild)

    rows = []

    for idx, case in enumerate(cases, start=1):
        case_id = case.get("id", f"case_{idx}")
        question = case.get("question", "")

        print(f"\n[eval] ({idx}/{len(cases)}) {case_id}: {question}")

        row = evaluate_one_case(case, top_k=args.top_k)
        rows.append(row)

        print(
            f"[eval] result: passed={row['passed']}, "
            f"error_type={row['error_type']}, "
            f"status={row['actual_status']}, "
            f"decision={row['actual_decision']}, "
            f"action={row['actual_action']}, "
            f"tool={row['actual_tool']}"
        )

    metrics = calc_metrics(rows)

    report = {
        "metrics": metrics,
        "results": rows,
    }

    json_path = output_dir / "agent_eval_results.json"
    csv_path = output_dir / "agent_eval_results.csv"

    write_json(json_path, report)
    write_csv(csv_path, rows)

    print_summary(metrics, rows)

    print("\n结果文件：")
    print(f"- JSON：{json_path}")
    print(f"- CSV ：{csv_path}")


if __name__ == "__main__":
    main()