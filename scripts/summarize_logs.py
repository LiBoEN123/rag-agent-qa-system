from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


SUCCESS_STATUSES = {"success", "ok", "answered"}

REFUSE_STATUSES = {
    "no_relevant_info",
    "refused",
    "reject",
    "insufficient",
}


def load_jsonl(file_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """
    读取 jsonl 文件。

    返回：
    - rows：成功解析出来的 JSON 对象列表
    - bad_lines：解析失败的行
    """
    if not file_path.exists():
        return [], [f"文件不存在：{file_path}"]

    rows: list[dict[str, Any]] = []
    bad_lines: list[str] = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad_lines.append(f"第 {line_no} 行不是合法 JSON：{line[:120]}")
                continue

            if isinstance(item, dict):
                rows.append(item)
            else:
                bad_lines.append(f"第 {line_no} 行 JSON 不是对象：{line[:120]}")

    return rows, bad_lines


def to_bool(value: Any) -> bool:
    """
    将 true/false、1/0、字符串 true/false 等统一转成 bool。
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    return bool(value)


def to_number(value: Any) -> float | None:
    """
    尝试把值转成数字。
    """
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_success(row: dict[str, Any]) -> bool:
    """
    判断 metrics.jsonl 中的一条请求是否成功。
    """
    status = str(row.get("status", "") or "").strip().lower()
    return status in SUCCESS_STATUSES


def is_refused(row: dict[str, Any]) -> bool:
    """
    判断是否属于拒答/未答。

    这里采用广义口径：
    - 没有检索到资料
    - 超出知识库范围
    - 问题为空
    - 问题需要澄清
    - 系统明确说无法根据文档回答
    """
    status = str(row.get("status", "") or "").strip().lower()
    action = str(row.get("action", "") or "").strip().lower()
    reason = str(row.get("reason", "") or "").strip()

    if status in REFUSE_STATUSES:
        return True

    if action in {"refuse", "clarify"}:
        return True

    if "无法根据提供的文档确定答案" in reason:
        return True

    if "问题主题超出当前知识库范围" in reason:
        return True

    if "问题存在明显指代不明" in reason:
        return True

    if "用户未输入问题" in reason:
        return True

    return False


def infer_error_type_from_metric(row: dict[str, Any]) -> str:
    """
    如果 metrics.jsonl 里没有 error_type，就根据 status/action/reason 推断失败类型。

    这个主要是兜底用。
    现在你有 bad_cases.jsonl 后，失败类型排行优先从 bad_cases.jsonl 统计。
    """
    error_type = str(row.get("error_type", "") or "").strip()
    if error_type:
        return error_type

    status = str(row.get("status", "") or "").strip().lower()
    action = str(row.get("action", "") or "").strip().lower()
    reason = str(row.get("reason", "") or "").strip()
    tool_name = str(row.get("tool_name", "") or "").strip()
    question_len = row.get("question_len")

    if status in SUCCESS_STATUSES:
        return ""

    if question_len == 0 or "用户未输入问题" in reason:
        return "empty_question"

    if status == "tool_error":
        if tool_name:
            return f"tool_error:{tool_name}"
        return "tool_error"

    if status == "internal_error":
        return "internal_error"

    if action == "clarify" or "指代不明" in reason or "缺少关键限定" in reason:
        return "ambiguous_error"

    if action == "refuse" or "超出当前知识库范围" in reason:
        return "out_of_scope_error"

    if (
        "检索结果为空" in reason
        or "top1 分数低于阈值" in reason
        or "证据分数低于阈值" in reason
    ):
        return "retrieval_error"

    if "引用" in reason or "支撑" in reason:
        return "bad_reference_error"

    if status:
        return status

    return "unknown_error"


def counter_to_markdown_table(counter: Counter, name_col: str, count_col: str, top_n: int) -> str:
    """
    Counter 转 Markdown 表格。
    """
    if not counter:
        return "无\n"

    lines = [
        f"| {name_col} | {count_col} |",
        "|---|---:|",
    ]

    for name, count in counter.most_common(top_n):
        lines.append(f"| {name} | {count} |")

    return "\n".join(lines) + "\n"


def build_summary(
    metric_rows: list[dict[str, Any]],
    bad_case_rows: list[dict[str, Any]],
    metric_bad_lines: list[str],
    bad_case_bad_lines: list[str],
    top_n: int,
) -> dict[str, Any]:
    """
    汇总 metrics.jsonl 和 bad_cases.jsonl。
    """
    total_requests = len(metric_rows)
    success_count = sum(1 for row in metric_rows if is_success(row))
    failure_count = total_requests - success_count

    latency_values = [
        value
        for row in metric_rows
        if (value := to_number(row.get("latency_ms"))) is not None
    ]

    avg_latency_ms = mean(latency_values) if latency_values else 0.0
    max_latency_ms = max(latency_values) if latency_values else 0.0
    min_latency_ms = min(latency_values) if latency_values else 0.0

    refuse_count = sum(1 for row in metric_rows if is_refused(row))

    tool_call_count = sum(
        1
        for row in metric_rows
        if to_bool(row.get("tool_called"))
    )

    tool_success_count = sum(
        1
        for row in metric_rows
        if to_bool(row.get("tool_called")) and to_bool(row.get("tool_success"))
    )

    tool_failure_count = sum(
        1
        for row in metric_rows
        if to_bool(row.get("tool_called")) and not to_bool(row.get("tool_success"))
    )

    status_counter = Counter(
        str(row.get("status", "") or "unknown")
        for row in metric_rows
    )

    action_counter = Counter(
        str(row.get("action", "") or "unknown")
        for row in metric_rows
    )

    tool_counter = Counter(
        str(row.get("tool_name", "") or "no_tool")
        for row in metric_rows
        if to_bool(row.get("tool_called"))
    )

    # 关键改动：
    # 失败类型排行优先从 bad_cases.jsonl 统计。
    # 因为 bad_cases.jsonl 里已经有明确的 error_type。
    if bad_case_rows:
        error_type_counter = Counter(
            str(row.get("error_type", "") or "unknown_error")
            for row in bad_case_rows
        )
    else:
        failure_metric_rows = [
            row
            for row in metric_rows
            if not is_success(row)
        ]

        error_type_counter = Counter(
            infer_error_type_from_metric(row)
            for row in failure_metric_rows
        )

    slow_requests = sorted(
        metric_rows,
        key=lambda row: to_number(row.get("latency_ms")) or 0,
        reverse=True,
    )[:5]

    return {
        "total_requests": total_requests,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / total_requests if total_requests else 0.0,
        "failure_rate": failure_count / total_requests if total_requests else 0.0,
        "avg_latency_ms": avg_latency_ms,
        "max_latency_ms": max_latency_ms,
        "min_latency_ms": min_latency_ms,
        "refuse_count": refuse_count,
        "tool_call_count": tool_call_count,
        "tool_success_count": tool_success_count,
        "tool_failure_count": tool_failure_count,
        "status_counter": status_counter,
        "action_counter": action_counter,
        "tool_counter": tool_counter,
        "error_type_counter": error_type_counter,
        "slow_requests": slow_requests,
        "bad_case_rows": bad_case_rows,
        "metric_bad_lines": metric_bad_lines,
        "bad_case_bad_lines": bad_case_bad_lines,
        "top_n": top_n,
    }


def render_report(
    summary: dict[str, Any],
    metrics_path: Path,
    bad_cases_path: Path,
) -> str:
    """
    生成 Markdown 报告。
    """
    top_n = summary["top_n"]

    lines: list[str] = []

    lines.append("# RAG/Agent 第九周运行汇总报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 指标日志：`{metrics_path}`")
    lines.append(f"- 失败案例日志：`{bad_cases_path}`")
    lines.append("")

    lines.append("## 1. 总览")
    lines.append("")
    lines.append(f"- 请求总数：{summary['total_requests']}")
    lines.append(f"- 成功次数：{summary['success_count']}")
    lines.append(f"- 失败次数：{summary['failure_count']}")
    lines.append(f"- 成功率：{summary['success_rate']:.2%}")
    lines.append(f"- 失败率：{summary['failure_rate']:.2%}")
    lines.append(f"- 平均耗时：{summary['avg_latency_ms']:.2f} ms")
    lines.append(f"- 最大耗时：{summary['max_latency_ms']:.2f} ms")
    lines.append(f"- 最小耗时：{summary['min_latency_ms']:.2f} ms")
    lines.append(f"- 拒答/未答次数：{summary['refuse_count']}")
    lines.append(f"- 工具调用次数：{summary['tool_call_count']}")
    lines.append(f"- 工具调用成功次数：{summary['tool_success_count']}")
    lines.append(f"- 工具调用失败次数：{summary['tool_failure_count']}")
    lines.append("")

    lines.append("## 2. status 分布")
    lines.append("")
    lines.append(
        counter_to_markdown_table(
            summary["status_counter"],
            "status",
            "count",
            top_n=top_n,
        )
    )

    lines.append("## 3. action 分布")
    lines.append("")
    lines.append(
        counter_to_markdown_table(
            summary["action_counter"],
            "action",
            "count",
            top_n=top_n,
        )
    )

    lines.append("## 4. 工具调用分布")
    lines.append("")
    lines.append(
        counter_to_markdown_table(
            summary["tool_counter"],
            "tool_name",
            "count",
            top_n=top_n,
        )
    )

    lines.append("## 5. error_type 失败类型排行")
    lines.append("")
    lines.append(
        counter_to_markdown_table(
            summary["error_type_counter"],
            "error_type",
            "count",
            top_n=top_n,
        )
    )

    lines.append("## 6. 最慢请求 Top 5")
    lines.append("")

    slow_requests = summary["slow_requests"]

    if not slow_requests:
        lines.append("无")
        lines.append("")
    else:
        lines.append("| trace_id | latency_ms | status | action | tool_name | reason |")
        lines.append("|---|---:|---|---|---|---|")

        for row in slow_requests:
            trace_id = row.get("trace_id", "")
            latency_ms = row.get("latency_ms", "")
            status = row.get("status", "")
            action = row.get("action", "")
            tool_name = row.get("tool_name", "")
            reason = str(row.get("reason", "")).replace("|", "/")

            lines.append(
                f"| {trace_id} | {latency_ms} | {status} | {action} | {tool_name} | {reason} |"
            )

        lines.append("")

    lines.append("## 7. 失败案例")
    lines.append("")

    bad_case_rows = summary["bad_case_rows"]

    if not bad_case_rows:
        lines.append("未读取到 bad_cases.jsonl 中的失败案例。")
        lines.append("")
    else:
        for index, row in enumerate(bad_case_rows[:top_n], start=1):
            question = str(row.get("question", "") or "")
            answer = str(row.get("answer", "") or "")
            note = str(row.get("note", "") or "")
            references = row.get("references", [])

            lines.append(f"### Case {index}")
            lines.append("")
            lines.append(f"- trace_id：`{row.get('trace_id', '')}`")
            lines.append(f"- created_at：`{row.get('created_at', '')}`")
            lines.append(f"- route：`{row.get('route', '')}`")
            lines.append(f"- error_type：`{row.get('error_type', '')}`")
            lines.append(f"- question：{question if question else '空问题'}")
            lines.append(f"- answer：{answer if answer else '空'}")
            lines.append(f"- references_count：{len(references) if isinstance(references, list) else 0}")
            lines.append(f"- note：{note}")
            lines.append("")

    lines.append("## 8. 日志格式问题")
    lines.append("")

    metric_bad_lines = summary["metric_bad_lines"]
    bad_case_bad_lines = summary["bad_case_bad_lines"]

    if not metric_bad_lines and not bad_case_bad_lines:
        lines.append("未发现 JSONL 格式错误。")
        lines.append("")
    else:
        if metric_bad_lines:
            lines.append("### metrics.jsonl 格式问题")
            lines.append("")
            for item in metric_bad_lines[:top_n]:
                lines.append(f"- {item}")
            lines.append("")

        if bad_case_bad_lines:
            lines.append("### bad_cases.jsonl 格式问题")
            lines.append("")
            for item in bad_case_bad_lines[:top_n]:
                lines.append(f"- {item}")
            lines.append("")

    lines.append("## 9. 统计口径说明")
    lines.append("")
    lines.append("- `请求总数`、`成功次数`、`失败次数`、`平均耗时`、`工具调用次数` 主要来自 `metrics.jsonl`。")
    lines.append("- `error_type 失败类型排行` 优先来自 `bad_cases.jsonl`。")
    lines.append("- 如果没有 `bad_cases.jsonl`，脚本会退回到 `metrics.jsonl` 中根据 `status/action/reason` 推断失败类型。")
    lines.append("- `失败案例` 直接展示 `bad_cases.jsonl` 中记录的问题、答案、路由、错误类型和备注。")

    return "\n".join(lines)


def main() -> None:
    """解析日志汇总参数，读取 JSONL 日志并生成 Markdown 报告。"""
    parser = argparse.ArgumentParser(
        description="统计 RAG/Agent 日志，生成第九周运行报告。"
    )

    parser.add_argument(
        "--metrics",
        default="logs/metrics.jsonl",
        help="指标日志路径，默认：logs/metrics.jsonl",
    )

    parser.add_argument(
        "--bad-cases",
        default="logs/bad_cases.jsonl",
        help="失败案例日志路径，默认：logs/bad_cases.jsonl",
    )

    parser.add_argument(
        "--output",
        default="outputs/week9_summary.md",
        help="输出报告路径，默认：outputs/week9_summary.md",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="排行榜和失败案例展示数量，默认：5",
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help="只输出到终端，不写入 Markdown 文件",
    )

    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    bad_cases_path = Path(args.bad_cases)
    output_path = Path(args.output)

    metric_rows, metric_bad_lines = load_jsonl(metrics_path)
    bad_case_rows, bad_case_bad_lines = load_jsonl(bad_cases_path)

    summary = build_summary(
        metric_rows=metric_rows,
        bad_case_rows=bad_case_rows,
        metric_bad_lines=metric_bad_lines,
        bad_case_bad_lines=bad_case_bad_lines,
        top_n=args.top_n,
    )

    report = render_report(
        summary=summary,
        metrics_path=metrics_path,
        bad_cases_path=bad_cases_path,
    )

    print(report)

    if not args.no_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print()
        print(f"[summarize_logs] 报告已生成：{output_path}")


if __name__ == "__main__":
    main()