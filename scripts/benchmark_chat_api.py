"""``/chat`` HTTP 接口的并发性能评测程序。

用途：在指定并发级别下重复调用已启动的聊天接口，观察吞吐、延迟与失败分布。
输入：服务地址、性能用例 JSON、并发级别、重复次数、预热次数和请求超时。
输出：逐请求 JSONL、机器可读汇总 JSON，以及便于人工阅读的 Markdown 报告。
边界：程序只通过 HTTP 调用现有服务，不负责启动 API、模型或 RAG；预热请求不计入结果。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx


DEFAULT_CASES = Path("data/eval/performance_cases.json")
DEFAULT_OUTPUT_DIR = Path("outputs/performance/latest")
REQUEST_FIELDS = (
    "trace_id",
    "concurrency",
    "round_index",
    "case_id",
    "question",
    "latency_ms",
    "http_status",
    "business_code",
    "status",
    "success",
    "answer_length",
    "reference_count",
    "error_type",
    "error_message",
    "started_at",
    "finished_at",
)


def utc_now() -> str:
    """返回精确到毫秒的稳定 ISO-8601 UTC 时间戳。"""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def parse_concurrency(value: str) -> list[int]:
    """解析逗号分隔的正整数并发级别，并按输入顺序去重。"""
    try:
        levels = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("concurrency 必须是逗号分隔的整数") from exc
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("concurrency 必须全部大于 0")
    return list(dict.fromkeys(levels))


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读取并校验性能用例，避免在程序内临时编造测试问题。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到性能测试 cases：{path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(values, list) or not values:
        raise ValueError("性能测试 cases 必须是非空 list 或包含 cases 的 object")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"第 {index} 个性能 case 必须是 object")
        case_id = str(value.get("id") or "").strip()
        question = str(value.get("question") or "").strip()
        if not case_id or not question:
            raise ValueError(f"第 {index} 个性能 case 缺少 id 或 question")
        if case_id in seen:
            raise ValueError(f"性能 case id 重复：{case_id}")
        seen.add(case_id)
        cases.append({"id": case_id, "question": question})
    return cases


def percentile(values: list[float], percent: float) -> float:
    """用确定性的线性插值计算指定百分位数。"""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    weight = position - lower
    return round(
        ordered[lower] * (1 - weight) + ordered[upper] * weight,
        2,
    )


def _business_success(code: object) -> bool:
    """兼容数字和字符串形式的成功业务码。"""
    return code in (0, "0")


async def send_request(
    client: httpx.AsyncClient,
    *,
    chat_url: str,
    case: dict[str, Any],
    concurrency: int,
    round_index: int,
    top_k: int,
) -> dict[str, Any]:
    """发送单次请求，并统一传输层、HTTP 层和业务层失败分类。"""
    trace_id = f"benchmark-{uuid4()}"
    started_at = utc_now()
    started = time.perf_counter()
    http_status: int | None = None
    business_code: int | str | None = None
    status = ""
    answer = ""
    references: list[Any] = []
    error_type = ""
    error_message = ""
    try:
        response = await client.post(
            chat_url,
            json={"question": case["question"], "top_k": top_k},
            headers={"X-Trace-Id": trace_id},
        )
        http_status = response.status_code
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            payload = {}
            error_type = "invalid_json"
            error_message = "response body is not a JSON object"
        # 只提取稳定响应合同中的字段，额外字段不进入性能结果。
        if isinstance(payload, dict):
            business_code = payload.get("code")
            data = payload.get("data")
            if isinstance(data, dict):
                status = str(data.get("status") or "")
                answer = str(data.get("answer") or "")
                raw_references = data.get("references")
                if isinstance(raw_references, list):
                    references = raw_references
        else:
            error_type = "invalid_json"
            error_message = "response JSON must be an object"
        if not error_type and http_status != 200:
            error_type = "http_error"
            error_message = f"HTTP {http_status}"
        elif not error_type and not _business_success(business_code):
            error_type = "business_error"
            error_message = f"business code={business_code!r}"
        elif not error_type and not answer:
            error_type = "empty_answer"
            error_message = "response answer is empty"
    except httpx.TimeoutException as exc:
        error_type = "timeout"
        error_message = str(exc) or type(exc).__name__
    except httpx.HTTPError as exc:
        error_type = "network_error"
        error_message = str(exc) or type(exc).__name__
    except Exception as exc:  # pragma: no cover - CLI 边界的兜底保护
        error_type = "unexpected_error"
        error_message = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "trace_id": trace_id,
        "concurrency": concurrency,
        "round_index": round_index,
        "case_id": case["id"],
        "question": case["question"],
        "latency_ms": latency_ms,
        "http_status": http_status,
        "business_code": business_code,
        "status": status,
        "success": not error_type,
        "answer_length": len(answer),
        "reference_count": len(references),
        "error_type": error_type,
        "error_message": error_message[:500],
        "started_at": started_at,
        "finished_at": utc_now(),
    }


async def run_concurrency_level(
    client: httpx.AsyncClient,
    *,
    chat_url: str,
    cases: list[dict[str, Any]],
    concurrency: int,
    repeats: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], float]:
    """用信号量限制并发，执行全部用例与重复轮次组合。"""
    # 任务可以一次创建，但同时在途的 HTTP 请求不会超过当前并发级别。
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(case: dict[str, Any], round_index: int) -> dict[str, Any]:
        """在当前并发配额内发送一条性能评测请求。"""
        async with semaphore:
            return await send_request(
                client,
                chat_url=chat_url,
                case=case,
                concurrency=concurrency,
                round_index=round_index,
                top_k=top_k,
            )

    started = time.perf_counter()
    rows = await asyncio.gather(
        *(
            bounded(case, round_index)
            for round_index in range(1, repeats + 1)
            for case in cases
        )
    )
    return list(rows), time.perf_counter() - started


def summarize_rows(
    rows: list[dict[str, Any]],
    duration_seconds: float,
) -> dict[str, Any]:
    """汇总成功率、吞吐、延迟、响应规模和错误分布。"""
    latencies = [float(row["latency_ms"]) for row in rows]
    successes = [row for row in rows if row["success"]]
    request_count = len(rows)
    success_count = len(successes)
    return {
        "request_count": request_count,
        "success_count": success_count,
        "failure_count": request_count - success_count,
        "success_rate": round(success_count / request_count, 6)
        if request_count
        else 0.0,
        "throughput_rps": round(request_count / duration_seconds, 4)
        if duration_seconds > 0
        else 0.0,
        "mean_latency_ms": round(statistics.mean(latencies), 2)
        if latencies
        else 0.0,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "min_latency_ms": round(min(latencies), 2) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        "mean_answer_length": round(
            statistics.mean(row["answer_length"] for row in rows),
            2,
        )
        if rows
        else 0.0,
        "mean_reference_count": round(
            statistics.mean(row["reference_count"] for row in rows),
            2,
        )
        if rows
        else 0.0,
        "http_status_distribution": dict(
            sorted(Counter(str(row["http_status"]) for row in rows).items())
        ),
        "business_status_distribution": dict(
            sorted(Counter(row["status"] or "<empty>" for row in rows).items())
        ),
        "error_type_distribution": dict(
            sorted(
                Counter(
                    row["error_type"] or "<none>" for row in rows
                ).items()
            )
        ),
    }


async def run_benchmark(
    *,
    base_url: str,
    cases: list[dict[str, Any]],
    concurrency_levels: list[int],
    repeats: int,
    warmup: int,
    timeout: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """复用一个异步客户端完成预热和各级并发的正式测量。"""
    chat_url = f"{base_url.rstrip('/')}/chat"
    limits = httpx.Limits(
        max_connections=max(concurrency_levels),
        max_keepalive_connections=max(concurrency_levels),
    )
    rows: list[dict[str, Any]] = []
    by_concurrency: dict[str, dict[str, Any]] = {}
    measured_seconds = 0.0
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # 预热只建立连接并激活服务缓存，不写入正式逐请求结果。
        for index in range(warmup):
            await send_request(
                client,
                chat_url=chat_url,
                case=cases[index % len(cases)],
                concurrency=0,
                round_index=index + 1,
                top_k=top_k,
            )
        for concurrency in concurrency_levels:
            level_rows, duration = await run_concurrency_level(
                client,
                chat_url=chat_url,
                cases=cases,
                concurrency=concurrency,
                repeats=repeats,
                top_k=top_k,
            )
            rows.extend(level_rows)
            measured_seconds += duration
            by_concurrency[str(concurrency)] = summarize_rows(
                level_rows,
                duration,
            )
    summary = summarize_rows(rows, measured_seconds)
    summary.update(
        {
            "schema_version": 1,
            "base_url": base_url.rstrip("/"),
            "concurrency_levels": concurrency_levels,
            "repeats": repeats,
            "warmup": warmup,
            "timeout_seconds": timeout,
            "top_k": top_k,
            "by_concurrency": by_concurrency,
        }
    )
    return rows, summary


def render_summary_markdown(summary: dict[str, Any]) -> str:
    """把机器汇总渲染为紧凑的 Markdown 性能报告。"""
    lines = [
        "# Chat API Benchmark",
        "",
        f"- Requests: {summary['request_count']}",
        f"- Success rate: {summary['success_rate']:.2%}",
        f"- Throughput: {summary['throughput_rps']:.4f} req/s",
        f"- Mean latency: {summary['mean_latency_ms']:.2f} ms",
        f"- P50 / P95 / P99: {summary['p50_latency_ms']:.2f} / "
        f"{summary['p95_latency_ms']:.2f} / {summary['p99_latency_ms']:.2f} ms",
        "",
        "## By concurrency",
        "",
        "| Concurrency | Requests | Success rate | Throughput | Mean | P95 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for concurrency, values in summary["by_concurrency"].items():
        lines.append(
            f"| {concurrency} | {values['request_count']} | "
            f"{values['success_rate']:.2%} | {values['throughput_rps']:.4f} | "
            f"{values['mean_latency_ms']:.2f} | {values['p95_latency_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Distributions",
            "",
            f"- HTTP: `{json.dumps(summary['http_status_distribution'], ensure_ascii=False)}`",
            f"- Business status: `{json.dumps(summary['business_status_distribution'], ensure_ascii=False)}`",
            f"- Errors: `{json.dumps(summary['error_type_distribution'], ensure_ascii=False)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """写入正式支持的逐请求、汇总 JSON 和 Markdown 三类产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {field: row[field] for field in REQUEST_FIELDS},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """构造性能评测命令行参数解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--concurrency",
        type=parse_concurrency,
        default=parse_concurrency("1,2,3,5"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """校验参数、运行评测并用退出码表达是否存在失败请求。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("repeats/timeout 必须大于 0，warmup 不能为负数")
    if not 1 <= args.top_k <= 10:
        parser.error("top-k 必须在 1 到 10 之间")
    cases = load_cases(args.cases)
    rows, summary = asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            cases=cases,
            concurrency_levels=args.concurrency,
            repeats=args.repeats,
            warmup=args.warmup,
            timeout=args.timeout,
            top_k=args.top_k,
        )
    )
    write_outputs(args.output_dir, rows, summary)
    print(f"Benchmark written to: {args.output_dir}")
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
