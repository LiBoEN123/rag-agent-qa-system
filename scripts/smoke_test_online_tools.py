"""受治理 Online Tools 的手动冒烟验证程序。

用途：通过正式 ToolRegistry/ToolRuntime 分别验证网页搜索、网页抓取和 PDF 抓取。
输入：search/fetch/pdf 子命令、目标查询或 URL，以及当前工具权限和网络配置。
输出：脱敏且限制长度的 JSON 摘要，并用退出码区分成功、执行失败和权限不足。
边界：会访问真实外部网络，仅供人工显式运行；不会由 Pytest 或 CI 自动触发。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.context import (  # noqa: E402
    ToolInvocationContext,
    build_default_tool_context,
)
from app.tools.registry import ToolRegistry, get_tool_registry  # noqa: E402


PERMISSIONS = {
    "web_search": "web:search",
    "web_fetch": "web:fetch",
    "web_pdf_fetch": "web:fetch",
}


def build_context(tool_name: str) -> ToolInvocationContext:
    """为受治理分发构造显式的默认主体调用上下文。"""
    source = "manual_online_tools_smoke"
    default = build_default_tool_context(source=source)
    return ToolInvocationContext(
        trace_id=f"manual-{tool_name.replace('_', '-')}-smoke",
        principal_id=default.principal_id,
        principal_type=default.principal_type,
        permissions=default.permissions,
        source=source,
    )


def safe_url_preview(value: object) -> str:
    """生成不含凭据、查询参数和片段的限长 URL 预览。"""
    if not isinstance(value, str):
        return ""
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        port = f":{parsed.port}" if parsed.port is not None else ""
        return urlunsplit(
            (parsed.scheme, f"{parsed.hostname}{port}", parsed.path, "", "")
        )[:240]
    except ValueError:
        return ""


def _command_spec(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """把 CLI 子命令转换为正式工具名和结构化参数。"""
    if args.command == "search":
        return "web_search", {
            "query": args.query,
            "max_results": args.max_results,
            "freshness": args.freshness,
        }
    if args.command == "fetch":
        return "web_fetch", {"url": args.url, "max_chars": args.max_chars}
    if args.command == "pdf":
        return "web_pdf_fetch", {
            "url": args.url,
            "start_page": args.start_page,
            "max_pages": args.max_pages,
            "max_chars": args.max_chars,
        }
    raise ValueError(f"unknown command: {args.command}")


def _print_result(
    command: str,
    result: dict[str, Any],
    elapsed_ms: int,
) -> None:
    """按工具类型筛选安全字段并打印统一 JSON 结果。"""
    output = result.get("output")
    output = output if isinstance(output, dict) else {}
    common = {
        "tool": result.get("tool_name", ""),
        "ok": result.get("ok") is True,
        "status": result.get("status", ""),
        "error_code": result.get("error_code", ""),
        "elapsed_ms": elapsed_ms,
    }
    if command == "search":
        details = {
            "provider": output.get("provider", ""),
            "count": output.get("count", 0),
            "results": [
                {
                    "rank": item.get("rank", ""),
                    "title": item.get("title", ""),
                    "url": safe_url_preview(item.get("url")),
                    "snippet": str(item.get("snippet", ""))[:240],
                }
                for item in output.get("results", [])
                if isinstance(item, dict)
            ],
        }
    elif command == "fetch":
        details = {
            "final_url": safe_url_preview(output.get("final_url")),
            "status_code": output.get("status_code", ""),
            "content_type": output.get("content_type", ""),
            "title": output.get("title", ""),
            "content_length": output.get("content_length", 0),
            "truncated": bool(output.get("truncated", False)),
            "content_preview": str(output.get("content", ""))[:500],
        }
    else:
        details = {
            "final_url": safe_url_preview(output.get("final_url")),
            "status_code": output.get("status_code", ""),
            "content_type": output.get("content_type", ""),
            "total_pages": output.get("total_pages", 0),
            "page_numbers": output.get("page_numbers", []),
            "content_length": output.get("content_length", 0),
            "truncated": bool(output.get("truncated", False)),
            "content_preview": str(output.get("content", ""))[:500],
        }
    print(json.dumps({**common, **details}, ensure_ascii=False, indent=2))


def run_command(
    args: argparse.Namespace,
    *,
    registry: ToolRegistry | None = None,
) -> int:
    """检查权限、经 Runtime 分发真实工具，并统一退出码。"""
    tool_name, tool_args = _command_spec(args)
    context = build_context(tool_name)
    required_permission = PERMISSIONS[tool_name]
    if required_permission not in context.permissions:
        print(
            f"{tool_name} requires {required_permission}; update "
            "TOOL_DEFAULT_PERMISSIONS for this explicit smoke test.",
            file=sys.stderr,
        )
        return 2
    started = time.perf_counter()
    try:
        # 这里刻意走正式注册表边界，确保权限、超时和错误合同都参与验证。
        result = (registry or get_tool_registry()).dispatch(
            tool_name,
            tool_args,
            context=context,
        )
    except Exception as exc:
        if args.debug:
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(f"elapsed_ms: {elapsed_ms}")
        print("runtime_error_code: execution_error")
        print(f"{tool_name} failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    _print_result(args.command, result, elapsed_ms)
    if result.get("ok") is True:
        return 0
    print(
        f"{tool_name} failed: "
        f"{result.get('error_code') or 'unknown_error'}",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    """构造三个 Online Tool 子命令及其参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Run governed web_search")
    search.add_argument("--query", required=True)
    search.add_argument("--max-results", type=int, default=3)
    search.add_argument(
        "--freshness",
        choices=("any", "day", "week", "month", "year"),
        default="any",
    )
    search.add_argument("--debug", action="store_true")

    fetch = subparsers.add_parser("fetch", help="Run governed web_fetch")
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--max-chars", type=int, default=4000)
    fetch.add_argument("--debug", action="store_true")

    pdf = subparsers.add_parser("pdf", help="Run governed web_pdf_fetch")
    pdf.add_argument("--url", required=True)
    pdf.add_argument("--start-page", type=int, default=1)
    pdf.add_argument("--max-pages", type=int, default=3)
    pdf.add_argument("--max-chars", type=int, default=8000)
    pdf.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数并运行一次显式的在线工具冒烟验证。"""
    return run_command(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
