"""通过项目正式 Provider/bridge 列出或调用受治理的 MCP Tool。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.tools.context import build_default_tool_context  # noqa: E402
from app.tools.mcp.bridge import invoke_mcp_tool  # noqa: E402
from app.tools.mcp.client import (  # noqa: E402
    close_mcp_runtime,
    initialize_mcp_runtime,
)
from app.tools.mcp.provider import get_mcp_tool_provider  # noqa: E402
from app.common.redaction import redact_sensitive_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构造列举或调用受治理 MCP Tool 的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="列出或调用当前部署配置允许的受治理 MCP read-only tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="列出当前权限可见的 MCP tools")
    invoke = subparsers.add_parser("invoke", help="调用一个当前权限可见的 MCP tool")
    invoke.add_argument("--tool", required=True, help="带 Server 前缀的工具名")
    invoke.add_argument(
        "--args-json",
        required=True,
        help="JSON object 形式的工具参数",
    )
    return parser


def _safe_result(value: dict[str, Any]) -> dict[str, Any]:
    """只打印有限结果字段，不回显原始参数、Header 或 Client 信息。"""
    runtime = value.get("runtime") if isinstance(value.get("runtime"), dict) else {}
    output = value.get("output")
    return {
        "status": value.get("status", ""),
        "ok": value.get("ok") is True,
        "tool_name": value.get("tool_name", ""),
        "output": redact_sensitive_data(
            output,
            max_depth=6,
            max_items=30,
            max_string_chars=2000,
            max_serialized_chars=20_000,
        ),
        "error": value.get("error", ""),
        "error_code": value.get("error_code", ""),
        "runtime": {
            "elapsed_ms": runtime.get("elapsed_ms", 0),
            "timed_out": bool(runtime.get("timed_out", False)),
        },
    }


async def _run(args: argparse.Namespace) -> int:
    """初始化正式 MCP Runtime，并按命令安全列举或调用可见工具。"""
    settings = get_settings()
    if not settings.mcp_tools_enabled:
        print("MCP tools are disabled")
        return 0
    try:
        await initialize_mcp_runtime(settings)
    except Exception as exc:
        print(f"MCP initialization failed: {type(exc).__name__}", file=sys.stderr)
        return 2

    try:
        provider = get_mcp_tool_provider()
        if provider is None:
            print("MCP runtime is unavailable", file=sys.stderr)
            return 2
        context = build_default_tool_context(
            source="mcp_smoke_test",
            settings=settings,
        )
        visible = {tool.name: tool for tool in provider.list_tools(context=context)}
        if args.command == "list":
            rows = [
                row
                for row in provider.catalog()
                if row["tool_name"] in visible
            ]
            print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        try:
            tool_args = json.loads(args.args_json)
        except json.JSONDecodeError:
            print("--args-json must be valid JSON", file=sys.stderr)
            return 2
        if not isinstance(tool_args, dict):
            print("--args-json must decode to a JSON object", file=sys.stderr)
            return 2
        tool = visible.get(args.tool)
        if tool is None:
            print("Configured MCP tool is unavailable", file=sys.stderr)
            return 2
        result = await invoke_mcp_tool(tool, tool_args, context=context)
        print(json.dumps(_safe_result(result), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("ok") is True else 1
    finally:
        await close_mcp_runtime()


def main(argv: list[str] | None = None) -> int:
    """解析命令行参数并在独立事件循环中运行冒烟流程。"""
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
