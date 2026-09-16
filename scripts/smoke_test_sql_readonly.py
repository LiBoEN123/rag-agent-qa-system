"""受治理只读 SQL 工具的手动冒烟程序。

用途：通过唯一本地执行入口手动验证表清单、有限 Schema 和只读查询。
输入：list/describe/query 子命令，以及当前 SQL_* 和 TOOL_DEFAULT_PERMISSIONS 配置。
输出：不含数据库 URL/凭据的有限 JSON 摘要，并用退出码表达配置或执行结果。
边界：可能连接显式配置的真实业务数据库，仅供人工运行；不会由 Pytest/CI 自动执行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.tools import ToolRuntime  # noqa: E402

from app.agent.framework_tools import build_local_tool  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.tools.context import build_default_tool_context  # noqa: E402
from app.tools.catalog import get_local_tool_specs  # noqa: E402


def _tool_call(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """把 CLI 子命令转换为正式 SQL 工具名和参数。"""
    if args.command == "list":
        return "list_sql_tables", {"schema": args.schema}
    if args.command == "describe":
        return "describe_sql_table", {"table_name": args.table}
    if args.command == "query":
        return "query_sql_readonly", {
            "sql": args.sql,
            "max_rows": args.max_rows,
        }
    raise ValueError(f"unknown command: {args.command}")


def _limited_output(command: str, output: object) -> dict[str, Any]:
    """筛选并限制人工终端输出，不打印 DSN 或无限结果集。"""
    value = output if isinstance(output, dict) else {}
    if command == "list":
        return {
            "dialect": value.get("dialect", ""),
            "schema": value.get("schema", ""),
            "count": value.get("count", 0),
            "tables": list(value.get("tables", []))[:50],
        }
    if command == "describe":
        return {
            "table_name": value.get("table_name", ""),
            "column_count": value.get("column_count", 0),
            "columns": list(value.get("columns", []))[:50],
        }
    rows = value.get("rows", [])
    preview_rows = [
        [str(cell)[:120] if isinstance(cell, str) else cell for cell in row]
        for row in rows[:5]
        if isinstance(row, list)
    ]
    return {
        "columns": list(value.get("columns", []))[:50],
        "row_count": value.get("row_count", 0),
        "rows_preview": preview_rows,
        "truncated": bool(value.get("truncated", False)),
        "dialect": value.get("dialect", ""),
        "execution_ms": value.get("execution_ms", 0),
    }


def run_command(args: argparse.Namespace) -> int:
    """检查显式启用与授权后，经唯一执行入口调用一次工具。"""
    settings = get_settings()
    if not settings.sql_tool_enabled:
        print(
            "SQL tools are disabled. Set SQL_TOOL_ENABLED=true, configure "
            "SQL_DATABASE_URL and SQL_ALLOWED_TABLES, then restart.",
            file=sys.stderr,
        )
        return 2
    context = build_default_tool_context(source="manual_sql_readonly_smoke")
    if "database:read" not in context.permissions:
        print(
            "SQL tools require database:read; add it explicitly to "
            "TOOL_DEFAULT_PERMISSIONS for this smoke test.",
            file=sys.stderr,
        )
        return 2

    tool_name, tool_args = _tool_call(args)
    registry = get_local_tool_specs(get_settings())
    spec = next((item for item in registry if item.name == tool_name), None)
    if spec is None:
        raise LookupError(f"unknown local tool: {tool_name}")
    result = build_local_tool(
        spec,
        settings=settings,
    ).invoke({**tool_args, "runtime": ToolRuntime(
        state={}, context=context, config={}, stream_writer=lambda _: None,
        tool_call_id="manual-smoke", store=None,
    )})
    summary = {
        "tool": tool_name,
        "ok": result.get("ok") is True,
        "status": result.get("status", ""),
        "error_code": result.get("error_code", ""),
        "output": _limited_output(args.command, result.get("output")),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if result.get("ok") is True:
        return 0
    print(
        f"{tool_name} failed: {result.get('error_code') or 'unknown_error'}",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    """构造 list、describe 和 query 三个子命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Run list_sql_tables")
    list_parser.add_argument("--schema")

    describe = subparsers.add_parser("describe", help="Run describe_sql_table")
    describe.add_argument("--table", required=True)

    query = subparsers.add_parser("query", help="Run query_sql_readonly")
    query.add_argument("--sql", required=True)
    query.add_argument("--max-rows", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数并执行一次显式只读 SQL 冒烟调用。"""
    return run_command(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
