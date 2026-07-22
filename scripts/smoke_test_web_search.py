"""通过 Registry/ToolRuntime 手动验证真实 DDGS Web Search。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.context import (  # noqa: E402
    ToolInvocationContext,
    build_default_tool_context,
)
from app.tools.registry import tool_registry  # noqa: E402


def _context() -> ToolInvocationContext:
    """从默认主体构造显式、不可变的手动冒烟上下文。"""
    default = build_default_tool_context(source="manual_web_search_smoke")
    return ToolInvocationContext(
        trace_id="manual-web-search-smoke",
        principal_id=default.principal_id,
        principal_type=default.principal_type,
        permissions=default.permissions,
        source="manual_web_search_smoke",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument(
        "--freshness",
        choices=("any", "day", "week", "month", "year"),
        default="any",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    context = _context()
    if "web:search" not in context.permissions:
        print(
            "web_search is not enabled for the default service principal; "
            "add web:search to TOOL_DEFAULT_PERMISSIONS.",
            file=sys.stderr,
        )
        return 2

    started = time.perf_counter()
    try:
        result = tool_registry.dispatch(
            "web_search",
            {
                "query": args.query,
                "max_results": args.max_results,
                "freshness": args.freshness,
            },
            context=context,
        )
    except Exception as exc:
        if args.debug:
            raise
        print(f"web_search failed: {type(exc).__name__}", file=sys.stderr)
        return 1

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print(f"query: {args.query}")
    print(f"provider: {(result.get('output') or {}).get('provider', 'ddgs')}")
    print(f"count: {(result.get('output') or {}).get('count', 0)}")
    for item in (result.get("output") or {}).get("results", []):
        print(f"[{item.get('rank', '')}] {item.get('title', '')}")
        print(f"url: {item.get('url', '')}")
        snippet = str(item.get("snippet", ""))
        print(f"snippet: {snippet[:240]}")
    print(f"elapsed_ms: {elapsed_ms}")
    print(f"runtime_error_code: {result.get('error_code', '')}")
    if result.get("ok"):
        return 0
    print(f"web_search failed: {result.get('error') or 'unknown error'}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
