"""通过 Registry/ToolRuntime 手动验证真实安全 Web Fetch。"""

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
    default = build_default_tool_context(source="manual_web_fetch_smoke")
    return ToolInvocationContext(
        trace_id="manual-web-fetch-smoke",
        principal_id=default.principal_id,
        principal_type=default.principal_type,
        permissions=default.permissions,
        source="manual_web_fetch_smoke",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    context = _context()
    if "web:fetch" not in context.permissions:
        print(
            "web_fetch is not enabled for the default service principal; "
            "add web:fetch to TOOL_DEFAULT_PERMISSIONS.",
            file=sys.stderr,
        )
        return 2

    started = time.perf_counter()
    try:
        result = tool_registry.dispatch(
            "web_fetch",
            {"url": args.url, "max_chars": args.max_chars},
            context=context,
        )
    except Exception as exc:
        if args.debug:
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(f"elapsed_ms: {elapsed_ms}")
        print("runtime_error_code: execution_error")
        print(f"web_fetch failed: {type(exc).__name__}", file=sys.stderr)
        return 1

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    output = result.get("output") if isinstance(result.get("output"), dict) else {}
    print(f"requested_url: {output.get('requested_url', args.url)}")
    print(f"final_url: {output.get('final_url', '')}")
    print(f"status_code: {output.get('status_code', '')}")
    print(f"content_type: {output.get('content_type', '')}")
    print(f"title: {output.get('title', '')}")
    print(f"content_length: {output.get('content_length', 0)}")
    print(f"truncated: {bool(output.get('truncated', False))}")
    preview = str(output.get("content", ""))[:500]
    print(f"content_preview: {preview}")
    print(f"runtime_error_code: {result.get('error_code', '')}")
    print(f"elapsed_ms: {elapsed_ms}")
    if result.get("ok"):
        return 0
    print(
        f"web_fetch failed: {result.get('error') or 'unknown error'}", file=sys.stderr
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
