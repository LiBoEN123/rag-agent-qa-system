"""把不可信 MCP Tool 输出压缩为有限、脱敏的项目数据。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mcp.types import CallToolResult, TextContent

from app.config import AppSettings
from app.tools.errors import ToolErrorCode
from app.tools.redaction import (
    TRUNCATED,
    redact_sensitive_data,
    redact_sensitive_text,
)


@dataclass(frozen=True, slots=True)
class McpSerializationOutcome:
    """MCP 原始结果的安全序列化结论。"""

    ok: bool
    output: dict[str, Any] | None = None
    error: str = ""
    error_code: str = ""


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    """在字符边界截断文本，并保证标记计入限制。"""
    if len(value) <= limit:
        return value, False
    marker = TRUNCATED
    if limit <= len(marker):
        return marker[:limit], True
    return value[: limit - len(marker)] + marker, True


def _bounded_structured(
    value: Any,
    *,
    limit: int,
) -> tuple[Any, bool]:
    """复用现有脱敏器，并逐级收紧为完整 JSON 值而非半截 JSON。"""
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return {
            "truncated": True,
            "unsupported_value": True,
            "message": "MCP structured content is not JSON-compatible",
        }, True

    attempts = (
        (8, 100, 4000),
        (6, 50, 2000),
        (5, 20, 1000),
        (4, 10, 500),
        (3, 5, 200),
    )
    for max_depth, max_items, max_string_chars in attempts:
        sanitized = redact_sensitive_data(
            value,
            max_depth=max_depth,
            max_items=max_items,
            max_string_chars=min(max_string_chars, limit),
            max_serialized_chars=max(limit * 4, limit + 1024),
        )
        serialized = json.dumps(
            sanitized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(serialized) <= limit:
            truncated = len(raw) > limit or TRUNCATED in serialized
            return sanitized, truncated
    summary = {
        "truncated": True,
        "type": type(value).__name__,
        "message": "MCP structured content exceeded the configured limit",
    }
    if len(json.dumps(summary, ensure_ascii=False)) <= limit:
        return summary, True
    return {"truncated": True}, True


def serialize_mcp_call_result(
    result: object,
    *,
    settings: AppSettings,
    server_alias: str,
    remote_tool_name: str,
) -> McpSerializationOutcome:
    """只接受 MCP SDK 结果中的文本和 JSON-compatible structured content。"""
    if not isinstance(result, CallToolResult):
        return McpSerializationOutcome(
            ok=False,
            error="MCP tool returned an invalid result",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )
    if result.isError:
        return McpSerializationOutcome(
            ok=False,
            error="MCP tool execution failed",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )

    text_parts: list[str] = []
    for block in result.content:
        if not isinstance(block, TextContent):
            return McpSerializationOutcome(
                ok=False,
                error="MCP tool returned unsupported content",
                error_code=ToolErrorCode.MCP_UNSUPPORTED_CONTENT.value,
            )
        text_parts.append(block.text)
    text, text_truncated = _truncate_text(
        redact_sensitive_text("\n".join(text_parts)),
        settings.mcp_max_text_chars,
    )

    output: dict[str, Any] = {
        "content": text,
        "content_is_untrusted": True,
        "server": server_alias,
        "tool": remote_tool_name,
        "truncated": text_truncated,
    }
    if result.structuredContent is not None:
        structured, structured_truncated = _bounded_structured(
            result.structuredContent,
            limit=settings.mcp_max_structured_chars,
        )
        output["structured_content"] = structured
        output["truncated"] = bool(text_truncated or structured_truncated)
    return McpSerializationOutcome(ok=True, output=output)


def summarize_mcp_output(value: object) -> dict[str, Any]:
    """生成不含正文的 MCP 审计输出摘要。"""
    if not isinstance(value, dict):
        return {"mode": "summary", "type": type(value).__name__}
    return {
        "mode": "summary",
        "type": "dict",
        "fields": sorted(str(key) for key in value)[:20],
        "content_chars": len(value.get("content", ""))
        if isinstance(value.get("content"), str)
        else 0,
        "truncated": bool(value.get("truncated", False)),
        "content_is_untrusted": bool(value.get("content_is_untrusted", False)),
    }


__all__ = [
    "McpSerializationOutcome",
    "serialize_mcp_call_result",
    "summarize_mcp_output",
]
