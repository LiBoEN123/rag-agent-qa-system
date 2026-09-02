"""把不可信 MCP Tool 输出压缩为有限、脱敏的项目数据。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from mcp import types

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


_MAX_ERROR_TEXT_CHARS = 2_000


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


def _safe_resource_uri(value: object) -> str:
    """移除资源 URI 的 userinfo、query 和 fragment，避免凭据进入模型上下文。"""
    try:
        parsed = urlsplit(str(value))
        port = parsed.port
    except ValueError:
        return "<invalid-resource-uri>"
    hostname = parsed.hostname or ""
    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = f"{hostname}:{port}" if port is not None else hostname
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _content_summary(block_type: str, **metadata: object) -> str:
    """把非文本 block 压缩为不含 payload 的 JSON 摘要。"""
    payload = {"type": block_type}
    payload.update({key: value for key, value in metadata.items() if value is not None})
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _serialize_content_block(block: object) -> str | None:  # noqa: PLR0911
    """依据官方 MCP SDK 类型提取文本，或生成不含二进制的安全摘要。"""
    if isinstance(block, types.TextContent):
        return block.text
    if isinstance(block, types.EmbeddedResource):
        resource = block.resource
        if isinstance(resource, types.TextResourceContents):
            return resource.text
        if isinstance(resource, types.BlobResourceContents):
            return _content_summary(
                "embedded_blob",
                mime_type=resource.mimeType,
                size_chars=len(resource.blob),
            )
        return None
    if isinstance(block, types.ResourceLink):
        return _content_summary(
            "resource_link",
            name=block.name,
            uri=_safe_resource_uri(block.uri),
            mime_type=block.mimeType,
        )
    if isinstance(block, types.ImageContent):
        return _content_summary(
            "image",
            mime_type=block.mimeType,
            data_chars=len(block.data),
        )
    if isinstance(block, types.AudioContent):
        return _content_summary(
            "audio",
            mime_type=block.mimeType,
            data_chars=len(block.data),
        )
    return None


def serialize_mcp_call_result(
    result: object,
    *,
    settings: AppSettings,
    server_alias: str,
    remote_tool_name: str,
) -> McpSerializationOutcome:
    """安全归一化 MCP SDK 结果中的官方 content blocks 和 structured content。"""
    if not isinstance(result, types.CallToolResult):
        return McpSerializationOutcome(
            ok=False,
            error="MCP tool returned an invalid result",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )

    text_parts: list[str] = []
    for block in result.content:
        serialized = _serialize_content_block(block)
        if serialized is None:
            block_type, _ = _truncate_text(
                redact_sensitive_text(type(block).__name__),
                128,
            )
            return McpSerializationOutcome(
                ok=False,
                error=f"MCP tool returned unsupported content type: {block_type}",
                error_code=ToolErrorCode.MCP_UNSUPPORTED_CONTENT.value,
            )
        text_parts.append(serialized)
    text, text_truncated = _truncate_text(
        redact_sensitive_text("\n".join(text_parts)),
        settings.mcp_max_text_chars,
    )

    if result.isError:
        error = "MCP tool execution failed"
        if text.strip():
            error = f"{error}: {text}"
        error, _ = _truncate_text(
            error,
            min(settings.mcp_max_text_chars, _MAX_ERROR_TEXT_CHARS),
        )
        return McpSerializationOutcome(
            ok=False,
            error=error,
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
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
