"""把不可信 MCP Tool 输出压缩为有限、脱敏的项目数据。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
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


class McpSerializationProfile(str, Enum):
    """区分 Agent 上下文输出与显式永久知识摄取输出。"""

    AGENT = "agent"
    INGESTION = "ingestion"


_MAX_ERROR_TEXT_CHARS = 2_000
MCP_INGESTION_MAX_TEXT_CHARS = 1_000_000


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


@dataclass(frozen=True, slots=True)
class _SerializedContentBlock:
    """一个官方 MCP content block 的安全文本与有限类型信息。"""

    text: str
    block: dict[str, Any]


def _text_block(block_type: str, text: str, **metadata: object) -> _SerializedContentBlock:
    return _SerializedContentBlock(
        text=text,
        block={"type": block_type, "text": text, **metadata},
    )


def _summary_block(block_type: str, **metadata: object) -> _SerializedContentBlock:
    safe_metadata = {key: value for key, value in metadata.items() if value is not None}
    return _SerializedContentBlock(
        text=_content_summary(block_type, **safe_metadata),
        block={"type": block_type, **safe_metadata},
    )


def _serialize_content_block(  # noqa: PLR0911
    block: object,
) -> _SerializedContentBlock | None:
    """依据官方 SDK 类型提取文本，并永久丢弃二进制 payload。"""
    if isinstance(block, types.TextContent):
        return _text_block("text", block.text, size_chars=len(block.text))
    if isinstance(block, types.EmbeddedResource):
        resource = block.resource
        if isinstance(resource, types.TextResourceContents):
            return _text_block(
                "embedded_resource_text",
                resource.text,
                uri=_safe_resource_uri(resource.uri),
                mime_type=resource.mimeType,
                size_chars=len(resource.text),
            )
        if isinstance(resource, types.BlobResourceContents):
            return _summary_block(
                "embedded_blob",
                mime_type=resource.mimeType,
                size_chars=len(resource.blob),
            )
        return None
    if isinstance(block, types.ResourceLink):
        return _summary_block(
            "resource_link",
            name=block.name,
            uri=_safe_resource_uri(block.uri),
            mime_type=block.mimeType,
        )
    if isinstance(block, types.ImageContent):
        return _summary_block(
            "image",
            mime_type=block.mimeType,
            data_chars=len(block.data),
        )
    if isinstance(block, types.AudioContent):
        return _summary_block(
            "audio",
            mime_type=block.mimeType,
            data_chars=len(block.data),
        )
    return None


def _safe_ingestion_blocks(
    blocks: list[_SerializedContentBlock],
    *,
    text_truncated: bool,
) -> list[dict[str, Any]]:
    """只在完整时保留脱敏文本；截断结果只暴露 block 类型摘要。"""
    safe_blocks: list[dict[str, Any]] = []
    for serialized in blocks:
        block: dict[str, Any] = {}
        for key, value in serialized.block.items():
            if value is None or (key == "text" and text_truncated):
                continue
            if key == "text" and isinstance(value, str):
                try:
                    decoded = json.loads(value)
                except (TypeError, ValueError):
                    value = redact_sensitive_text(value)
                else:
                    value = json.dumps(
                        redact_sensitive_data(
                            decoded,
                            max_depth=8,
                            max_items=100,
                            max_string_chars=MCP_INGESTION_MAX_TEXT_CHARS,
                            max_serialized_chars=MCP_INGESTION_MAX_TEXT_CHARS * 2,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
            elif isinstance(value, str):
                value = redact_sensitive_text(value)
            block[key] = value
        if text_truncated and "text" in serialized.block:
            block["truncated"] = True
        safe_blocks.append(block)
    return safe_blocks


def serialize_mcp_call_result(
    result: object,
    *,
    settings: AppSettings,
    server_alias: str,
    remote_tool_name: str,
    profile: McpSerializationProfile = McpSerializationProfile.AGENT,
) -> McpSerializationOutcome:
    """安全归一化 MCP SDK 结果中的官方 content blocks 和 structured content。"""
    if not isinstance(result, types.CallToolResult):
        return McpSerializationOutcome(
            ok=False,
            error="MCP tool returned an invalid result",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )

    try:
        resolved_profile = McpSerializationProfile(profile)
    except (TypeError, ValueError):
        return McpSerializationOutcome(
            ok=False,
            error="MCP serialization profile is invalid",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )

    serialized_blocks: list[_SerializedContentBlock] = []
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
        serialized_blocks.append(serialized)
    text_limit = (
        settings.mcp_max_text_chars
        if resolved_profile is McpSerializationProfile.AGENT
        else MCP_INGESTION_MAX_TEXT_CHARS
    )
    text, text_truncated = _truncate_text(
        redact_sensitive_text("\n".join(block.text for block in serialized_blocks)),
        text_limit,
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
    if resolved_profile is McpSerializationProfile.INGESTION:
        output["serialization_profile"] = resolved_profile.value
        output["content_blocks"] = _safe_ingestion_blocks(
            serialized_blocks,
            text_truncated=text_truncated,
        )
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
    "MCP_INGESTION_MAX_TEXT_CHARS",
    "McpSerializationOutcome",
    "McpSerializationProfile",
    "serialize_mcp_call_result",
    "summarize_mcp_output",
]
