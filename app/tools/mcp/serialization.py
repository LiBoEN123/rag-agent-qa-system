"""知识入库 MCP 结果的脱敏、完整性标记与体积限制。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from mcp import types

from app.tools.errors import ToolErrorCode
from app.common.redaction import TRUNCATED, redact_sensitive_data, redact_sensitive_text


_MAX_ERROR_TEXT_CHARS = 2_000
MCP_INGESTION_MAX_TEXT_CHARS = 1_000_000
# Transport 可同时包含 Markdown、HTML 和 metadata；最终正文仍由 ingestion policy 限制。
MCP_INGESTION_MAX_STRUCTURED_CHARS = 2_000_000


@dataclass(frozen=True, slots=True)
class McpSanitizationOutcome:
    """净化后的官方 MCP result，以及 ingestion 所需的安全原始元数据。"""

    ok: bool
    result: types.CallToolResult | None = None
    text: str = ""
    truncated: bool = False
    ingestion_output: dict[str, Any] | None = None
    error: str = ""
    error_code: str = ""


# 保存清理后的文本、原生 MCP 内容块和入库专用内容表示。
@dataclass(frozen=True, slots=True)
class _SafeBlock:
    text: str
    mcp_block: object
    ingestion_block: dict[str, Any]


# 按字符上限截断文本并添加可容纳的截断标记，同时返回截断状态。
def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    if limit <= len(TRUNCATED):
        return TRUNCATED[:limit], True
    return value[: limit - len(TRUNCATED)] + TRUNCATED, True


# 入库在总预算内保留完整结构；超过预算或不能安全净化时标记不完整。
def _bounded_structured(value: Any, *, limit: int) -> tuple[Any, bool]:
    try:
        raw = json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        if len(raw) > limit:
            return {"truncated": True}, True
        sanitized = redact_sensitive_data(
            value, max_depth=limit, max_items=limit, max_string_chars=limit,
            max_serialized_chars=max(limit * 4, limit + 1024),
        )
        serialized = json.dumps(
            sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError):
        return {"truncated": True, "unsupported_value": True}, True
    if len(serialized) > limit:
        return {"truncated": True}, True
    return sanitized, TRUNCATED in serialized


# 重建资源地址以移除用户凭据、查询参数和片段，格式异常时返回占位符。
def _safe_resource_uri(value: object) -> str:
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


# 将内容块类型和非空元数据编码为稳定、紧凑的 JSON 摘要。
def _summary(block_type: str, **metadata: object) -> str:
    payload = {"type": block_type}
    payload.update({key: value for key, value in metadata.items() if value is not None})
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


# 脱敏文本和资源地址，将二进制等内容转为摘要，不支持的类型返回空值。
def _safe_block(block: object) -> _SafeBlock | None:  # noqa: PLR0911
    if isinstance(block, types.TextContent):
        text = redact_sensitive_text(block.text)
        return _SafeBlock(
            text,
            block.model_copy(update={"text": text}),
            {"type": "text", "text": text, "size_chars": len(block.text)},
        )
    if isinstance(block, types.EmbeddedResource):
        resource = block.resource
        if isinstance(resource, types.TextResourceContents):
            text = redact_sensitive_text(resource.text)
            uri = _safe_resource_uri(resource.uri)
            safe_resource = resource.model_copy(update={"text": text, "uri": uri})
            return _SafeBlock(
                text,
                block.model_copy(update={"resource": safe_resource}),
                {
                    "type": "embedded_resource_text",
                    "text": text,
                    "uri": uri,
                    "mime_type": resource.mimeType,
                    "size_chars": len(resource.text),
                },
            )
        if isinstance(resource, types.BlobResourceContents):
            text = _summary(
                "embedded_blob",
                mime_type=resource.mimeType,
                size_chars=len(resource.blob),
            )
            return _SafeBlock(
                text, types.TextContent(type="text", text=text), json.loads(text)
            )
        return None
    if isinstance(block, types.ResourceLink):
        text = _summary(
            "resource_link",
            name=redact_sensitive_text(block.name),
            uri=_safe_resource_uri(block.uri),
            mime_type=block.mimeType,
        )
    elif isinstance(block, types.ImageContent):
        text = _summary("image", mime_type=block.mimeType, data_chars=len(block.data))
    elif isinstance(block, types.AudioContent):
        text = _summary("audio", mime_type=block.mimeType, data_chars=len(block.data))
    else:
        return None
    return _SafeBlock(text, types.TextContent(type="text", text=text), json.loads(text))


def sanitize_mcp_call_result(
    value: object,
    *,
    server_alias: str,
    remote_tool_name: str,
) -> McpSanitizationOutcome:
    """把不可信 payload 净化为仍由官方 adapter 转换的 CallToolResult。"""
    if not isinstance(value, types.CallToolResult):
        return McpSanitizationOutcome(
            ok=False,
            error="MCP tool returned an invalid result",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )
    blocks: list[_SafeBlock] = []
    for block in value.content:
        sanitized = _safe_block(block)
        if sanitized is None:
            block_type, _ = _truncate_text(
                redact_sensitive_text(type(block).__name__), 128
            )
            return McpSanitizationOutcome(
                ok=False,
                error=f"MCP tool returned unsupported content type: {block_type}",
                error_code=ToolErrorCode.MCP_UNSUPPORTED_CONTENT.value,
            )
        blocks.append(sanitized)

    text_limit = MCP_INGESTION_MAX_TEXT_CHARS
    if value.isError:
        text_limit = min(text_limit, _MAX_ERROR_TEXT_CHARS)
    text, text_truncated = _truncate_text(
        "\n".join(block.text for block in blocks), text_limit
    )
    safe_content = (
        [block.mcp_block for block in blocks]
        if not text_truncated
        else [types.TextContent(type="text", text=text)]
    )

    structured = None
    structured_truncated = False
    if value.structuredContent is not None:
        structured, structured_truncated = _bounded_structured(
            value.structuredContent,
            limit=MCP_INGESTION_MAX_STRUCTURED_CHARS,
        )
    safe_result = value.model_copy(
        update={"content": safe_content, "structuredContent": structured}
    )
    truncated = bool(text_truncated or structured_truncated)
    ingestion_blocks = []
    for block in blocks:
        item = {
            key: val for key, val in block.ingestion_block.items() if val is not None
        }
        if text_truncated and "text" in item:
            item.pop("text")
            item["truncated"] = True
        ingestion_blocks.append(item)
    ingestion_output = {
        "content": text,
        "content_is_untrusted": True,
        "server": server_alias,
        "tool": remote_tool_name,
        "truncated": truncated,
        "content_blocks": ingestion_blocks,
    }
    if structured is not None:
        ingestion_output["structured_content"] = structured
    return McpSanitizationOutcome(
        ok=True,
        result=safe_result,
        text=text,
        truncated=truncated,
        ingestion_output=ingestion_output,
    )


# 生成 MCP 输出的类型、字段、文本长度和信任标记摘要，供审计使用。
def summarize_mcp_output(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"mode": "summary", "type": type(value).__name__}
    content = value.get("content")
    return {
        "mode": "summary",
        "type": "dict",
        "fields": sorted(str(key) for key in value)[:20],
        "content_chars": len(content) if isinstance(content, str) else 0,
        "truncated": bool(value.get("truncated", False)),
        "content_is_untrusted": bool(value.get("content_is_untrusted", False)),
    }
