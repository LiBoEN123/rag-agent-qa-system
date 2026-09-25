"""MCP 原始结果的安全净化与官方 ToolMessage 的项目结果归一化。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from langchain_core.messages import ToolMessage
from mcp import types

from app.config import AppSettings
from app.tools.errors import ToolErrorCode
from app.common.redaction import TRUNCATED, redact_sensitive_data, redact_sensitive_text
from app.tools.result import tool_error, tool_success


class McpSerializationProfile(str, Enum):
    """区分 Agent 上下文输出与显式永久知识摄取输出。"""

    AGENT = "agent"
    INGESTION = "ingestion"


_MAX_ERROR_TEXT_CHARS = 2_000
_PROJECT_META_PREFIX = "__project_mcp_meta__:"
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


# Agent 逐步裁剪；入库在总预算内保留完整结构，两者复用同一脱敏路径。
def _bounded_structured(
    value: Any, *, limit: int, preserve_content: bool = False,
) -> tuple[Any, bool]:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        return {"truncated": True, "unsupported_value": True}, True
    if preserve_content and len(raw) > limit:
        return {"truncated": True}, True
    # JSON 总长度已限制深度、项目数和字符串长度，无需再套用 Agent 摘要阈值。
    bounds = ((limit, limit, limit),) if preserve_content else (
        (8, 100, 4000),
        (6, 50, 2000),
        (5, 20, 1000),
        (4, 10, 500),
        (3, 5, 200),
    )
    for max_depth, max_items, max_string_chars in bounds:
        try:
            sanitized = redact_sensitive_data(
                value,
                max_depth=max_depth,
                max_items=max_items,
                max_string_chars=min(max_string_chars, limit),
                max_serialized_chars=max(limit * 4, limit + 1024),
            )
        except RecursionError:
            return {"truncated": True, "unsupported_value": True}, True
        serialized = json.dumps(
            sanitized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(serialized) <= limit:
            return sanitized, len(raw) > limit or TRUNCATED in serialized
    return {"truncated": True}, True


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
    settings: AppSettings,
    server_alias: str,
    remote_tool_name: str,
    profile: McpSerializationProfile = McpSerializationProfile.AGENT,
) -> McpSanitizationOutcome:
    """把不可信 payload 净化为仍由官方 adapter 转换的 CallToolResult。"""
    if not isinstance(value, types.CallToolResult):
        return McpSanitizationOutcome(
            ok=False,
            error="MCP tool returned an invalid result",
            error_code=ToolErrorCode.MCP_EXECUTION_ERROR.value,
        )
    try:
        resolved_profile = McpSerializationProfile(profile)
    except (TypeError, ValueError):
        return McpSanitizationOutcome(
            ok=False,
            error="MCP serialization profile is invalid",
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

    text_limit = (
        settings.mcp_max_text_chars
        if resolved_profile is McpSerializationProfile.AGENT
        else MCP_INGESTION_MAX_TEXT_CHARS
    )
    structured_limit = (
        settings.mcp_max_structured_chars
        if resolved_profile is McpSerializationProfile.AGENT
        else MCP_INGESTION_MAX_STRUCTURED_CHARS
    )
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
            limit=structured_limit,
            preserve_content=resolved_profile is McpSerializationProfile.INGESTION,
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
        "serialization_profile": resolved_profile.value,
        "content_blocks": ingestion_blocks,
    }
    if structured is not None:
        ingestion_output["structured_content"] = structured
    return McpSanitizationOutcome(
        ok=True,
        result=safe_result,
        text=text,
        truncated=truncated,
        ingestion_output=(
            ingestion_output
            if resolved_profile is McpSerializationProfile.INGESTION
            else None
        ),
    )


def attach_mcp_execution_metadata(
    result: types.CallToolResult,
    metadata: dict[str, Any],
) -> types.CallToolResult:
    """以安全文本标记携带项目元数据，随后仍由官方 adapter 生成 ToolMessage。"""
    marker = types.TextContent(
        type="text",
        text=_PROJECT_META_PREFIX
        + json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    )
    return result.model_copy(update={"content": [*result.content, marker]})


def normalize_mcp_tool_message(
    message: ToolMessage,
    *,
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any] | None:
    """从官方 ToolMessage 读取安全内容/artifact，并构造一次项目 ToolResult。"""
    if not isinstance(message.content, list):
        return None
    metadata = None
    text_parts: list[str] = []
    for block in message.content:
        if not isinstance(block, dict) or block.get("type") != "text":
            return None
        text = block.get("text")
        if not isinstance(text, str):
            return None
        if text.startswith(_PROJECT_META_PREFIX):
            if metadata is not None:
                return None
            try:
                candidate = json.loads(text.removeprefix(_PROJECT_META_PREFIX))
            except (TypeError, ValueError):
                return None
            if not isinstance(candidate, dict):
                return None
            metadata = candidate
        else:
            text_parts.append(text)
    if not isinstance(metadata, dict):
        return None

    runtime = metadata.get("runtime")
    governance = metadata.get("governance")
    audit = metadata.get("audit")
    if not all(isinstance(value, dict) for value in (runtime, governance, audit)):
        return None
    error_code = metadata.get("error_code")
    if message.status == "error" or isinstance(error_code, str) and error_code:
        result = tool_error(
            tool_name,
            args,
            str(metadata.get("error") or "MCP tool execution failed"),
            error_code=str(error_code or ToolErrorCode.MCP_EXECUTION_ERROR.value),
            retryable=False,
            runtime=runtime,
            governance=governance,
        )
    else:
        output = metadata.get("output")
        if not isinstance(output, dict):
            return None
        output = {**output, "content": "\n".join(text_parts)}
        artifact = message.artifact
        if isinstance(artifact, dict) and "structured_content" in artifact:
            output["structured_content"] = artifact["structured_content"]
        result = tool_success(
            tool_name,
            args,
            output,
            runtime=runtime,
            governance=governance,
        )
    result["tool_call_id"] = message.tool_call_id
    result["audit"] = audit
    return result


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
