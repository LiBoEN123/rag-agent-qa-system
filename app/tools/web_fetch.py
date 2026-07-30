"""使用共享 hardened 下载层并由 Trafilatura 提取单个公开网页。"""

from __future__ import annotations

import json
import re
from typing import Any

from trafilatura import extract

from app.config import get_settings
from app.tools.context import ToolInvocationContext
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.result import tool_success
from app.tools.web_download import (
    DownloadedWebResource,
    download_public_web_resource,
    normalize_public_web_url,
)


_ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "text/plain",
    }
)
_MULTIPLE_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def _decode_body(page: DownloadedWebResource) -> str:
    """按响应声明编码解码，未知编码时安全回退 UTF-8。"""
    try:
        return page.body.decode(page.encoding, errors="replace")
    except LookupError:
        return page.body.decode("utf-8", errors="replace")


def _normalize_content(value: str) -> str:
    """仅统一换行、首尾空白和连续空行，不改写正文。"""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return _MULTIPLE_BLANK_LINES.sub("\n\n", normalized)


def _metadata_text(value: object, *, max_chars: int) -> str:
    """把 Trafilatura 可选元数据规范化为有限单行字符串。"""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:max_chars]


def _extract_html(
    page: DownloadedWebResource,
) -> tuple[str, str, str, str, str]:
    """通过 Trafilatura 公共 API 提取正文和有限元数据。"""
    extracted = extract(
        _decode_body(page),
        url=page.final_url,
        output_format="json",
        with_metadata=True,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=True,
    )
    if not extracted:
        raise ToolExecutionError(
            "Trafilatura could not extract webpage content",
            code=ToolErrorCode.INVALID_RESULT,
            retryable=False,
        )
    try:
        document = json.loads(extracted)
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(
            "Trafilatura returned an invalid extraction result",
            code=ToolErrorCode.INVALID_RESULT,
            retryable=False,
        ) from exc
    if not isinstance(document, dict):
        raise ToolExecutionError(
            "Trafilatura returned an invalid extraction result",
            code=ToolErrorCode.INVALID_RESULT,
            retryable=False,
        )
    content = document.get("text")
    if not isinstance(content, str) or not _normalize_content(content):
        raise ToolExecutionError(
            "Trafilatura could not extract webpage content",
            code=ToolErrorCode.INVALID_RESULT,
            retryable=False,
        )
    return (
        _normalize_content(content),
        _metadata_text(document.get("title"), max_chars=500),
        _metadata_text(document.get("author"), max_chars=300),
        _metadata_text(document.get("date"), max_chars=100),
        _metadata_text(document.get("language"), max_chars=50),
    )


def _extract_page(
    page: DownloadedWebResource,
) -> tuple[str, str, str, str, str]:
    """按受支持内容类型选择纯文本规范化或 Trafilatura 提取。"""
    if page.content_type == "text/plain":
        content = _normalize_content(_decode_body(page))
        if not content:
            raise ToolExecutionError(
                "Web fetch response did not contain text",
                code=ToolErrorCode.INVALID_RESULT,
                retryable=False,
            )
        return content, "", "", "", ""
    return _extract_html(page)


def web_fetch(args: dict[str, Any], context: ToolInvocationContext) -> dict[str, Any]:
    """安全下载单页正文并返回项目统一成功结果。"""
    del context
    settings = get_settings()
    max_chars = args["max_chars"]
    page = download_public_web_resource(
        normalize_public_web_url(args.get("url")),
        allowed_content_types=_ALLOWED_CONTENT_TYPES,
        max_response_bytes=settings.web_fetch_max_response_bytes,
        settings=settings,
    )
    content, title, author, published_date, language = _extract_page(page)
    truncated = len(content) > max_chars
    content = content[:max_chars]
    output = {
        "requested_url": page.requested_url,
        "final_url": page.final_url,
        "status_code": page.status_code,
        "content_type": page.content_type,
        "title": title,
        "author": author,
        "published_date": published_date,
        "language": language,
        "content": content,
        "content_length": len(content),
        "truncated": truncated,
        "extractor": "trafilatura",
        "content_is_untrusted": True,
    }
    return tool_success("web_fetch", args, output)


__all__ = ["web_fetch"]
