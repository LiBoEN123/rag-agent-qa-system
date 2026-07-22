"""使用 requests-hardened 下载并由 Trafilatura 提取单个公开网页。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from requests_hardened import Config, Manager
from requests_hardened.ip_filter import InvalidIPAddress
from trafilatura import extract

from app.config import Settings
from app.tools.context import ToolInvocationContext
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.result import tool_success


_ALLOWED_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
}
_ACCEPT_HEADER = "text/html, application/xhtml+xml, text/plain"
_CHUNK_SIZE = 64 * 1024
_URL_MAX_CHARS = 2048
_MULTIPLE_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


@dataclass(frozen=True, slots=True)
class _DownloadedPage:
    """只在 handler 内部传递的有限响应数据。"""

    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    encoding: str
    body: bytes


def _url_error(message: str, *, code: ToolErrorCode) -> ToolExecutionError:
    """构造不包含目标地址、IP 或网络拓扑的 URL 错误。"""
    return ToolExecutionError(message, code=code, retryable=False)


def _normalize_web_url(value: object, *, final: bool = False) -> str:
    """实施少量应用层 URL 规则，并移除 fragment。"""
    code = ToolErrorCode.INVALID_RESULT if final else ToolErrorCode.INVALID_ARGUMENTS
    message = (
        "Web fetch returned an invalid final URL"
        if final
        else "Web fetch URL is not a supported public webpage URL"
    )
    if not isinstance(value, str) or not value or len(value) > _URL_MAX_CHARS:
        raise _url_error(message, code=code)
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise _url_error(message, code=code)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise _url_error(message, code=code) from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise _url_error(message, code=code)
    if parsed.username is not None or parsed.password is not None:
        raise _url_error(message, code=code)
    if port is not None and port not in {80, 443}:
        raise _url_error(message, code=code)
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path,
            parsed.query,
            "",
        )
    )


def _build_hardened_manager() -> Manager:
    """按可信服务器配置构造启用 SSRF 过滤的 HTTP Manager。"""
    return Manager(
        Config(
            ip_filter_enable=True,
            ip_filter_allow_loopback_ips=False,
            never_redirect=False,
            default_timeout=(
                Settings.WEB_FETCH_CONNECT_TIMEOUT,
                Settings.WEB_FETCH_READ_TIMEOUT,
            ),
            user_agent_override=Settings.WEB_FETCH_USER_AGENT,
        )
    )


def _request_kwargs() -> dict[str, Any]:
    """构造不包含用户 headers、Cookie、认证或隐式代理的 GET 参数。"""
    kwargs: dict[str, Any] = {
        "allow_redirects": True,
        "headers": {
            "Accept": _ACCEPT_HEADER,
            "User-Agent": Settings.WEB_FETCH_USER_AGENT,
        },
        "stream": True,
        "timeout": (
            Settings.WEB_FETCH_CONNECT_TIMEOUT,
            Settings.WEB_FETCH_READ_TIMEOUT,
        ),
    }
    if Settings.WEB_FETCH_PROXY is not None:
        kwargs["proxies"] = {
            "http": Settings.WEB_FETCH_PROXY,
            "https": Settings.WEB_FETCH_PROXY,
        }
    return kwargs


def _content_type(response: requests.Response) -> str:
    """读取并规范化 Content-Type 主类型，缺失时保守拒绝。"""
    raw_value = response.headers.get("Content-Type", "")
    normalized = raw_value.split(";", 1)[0].strip().lower()
    if normalized not in _ALLOWED_CONTENT_TYPES:
        raise ToolExecutionError(
            "Web fetch response content type is not supported",
            code=ToolErrorCode.INVALID_RESULT,
            retryable=False,
        )
    return normalized


def _check_status(response: requests.Response) -> int:
    """把 HTTP 状态映射为稳定 ToolRuntime 错误契约。"""
    status_code = response.status_code
    if not isinstance(status_code, int):
        raise ToolExecutionError(
            "Web fetch response status is invalid",
            code=ToolErrorCode.INVALID_RESULT,
            retryable=False,
        )
    if 200 <= status_code <= 299:
        return status_code
    if status_code == 429:
        raise ToolExecutionError(
            "Web fetch was rate limited by the upstream website",
            code=ToolErrorCode.RATE_LIMITED,
            retryable=True,
        )
    if status_code >= 500:
        raise ToolExecutionError(
            "Web fetch upstream website is temporarily unavailable",
            code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            retryable=True,
        )
    raise ToolExecutionError(
        "Web fetch upstream website rejected the request",
        code=ToolErrorCode.EXECUTION_ERROR,
        retryable=False,
    )


def _read_limited_body(response: requests.Response) -> bytes:
    """流式累计解压后字节，并在超过服务端上限时立即停止。"""
    limit = Settings.WEB_FETCH_MAX_RESPONSE_BYTES
    declared_length = response.headers.get("Content-Length")
    if declared_length:
        try:
            if int(declared_length) > limit:
                raise ToolExecutionError(
                    "Web fetch response exceeded the configured size limit",
                    code=ToolErrorCode.INVALID_RESULT,
                    retryable=False,
                )
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
        if not chunk:
            continue
        if not isinstance(chunk, bytes):
            raise ToolExecutionError(
                "Web fetch response body is invalid",
                code=ToolErrorCode.INVALID_RESULT,
                retryable=False,
            )
        total += len(chunk)
        if total > limit:
            raise ToolExecutionError(
                "Web fetch response exceeded the configured size limit",
                code=ToolErrorCode.INVALID_RESULT,
                retryable=False,
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _download_page(requested_url: str) -> _DownloadedPage:
    """通过 hardened Session 下载单个有限响应并保证资源关闭。"""
    manager = _build_hardened_manager()
    session = manager.get_session()
    response: requests.Response | None = None
    try:
        session.trust_env = False
        session.max_redirects = Settings.WEB_FETCH_MAX_REDIRECTS
        response = session.request("GET", requested_url, **_request_kwargs())
        status_code = _check_status(response)
        content_type = _content_type(response)
        final_url = _normalize_web_url(response.url, final=True)
        body = _read_limited_body(response)
        return _DownloadedPage(
            requested_url=requested_url,
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
            encoding=response.encoding or "utf-8",
            body=body,
        )
    except ToolExecutionError:
        raise
    except InvalidIPAddress as exc:
        raise ToolExecutionError(
            "Web fetch target was denied by the network safety policy",
            code=ToolErrorCode.POLICY_DENIED,
            retryable=False,
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ToolExecutionError(
            "Web fetch request timed out",
            code=ToolErrorCode.TIMEOUT,
            retryable=True,
        ) from exc
    except (
        requests.exceptions.InvalidSchema,
        requests.exceptions.InvalidURL,
        requests.exceptions.MissingSchema,
    ) as exc:
        raise ToolExecutionError(
            "Web fetch URL was rejected by the HTTP client",
            code=ToolErrorCode.INVALID_ARGUMENTS,
            retryable=False,
        ) from exc
    except requests.exceptions.TooManyRedirects as exc:
        raise ToolExecutionError(
            "Web fetch exceeded the configured redirect limit",
            code=ToolErrorCode.EXECUTION_ERROR,
            retryable=False,
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise ToolExecutionError(
            "Web fetch dependency is temporarily unavailable",
            code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            retryable=True,
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ToolExecutionError(
            "Web fetch dependency failed to return a valid response",
            code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            retryable=True,
        ) from exc
    finally:
        if response is not None:
            response.close()
        session.close()


def _decode_body(page: _DownloadedPage) -> str:
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


def _extract_html(page: _DownloadedPage) -> tuple[str, str, str, str, str]:
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


def _extract_page(page: _DownloadedPage) -> tuple[str, str, str, str, str]:
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
    requested_url = _normalize_web_url(args.get("url"))
    max_chars = args["max_chars"]
    page = _download_page(requested_url)
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
