"""HTML 与 PDF Web 工具共用的有限 hardened 下载适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from requests_hardened import Config, Manager
from requests_hardened.ip_filter import InvalidIPAddress

from app.config import AppSettings, get_settings
from app.tools.errors import ToolErrorCode, ToolExecutionError


_CHUNK_SIZE = 64 * 1024
_URL_MAX_CHARS = 2048
_ACCEPT_TYPE_ORDER = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
)


@dataclass(frozen=True, slots=True)
class DownloadedWebResource:
    """只向调用工具暴露的有限、安全响应数据。"""

    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    encoding: str
    body: bytes


def _url_error(message: str, *, code: ToolErrorCode) -> ToolExecutionError:
    """构造不包含目标地址、IP 或网络拓扑的 URL 错误。"""
    return ToolExecutionError(message, code=code, retryable=False)


def normalize_public_web_url(
    value: object,
    *,
    final: bool = False,
) -> str:
    """实施应用层公共 Web URL 规则并移除 fragment。"""
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


def _build_hardened_manager(
    settings: AppSettings | None = None,
) -> Manager:
    """按默认安全模式或显式受信代理模式构造 HTTP Manager。"""
    resolved = settings or get_settings()
    delegate_network_policy = (
        resolved.web_trusted_proxy_mode
        and resolved.resolved_web_fetch_proxy is not None
    )
    return Manager(
        Config(
            ip_filter_enable=not delegate_network_policy,
            ip_filter_allow_loopback_ips=False,
            never_redirect=False,
            default_timeout=(
                resolved.web_fetch_connect_timeout,
                resolved.web_fetch_read_timeout,
            ),
            user_agent_override=resolved.web_fetch_user_agent,
        )
    )


def _request_kwargs(
    allowed_content_types: frozenset[str],
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """构造不含用户 headers、Cookie、认证或隐式代理的 GET 参数。"""
    resolved = settings or get_settings()
    ordered_types = [
        content_type
        for content_type in _ACCEPT_TYPE_ORDER
        if content_type in allowed_content_types
    ]
    ordered_types.extend(
        sorted(allowed_content_types - set(ordered_types))
    )
    kwargs: dict[str, Any] = {
        "allow_redirects": True,
        "headers": {
            "Accept": ", ".join(ordered_types),
            "User-Agent": resolved.web_fetch_user_agent,
        },
        "stream": True,
        "timeout": (
            resolved.web_fetch_connect_timeout,
            resolved.web_fetch_read_timeout,
        ),
    }
    proxy = resolved.resolved_web_fetch_proxy
    if proxy is not None:
        kwargs["proxies"] = {
            "http": proxy,
            "https": proxy,
        }
    return kwargs


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


def _content_type(
    response: requests.Response,
    *,
    allowed_content_types: frozenset[str],
) -> str:
    """读取 Content-Type 主类型并按调用工具的白名单校验。"""
    raw_value = response.headers.get("Content-Type", "")
    normalized = (
        raw_value.split(";", 1)[0].strip().lower()
        if isinstance(raw_value, str)
        else ""
    )
    if normalized not in allowed_content_types:
        raise ToolExecutionError(
            "Web fetch response content type is not supported",
            code=ToolErrorCode.INVALID_RESULT,
            retryable=False,
        )
    return normalized


def _read_limited_body(
    response: requests.Response,
    *,
    max_response_bytes: int,
) -> bytes:
    """流式累计解压后字节，并在超过调用工具上限时立即停止。"""
    declared_length = response.headers.get("Content-Length")
    if declared_length:
        try:
            if int(declared_length) > max_response_bytes:
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
        if total > max_response_bytes:
            raise ToolExecutionError(
                "Web fetch response exceeded the configured size limit",
                code=ToolErrorCode.INVALID_RESULT,
                retryable=False,
            )
        chunks.append(chunk)
    return b"".join(chunks)


def download_public_web_resource(
    url: str,
    *,
    allowed_content_types: frozenset[str],
    max_response_bytes: int,
    settings: AppSettings | None = None,
) -> DownloadedWebResource:
    """通过 requests-hardened 下载单个有限响应并保证资源关闭。"""
    resolved = settings or get_settings()
    if not allowed_content_types:
        raise ValueError("allowed_content_types 不能为空")
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or max_response_bytes < 1
    ):
        raise ValueError("max_response_bytes 必须是正整数")

    requested_url = normalize_public_web_url(url)
    manager = _build_hardened_manager(resolved)
    session = manager.get_session()
    response: requests.Response | None = None
    try:
        session.trust_env = False
        session.max_redirects = resolved.web_fetch_max_redirects
        response = session.request(
            "GET",
            requested_url,
            **_request_kwargs(allowed_content_types, resolved),
        )
        status_code = _check_status(response)
        content_type = _content_type(
            response,
            allowed_content_types=allowed_content_types,
        )
        final_url = normalize_public_web_url(response.url, final=True)
        body = _read_limited_body(
            response,
            max_response_bytes=max_response_bytes,
        )
        return DownloadedWebResource(
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


__all__ = [
    "DownloadedWebResource",
    "download_public_web_resource",
    "normalize_public_web_url",
]
