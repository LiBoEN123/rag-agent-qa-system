"""基于 DDGS 公共 Python API 的公开互联网搜索工具实现。"""

from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import urlsplit

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from app.config import Settings
from app.tools.context import ToolInvocationContext
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.result import tool_success


_FRESHNESS_TO_TIMELIMIT = {
    "any": None,
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
}
_WHITESPACE = re.compile(r"\s+")
_TITLE_MAX_CHARS = 300
_SNIPPET_MAX_CHARS = 1000
_URL_MAX_CHARS = 2048


def _clean_text(value: object, *, max_chars: int) -> str:
    """使用标准库解码实体、合并空白并限制文本体量。"""
    if not isinstance(value, str):
        return ""
    cleaned = _WHITESPACE.sub(" ", html.unescape(value).strip())
    return cleaned[:max_chars]


def _normalize_url(value: object) -> str:
    """规范化并校验公开 HTTP(S) URL。"""
    url = _clean_text(value, max_chars=_URL_MAX_CHARS)
    if not url or any(character.isspace() for character in url):
        return ""
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def _normalize_results(raw_results: object, *, max_results: int) -> list[dict[str, Any]]:
    """把 DDGS 文本结果转换为稳定、去重且有限的公开结构。"""
    if not isinstance(raw_results, list):
        raise ToolExecutionError(
            "Web search returned an invalid result structure",
            code=ToolErrorCode.INVALID_RESULT,
            retryable=False,
        )

    normalized: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for raw_item in raw_results:
        if not isinstance(raw_item, dict):
            continue
        url = _normalize_url(raw_item.get("href"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        normalized.append(
            {
                "rank": len(normalized) + 1,
                "title": _clean_text(
                    raw_item.get("title"), max_chars=_TITLE_MAX_CHARS
                ),
                "url": url,
                "snippet": _clean_text(
                    raw_item.get("body"), max_chars=_SNIPPET_MAX_CHARS
                ),
            }
        )
        if len(normalized) >= max_results:
            break
    return normalized


def web_search(args: dict[str, Any], context: ToolInvocationContext) -> dict[str, Any]:
    """调用 DDGS 文本搜索并返回项目统一成功结果。"""
    del context
    query = args["query"]
    max_results = args["max_results"]
    freshness = args["freshness"]
    try:
        raw_results = DDGS(
            proxy=Settings.WEB_SEARCH_PROXY,
            timeout=Settings.WEB_SEARCH_TIMEOUT,
        ).text(
            query,
            region=Settings.WEB_SEARCH_REGION,
            safesearch=Settings.WEB_SEARCH_SAFESEARCH,
            timelimit=_FRESHNESS_TO_TIMELIMIT[freshness],
            max_results=max_results,
            backend="auto",
        )
    except RatelimitException as exc:
        raise ToolExecutionError(
            "Web search was rate limited by the upstream service",
            code=ToolErrorCode.RATE_LIMITED,
            retryable=True,
        ) from exc
    except TimeoutException as exc:
        raise ToolExecutionError(
            "Web search request timed out",
            code=ToolErrorCode.TIMEOUT,
            retryable=True,
        ) from exc
    except DDGSException as exc:
        raise ToolExecutionError(
            "Web search dependency is temporarily unavailable",
            code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            retryable=True,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(
            "Web search request arguments were rejected",
            code=ToolErrorCode.INVALID_ARGUMENTS,
            retryable=False,
        ) from exc

    results = _normalize_results(raw_results, max_results=max_results)
    output = {
        "query": query,
        "count": len(results),
        "provider": "ddgs",
        "content_is_untrusted": True,
        "results": results,
    }
    return tool_success("web_search", args, output)


__all__ = ["web_search"]
