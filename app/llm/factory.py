"""按模型角色创建并缓存 LangChain ChatOpenAI 客户端。"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.config import Settings
from app.llm.errors import ModelCallError

try:
    from langchain_openai import ChatOpenAI
except ImportError as exc:  # pragma: no cover - 仅在依赖尚未安装时触发
    ChatOpenAI = None  # type: ignore[assignment,misc]
    _LANGCHAIN_IMPORT_ERROR: Exception | None = exc
else:
    _LANGCHAIN_IMPORT_ERROR = None


def _openai_base_url(api_url: str) -> str:
    """把现有 Chat Completions 完整地址转换为 ChatOpenAI 的 base_url。"""
    value = (api_url or "").strip().rstrip("/")
    if not value:
        return ""

    parsed = urlsplit(value)
    path = parsed.path.rstrip("/")
    suffix = "/chat/completions"
    if path.endswith(suffix):
        path = path[: -len(suffix)]
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)).rstrip("/")


def _parameter_name(modern: str, legacy: str) -> str:
    """根据已安装 ChatOpenAI 签名选择新旧版本都可接受的参数名。"""
    if ChatOpenAI is None:
        return modern
    try:
        parameters = inspect.signature(ChatOpenAI).parameters
    except (TypeError, ValueError):
        return modern
    if modern in parameters or any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in parameters.values()
    ):
        return modern
    return legacy if legacy in parameters else modern


def _create_model(
    *,
    role: str,
    api_key: str | None,
    api_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int | float,
    max_retries: int,
) -> Any | None:
    """创建单个角色的 ChatOpenAI；缺少模型或 URL 时明确返回 None。"""
    api_url = (api_url or "").strip()
    model_name = (model_name or "").strip()
    if not api_url or not model_name:
        return None
    if ChatOpenAI is None:
        raise ModelCallError(
            "LangChain 模型客户端不可用，请安装 langchain-openai。",
            detail=type(_LANGCHAIN_IMPORT_ERROR).__name__ if _LANGCHAIN_IMPORT_ERROR else "ImportError",
            provider="langchain",
            model_name=model_name,
            retryable=False,
        )

    kwargs: dict[str, Any] = {
        _parameter_name("model", "model_name"): model_name,
        _parameter_name("api_key", "openai_api_key"): api_key or "not-required",
        _parameter_name("base_url", "openai_api_base"): _openai_base_url(api_url),
        "temperature": temperature,
        "max_tokens": max_tokens,
        _parameter_name("timeout", "request_timeout"): timeout,
        "max_retries": max(0, int(max_retries)),
    }
    try:
        return ChatOpenAI(**kwargs)
    except Exception as exc:
        raise ModelCallError(
            f"{role} 模型客户端初始化失败，请检查模型配置。",
            detail=type(exc).__name__,
            provider="langchain",
            model_name=model_name,
            retryable=False,
        ) from exc


@lru_cache(maxsize=1)
def get_answer_model() -> Any | None:
    """返回 RAG 最终回答模型。"""
    return _create_model(
        role="answer",
        api_key=Settings.LLM_API_KEY,
        api_url=Settings.LLM_API_URL,
        model_name=Settings.MODEL_NAME,
        temperature=Settings.TEMPERATURE,
        max_tokens=Settings.LLM_MAX_TOKENS,
        timeout=Settings.LLM_TIMEOUT,
        max_retries=Settings.LLM_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_router_model() -> Any | None:
    """返回 Agent Router 模型。"""
    return _create_model(
        role="router",
        api_key=Settings.ROUTER_API_KEY or Settings.LLM_API_KEY,
        api_url=Settings.ROUTER_API_URL,
        model_name=Settings.ROUTER_MODEL_NAME,
        temperature=Settings.ROUTER_TEMPERATURE,
        max_tokens=Settings.ROUTER_MAX_TOKENS,
        timeout=Settings.ROUTER_TIMEOUT,
        max_retries=Settings.ROUTER_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_query_rewrite_model() -> Any | None:
    """返回多轮 Query Rewrite 模型。"""
    return _create_model(
        role="query_rewrite",
        api_key=Settings.QUERY_REWRITE_API_KEY or Settings.LLM_API_KEY,
        api_url=Settings.QUERY_REWRITE_API_URL,
        model_name=Settings.QUERY_REWRITE_MODEL_NAME,
        temperature=Settings.QUERY_REWRITE_TEMPERATURE,
        max_tokens=Settings.QUERY_REWRITE_MAX_TOKENS,
        timeout=Settings.QUERY_REWRITE_TIMEOUT,
        max_retries=0,
    )


@lru_cache(maxsize=1)
def get_decompose_model() -> Any | None:
    """返回 Query Decompose 模型，并保留其现有独立 Key 语义。"""
    return _create_model(
        role="decompose",
        api_key=Settings.DECOMPOSE_API_KEY,
        api_url=Settings.DECOMPOSE_API_URL,
        model_name=Settings.DECOMPOSE_MODEL_NAME,
        temperature=0.0,
        max_tokens=Settings.DECOMPOSE_MAX_TOKENS,
        timeout=Settings.DECOMPOSE_TIMEOUT,
        max_retries=0,
    )


@lru_cache(maxsize=1)
def get_compare_model() -> Any | None:
    """返回文档对比模型，并保留其独立推理参数。"""
    return _create_model(
        role="compare",
        api_key=Settings.LLM_API_KEY,
        api_url=Settings.LLM_API_URL,
        model_name=Settings.MODEL_NAME,
        temperature=0.0,
        max_tokens=min(Settings.LLM_MAX_TOKENS, 1024),
        timeout=Settings.LLM_TIMEOUT,
        max_retries=Settings.LLM_MAX_RETRIES,
    )


@lru_cache(maxsize=8)
def get_judge_model(
    *,
    api_key: str | None = None,
    api_url: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | float | None = None,
    max_retries: int | None = None,
) -> Any | None:
    """返回离线评测 Judge 模型，并允许保留 CLI 配置覆盖。"""
    return _create_model(
        role="judge",
        api_key=(
            Settings.JUDGE_API_KEY or Settings.LLM_API_KEY
            if api_key is None
            else api_key
        ),
        api_url=Settings.JUDGE_API_URL if api_url is None else api_url,
        model_name=Settings.JUDGE_MODEL if model_name is None else model_name,
        temperature=0.0 if temperature is None else temperature,
        max_tokens=Settings.JUDGE_MAX_TOKENS if max_tokens is None else max_tokens,
        timeout=Settings.JUDGE_TIMEOUT if timeout is None else timeout,
        max_retries=(
            Settings.JUDGE_MAX_RETRIES if max_retries is None else max_retries
        ),
    )


def clear_model_factory_cache() -> None:
    """清理全部角色的模型缓存，供测试或配置刷新使用。"""
    get_answer_model.cache_clear()
    get_router_model.cache_clear()
    get_query_rewrite_model.cache_clear()
    get_decompose_model.cache_clear()
    get_compare_model.cache_clear()
    get_judge_model.cache_clear()
