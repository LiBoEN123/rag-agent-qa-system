"""LangChain 优先、legacy 可回退的统一模型调用网关。"""

from __future__ import annotations

import json
import inspect
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.common.logging import get_logger
from app.config import Settings
from app.llm.errors import ModelCallError


logger = get_logger("chat_service")

LegacyInvoke = Callable[..., str]


def _status_code(exc: BaseException) -> int | None:
    """从 OpenAI、httpx 或 requests 风格异常中提取 HTTP 状态码。"""
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_timeout_or_connection_error(exc: BaseException) -> bool:
    """按异常类型识别无需依赖具体 SDK 版本的网络类错误。"""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name or "connect" in name


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否属于可重试的临时服务故障。"""
    status = _status_code(exc)
    return bool((status is not None and status >= 500) or _is_timeout_or_connection_error(exc))


def _should_fallback(exc: BaseException) -> bool:
    """决定 LangChain 失败后是否值得再尝试 legacy 客户端。"""
    status = _status_code(exc)
    if status in {400, 401, 403, 404, 429}:
        return False
    return True


def _user_error_message(exc: BaseException) -> str:
    """把底层调用异常转换成稳定、可读且不含敏感信息的用户提示。"""
    status = _status_code(exc)
    messages = {
        400: "模型请求参数错误，请检查 model、messages、max_tokens 等配置。",
        401: "模型调用未授权，请检查 API Key 是否正确。",
        403: "模型调用被拒绝，请检查账号权限或模型访问权限。",
        404: "模型接口地址或模型名称不存在，请检查 LLM_API_URL 和 MODEL_NAME。",
        429: "模型请求过于频繁或额度不足，请稍后重试或检查账户额度。",
    }
    if status in messages:
        return messages[status]
    if status is not None and status >= 500:
        return "模型服务端异常，请稍后重试。"
    if _is_timeout_or_connection_error(exc):
        if "timeout" in type(exc).__name__.lower() or isinstance(exc, TimeoutError):
            return "模型调用超时，请检查模型服务是否响应过慢，或适当调大 LLM_TIMEOUT。"
        return "无法连接模型服务，请检查 LLM_API_URL 是否正确，以及模型服务是否已启动。"
    return "模型请求失败，请检查模型客户端和服务配置。"


def _model_name(model: Any) -> str:
    """安全读取 LangChain 模型名称，仅用于日志和异常元数据。"""
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, "")
        if value:
            return str(value)
    return ""


def _extract_text_content(response: Any) -> str:
    """从 AIMessage 或兼容对象中提取文本 content。"""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("模型返回格式不符合预期，缺少文本 content。")


def _to_langchain_messages(messages: list[dict[str, str]]) -> list[Any]:
    """把项目统一消息字典转换为 LangChain 消息对象。"""
    message_types = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    converted = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        message_type = message_types.get(role)
        if message_type is None:
            raise ValueError(f"不支持的消息角色: {role or '<empty>'}")
        converted.append(message_type(content=str(message.get("content", ""))))
    return converted


def _validate_schema(schema: type[BaseModel], value: Any) -> BaseModel:
    """兼容 Pydantic v1/v2 地校验结构化模型输出。"""
    if isinstance(value, schema):
        return value
    validator = getattr(schema, "model_validate", None)
    if callable(validator):
        return validator(value)
    return schema.parse_obj(value)


def _parse_json_object(text: str) -> dict[str, Any]:
    """从纯 JSON、Markdown 代码块或夹杂说明的文本中提取 JSON 对象。"""
    content = (text or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content).strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型未返回 JSON 对象")
        value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型结构化输出的 JSON 根节点不是对象")
    return value


def _call_legacy_callable(
    legacy_invoke: LegacyInvoke,
    messages: list[dict[str, str]],
    trace_id: str,
) -> Any:
    """兼容 legacy callable 的无参、仅 messages 和 messages+trace_id 签名。"""
    try:
        parameters = inspect.signature(legacy_invoke).parameters
    except (TypeError, ValueError):
        return legacy_invoke(messages, trace_id=trace_id)

    if not parameters:
        return legacy_invoke()
    if "trace_id" in parameters or any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in parameters.values()
    ):
        return legacy_invoke(messages, trace_id=trace_id)
    positional = [
        item
        for item in parameters.values()
        if item.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if len(positional) >= 2:
        return legacy_invoke(messages, trace_id)
    return legacy_invoke(messages)


class ModelGateway:
    """在统一消息协议上执行 LangChain 调用并按错误类型回退 legacy。"""

    def __init__(
        self,
        *,
        role: str,
        langchain_model: Any = None,
        legacy_invoke: LegacyInvoke | None = None,
        client_mode: str | None = None,
    ) -> None:
        self.role = str(role)
        self.langchain_model = langchain_model
        self.legacy_invoke = legacy_invoke
        self.client_mode = (client_mode or Settings.LLM_CLIENT_MODE or "langchain").strip().lower()
        if self.client_mode not in {"langchain", "legacy"}:
            raise ModelCallError(
                "LLM_CLIENT_MODE 配置无效，仅支持 langchain 或 legacy。",
                detail="invalid client mode",
                provider="configuration",
                model_name=_model_name(langchain_model),
                retryable=False,
            )

    def _log_call(
        self,
        *,
        trace_id: str,
        client_mode: str,
        prompt_chars: int,
        fallback: bool,
        error: BaseException | None = None,
    ) -> None:
        """记录不含 Prompt、响应和凭据的安全调用元数据。"""
        logger.info(
            "[trace_id=%s] [model_gateway] role=%s model=%s attempt=1 "
            "client_mode=%s fallback=%s prompt_chars=%s error_type=%s",
            trace_id,
            self.role,
            _model_name(self.langchain_model),
            client_mode,
            fallback,
            prompt_chars,
            type(error).__name__ if error is not None else "",
        )

    def _invoke_legacy(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        fallback: bool,
    ) -> str:
        """调用注入的 legacy 客户端并统一其异常。"""
        if self.legacy_invoke is None:
            raise ModelCallError(
                "模型客户端不可用，请检查 LLM 配置。",
                detail="legacy fallback is not configured",
                provider="legacy",
                model_name=_model_name(self.langchain_model),
                retryable=False,
            )
        prompt_chars = sum(len(str(item.get("content", ""))) for item in messages)
        self._log_call(
            trace_id=trace_id,
            client_mode="legacy",
            prompt_chars=prompt_chars,
            fallback=fallback,
        )
        try:
            return _extract_text_content(
                _call_legacy_callable(self.legacy_invoke, messages, trace_id)
            )
        except ModelCallError:
            raise
        except Exception as exc:
            raise ModelCallError(
                _user_error_message(exc),
                detail=type(exc).__name__,
                provider="legacy",
                model_name=_model_name(self.langchain_model),
                retryable=_is_retryable(exc),
            ) from exc

    def invoke_text(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str = "",
    ) -> str:
        """调用文本模型，临时故障或兼容性故障时按策略回退 legacy。"""
        if self.client_mode == "legacy":
            return self._invoke_legacy(messages, trace_id=trace_id, fallback=False)

        if self.langchain_model is None:
            return self._invoke_legacy(messages, trace_id=trace_id, fallback=True)

        prompt_chars = sum(len(str(item.get("content", ""))) for item in messages)
        try:
            converted = _to_langchain_messages(messages)
            self._log_call(
                trace_id=trace_id,
                client_mode="langchain",
                prompt_chars=prompt_chars,
                fallback=False,
            )
            response = self.langchain_model.invoke(converted)
            return _extract_text_content(response)
        except ModelCallError:
            raise
        except Exception as exc:
            self._log_call(
                trace_id=trace_id,
                client_mode="langchain",
                prompt_chars=prompt_chars,
                fallback=False,
                error=exc,
            )
            if self.legacy_invoke is not None and _should_fallback(exc):
                return self._invoke_legacy(messages, trace_id=trace_id, fallback=True)
            raise ModelCallError(
                _user_error_message(exc),
                detail=type(exc).__name__,
                provider="langchain",
                model_name=_model_name(self.langchain_model),
                retryable=_is_retryable(exc),
            ) from exc

    def invoke_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        *,
        trace_id: str = "",
    ) -> BaseModel:
        """优先使用 LangChain 结构化输出，不兼容时降级为文本 JSON 解析。"""
        if self.client_mode == "langchain" and self.langchain_model is not None:
            prompt_chars = sum(len(str(item.get("content", ""))) for item in messages)
            try:
                structured_model = self.langchain_model.with_structured_output(schema)
                self._log_call(
                    trace_id=trace_id,
                    client_mode="langchain_structured",
                    prompt_chars=prompt_chars,
                    fallback=False,
                )
                value = structured_model.invoke(_to_langchain_messages(messages))
                return _validate_schema(schema, value)
            except Exception as exc:
                self._log_call(
                    trace_id=trace_id,
                    client_mode="langchain_structured",
                    prompt_chars=prompt_chars,
                    fallback=True,
                    error=exc,
                )
                if _status_code(exc) in {401, 403, 404, 429}:
                    raise ModelCallError(
                        _user_error_message(exc),
                        detail=type(exc).__name__,
                        provider="langchain",
                        model_name=_model_name(self.langchain_model),
                        retryable=_is_retryable(exc),
                    ) from exc

        raw_text = self.invoke_text(messages, trace_id=trace_id)
        try:
            return _validate_schema(schema, _parse_json_object(raw_text))
        except Exception as exc:
            raise ModelCallError(
                "模型返回的结构化内容不符合预期。",
                detail=type(exc).__name__,
                provider=self.client_mode,
                model_name=_model_name(self.langchain_model),
                retryable=False,
            ) from exc
