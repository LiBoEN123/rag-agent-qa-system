"""基于 LangChain 的统一模型调用网关。"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.common.logging import get_logger
from app.llm.errors import ModelCallError, StructuredOutputError


logger = get_logger("chat_service")
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _status_code(exc: BaseException) -> int | None:
    """从 OpenAI 或 httpx 风格异常中提取 HTTP 状态码。"""
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


def _build_model_call_error(exc: BaseException, model: Any) -> ModelCallError:
    """将底层模型异常统一映射为 ModelCallError。"""
    if isinstance(exc, ModelCallError):
        return exc
    return ModelCallError(
        _user_error_message(exc),
        detail=type(exc).__name__,
        provider="langchain",
        model_name=_model_name(model),
        retryable=_is_retryable(exc),
    )


def _is_structured_output_unsupported(exc: BaseException) -> bool:
    """仅识别明确的 structured output 能力不兼容错误。"""
    status = _status_code(exc)
    if _is_timeout_or_connection_error(exc):
        return False
    if status is not None and status not in {400, 422}:
        return False
    if status is None and isinstance(exc, NotImplementedError):
        return True

    message = str(exc).strip().lower()
    if (
        status is None
        and isinstance(exc, AttributeError)
        and "with_structured_output" in message
    ):
        return True

    normalized = re.sub(r"[\s_-]+", " ", message)
    capability_markers = (
        "structured output",
        "response format",
        "json schema",
    )
    unsupported_markers = (
        "unsupported",
        "not supported",
        "does not support",
        "unavailable",
        "not implemented",
        "unknown parameter",
        "unrecognized parameter",
    )
    explicitly_unsupported = bool(
        any(marker in normalized for marker in capability_markers)
        and any(marker in normalized for marker in unsupported_markers)
    )
    if status is None:
        return explicitly_unsupported
    return status in {400, 422} and explicitly_unsupported


def _extract_text_content(response: Any) -> str:
    """从 AIMessage 或兼容对象中提取文本 content。"""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
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


def _validate_schema(schema: type[SchemaT], value: Any) -> SchemaT:
    """兼容 Pydantic v1/v2 地校验结构化模型输出。"""
    if isinstance(value, schema):
        return value
    validator = getattr(schema, "model_validate", None)
    if callable(validator):
        return validator(value)
    return schema.parse_obj(value)


def _build_structured_output_error(
    exc: BaseException,
    schema: type[BaseModel],
) -> StructuredOutputError:
    """把解析或 Schema 校验失败转换成独立的结构化输出错误。"""
    return StructuredOutputError(
        "模型返回的结构化内容不符合预期。",
        detail=type(exc).__name__,
        schema_name=schema.__name__,
    )


def _is_native_output_validation_error(exc: BaseException) -> bool:
    """识别发生在原生结构化响应解析阶段的错误。"""
    return isinstance(
        exc,
        (json.JSONDecodeError, OutputParserException, ValidationError),
    )


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


class ModelGateway:
    """在统一消息协议上执行 LangChain 调用并规整结果与异常。"""

    # 保存模型用途角色和底层 LangChain 模型实例。
    def __init__(
        self,
        *,
        role: str,
        langchain_model: Any = None,
    ) -> None:
        self.role = str(role)
        self.langchain_model = langchain_model

    def _log_call(
        self,
        *,
        trace_id: str,
        invocation: str,
        prompt_chars: int,
        fallback: bool,
        error: BaseException | None = None,
    ) -> None:
        """记录不含 Prompt、响应和凭据的安全调用元数据。"""
        logger.info(
            "[trace_id=%s] [model_gateway] role=%s model=%s attempt=1 "
            "invocation=%s fallback=%s prompt_chars=%s error_type=%s",
            trace_id,
            self.role,
            _model_name(self.langchain_model),
            invocation,
            fallback,
            prompt_chars,
            type(error).__name__ if error is not None else "",
        )

    def invoke_text(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str = "",
    ) -> str:
        """调用 LangChain 文本模型并统一转换异常。"""
        if self.langchain_model is None:
            raise ModelCallError(
                "模型客户端不可用，请检查 LLM 配置。",
                detail="langchain model is not configured",
                provider="langchain",
                model_name="",
                retryable=False,
            )

        prompt_chars = sum(len(str(item.get("content", ""))) for item in messages)
        try:
            converted = _to_langchain_messages(messages)
            self._log_call(
                trace_id=trace_id,
                invocation="text",
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
                invocation="text",
                prompt_chars=prompt_chars,
                fallback=False,
                error=exc,
            )
            raise _build_model_call_error(exc, self.langchain_model) from exc

    def invoke_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[SchemaT],
        *,
        trace_id: str = "",
    ) -> SchemaT:
        """优先使用 LangChain 结构化输出，不兼容时降级为文本 JSON 解析。"""
        if self.langchain_model is not None:
            prompt_chars = sum(len(str(item.get("content", ""))) for item in messages)
            try:
                structured_model = self.langchain_model.with_structured_output(schema)
            except Exception as exc:
                fallback = _is_structured_output_unsupported(exc)
                self._log_call(
                    trace_id=trace_id,
                    invocation="structured",
                    prompt_chars=prompt_chars,
                    fallback=fallback,
                    error=exc,
                )
                if not fallback:
                    raise _build_model_call_error(
                        exc,
                        self.langchain_model,
                    ) from exc
            else:
                self._log_call(
                    trace_id=trace_id,
                    invocation="structured",
                    prompt_chars=prompt_chars,
                    fallback=False,
                )
                try:
                    value = structured_model.invoke(_to_langchain_messages(messages))
                except Exception as exc:
                    fallback = _is_structured_output_unsupported(exc)
                    self._log_call(
                        trace_id=trace_id,
                        invocation="structured",
                        prompt_chars=prompt_chars,
                        fallback=fallback,
                        error=exc,
                    )
                    if not fallback:
                        if _is_native_output_validation_error(exc):
                            raise _build_structured_output_error(exc, schema) from exc
                        raise _build_model_call_error(
                            exc,
                            self.langchain_model,
                        ) from exc
                else:
                    try:
                        return _validate_schema(schema, value)
                    except Exception as exc:
                        raise _build_structured_output_error(exc, schema) from exc

        raw_text = self.invoke_text(messages, trace_id=trace_id)
        try:
            return _validate_schema(schema, _parse_json_object(raw_text))
        except Exception as exc:
            raise _build_structured_output_error(exc, schema) from exc
