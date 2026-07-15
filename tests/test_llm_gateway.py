"""统一 ModelGateway 及 RAG generation 接入的回归测试。"""

import inspect

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm.errors import ModelCallError
from app.llm.gateway import ModelGateway
from app.llm.schemas import AnswerOutput
from app.rag import generation


MESSAGES = [
    {"role": "system", "content": "system rules"},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "previous answer"},
]


class FakeStatusError(Exception):
    """模拟 OpenAI SDK 携带 status_code 的异常。"""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class RecordingModel:
    """记录 LangChain 消息并返回预设结果或异常。"""

    model_name = "test-model"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


def test_langchain_text_call_extracts_ai_message_and_converts_roles():
    """LangChain 成功时提取 AIMessage.content 并正确转换三种角色。"""
    model = RecordingModel(AIMessage(content="model answer"))
    gateway = ModelGateway(role="answer", langchain_model=model, client_mode="langchain")

    result = gateway.invoke_text(MESSAGES, trace_id="trace-1")

    assert result == "model answer"
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[0][1], HumanMessage)
    assert isinstance(model.calls[0][2], AIMessage)


def test_legacy_mode_only_calls_legacy_client():
    """legacy 模式不得调用 LangChain 模型。"""
    model = RecordingModel(error=AssertionError("langchain must not run"))
    calls = []

    def legacy(messages, trace_id=""):
        calls.append((messages, trace_id))
        return "legacy answer"

    gateway = ModelGateway(
        role="answer",
        langchain_model=model,
        legacy_invoke=legacy,
        client_mode="legacy",
    )

    assert gateway.invoke_text(MESSAGES, trace_id="legacy-trace") == "legacy answer"
    assert model.calls == []
    assert calls == [(MESSAGES, "legacy-trace")]


def test_legacy_callable_without_trace_id_is_supported():
    """旧客户端只接收 messages 时也能通过 Gateway 调用。"""
    gateway = ModelGateway(
        role="answer",
        legacy_invoke=lambda messages: f"legacy received {len(messages)} messages",
        client_mode="legacy",
    )

    assert gateway.invoke_text(MESSAGES) == "legacy received 3 messages"


def test_langchain_timeout_falls_back_to_legacy():
    """LangChain 超时属于临时故障，应允许 legacy fallback。"""
    model = RecordingModel(error=TimeoutError("slow service"))
    gateway = ModelGateway(
        role="answer",
        langchain_model=model,
        legacy_invoke=lambda messages, trace_id="": "fallback answer",
        client_mode="langchain",
    )

    assert gateway.invoke_text(MESSAGES) == "fallback answer"
    assert len(model.calls) == 1


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 429])
def test_non_retryable_http_errors_do_not_call_legacy(status_code):
    """认证、参数、地址和限流错误不应通过相同配置做无意义二次调用。"""
    model = RecordingModel(error=FakeStatusError(status_code))
    legacy_calls = []
    gateway = ModelGateway(
        role="answer",
        langchain_model=model,
        legacy_invoke=lambda messages, trace_id="": legacy_calls.append(messages),
        client_mode="langchain",
    )

    with pytest.raises(ModelCallError) as error:
        gateway.invoke_text(MESSAGES)

    assert error.value.retryable is False
    assert len(model.calls) == 1
    assert legacy_calls == []


def test_server_error_can_fallback_successfully():
    """5xx 服务错误允许 legacy fallback 并返回其结果。"""
    gateway = ModelGateway(
        role="answer",
        langchain_model=RecordingModel(error=FakeStatusError(503)),
        legacy_invoke=lambda messages, trace_id="": "legacy recovered",
        client_mode="langchain",
    )

    assert gateway.invoke_text(MESSAGES) == "legacy recovered"


def test_both_clients_fail_with_unified_error():
    """LangChain 和 legacy 都失败后必须抛出统一 ModelCallError。"""
    gateway = ModelGateway(
        role="answer",
        langchain_model=RecordingModel(error=TimeoutError("slow")),
        legacy_invoke=lambda messages, trace_id="": (_ for _ in ()).throw(
            ConnectionError("offline")
        ),
        client_mode="langchain",
    )

    with pytest.raises(ModelCallError) as error:
        gateway.invoke_text(MESSAGES)

    assert error.value.provider == "legacy"
    assert error.value.retryable is True
    assert error.value.user_message


def test_structured_call_returns_schema_instance():
    """兼容服务支持 LangChain structured output 时直接返回 Pydantic 实例。"""
    expected = AnswerOutput(answer="structured", references=["Reference 1"])

    class StructuredRunnable:
        def invoke(self, messages):
            return expected

    class StructuredModel(RecordingModel):
        def with_structured_output(self, schema):
            assert schema is AnswerOutput
            return StructuredRunnable()

    gateway = ModelGateway(
        role="answer",
        langchain_model=StructuredModel(),
        client_mode="langchain",
    )

    result = gateway.invoke_structured(MESSAGES, AnswerOutput)

    assert result is expected


def test_structured_incompatibility_degrades_to_text_json():
    """结构化能力不兼容时退回普通文本调用和安全 JSON 解析。"""

    class TextFallbackModel(RecordingModel):
        def with_structured_output(self, schema):
            raise NotImplementedError("structured output is unsupported")

    model = TextFallbackModel(
        response=AIMessage(
            content='```json\n{"answer":"text fallback","references":[]}\n```'
        )
    )
    gateway = ModelGateway(role="answer", langchain_model=model, client_mode="langchain")

    result = gateway.invoke_structured(MESSAGES, AnswerOutput)

    assert result.answer == "text fallback"
    assert result.references == []


def test_gateway_logs_do_not_include_api_key(monkeypatch):
    """安全日志仅记录异常类型，不记录异常文本中的 API Key。"""
    api_key = "test-super-secret-api-key"
    records = []

    class CaptureLogger:
        def info(self, message, *args):
            records.append(message % args)

    from app.llm import gateway as gateway_module

    monkeypatch.setattr(gateway_module, "logger", CaptureLogger())
    model = RecordingModel(error=RuntimeError(f"provider rejected {api_key}"))
    gateway = ModelGateway(
        role="answer",
        langchain_model=model,
        legacy_invoke=lambda messages, trace_id="": "safe fallback",
        client_mode="langchain",
    )

    assert gateway.invoke_text(MESSAGES) == "safe fallback"
    assert api_key not in "\n".join(records)
    assert "RuntimeError" in "\n".join(records)


def test_model_call_error_redacts_api_key_from_user_and_debug_messages(monkeypatch):
    """统一异常的用户提示与调试详情都不能保存环境中的 API Key。"""
    monkeypatch.setenv("LLM_API_KEY", "environment-secret-key")

    error = ModelCallError(
        "request failed with environment-secret-key",
        detail="Authorization: Bearer environment-secret-key",
    )

    assert "environment-secret-key" not in error.user_message
    assert "environment-secret-key" not in error.detail
    assert "[REDACTED]" in error.user_message


def test_generation_call_llm_signature_is_unchanged():
    """RAG 对外调用参数仍是 question、context、trace_id。"""
    assert list(inspect.signature(generation.call_llm).parameters) == [
        "question",
        "context",
        "trace_id",
    ]


def test_generation_legacy_mode_returns_original_string(monkeypatch):
    """RAG legacy 模式仍使用原 requests payload 并返回 content 字符串。"""
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "legacy content"}}]}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(generation.Settings, "LLM_CLIENT_MODE", "legacy")
    monkeypatch.setattr(generation.Settings, "LLM_MAX_RETRIES", 0)
    monkeypatch.setattr(generation.requests, "post", fake_post)

    result = generation.call_llm("question", "context", trace_id="legacy")

    assert result == "legacy content"
    assert isinstance(result, str)
    assert captured["payload"]["messages"] == generation._build_answer_messages(
        "question", "context"
    )


def test_generation_langchain_mode_uses_gateway_model(monkeypatch):
    """RAG langchain 模式经过 Gateway，成功时不触发 legacy。"""
    model = RecordingModel(AIMessage(content="langchain content"))
    monkeypatch.setattr(generation.Settings, "LLM_CLIENT_MODE", "langchain")
    monkeypatch.setattr(generation, "get_answer_model", lambda: model)
    monkeypatch.setattr(
        generation,
        "_legacy_call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy should not run")
        ),
    )

    assert generation.call_llm("question", "context") == "langchain content"
    assert len(model.calls) == 1


def test_generation_timeout_message_remains_readable(monkeypatch):
    """legacy 超时仍转换成原有可读提示。"""
    monkeypatch.setattr(generation.Settings, "LLM_CLIENT_MODE", "legacy")
    monkeypatch.setattr(generation.Settings, "LLM_MAX_RETRIES", 0)
    monkeypatch.setattr(
        generation.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout("slow")),
    )

    with pytest.raises(ModelCallError, match="模型调用超时"):
        generation.call_llm("question", "context")


@pytest.mark.parametrize("status_code", [401, 404, 429])
def test_generation_non_retryable_status_is_not_retried(monkeypatch, status_code):
    """legacy 的 401、404、429 继续保持立即失败行为。"""
    calls = []

    class FakeResponse:
        text = "failure"

        def __init__(self):
            self.status_code = status_code

    monkeypatch.setattr(generation.Settings, "LLM_CLIENT_MODE", "legacy")
    monkeypatch.setattr(generation.Settings, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(
        generation.requests,
        "post",
        lambda *args, **kwargs: calls.append(1) or FakeResponse(),
    )

    with pytest.raises(ModelCallError):
        generation.call_llm("question", "context")

    assert len(calls) == 1


def test_generation_server_error_retries_then_succeeds(monkeypatch):
    """legacy 5xx 仍按 LLM_MAX_RETRIES 重试。"""
    calls = []

    class FakeResponse:
        text = "server error"

        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {"choices": [{"message": {"content": "recovered"}}]}

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(500 if len(calls) == 1 else 200)

    monkeypatch.setattr(generation.Settings, "LLM_CLIENT_MODE", "legacy")
    monkeypatch.setattr(generation.Settings, "LLM_MAX_RETRIES", 1)
    monkeypatch.setattr(generation.Settings, "LLM_RETRY_SLEEP", 0)
    monkeypatch.setattr(generation.requests, "post", fake_post)

    assert generation.call_llm("question", "context") == "recovered"
    assert len(calls) == 2


def test_parse_llm_output_keeps_existing_fallback_behavior():
    """统一调用层不改变现有回答 JSON 解析规则。"""
    assert generation.parse_llm_output(
        '```json\n{"answer":"ok","references":"invalid"}\n```'
    ) == {"ok": True, "answer": "ok", "references": []}
    assert generation.parse_llm_output("not-json")["ok"] is False
