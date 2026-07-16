"""统一 ModelGateway 及 RAG generation 接入的回归测试。"""

import pytest
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


def test_text_call_extracts_ai_message_and_converts_all_roles():
    """文本调用提取 AIMessage.content 并正确转换三种消息角色。"""
    model = RecordingModel(AIMessage(content="model answer"))
    gateway = ModelGateway(role="answer", langchain_model=model)

    result = gateway.invoke_text(MESSAGES, trace_id="trace-1")

    assert result == "model answer"
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[0][1], HumanMessage)
    assert isinstance(model.calls[0][2], AIMessage)
    assert [message.content for message in model.calls[0]] == [
        "system rules",
        "question",
        "previous answer",
    ]


def test_text_call_accepts_plain_string_response():
    """兼容模型直接返回字符串的情况。"""
    gateway = ModelGateway(
        role="answer",
        langchain_model=RecordingModel(response="plain answer"),
    )

    assert gateway.invoke_text(MESSAGES) == "plain answer"


def test_text_call_safely_normalizes_complex_content_blocks():
    """复杂 content 仅提取文本块并忽略非文本块。"""

    class ContentResponse:
        content = [
            {"type": "text", "text": "first"},
            {"type": "image", "image_url": "secret-local-path"},
            " second",
            {"content": " third"},
        ]

    gateway = ModelGateway(
        role="answer",
        langchain_model=RecordingModel(response=ContentResponse()),
    )

    assert gateway.invoke_text(MESSAGES) == "first second third"


def test_unsupported_message_role_maps_to_model_call_error():
    """非法项目消息不会以裸 ValueError 穿透业务层。"""
    model = RecordingModel(AIMessage(content="unused"))
    gateway = ModelGateway(role="answer", langchain_model=model)

    with pytest.raises(ModelCallError) as error:
        gateway.invoke_text([{"role": "tool", "content": "result"}])

    assert error.value.provider == "langchain"
    assert model.calls == []


def test_missing_model_returns_clear_model_call_error():
    """未配置 LangChain 模型时返回明确的统一错误。"""
    gateway = ModelGateway(role="answer", langchain_model=None)

    with pytest.raises(ModelCallError, match="模型客户端不可用") as error:
        gateway.invoke_text(MESSAGES)

    assert error.value.provider == "langchain"
    assert error.value.retryable is False


@pytest.mark.parametrize(
    ("exception", "message", "retryable"),
    [
        (FakeStatusError(400), "请求参数错误", False),
        (FakeStatusError(401), "未授权", False),
        (FakeStatusError(403), "被拒绝", False),
        (FakeStatusError(404), "不存在", False),
        (FakeStatusError(429), "过于频繁", False),
        (FakeStatusError(503), "服务端异常", True),
        (TimeoutError("slow"), "调用超时", True),
        (ConnectionError("offline"), "无法连接", True),
    ],
)
def test_model_errors_are_classified_and_mapped(exception, message, retryable):
    """常见供应商错误统一映射为 ModelCallError。"""
    gateway = ModelGateway(
        role="answer",
        langchain_model=RecordingModel(error=exception),
    )

    with pytest.raises(ModelCallError, match=message) as error:
        gateway.invoke_text(MESSAGES)

    assert error.value.provider == "langchain"
    assert error.value.retryable is retryable
    assert error.value.detail == type(exception).__name__


def test_gateway_logs_trace_id_and_redacts_exception_text(monkeypatch):
    """安全日志保留 trace_id 和异常类型，但不记录异常文本中的 API Key。"""
    api_key = "test-super-secret-api-key"
    records = []

    class CaptureLogger:
        def info(self, message, *args):
            records.append(message % args)

    from app.llm import gateway as gateway_module

    monkeypatch.setattr(gateway_module, "logger", CaptureLogger())
    gateway = ModelGateway(
        role="answer",
        langchain_model=RecordingModel(
            error=RuntimeError(f"provider rejected {api_key}")
        ),
    )

    with pytest.raises(ModelCallError):
        gateway.invoke_text(MESSAGES, trace_id="safe-trace")

    logs = "\n".join(records)
    assert "safe-trace" in logs
    assert "RuntimeError" in logs
    assert api_key not in logs


def test_structured_call_returns_schema_instance():
    """服务支持 structured output 时直接返回 Pydantic 实例。"""
    expected = AnswerOutput(answer="structured", references=["Reference 1"])

    class StructuredRunnable:
        def invoke(self, messages):
            assert isinstance(messages[0], SystemMessage)
            return expected

    class StructuredModel(RecordingModel):
        def with_structured_output(self, schema):
            assert schema is AnswerOutput
            return StructuredRunnable()

    gateway = ModelGateway(role="answer", langchain_model=StructuredModel())

    assert gateway.invoke_structured(MESSAGES, AnswerOutput) is expected


def test_structured_incompatibility_degrades_to_same_model_text_json():
    """结构化能力不兼容时退回同一个 LangChain 模型的文本 JSON。"""

    class TextFallbackModel(RecordingModel):
        def with_structured_output(self, schema):
            raise NotImplementedError("structured output is unsupported")

    model = TextFallbackModel(
        response=AIMessage(
            content='```json\n{"answer":"text fallback","references":[]}\n```'
        )
    )
    gateway = ModelGateway(role="answer", langchain_model=model)

    result = gateway.invoke_structured(MESSAGES, AnswerOutput)

    assert result.answer == "text fallback"
    assert result.references == []
    assert len(model.calls) == 1


def test_missing_structured_method_degrades_to_same_model_text_json():
    """模型缺少 structured 方法时退回同一模型的文本 JSON。"""
    model = RecordingModel(
        response=AIMessage(content='{"answer":"attribute fallback","references":[]}')
    )
    gateway = ModelGateway(role="answer", langchain_model=model)

    result = gateway.invoke_structured(MESSAGES, AnswerOutput)

    assert result.answer == "attribute fallback"
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "exception",
    [
        FakeStatusError(401),
        FakeStatusError(429),
        FakeStatusError(503),
        TimeoutError("slow"),
        ConnectionError("offline"),
    ],
)
def test_structured_transport_errors_do_not_retry_as_text(exception):
    """认证、限流、服务端和网络错误不会重复发送文本请求。"""

    class StructuredModel(RecordingModel):
        def with_structured_output(self, schema):
            raise exception

    model = StructuredModel(response=AIMessage(content="must not run"))
    gateway = ModelGateway(role="answer", langchain_model=model)

    with pytest.raises(ModelCallError):
        gateway.invoke_structured(MESSAGES, AnswerOutput)

    assert model.calls == []


def test_invalid_native_structured_result_does_not_retry_as_text():
    """原生 structured 返回值校验失败时不再执行文本模型。"""

    class InvalidStructuredRunnable:
        def invoke(self, messages):
            return {"answer": 123, "references": "not-a-list"}

    class StructuredModel(RecordingModel):
        def with_structured_output(self, schema):
            return InvalidStructuredRunnable()

    model = StructuredModel(response=AIMessage(content="must not run"))
    gateway = ModelGateway(role="answer", langchain_model=model)

    with pytest.raises(ModelCallError):
        gateway.invoke_structured(MESSAGES, AnswerOutput)

    assert model.calls == []


def test_invalid_structured_text_maps_to_model_call_error():
    """文本 fallback 的非法 JSON 继续转换为统一结构化输出错误。"""

    class TextFallbackModel(RecordingModel):
        def with_structured_output(self, schema):
            raise NotImplementedError

    gateway = ModelGateway(
        role="answer",
        langchain_model=TextFallbackModel(response=AIMessage(content="not-json")),
    )

    with pytest.raises(ModelCallError, match="结构化内容不符合预期") as error:
        gateway.invoke_structured(MESSAGES, AnswerOutput)

    assert error.value.provider == "langchain"
    assert error.value.retryable is False


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


def test_generation_call_llm_uses_answer_factory_and_keeps_prompt(monkeypatch):
    """Answer 调用使用对应工厂，Prompt 内容和返回字符串保持不变。"""
    model = RecordingModel(AIMessage(content="answer content"))
    monkeypatch.setattr(generation, "get_answer_model", lambda: model)

    result = generation.call_llm("question", "context", trace_id="answer-trace")

    assert result == "answer content"
    assert isinstance(result, str)
    assert len(model.calls) == 1
    assert [message.content for message in model.calls[0]] == [
        message["content"]
        for message in generation._build_answer_messages("question", "context")
    ]


def test_parse_llm_output_keeps_existing_fallback_behavior():
    """统一调用层不改变现有回答 JSON 解析规则。"""
    assert generation.parse_llm_output(
        '```json\n{"answer":"ok","references":"invalid"}\n```'
    ) == {"ok": True, "answer": "ok", "references": []}
    assert generation.parse_llm_output("not-json")["ok"] is False
