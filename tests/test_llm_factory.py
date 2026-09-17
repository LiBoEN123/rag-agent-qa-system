"""统一 LLM Factory 的配置映射与缓存测试。"""

import socket

import pytest
from langchain_openai import ChatOpenAI

from app.llm import factory
from app.llm.errors import ModelCallError


class FakeChatOpenAI:
    """记录构造参数，避免单元测试连接真实模型服务。"""

    instances: list["FakeChatOpenAI"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)


@pytest.fixture(autouse=True)
def clear_factory_cache(monkeypatch):
    """隔离各用例对 Settings 和模型缓存的修改。"""
    FakeChatOpenAI.instances.clear()
    monkeypatch.setattr(factory, "ChatOpenAI", FakeChatOpenAI)
    factory.clear_model_factory_cache()
    yield
    factory.clear_model_factory_cache()


def test_answer_model_uses_existing_configuration(settings_env):
    """Answer 模型应完整传递名称、base URL、温度和调用控制参数。"""
    settings_env(
        llm_api_key="answer-key",
        llm_api_url="https://models.example.test/v1/chat/completions",
        model_name="answer-model",
        temperature=0.25,
        llm_max_tokens=987,
        llm_timeout=42,
        llm_max_retries=3,
    )

    model = factory.get_answer_model()

    assert model.kwargs == {
        "model": "answer-model",
        "api_key": "answer-key",
        "base_url": "https://models.example.test/v1",
        "temperature": 0.25,
        "max_tokens": 987,
        "timeout": 42,
        "max_retries": 3,
    }


@pytest.mark.parametrize(
    ("getter_name", "key_name", "url_name", "model_name"),
    [
        (
            "get_query_rewrite_model",
            "QUERY_REWRITE_API_KEY",
            "QUERY_REWRITE_API_URL",
            "QUERY_REWRITE_MODEL_NAME",
        ),
        ("get_judge_model", "JUDGE_API_KEY", "JUDGE_API_URL", "JUDGE_MODEL"),
    ],
)
def test_role_models_reuse_answer_key_when_independent_key_is_empty(
    settings_env,
    getter_name,
    key_name,
    url_name,
    model_name,
):
    """保持 Rewrite、Judge 已有的 Answer Key 复用策略。"""
    settings_env(
        llm_api_key="shared-answer-key",
        **{
            key_name.lower(): "",
            url_name.lower(): "https://models.example.test/v1",
            model_name.lower(): "role-model",
        },
    )

    model = getattr(factory, getter_name)()

    assert model.kwargs["api_key"] == "shared-answer-key"


def test_decompose_model_keeps_independent_key_semantics(settings_env):
    """Decompose 原先允许无独立 Key，Factory 不改变该配置含义。"""
    settings_env(
        llm_api_key="shared-answer-key",
        decompose_api_key="",
        decompose_api_url="http://localhost:11434/v1",
        decompose_model_name="qwen-test",
    )

    model = factory.get_decompose_model()

    assert model.kwargs["api_key"] == "not-required"
    assert model.kwargs["temperature"] == 0.0


def test_judge_model_accepts_cli_configuration_overrides():
    """离线 Judge 的 CLI 配置覆盖会完整传入 ChatOpenAI。"""
    model = factory.get_judge_model(
        api_key="cli-key",
        api_url="https://judge.example.test/v1/chat/completions",
        model_name="cli-judge",
        temperature=0.15,
        max_tokens=333,
        timeout=9,
        max_retries=4,
    )

    assert model.kwargs == {
        "model": "cli-judge",
        "api_key": "cli-key",
        "base_url": "https://judge.example.test/v1",
        "temperature": 0.15,
        "max_tokens": 333,
        "timeout": 9,
        "max_retries": 4,
    }


@pytest.mark.parametrize("missing_field", ["LLM_API_URL", "MODEL_NAME"])
def test_missing_answer_model_configuration_returns_none(
    settings_env,
    missing_field,
):
    """模型名或 URL 缺失时返回 None，供 Gateway 生成清晰配置错误。"""
    values = {
        "llm_api_url": "https://models.example.test/v1",
        "model_name": "answer-model",
        missing_field.lower(): "",
    }
    settings_env(**values)

    assert factory.get_answer_model() is None


def test_factory_cache_can_be_cleared(settings_env):
    """重复读取复用实例，显式清缓存后按新配置重新创建。"""
    settings_env(
        llm_api_url="https://models.example.test/v1",
        model_name="model-a",
    )

    first = factory.get_answer_model()
    second = factory.get_answer_model()
    settings_env(model_name="model-b")
    third = factory.get_answer_model()

    assert first is second
    assert third is not first
    assert third.kwargs["model"] == "model-b"
    assert len(FakeChatOpenAI.instances) == 2


def test_real_chat_openai_constructor_without_network(monkeypatch, settings_env):
    """锁定 SDK 接受 Factory 参数；仅构造客户端，并禁止网络连接。"""
    def deny_network(*args, **kwargs):
        raise AssertionError("constructor smoke must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(factory, "ChatOpenAI", ChatOpenAI)
    settings_env(
        llm_api_key="test-key",
        llm_api_url="http://localhost:12345/v1/chat/completions",
        model_name="test-model",
        temperature=0,
        llm_max_tokens=10,
        llm_timeout=1,
        llm_max_retries=0,
    )

    model = factory.get_answer_model()

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "test-model"
    assert model.openai_api_key.get_secret_value() == "test-key"
    assert model.openai_api_base == "http://localhost:12345/v1"
    assert model.temperature == 0
    assert model.max_tokens == 10
    assert model.request_timeout == 1
    assert model.max_retries == 0
    assert model.model_kwargs == {}


def test_constructor_failure_maps_to_model_call_error(monkeypatch, settings_env):
    """初始化异常仍统一映射为不可重试的 ModelCallError。"""
    failure = ValueError("invalid constructor configuration")

    def fail_constructor(**kwargs):
        raise failure

    monkeypatch.setattr(factory, "ChatOpenAI", fail_constructor)
    settings_env(
        llm_api_url="http://localhost:12345/v1",
        model_name="test-model",
    )

    with pytest.raises(ModelCallError) as error:
        factory.get_answer_model()

    assert error.value.user_message == "answer 模型客户端初始化失败，请检查模型配置。"
    assert error.value.detail == "ValueError"
    assert error.value.provider == "langchain"
    assert error.value.model_name == "test-model"
    assert error.value.retryable is False
    assert error.value.__cause__ is failure
