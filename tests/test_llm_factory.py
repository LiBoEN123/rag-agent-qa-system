"""统一 LLM Factory 的配置映射与缓存测试。"""

import pytest

from app.llm import factory


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


def test_answer_model_uses_existing_configuration(monkeypatch):
    """Answer 模型应完整传递名称、base URL、温度和调用控制参数。"""
    monkeypatch.setattr(factory.Settings, "LLM_API_KEY", "answer-key")
    monkeypatch.setattr(
        factory.Settings,
        "LLM_API_URL",
        "https://models.example.test/v1/chat/completions",
    )
    monkeypatch.setattr(factory.Settings, "MODEL_NAME", "answer-model")
    monkeypatch.setattr(factory.Settings, "TEMPERATURE", 0.25)
    monkeypatch.setattr(factory.Settings, "LLM_MAX_TOKENS", 987)
    monkeypatch.setattr(factory.Settings, "LLM_TIMEOUT", 42)
    monkeypatch.setattr(factory.Settings, "LLM_MAX_RETRIES", 3)

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
        ("get_router_model", "ROUTER_API_KEY", "ROUTER_API_URL", "ROUTER_MODEL_NAME"),
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
    monkeypatch,
    getter_name,
    key_name,
    url_name,
    model_name,
):
    """保持 Router、Rewrite、Judge 已有的 Answer Key 复用策略。"""
    monkeypatch.setattr(factory.Settings, "LLM_API_KEY", "shared-answer-key")
    monkeypatch.setattr(factory.Settings, key_name, "")
    monkeypatch.setattr(factory.Settings, url_name, "https://models.example.test/v1")
    monkeypatch.setattr(factory.Settings, model_name, "role-model")

    model = getattr(factory, getter_name)()

    assert model.kwargs["api_key"] == "shared-answer-key"


def test_decompose_model_keeps_independent_key_semantics(monkeypatch):
    """Decompose 原先允许无独立 Key，Factory 不改变该配置含义。"""
    monkeypatch.setattr(factory.Settings, "LLM_API_KEY", "shared-answer-key")
    monkeypatch.setattr(factory.Settings, "DECOMPOSE_API_KEY", "")
    monkeypatch.setattr(factory.Settings, "DECOMPOSE_API_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(factory.Settings, "DECOMPOSE_MODEL_NAME", "qwen-test")

    model = factory.get_decompose_model()

    assert model.kwargs["api_key"] == "not-required"
    assert model.kwargs["temperature"] == 0.0


@pytest.mark.parametrize("missing_field", ["LLM_API_URL", "MODEL_NAME"])
def test_missing_answer_model_configuration_returns_none(monkeypatch, missing_field):
    """模型名或 URL 缺失时返回 None，便于 Gateway 清晰地选择 fallback。"""
    monkeypatch.setattr(factory.Settings, "LLM_API_URL", "https://models.example.test/v1")
    monkeypatch.setattr(factory.Settings, "MODEL_NAME", "answer-model")
    monkeypatch.setattr(factory.Settings, missing_field, "")

    assert factory.get_answer_model() is None


def test_factory_cache_can_be_cleared(monkeypatch):
    """重复读取复用实例，显式清缓存后按新配置重新创建。"""
    monkeypatch.setattr(factory.Settings, "LLM_API_URL", "https://models.example.test/v1")
    monkeypatch.setattr(factory.Settings, "MODEL_NAME", "model-a")

    first = factory.get_answer_model()
    second = factory.get_answer_model()
    monkeypatch.setattr(factory.Settings, "MODEL_NAME", "model-b")
    factory.clear_model_factory_cache()
    third = factory.get_answer_model()

    assert first is second
    assert third is not first
    assert third.kwargs["model"] == "model-b"
    assert len(FakeChatOpenAI.instances) == 2
