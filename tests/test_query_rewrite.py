import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.agent import query_rewrite
from app.llm.errors import StructuredOutputError
from app.llm.schemas import QueryRewriteOutput


HISTORY = [
    {"role": "user", "content": "普通本科四年制专业最长学习年限是多少？"},
    {"role": "assistant", "content": "一般学生为 3～6 年，休学创业学生最长为 8 年。"},
]


def test_query_rewrite_llm_uses_role_factory_and_preserves_messages(monkeypatch):
    """Query Rewrite 通过对应工厂调用并保持 system/user 顺序。"""

    class FakeRewriteModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(
                content=(
                    '{"need_rewrite":false,'
                    '"rewritten_question":"独立问题",'
                    '"reason":"无需改写"}'
                )
            )

    model = FakeRewriteModel()
    monkeypatch.setattr(query_rewrite, "get_query_rewrite_model", lambda: model)

    output = query_rewrite.call_query_rewrite_llm("rewrite prompt", trace_id="rw")

    assert isinstance(output, QueryRewriteOutput)
    assert output.model_dump() == {
        "need_rewrite": False,
        "rewritten_question": "独立问题",
        "reason": "无需改写",
    }
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[0][1], HumanMessage)
    assert model.calls[0][0].content == "你只输出合法 JSON。"
    assert model.calls[0][1].content == "rewrite prompt"


def test_should_not_rewrite_without_history():
    """验证没有历史时不改写。"""
    need_check, reason = query_rewrite.should_check_rewrite_with_history(
        "那休学创业呢？",
        [],
    )

    assert need_check is False
    assert reason == "no history"


def test_should_rewrite_short_follow_up_with_history():
    """验证短追问会触发改写。"""
    need_check, reason = query_rewrite.should_check_rewrite_with_history(
        "那休学创业呢？",
        HISTORY,
    )

    assert need_check is True
    assert reason


def test_short_question_with_history_enters_llm_check():
    """验证短句即使看似完整，也交给 LLM 判断是否依赖历史。"""
    question = "普通本科四年制专业最长学习年限是多少？"

    need_check, reason = query_rewrite.should_check_rewrite_with_history(
        question,
        HISTORY,
    )

    assert need_check is True
    assert reason == "short ambiguous question with history"


def test_should_rewrite_short_reference_marker_with_history():
    """验证包含明显指代词的短问题会触发改写。"""
    need_check, reason = query_rewrite.should_check_rewrite_with_history(
        "这个是多少？",
        HISTORY,
    )

    assert need_check is True
    assert reason == "obvious follow-up marker"


def test_obvious_short_follow_ups_enter_llm_check():
    """验证“还有吗”“条件呢”等典型短追问都会进入 LLM 判断。"""
    for question in ["没有遗漏的吗", "还有吗", "条件呢"]:
        need_check, reason = query_rewrite.should_check_rewrite_with_history(
            question,
            HISTORY,
        )

        assert need_check is True
        assert reason == "obvious follow-up marker"


def test_rewrite_obvious_follow_up_from_llm(monkeypatch, settings_env):
    """验证 LLM 可把“没有遗漏的吗”改成可独立检索的问题。"""
    settings_env(
        enable_query_rewrite=True,
        query_rewrite_api_key="test-key",
        query_rewrite_api_url="http://test.invalid",
        query_rewrite_model_name="test-model",
    )
    monkeypatch.setattr(
        query_rewrite,
        "call_query_rewrite_llm",
        lambda prompt, trace_id="": QueryRewriteOutput(
            need_rewrite=True,
            rewritten_question=(
                "普通本科四年制专业最长学习年限是否有遗漏？"
                "请根据知识库继续核对并补充完整。"
            ),
            reason="当前问题依赖上一轮主题",
        ),
    )

    result = query_rewrite.rewrite_query_with_history("没有遗漏的吗", HISTORY)

    assert result["need_check"] is True
    assert result["need_rewrite"] is True
    assert result["rewritten"] is True
    assert result["history_used"] is True
    assert result["rewrite_llm_used"] is True
    assert "最长学习年限是否有遗漏" in result["rewritten_question"]


def test_standalone_new_question_is_not_forced_to_use_history(
    monkeypatch,
    settings_env,
):
    """验证 LLM 判定独立问题后保留原问题，不拼接上一轮历史。"""
    question = "Docker容器怎么运行到别人电脑上"
    settings_env(
        enable_query_rewrite=True,
        query_rewrite_api_key="test-key",
        query_rewrite_api_url="http://test.invalid",
        query_rewrite_model_name="test-model",
    )
    monkeypatch.setattr(
        query_rewrite,
        "call_query_rewrite_llm",
        lambda prompt, trace_id="": QueryRewriteOutput(
            need_rewrite=False,
            rewritten_question=question,
            reason="当前问题语义完整，不依赖历史",
        ),
    )

    result = query_rewrite.rewrite_query_with_history(question, HISTORY)

    assert result["need_check"] is True
    assert result["need_rewrite"] is False
    assert result["rewritten_question"] == question
    assert result["history_used"] is False
    assert result["rewritten"] is False
    assert result["rewrite_llm_used"] is True


def test_no_history_math_and_system_question_skip_rewrite(settings_env):
    """验证无历史、简单数学和系统能力问题仍由硬规则直接跳过。"""
    settings_env(enable_query_rewrite=True)

    no_history_result = query_rewrite.rewrite_query_with_history(
        "没有遗漏的吗",
        [],
    )
    math_result = query_rewrite.rewrite_query_with_history("1+1", HISTORY)
    system_result = query_rewrite.rewrite_query_with_history("你能做什么？", HISTORY)

    assert no_history_result["rewritten"] is False
    assert no_history_result["reason"] == "no history"
    assert math_result["rewritten"] is False
    assert math_result["need_check"] is False
    assert math_result["rewrite_llm_used"] is False
    assert math_result["reason"] == "simple math question"
    assert system_result["rewritten"] is False
    assert system_result["need_check"] is False
    assert system_result["rewrite_llm_used"] is False
    assert system_result["reason"] == "system capability question"


@pytest.mark.parametrize(
    "question",
    [
        "你是谁",
        "你是谁?",
        "你是谁？",
        "你能做什么",
        "你能做什么?",
        "你能做什么？",
    ],
)
def test_system_capability_variants_skip_rewrite_llm(
    question,
    monkeypatch,
    settings_env,
):
    """有历史时系统能力问题的标点变体仍不调用改写模型。"""
    settings_env(enable_query_rewrite=True)

    def fail_model_call(*args, **kwargs):
        raise AssertionError("系统能力问题不应调用 Query Rewrite LLM")

    monkeypatch.setattr(query_rewrite, "call_query_rewrite_llm", fail_model_call)

    result = query_rewrite.rewrite_query_with_history(question, HISTORY)

    assert result["need_check"] is False
    assert result["rewrite_llm_used"] is False
    assert result["reason"] == "system capability question"


def test_ambiguous_short_question_without_llm_is_not_force_concatenated(
    settings_env,
):
    """验证仅模糊但无明显追问词时，LLM 未配置不会硬拼历史。"""
    settings_env(
        enable_query_rewrite=True,
        query_rewrite_api_key="",
        llm_api_key="",
    )

    result = query_rewrite.rewrite_query_with_history("怎么申请？", HISTORY)

    assert result["rewritten_question"] == "怎么申请？"
    assert result["rewritten"] is False
    assert result["history_used"] is False
    assert result["reason"] == (
        "ambiguous question but query rewrite llm is not configured"
    )


def test_query_rewrite_schema_requires_strict_need_rewrite():
    """改写协议拒绝缺失或非 bool 的 need_rewrite。"""
    for payload in [
        {"rewritten_question": "完整的新问题", "reason": "缺少字段"},
        {
            "need_rewrite": "true",
            "rewritten_question": "完整的新问题",
            "reason": "类型错误",
        },
    ]:
        with pytest.raises(ValidationError):
            QueryRewriteOutput.model_validate(payload)


def test_rewrite_query_fallback_when_structured_result_is_invalid(
    monkeypatch,
    settings_env,
):
    """结构化输出无效时沿用明显追问的现有 fallback。"""
    settings_env(
        enable_query_rewrite=True,
        query_rewrite_api_key="test-key",
        query_rewrite_api_url="http://test.invalid",
        query_rewrite_model_name="test-model",
    )

    def fail_validation(prompt, trace_id=""):
        raise StructuredOutputError(
            "模型返回的结构化内容不符合预期。",
            schema_name="QueryRewriteOutput",
        )

    monkeypatch.setattr(
        query_rewrite,
        "call_query_rewrite_llm",
        fail_validation,
    )

    result = query_rewrite.rewrite_query_with_history("那休学创业呢？", HISTORY)

    assert result["need_rewrite"] is True
    assert result["history_used"] is True
    assert result["rewrite_llm_used"] is True
    assert result["reason"] == "fallback rewrite after llm error"
    assert "结构化内容不符合预期" in result["error"]



def test_rewrite_query_fallback_when_llm_fails(monkeypatch, settings_env):
    """验证改写 LLM 失败时会 fallback 且不抛错。"""
    settings_env(
        enable_query_rewrite=True,
        query_rewrite_api_key="test-key",
        query_rewrite_api_url="http://test.invalid",
        query_rewrite_model_name="test-model",
    )

    def fail_llm(prompt, trace_id=""):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(query_rewrite, "call_query_rewrite_llm", fail_llm)

    result = query_rewrite.rewrite_query_with_history("那休学创业呢？", HISTORY)

    assert result["original_question"] == "那休学创业呢？"
    assert result["rewritten"] is True
    assert result["rewritten_question"]
    assert "普通本科四年制专业最长学习年限是多少" in result["rewritten_question"]
    assert result["rewrite_llm_used"] is True
    assert result["error"] == "llm unavailable"
