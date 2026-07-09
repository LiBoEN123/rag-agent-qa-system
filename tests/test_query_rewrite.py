import pytest

from app.agent import query_rewrite


HISTORY = [
    {"role": "user", "content": "普通本科四年制专业最长学习年限是多少？"},
    {"role": "assistant", "content": "一般学生为 3～6 年，休学创业学生最长为 8 年。"},
]


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


def test_rewrite_obvious_follow_up_from_llm(monkeypatch):
    """验证 LLM 可把“没有遗漏的吗”改成可独立检索的问题。"""
    monkeypatch.setattr(query_rewrite.Settings, "ENABLE_QUERY_REWRITE", True, raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_API_URL", "http://test.invalid", raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_MODEL_NAME", "test-model", raising=False)
    monkeypatch.setattr(
        query_rewrite,
        "call_query_rewrite_llm",
        lambda prompt, trace_id="": (
            '{"need_rewrite":true,'
            '"rewritten_question":"普通本科四年制专业最长学习年限是否有遗漏？'
            '请根据知识库继续核对并补充完整。",'
            '"reason":"当前问题依赖上一轮主题"}'
        ),
    )

    result = query_rewrite.rewrite_query_with_history("没有遗漏的吗", HISTORY)

    assert result["need_check"] is True
    assert result["need_rewrite"] is True
    assert result["rewritten"] is True
    assert result["history_used"] is True
    assert result["rewrite_llm_used"] is True
    assert "最长学习年限是否有遗漏" in result["rewritten_question"]


def test_standalone_new_question_is_not_forced_to_use_history(monkeypatch):
    """验证 LLM 判定独立问题后保留原问题，不拼接上一轮历史。"""
    question = "Docker容器怎么运行到别人电脑上"
    monkeypatch.setattr(query_rewrite.Settings, "ENABLE_QUERY_REWRITE", True, raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_API_URL", "http://test.invalid", raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_MODEL_NAME", "test-model", raising=False)
    monkeypatch.setattr(
        query_rewrite,
        "call_query_rewrite_llm",
        lambda prompt, trace_id="": (
            '{"need_rewrite":false,'
            f'"rewritten_question":"{question}",'
            '"reason":"当前问题语义完整，不依赖历史"}'
        ),
    )

    result = query_rewrite.rewrite_query_with_history(question, HISTORY)

    assert result["need_check"] is True
    assert result["need_rewrite"] is False
    assert result["rewritten_question"] == question
    assert result["history_used"] is False
    assert result["rewritten"] is False
    assert result["rewrite_llm_used"] is True


def test_no_history_and_simple_math_skip_rewrite(monkeypatch):
    """验证无历史和简单数学仍由硬规则直接跳过。"""
    monkeypatch.setattr(query_rewrite.Settings, "ENABLE_QUERY_REWRITE", True, raising=False)

    no_history_result = query_rewrite.rewrite_query_with_history(
        "没有遗漏的吗",
        [],
    )
    math_result = query_rewrite.rewrite_query_with_history("1+1", HISTORY)

    assert no_history_result["rewritten"] is False
    assert no_history_result["reason"] == "no history"
    assert math_result["rewritten"] is False
    assert math_result["need_check"] is False
    assert math_result["rewrite_llm_used"] is False
    assert math_result["reason"] == "simple math question"


def test_ambiguous_short_question_without_llm_is_not_force_concatenated(monkeypatch):
    """验证仅模糊但无明显追问词时，LLM 未配置不会硬拼历史。"""
    monkeypatch.setattr(query_rewrite.Settings, "ENABLE_QUERY_REWRITE", True, raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_API_KEY", "", raising=False)

    result = query_rewrite.rewrite_query_with_history("怎么申请？", HISTORY)

    assert result["rewritten_question"] == "怎么申请？"
    assert result["rewritten"] is False
    assert result["history_used"] is False
    assert result["reason"] == (
        "ambiguous question but query rewrite llm is not configured"
    )


def test_parse_rewrite_json_requires_need_rewrite():
    """验证新协议必须明确返回布尔类型 need_rewrite。"""
    with pytest.raises(ValueError, match="invalid need_rewrite"):
        query_rewrite._parse_rewrite_json(
            '{"rewritten_question":"完整的新问题","reason":"缺少字段"}',
        )



def test_rewrite_query_fallback_when_llm_fails(monkeypatch):
    """验证改写 LLM 失败时会 fallback 且不抛错。"""
    monkeypatch.setattr(query_rewrite.Settings, "ENABLE_QUERY_REWRITE", True, raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_API_URL", "http://test.invalid", raising=False)
    monkeypatch.setattr(query_rewrite.Settings, "QUERY_REWRITE_MODEL_NAME", "test-model", raising=False)

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
