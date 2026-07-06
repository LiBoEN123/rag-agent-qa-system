from app import query_rewrite


HISTORY = [
    {"role": "user", "content": "普通本科四年制专业最长学习年限是多少？"},
    {"role": "assistant", "content": "一般学生为 3～6 年，休学创业学生最长为 8 年。"},
]


def test_should_not_rewrite_without_history():
    """验证没有历史时不改写。"""
    need_rewrite, reason = query_rewrite.should_rewrite_with_history("那休学创业呢？", [])

    assert need_rewrite is False
    assert reason == "no history"


def test_should_rewrite_short_follow_up_with_history():
    """验证短追问会触发改写。"""
    need_rewrite, reason = query_rewrite.should_rewrite_with_history("那休学创业呢？", HISTORY)

    assert need_rewrite is True
    assert reason


def test_should_not_rewrite_standalone_question_with_history():
    """验证完整问题不触发改写。"""
    question = "普通本科四年制专业最长学习年限是多少？"

    need_rewrite, reason = query_rewrite.should_rewrite_with_history(question, HISTORY)

    assert need_rewrite is False
    assert reason == "not a follow-up question"


def test_should_rewrite_short_reference_marker_with_history():
    """验证包含明显指代词的短问题会触发改写。"""
    need_rewrite, reason = query_rewrite.should_rewrite_with_history("这个是多少？", HISTORY)

    assert need_rewrite is True
    assert "reference marker" in reason


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
