import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.rag import decomposition
from app.rag.decomposition import (
    parse_decompose_llm_json,
    should_decompose_query,
)
from app.rag.retrieval import should_refuse


def test_should_refuse_empty_results():
    """验证没有召回结果时拒答。"""
    assert should_refuse([], vector_threshold=0.35) is True


def test_should_refuse_accepts_high_vector_score():
    """验证向量分数达标时直接放行。"""
    chunks = [
        {
            "vector_score": 0.42,
            "bm25_score": 0.0,
            "hybrid_score": 0.0,
            "rerank_score": 0.0,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is False


def test_should_refuse_rejects_weak_evidence():
    """验证多路分数都弱时拒答。"""
    chunks = [
        {
            "vector_score": 0.12,
            "bm25_score": 0.0,
            "hybrid_score": 0.1,
            "rerank_score": 0.1,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is True


def test_should_refuse_accepts_strong_rerank_with_near_vector_score():
    """验证向量略低但重排和混合分数较强时放行。"""
    chunks = [
        {
            "vector_score": 0.28,
            "bm25_score": 0.0,
            "hybrid_score": 0.38,
            "rerank_score": 0.5,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is False


def test_should_refuse_rejects_near_vector_with_weak_keyword_signal():
    """验证向量略低时不会被很弱的关键词命中误放行。"""
    chunks = [
        {
            "vector_score": 0.28,
            "bm25_score": 0.1,
            "hybrid_score": 0.05,
            "rerank_score": 0.1,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is True


def test_should_refuse_accepts_strong_keyword_and_rerank_signal():
    """验证关键词命中和重排分数都强时不被向量阈值误拒。"""
    chunks = [
        {
            "vector_score": 0.0,
            "bm25_score": 8.2,
            "hybrid_score": 0.24,
            "rerank_score": 0.49,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is False


def test_should_decompose_query_skips_short_single_intent_question():
    """验证普通短问题不触发拆分模型。"""
    should_decompose, reason = should_decompose_query("最长学习年限是几年？")

    assert should_decompose is False
    assert "single-intent" in reason


def test_should_decompose_query_triggers_for_multi_intent_keywords():
    """验证对比、新旧、分别、以及等关键词会触发拆分。"""
    examples = [
        "新旧政策对比有哪些变化？",
        "请分别说明缓考入口以及申请时间。",
        "毕业条件和材料分别是什么？",
    ]

    for question in examples:
        should_decompose, reason = should_decompose_query(question)
        assert should_decompose is True, (question, reason)


def test_parse_decompose_llm_json_accepts_valid_compact_json():
    """验证拆分模型返回标准 JSON 时能解析并规整。"""
    result = parse_decompose_llm_json(
        '{"need_decompose":true,"reason":"多证据",'
        '"sub_queries":[{"id":1,"sub_question":"入口是什么？",'
        '"search_query":"缓考 免考 线上填报 入口",'
        '"must_answer":true}]}'
    )

    assert result["need_decompose"] is True
    assert result["reason"] == "多证据"
    assert result["sub_queries"] == [
        {
            "id": 1,
            "sub_question": "入口是什么？",
            "search_query": "缓考 免考 线上填报 入口",
            "must_answer": True,
        }
    ]


def test_parse_decompose_llm_json_extracts_object_from_extra_text():
    """验证模型在 JSON 前后输出多余文本时仍能提取 JSON 对象。"""
    result = parse_decompose_llm_json(
        '下面是结果：\n```json\n'
        '{"need_decompose":false,"reason":"单一问题","sub_queries":[]}'
        '\n```\n请使用。'
    )

    assert result == {
        "need_decompose": False,
        "reason": "单一问题",
        "sub_queries": [],
    }


def test_parse_decompose_llm_json_compacts_unescaped_newlines_in_fields():
    """验证模型把换行放进字符串字段时可容错并压回单行。"""
    result = parse_decompose_llm_json(
        '{"need_decompose":true,"reason":"多方面",'
        '"sub_queries":[{"id":1,"sub_question":"缓考\\n免考入口是什么？",'
        '"search_query":"学生\n缓考 免考 线上填报",'
        '"must_answer":"true"}]}'
    )

    assert result["need_decompose"] is True
    assert result["sub_queries"][0]["sub_question"] == "缓考 免考入口是什么？"
    assert result["sub_queries"][0]["search_query"] == "学生 缓考 免考 线上填报"
    assert result["sub_queries"][0]["must_answer"] is True


def test_parse_decompose_llm_json_reports_truncated_object():
    """验证真正被截断的 JSON 会给出明确错误。"""
    with pytest.raises(ValueError, match="不完整"):
        parse_decompose_llm_json(
            '{"need_decompose":true,"reason":"多方面",'
            '"sub_queries":[{"id":1,"sub_question":"入口是什么？",'
            '"search_query":"学生'
        )


def test_decompose_query_with_llm_uses_role_factory_and_parses_result(monkeypatch):
    """拆分模型通过对应工厂调用，并保留消息顺序与业务解析。"""

    class FakeDecomposeModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(
                content=(
                    '{"need_decompose":false,'
                    '"reason":"单一问题","sub_queries":[]}'
                )
            )

    model = FakeDecomposeModel()
    monkeypatch.setattr(decomposition, "get_decompose_model", lambda: model)

    result = decomposition.decompose_query_with_llm("缓考和免考入口是什么？")

    assert result["need_decompose"] is False
    assert result["reason"] == "单一问题"
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[0][1], HumanMessage)
    assert model.calls[0][1].content == "缓考和免考入口是什么？"
