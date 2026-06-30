import pytest

import app.rag_retrieval as rag_retrieval
from app.rag_retrieval import parse_decompose_llm_json


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


def test_decompose_query_with_llm_uses_configured_max_tokens(monkeypatch):
    """验证调用拆分模型时使用可配置的输出 token 上限。"""
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"need_decompose":false,'
                                '"reason":"单一问题","sub_queries":[]}'
                            )
                        }
                    }
                ]
            }

    def fake_post(api_url, headers, json, timeout):
        captured["api_url"] = api_url
        captured["headers"] = headers
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(rag_retrieval.Settings, "DECOMPOSE_MODEL_NAME", "fake-model")
    monkeypatch.setattr(rag_retrieval.Settings, "DECOMPOSE_API_URL", "http://example.test/v1/chat/completions")
    monkeypatch.setattr(rag_retrieval.Settings, "DECOMPOSE_API_KEY", "")
    monkeypatch.setattr(rag_retrieval.Settings, "DECOMPOSE_TIMEOUT", 3)
    monkeypatch.setattr(rag_retrieval.Settings, "DECOMPOSE_MAX_TOKENS", 777)
    monkeypatch.setattr(rag_retrieval.requests, "post", fake_post)

    result = rag_retrieval.decompose_query_with_llm("缓考和免考入口是什么？")

    assert result["need_decompose"] is False
    assert captured["payload"]["max_tokens"] == 777
    assert captured["payload"]["stream"] is False
    assert captured["timeout"] == 3
