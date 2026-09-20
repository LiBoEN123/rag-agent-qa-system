import pytest
from pydantic import ValidationError

from app.tools.schemas import (

    GetChunkContextArgs,
    SearchKnowledgeBaseArgs,
)


def test_search_knowledge_base_args_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs(query="测试问题", top_k=0)

    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs(query="测试问题", top_k=11)


@pytest.mark.parametrize("query", ["", "   "])
def test_search_knowledge_base_args_rejects_blank_query(query):
    """空字符串和纯空白查询都在参数模型边界拒绝。"""
    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs(query=query)


def test_search_knowledge_base_args_normalizes_query_and_top_k():
    """查询 trim 与数字字符串转换只由参数模型负责。"""
    args = SearchKnowledgeBaseArgs(query="  测试问题  ", top_k="5")

    assert args.model_dump(mode="json") == {"query": "测试问题", "top_k": 5}


def test_get_chunk_context_args_parses_positive_integer_locator():
    assert GetChunkContextArgs(chunk_id=1).chunk_id == 1
    assert GetChunkContextArgs(chunk_id=382).chunk_id == 382
    assert GetChunkContextArgs(chunk_id="382").chunk_id == 382


@pytest.mark.parametrize("chunk_id", [0, -1, "", "   ", "abc", "knowledge_base:ef463ed6"])
def test_get_chunk_context_args_rejects_invalid_locator(chunk_id):
    """拒绝非正整数与 evidence identity。"""
    with pytest.raises(ValidationError):
        GetChunkContextArgs(chunk_id=chunk_id)


