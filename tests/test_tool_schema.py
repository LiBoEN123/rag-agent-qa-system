import pytest
from pydantic import ValidationError

from app.tools.schemas import (
    CompareDocumentsArgs,
    GetChunkTextArgs,
    ListDocumentsArgs,
    SearchKnowledgeBaseArgs,
    SimpleCalculatorArgs,
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


def test_compare_documents_args_requires_docs():
    with pytest.raises(ValidationError):
        CompareDocumentsArgs(query="对比学习年限", doc_a="学生手册")


@pytest.mark.parametrize("field_name", ["query", "doc_a", "doc_b"])
def test_compare_documents_args_rejects_blank_business_fields(field_name):
    """对比工具的三个基础文本字段都拒绝纯空白。"""
    payload = {"query": "对比主题", "doc_a": "old.md", "doc_b": "new.md"}
    payload[field_name] = "   "

    with pytest.raises(ValidationError):
        CompareDocumentsArgs.model_validate(payload)


@pytest.mark.parametrize("top_k", [0, 11])
def test_compare_documents_args_rejects_out_of_range_top_k(top_k):
    """每份文档的证据数量只能位于 1 到 10。"""
    with pytest.raises(ValidationError):
        CompareDocumentsArgs(
            query="对比主题",
            doc_a="old.md",
            doc_b="new.md",
            top_k=top_k,
        )


def test_compare_documents_args_normalizes_text_and_top_k():
    """基础文本 trim 与数字字符串转换在参数模型中完成。"""
    args = CompareDocumentsArgs(
        query="  对比主题  ",
        doc_a="  old.md  ",
        doc_b="  new.md  ",
        top_k="5",
    )

    assert args.model_dump(mode="json") == {
        "query": "对比主题",
        "doc_a": "old.md",
        "doc_b": "new.md",
        "top_k": 5,
    }


def test_simple_calculator_args_requires_expression():
    for expression in ("", "   "):
        with pytest.raises(ValidationError):
            SimpleCalculatorArgs(expression=expression)

    assert SimpleCalculatorArgs(expression=" 1 + 2 ").expression == "1 + 2"


def test_get_chunk_text_args_accepts_int_or_str():
    assert GetChunkTextArgs(chunk_id=1).chunk_id == 1
    assert GetChunkTextArgs(chunk_id=0).chunk_id == 0
    assert GetChunkTextArgs(chunk_id=" chunk-abc ").chunk_id == "chunk-abc"


@pytest.mark.parametrize("chunk_id", ["", "   "])
def test_get_chunk_text_args_rejects_blank_string(chunk_id):
    """字符串 chunk_id 必须在 trim 后仍非空。"""
    with pytest.raises(ValidationError):
        GetChunkTextArgs(chunk_id=chunk_id)


def test_list_documents_args_keeps_extra_compatibility():
    """无参工具继续忽略框架可能注入的额外字段。"""
    args = ListDocumentsArgs.model_validate({"runtime": "injected"})

    assert args.model_dump(mode="json") == {}
