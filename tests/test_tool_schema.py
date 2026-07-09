import pytest
from pydantic import ValidationError

from app.tools.schemas import (
    CompareDocumentsArgs,
    GetChunkTextArgs,
    ListDocumentsArgs,
    SearchKnowledgeBaseArgs,
    SimpleCalculatorArgs,
    ToolResult,
)


def test_tool_result_defaults():
    result = ToolResult(
        status="success",
        ok=True,
        tool_name="simple_calculator",
    )

    assert result.status == "success"
    assert result.ok is True
    assert result.tool_name == "simple_calculator"
    assert result.tool_input == {}
    assert result.output == {}
    assert result.error == ""
    assert result.cost_ms == 0


def test_search_knowledge_base_args_valid():
    args = SearchKnowledgeBaseArgs(query="测试问题", top_k=5)
    assert args.query == "测试问题"
    assert args.top_k == 5


def test_search_knowledge_base_args_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs(query="测试问题", top_k=0)

    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs(query="测试问题", top_k=11)


def test_compare_documents_args_requires_docs():
    with pytest.raises(ValidationError):
        CompareDocumentsArgs(query="对比学习年限", doc_a="学生手册")


def test_simple_calculator_args_requires_expression():
    with pytest.raises(ValidationError):
        SimpleCalculatorArgs(expression="")


def test_get_chunk_text_args_accepts_int_or_str():
    assert GetChunkTextArgs(chunk_id=1).chunk_id == 1
    assert GetChunkTextArgs(chunk_id="chunk-abc").chunk_id == "chunk-abc"


def test_list_documents_args_empty():
    args = ListDocumentsArgs()
    assert args.model_dump() == {}
