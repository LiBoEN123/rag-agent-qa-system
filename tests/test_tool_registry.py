import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.tools.context import ToolInvocationContext
from app.tools.registry import ToolRegistry, ToolSpec
from app.tools.result import tool_success
from tests.support.local_tools import invoke_local_tool
from app.tools.schemas import (
    SearchKnowledgeBaseArgs,
    SearchKnowledgeBaseOutput,
    SimpleCalculatorOutput,
)


def test_tool_registry_names_include_builtin_tools(tool_registry):
    """验证内置工具已注册到统一注册中心。"""
    names = tool_registry.names()

    assert "search_knowledge_base" in names
    assert "simple_calculator" in names
    assert "list_documents" in names
    assert "get_chunk_text" in names
    assert "compare_documents" in names
    assert "web_search" in names
    assert "web_fetch" in names
    assert "web_pdf_fetch" in names


def test_builtin_tools_publish_v2_contract_metadata(tool_registry):
    """验证八个注册生产工具发布完整且符合要求的 V2 契约。"""
    specs = tool_registry.contract_specs()
    specs_by_name = {spec["name"]: spec for spec in specs}
    expected_permissions = {
        "search_knowledge_base": ["knowledge_base:read"],
        "compare_documents": ["knowledge_base:read"],
        "simple_calculator": [],
        "list_documents": ["knowledge_base:read"],
        "get_chunk_text": ["knowledge_base:read"],
        "web_search": ["web:search"],
        "web_fetch": ["web:fetch"],
        "web_pdf_fetch": ["web:fetch"],
    }
    expected_timeouts = {
        "search_knowledge_base": 120.0,
        "compare_documents": 20.0,
        "simple_calculator": 3.0,
        "list_documents": 5.0,
        "get_chunk_text": 5.0,
        "web_search": 10.0,
        "web_fetch": 30.0,
        "web_pdf_fetch": 45.0,
    }

    assert len(specs) == len(tool_registry.names())
    assert set(specs_by_name) == set(tool_registry.names())
    assert [spec["name"] for spec in specs] == tool_registry.names()
    json.dumps(specs)

    for name, spec in specs_by_name.items():
        registered = tool_registry.get(name)
        assert registered is not None
        assert spec["version"] == "1.0.0"
        assert spec["risk_level"] == "read_only"
        assert spec["required_permissions"] == expected_permissions[name]
        assert spec["timeout"] == expected_timeouts[name]
        assert spec["max_retries"] == (
            1 if name in {"web_search", "web_fetch", "web_pdf_fetch"} else 0
        )
        assert spec["idempotent"] is True
        assert spec["requires_approval"] is False
        assert spec["output_model"] == {
            "search_knowledge_base": "SearchKnowledgeBaseOutput",
            "compare_documents": "CompareDocumentsOutput",
            "simple_calculator": "SimpleCalculatorOutput",
            "list_documents": "ListDocumentsOutput",
            "get_chunk_text": "GetChunkTextOutput",
            "web_search": "WebSearchOutput",
            "web_fetch": "WebFetchOutput",
            "web_pdf_fetch": "WebPdfFetchOutput",
        }[name]
        assert spec["args_schema"] == registered.args_model.model_json_schema()
        if registered.output_model is not None:
            assert (
                spec["output_schema"]
                == registered.output_model.model_json_schema()
            )
        else:
            assert spec["output_schema"] == registered.output_schema
        assert "handler" not in spec

    web = specs_by_name["web_search"]
    assert web["sensitive_arg_fields"] == ["query"]
    assert web["audit_output_mode"] == "summary"
    fetch = specs_by_name["web_fetch"]
    assert fetch["sensitive_arg_fields"] == ["url"]
    assert fetch["audit_output_mode"] == "summary"
    pdf = specs_by_name["web_pdf_fetch"]
    assert pdf["sensitive_arg_fields"] == ["url"]
    assert pdf["audit_output_mode"] == "summary"
    knowledge = specs_by_name["search_knowledge_base"]
    knowledge_schema = knowledge["output_schema"]
    assert set(knowledge_schema["properties"]) == {
        "count",
        "chunks",
        "top_chunk_ids",
        "top_scores",
        "rag_result",
    }
    assert set(knowledge_schema["required"]) == set(
        knowledge_schema["properties"]
    )
    assert knowledge_schema["properties"]["rag_result"] == {
        "$ref": "#/$defs/RAGResult"
    }


def test_invoke_local_tool_runs_simple_calculator(tool_registry):
    """验证唯一执行入口可以调用已注册计算器工具。"""
    result = invoke_local_tool(
        registry=tool_registry,
        tool_name="simple_calculator",
        tool_args={"expression": "1 + 2 * 3"},
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"]["result"] == 7


def test_migrated_calculator_rejects_malformed_output(tool_registry):
    """生产计算器契约会在 Runtime 拒绝结构错误的成功输出。"""
    spec = tool_registry.get("simple_calculator")
    assert spec is not None
    assert spec.output_model is SimpleCalculatorOutput

    registry = ToolRegistry()
    registry.register(
        replace(
            spec,
            handler=lambda args, context: tool_success(
                "simple_calculator",
                args,
                {"expression": args["expression"], "result": {"invalid": True}},
            ),
        )
    )

    result = invoke_local_tool(
        registry=registry,
        tool_name="simple_calculator",
        tool_args={"expression": "1 + 2"},
    )

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["error_code"] == "output_validation_error"
    assert result["output"] is None


def test_invoke_local_tool_runs_search_knowledge_base(
    monkeypatch,
    tool_registry,
):
    """Graph 输入保留预检索候选，对外 evidence 只使用最终候选。"""
    import app.rag as rag

    prefetched_chunks = [
        {"chunk_id": "initial-a", "text": "初始候选 A", "hybrid_score": 0.4},
        {"chunk_id": "initial-b", "text": "初始候选 B", "hybrid_score": 0.3},
    ]
    final_chunks = [
        {"chunk_id": "final-c", "text": "最终候选 C", "hybrid_score": 0.87654},
        {
            "metadata": {
                "chunk_id": "final-d",
                "hybrid_score": "invalid-score",
            },
            "text": "最终候选 D",
        },
    ]
    rag_call = {}

    def fake_search_relevant_chunks(query, top_k=5):
        assert query == "普通本科四年制专业最长学习年限是多少？"
        assert top_k == 5
        return prefetched_chunks

    def fake_run_rag_subgraph(**kwargs):
        rag_call.update(kwargs)
        return {
            "status": "OK",
            "answer": "测试 RAG 回答",
            "references": [],
            "final_chunks": final_chunks,
            "retrieval_confidence": 0.91,
            "answer_confidence": 0.8,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "测试证据充分",
            "timings": {},
            "decompose_llm_used": False,
            "answer_llm_used": True,
            "llm_call_count": 1,
        }

    monkeypatch.setattr(
        rag,
        "search_relevant_chunks",
        fake_search_relevant_chunks,
        raising=False
    )
    monkeypatch.setattr(
        "app.rag.graph.run_rag_subgraph",
        fake_run_rag_subgraph,
    )

    result = invoke_local_tool(
        registry=tool_registry,
        tool_name="search_knowledge_base",
        tool_args={
            "query": "  普通本科四年制专业最长学习年限是多少？  ",
            "top_k": "5",
        },
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["tool_name"] == "search_knowledge_base"
    assert result["tool_input"] == {
        "query": "普通本科四年制专业最长学习年限是多少？",
        "top_k": 5,
    }
    assert rag_call["retrieved_chunks"] is prefetched_chunks
    assert result["output"]["count"] == 2
    assert result["output"]["chunks"] == final_chunks
    assert result["output"]["top_chunk_ids"] == ["final-c", "final-d"]
    assert result["output"]["top_scores"] == [0.8765, 0.0]
    assert result["output"]["rag_result"]["answer"] == "测试 RAG 回答"


def test_search_knowledge_base_exposes_multi_query_final_chunks(
    monkeypatch,
    tool_registry,
):
    """模拟多查询改变候选集合后，公开 evidence 不回退到预检索结果。"""
    import app.rag as rag

    prefetched_chunks = [
        {"chunk_id": "initial-a", "text": "初始候选 A", "hybrid_score": 0.4}
    ]
    final_chunks = [
        {"chunk_id": "final-b", "text": "最终候选 B", "hybrid_score": 0.82},
        {"chunk_id": "final-c", "text": "最终候选 C", "hybrid_score": 0.73},
    ]
    rag_call = {}

    monkeypatch.setattr(
        rag,
        "search_relevant_chunks",
        lambda query, top_k=5: prefetched_chunks,
        raising=False,
    )

    def fake_run_rag_subgraph(**kwargs):
        rag_call.update(kwargs)
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": "没有足够证据支持回答。",
            "references": [],
            "final_chunks": final_chunks,
            "retrieval_confidence": 0.5,
            "answer_confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "多查询候选不足以支持回答",
            "timings": {},
            "decompose_llm_used": True,
            "answer_llm_used": False,
            "llm_call_count": 1,
        }

    monkeypatch.setattr(
        "app.rag.graph.run_rag_subgraph",
        fake_run_rag_subgraph,
    )

    result = invoke_local_tool(
        registry=tool_registry,
        tool_name="search_knowledge_base",
        tool_args={"query": "对比两个政策", "top_k": 1},
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert rag_call["retrieved_chunks"] is prefetched_chunks
    assert result["output"]["chunks"] == final_chunks
    assert result["output"]["count"] == 2
    assert result["output"]["top_chunk_ids"] == ["final-b", "final-c"]
    assert result["output"]["top_scores"] == [0.82, 0.73]
    assert result["output"]["rag_result"]["status"] == "NO_RELEVANT_INFO"


def test_search_knowledge_base_rejects_inconsistent_output_counts(tool_registry):
    """知识库生产输出模型拒绝证据数量不一致的成功结果。"""
    spec = tool_registry.get("search_knowledge_base")
    assert spec is not None
    assert spec.output_model is SearchKnowledgeBaseOutput
    rag_result = {
        "status": "OK",
        "answer": "测试回答",
        "references": [],
        "final_chunks": [],
        "retrieval_confidence": 0.0,
        "answer_confidence": 0.0,
        "need_clarification": False,
        "clarification_question": "",
        "reason": "测试",
        "timings": {},
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
    }
    registry = ToolRegistry()
    registry.register(
        replace(
            spec,
            handler=lambda args, context: tool_success(
                "search_knowledge_base",
                args,
                {
                    "count": 1,
                    "chunks": [],
                    "top_chunk_ids": [],
                    "top_scores": [],
                    "rag_result": rag_result,
                },
            ),
        )
    )

    result = invoke_local_tool(
        registry=registry,
        tool_name="search_knowledge_base",
        tool_args={"query": "测试", "top_k": 1},
    )

    assert result["ok"] is False
    assert result["error_code"] == "output_validation_error"
    assert result["output"] is None


@pytest.mark.parametrize(
    "tool_args",
    [
        {"query": "", "top_k": 5},
        {"query": "   ", "top_k": 5},
        {"query": "测试", "top_k": 0},
        {"query": "测试", "top_k": 11},
    ],
)
def test_search_knowledge_base_rejects_invalid_args_in_structured_tool(
    tool_registry,
    tool_args,
):
    """知识库参数非法时由 StructuredTool/Pydantic 拒绝。"""
    with pytest.raises(ValidationError):
        invoke_local_tool(
            registry=tool_registry,
            tool_name="search_knowledge_base",
            tool_args=tool_args,
        )


@pytest.mark.parametrize("chunk_id", [0, 1, " chunk-abc "])
def test_get_chunk_text_keeps_integer_and_string_ids(
    monkeypatch,
    tool_registry,
    chunk_id,
):
    """合法整数和字符串 ID 经 Schema 规范化后进入真实 handler。"""
    import app.rag as rag

    received = []

    def fake_get_chunk_full_text(value):
        received.append(value)
        return {"found": True, "chunk_id": value, "text": "测试正文"}

    monkeypatch.setattr(rag, "get_chunk_full_text", fake_get_chunk_full_text)
    result = invoke_local_tool(
        registry=tool_registry,
        tool_name="get_chunk_text",
        tool_args={"chunk_id": chunk_id},
    )

    expected = chunk_id.strip() if isinstance(chunk_id, str) else chunk_id
    assert result["ok"] is True
    assert received == [expected]
    assert result["tool_input"] == {"chunk_id": expected}
    assert result["output"] == {
        "chunk_id": expected,
        "chunk": {"found": True, "chunk_id": expected, "text": "测试正文"},
    }


def test_list_documents_output_model_accepts_current_preview_shape(
    monkeypatch,
    tool_registry,
):
    """文档列表生产输出模型接受当前普通字典预览结构。"""
    import app.rag as rag

    documents = [
        {"file_name": "a.md", "source_path": "data/a.md", "chunk_count": 2},
        {"file_name": "b.md", "source_path": "data/b.md", "chunk_count": 1},
    ]
    monkeypatch.setattr(rag, "preview_docs", lambda: documents)

    result = invoke_local_tool(
        registry=tool_registry,
        tool_name="list_documents",
        tool_args={},
    )

    assert result["ok"] is True
    assert result["output"] == {"count": 2, "documents": documents}


@pytest.mark.parametrize("chunk_id", ["", "   "])
def test_get_chunk_text_rejects_blank_string_in_structured_tool(tool_registry, chunk_id):
    """空白字符串 ID 在提交 handler 前由 Pydantic 拒绝。"""
    with pytest.raises(ValidationError):
        invoke_local_tool(
            registry=tool_registry,
            tool_name="get_chunk_text",
            tool_args={"chunk_id": chunk_id},
        )


@pytest.mark.parametrize("expression", ["", "   "])
def test_simple_calculator_rejects_blank_expression_in_structured_tool(
    tool_registry,
    expression,
):
    """空白表达式由参数模型拒绝，计算器只处理已校验表达式。"""
    with pytest.raises(ValidationError):
        invoke_local_tool(
            registry=tool_registry,
            tool_name="simple_calculator",
            tool_args={"expression": expression},
        )


def test_simple_calculator_keeps_ast_safety_rules(tool_registry):
    """表达式结构合法后仍由计算器 AST 白名单拒绝代码执行。"""
    result = invoke_local_tool(
        registry=tool_registry,
        tool_name="simple_calculator",
        tool_args={"expression": "__import__('os').getcwd()"},
    )

    assert result["error_code"] == "execution_error"
    assert "只支持简单数字四则运算" in result["error"]


def test_invoke_local_tool_rejects_unknown_tool(tool_registry):
    """直接调用只能构造已注册的 BaseTool；未知工具不另设 Runtime API。"""
    with pytest.raises(LookupError, match="unknown local tool"):
        invoke_local_tool(
            registry=tool_registry,
            tool_name="unknown_tool",
            tool_args={},
        )


def test_execution_entry_validates_args_before_handler():
    """验证唯一执行入口在调用 handler 前执行参数校验。"""
    registry = ToolRegistry()
    seen = {}

    def fake_handler(args, context):
        seen["args"] = args
        seen["context"] = context
        return {
            "status": "success",
            "ok": True,
            "tool_name": "search_knowledge_base",
            "tool_input": args,
            "output": {"count": 0},
            "error": "",
        }

    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            version="1.0.0",
            description="检索知识库",
            output_schema={},
            handler=fake_handler,
            args_model=SearchKnowledgeBaseArgs,
        )
    )

    context = ToolInvocationContext(trace_id="trace-registry")
    result = invoke_local_tool(
        registry=registry,
        tool_name="search_knowledge_base",
        tool_args={"query": "测试", "top_k": 5},
        context=context,
    )

    assert result["status"] == "success"
    assert seen["args"] == {"query": "测试", "top_k": 5}
    assert seen["context"] is context


def test_execution_entry_uses_structured_tool_validation_before_handler():
    """验证参数非法时由 StructuredTool 拒绝且不调用 handler。"""
    registry = ToolRegistry()
    called = {"count": 0}

    def fake_handler(args, context):
        called["count"] += 1
        return {"status": "success", "ok": True, "tool_name": "search_knowledge_base"}

    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            version="1.0.0",
            description="检索知识库",
            output_schema={},
            handler=fake_handler,
            args_model=SearchKnowledgeBaseArgs,
        )
    )

    with pytest.raises(ValidationError):
        invoke_local_tool(
            registry=registry,
            tool_name="search_knowledge_base",
            tool_args={"query": "测试", "top_k": 999},
        )

    assert called["count"] == 0


def test_execution_entry_accepts_explicit_context(tool_registry):
    """验证唯一执行入口保留显式请求上下文。"""
    context = ToolInvocationContext(trace_id="test-trace")
    result = invoke_local_tool(
        registry=tool_registry,
        tool_name="simple_calculator",
        tool_args={"expression": "2 + 3"},
        context=context,
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"]["result"] == 5
