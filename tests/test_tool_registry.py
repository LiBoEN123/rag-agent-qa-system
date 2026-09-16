import json

from pydantic import RootModel

import pytest
from pydantic import ValidationError

from app.tools.context import ToolInvocationContext
from app.tools.catalog import ToolSpec
from app.tools.result import tool_success
from tests.support.local_tools import invoke_local_tool
from app.tools.schemas import (
    SearchKnowledgeBaseArgs,
    SearchKnowledgeBaseOutput,
    ListDocumentsOutput,
)


def test_tool_registry_names_include_builtin_tools(tool_registry):
    """验证内置工具已注册到统一注册中心。"""
    names = [item.name for item in tool_registry]

    assert "search_knowledge_base" in names
    assert "list_documents" in names
    assert "get_chunk_text" in names
    assert "compare_documents" in names
    assert "web_search" in names
    assert "web_fetch" in names
    assert "web_pdf_fetch" in names


def test_builtin_tools_publish_v2_contract_metadata(tool_registry):
    """验证七个注册生产工具发布完整且符合要求的 V2 契约。"""
    from app.agent.framework_tools import build_local_tool, local_tool_contract

    specs = [local_tool_contract(build_local_tool(item)) for item in tool_registry]
    specs_by_name = {spec["name"]: spec for spec in specs}
    expected_permissions = {
        "search_knowledge_base": ["knowledge_base:read"],
        "compare_documents": ["knowledge_base:read"],
        "list_documents": ["knowledge_base:read"],
        "get_chunk_text": ["knowledge_base:read"],
        "web_search": ["web:search"],
        "web_fetch": ["web:fetch"],
        "web_pdf_fetch": ["web:fetch"],
    }
    expected_timeouts = {
        "search_knowledge_base": 120.0,
        "compare_documents": 20.0,
        "list_documents": 5.0,
        "get_chunk_text": 5.0,
        "web_search": 10.0,
        "web_fetch": 30.0,
        "web_pdf_fetch": 45.0,
    }

    assert len(specs) == len([item.name for item in tool_registry])
    assert set(specs_by_name) == set([item.name for item in tool_registry])
    assert [spec["name"] for spec in specs] == [item.name for item in tool_registry]
    json.dumps(specs)

    for name, spec in specs_by_name.items():
        registered = next((item for item in tool_registry if item.name == name), None)
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
            "list_documents": "ListDocumentsOutput",
            "get_chunk_text": "GetChunkTextOutput",
            "web_search": "WebSearchOutput",
            "web_fetch": "WebFetchOutput",
            "web_pdf_fetch": "WebPdfFetchOutput",
        }[name]
        assert spec["args_schema"] == registered.args_model.model_json_schema()
        assert spec["output_schema"] == registered.output_model.model_json_schema()
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
    }
    assert set(knowledge_schema["required"]) == set(
        knowledge_schema["properties"]
    )
    assert "RAGResult" not in json.dumps(knowledge_schema)


def test_list_documents_rejects_malformed_output(tool_registry):
    """生产文档列表契约会在 Runtime 拒绝结构错误的成功输出。"""
    spec = next((item for item in tool_registry if item.name == "list_documents"), None)
    assert spec is not None
    assert spec.output_model is ListDocumentsOutput

    registry = []
    registry.append(
        spec.model_copy(update={'handler': lambda args, context: tool_success(
                "list_documents",
                args,
                {"count": 1, "documents": []},
            )})
    )

    result = invoke_local_tool(
        registry=registry,
        tool_name="list_documents",
        tool_args={},
    )

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["error_code"] == "output_validation_error"
    assert result["output"] is None


def test_invoke_local_tool_runs_search_and_rerank(monkeypatch, tool_registry):
    """真实工具保留参数规范化、重排分数及 metadata 字段回退。"""
    from app.rag import pipeline

    candidates = [
        {"chunk_id": "a", "text": "候选 A", "hybrid_score": 0.87654},
        {"metadata": {"chunk_id": "b", "hybrid_score": "invalid-score"}, "text": "候选 B"},
    ]
    calls = []

    def search(query, top_k):
        calls.append((query, top_k))
        return candidates

    def rerank(chunks, query):
        assert chunks is candidates
        assert query == "学习年限"
        return [{**chunk, "rerank_score": 0.9} for chunk in chunks]

    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "search_relevant_chunks", search)
    monkeypatch.setattr(pipeline, "rerank", rerank)
    result = invoke_local_tool(
        registry=tool_registry, tool_name="search_knowledge_base",
        tool_args={"query": "  学习年限  ", "top_k": "5"},
    )

    assert result["ok"] is True and result["status"] == "success"
    assert calls == [("学习年限", 5)]
    assert result["tool_input"] == {"query": "学习年限", "top_k": 5}
    assert result["output"] == {
        "count": 2, "chunks": [{**chunk, "rerank_score": 0.9} for chunk in candidates],
        "top_chunk_ids": ["a", "b"], "top_scores": [0.8765, 0.0],
    }


def test_search_knowledge_base_applies_tool_top_k_after_rerank(monkeypatch, tool_registry):
    from app.rag import pipeline

    chunks = [
        {"chunk_id": "a", "text": "候选 A", "hybrid_score": 0.9},
        {"chunk_id": "b", "text": "候选 B", "hybrid_score": 0.8},
    ]
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda query, top_k: chunks)
    monkeypatch.setattr(pipeline, "rerank", lambda candidates, query: candidates[::-1])
    result = invoke_local_tool(
        registry=tool_registry, tool_name="search_knowledge_base",
        tool_args={"query": "政策", "top_k": 1},
        context=ToolInvocationContext(
            permissions=("knowledge_base:read",), effective_question="原问题", top_k=9,
        ),
    )

    assert result["ok"] is True
    assert result["output"] == {
        "count": 1, "chunks": [chunks[1]], "top_chunk_ids": ["b"], "top_scores": [0.8],
    }


def test_search_knowledge_base_rejects_inconsistent_output_counts(tool_registry):
    """知识库生产输出模型拒绝证据数量不一致的成功结果。"""
    spec = next((item for item in tool_registry if item.name == "search_knowledge_base"), None)
    assert spec is not None
    assert spec.output_model is SearchKnowledgeBaseOutput
    registry = []
    registry.append(
        spec.model_copy(update={'handler': lambda args, context: tool_success(
                "search_knowledge_base",
                args,
                {
                    "count": 1,
                    "chunks": [],
                    "top_chunk_ids": [],
                    "top_scores": [],
                },
            )})
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
    from app.rag import pipeline

    received = []

    def fake_get_chunk_full_text(value):
        received.append(value)
        return {"found": True, "chunk_id": value, "text": "测试正文"}

    monkeypatch.setattr(pipeline, "get_chunk_full_text", fake_get_chunk_full_text)
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
    from app.rag import pipeline

    documents = [
        {"file_name": "a.md", "source_path": "data/a.md", "chunk_count": 2},
        {"file_name": "b.md", "source_path": "data/b.md", "chunk_count": 1},
    ]
    monkeypatch.setattr(pipeline, "preview_docs", lambda: documents)

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
    registry = []
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

    registry.append(
        ToolSpec(
            name="search_knowledge_base",
            version="1.0.0",
            description="检索知识库",
            output_model=RootModel[dict],
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
    registry = []
    called = {"count": 0}

    def fake_handler(args, context):
        called["count"] += 1
        return {"status": "success", "ok": True, "tool_name": "search_knowledge_base"}

    registry.append(
        ToolSpec(
            name="search_knowledge_base",
            version="1.0.0",
            description="检索知识库",
            output_model=RootModel[dict],
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


def test_execution_entry_accepts_explicit_context(tool_registry, monkeypatch):
    """验证唯一执行入口保留显式请求上下文。"""
    monkeypatch.setattr("app.rag.pipeline.preview_docs", lambda: [])
    context = ToolInvocationContext(trace_id="test-trace", permissions=("knowledge_base:read",))
    result = invoke_local_tool(
        registry=tool_registry,
        tool_name="list_documents",
        tool_args={},
        context=context,
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"] == {"count": 0, "documents": []}
