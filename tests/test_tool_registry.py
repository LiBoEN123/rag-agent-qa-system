import json
from copy import deepcopy

from pydantic import RootModel

import pytest

from app.tools.context import ToolInvocationContext
from app.tools.catalog import ToolSpec
from app.tools.result import tool_success
from app.tools.knowledge_base import chunk_to_evidence_document
from tests.support.local_tools import invoke_local_tool
from app.tools.schemas import (
    SearchKnowledgeBaseArgs,
    SearchKnowledgeBaseOutput,
)


def test_tool_registry_names_include_builtin_tools(tool_registry):
    """验证内置工具已注册到统一注册中心。"""
    names = [item.name for item in tool_registry]

    assert "search_knowledge_base" in names
    assert "web_search" in names
    assert "web_fetch" in names
    assert "web_pdf_fetch" in names


def test_builtin_tools_publish_v2_contract_metadata(tool_registry):
    """验证注册生产工具发布完整且符合要求的 V2 契约。"""
    from app.agent.framework_tools import local_tool_contract

    specs = [local_tool_contract(item) for item in tool_registry]
    specs_by_name = {spec["name"]: spec for spec in specs}
    expected_permissions = {
        "search_knowledge_base": ["knowledge_base:read"],
        "find_documents": ["knowledge_base:read"],
        "search_document": ["knowledge_base:read"],

        "get_chunk_context": ["knowledge_base:read"],
        "web_search": ["web:search"],
        "web_fetch": ["web:fetch"],
        "web_pdf_fetch": ["web:fetch"],
    }
    expected_timeouts = {
        "search_knowledge_base": 120.0,
        "find_documents": 5.0,
        "search_document": 120.0,

        "get_chunk_context": 5.0,
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
        assert spec["required_permissions"] == expected_permissions[name]
        assert spec["timeout"] == expected_timeouts[name]
        assert spec["max_retries"] == (
            1 if name in {"web_search", "web_fetch", "web_pdf_fetch"} else 0
        )
        assert spec["output_model"] == {
            "search_knowledge_base": "SearchKnowledgeBaseOutput",
            "find_documents": "FindDocumentsOutput",
            "search_document": "SearchDocumentOutput",

            "get_chunk_context": "GetChunkContextOutput",
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
        "query",
        "count",
        "evidence",
    }
    assert set(knowledge_schema["required"]) == set(
        knowledge_schema["properties"]
    )
    assert "RAGResult" not in json.dumps(knowledge_schema)


def test_invoke_local_tool_runs_search_and_rerank(monkeypatch, tool_registry):
    """真实工具保留参数规范化、重排分数及 metadata 字段回退。"""
    from app.rag import runtime as pipeline

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
    output = result["output"]
    assert set(output) == {"query", "count", "evidence"}
    assert output["query"] == "学习年限"
    assert output["count"] == len(output["evidence"]) == 2
    assert [doc["page_content"] for doc in output["evidence"]] == ["候选 A", "候选 B"]
    metadata = [doc["metadata"] for doc in output["evidence"]]
    assert [item["chunk_id"] for item in metadata] == ["a", "b"]
    assert [item["hybrid_score"] for item in metadata] == [0.87654, "invalid-score"]
    assert all(item["rerank_score"] == 0.9 for item in metadata)


def test_search_knowledge_base_applies_tool_top_k_after_rerank(monkeypatch, tool_registry):
    from app.rag import runtime as pipeline

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
            permissions=("knowledge_base:read",), effective_question="原问题",
        ),
    )

    assert result["ok"] is True
    assert result["output"] == {
        "query": "政策", "count": 1,
        "evidence": [chunk_to_evidence_document(chunks[1]).model_dump(mode="json")],
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
                    "query": args["query"],
                    "evidence": [],
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
    result = invoke_local_tool(
            registry=tool_registry,
            tool_name="search_knowledge_base",
            tool_args=tool_args,
    )
    assert result["error_code"] == "invalid_arguments"


@pytest.mark.parametrize("source_id", ["source-policy"])
def test_retrieval_and_real_chunk_lookup_share_identity_and_provenance(
    monkeypatch, tool_registry, source_id,
):
    from app.rag import runtime as pipeline

    metadata = {
        "chunk_id": 382, "source_path": "data/policy.pdf", "position": 2,
        "file_name": "policy.pdf", "file_type": "pdf", "page_number": 3,
        "document_uid": "document-policy", "title_path": "政策 > 申请",
        "block_id": "b9", "table_id": "t1", "row_start": 4, "row_end": 6,
        "source_type": "local_file",
    }
    if source_id:
        metadata.update(source_id=source_id, source_uri="file:///data/policy.pdf")
    chunk = {"text": "完整申请要求", "metadata": metadata}
    original = deepcopy(chunk)
    monkeypatch.setattr(pipeline, "chunks", [chunk])
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda query, top_k: [chunk])
    scores = {"bm25_score": 1.2, "vector_score": 0.4, "hybrid_score": 0.72, "rerank_score": 0.9}
    monkeypatch.setattr(pipeline, "rerank", lambda chunks, query: [{**chunk, **scores}])

    search = invoke_local_tool(tool_registry, "search_knowledge_base", {"query": "申请", "top_k": 1})
    lookup = invoke_local_tool(tool_registry, "get_chunk_context", {"chunk_id": 382, "before": 0, "after": 0})
    assert search["ok"] is lookup["ok"] is True
    retrieved = search["output"]["evidence"][0]
    fetched = lookup["output"]["evidence"][0]
    assert retrieved["id"] == fetched["id"]
    assert retrieved["page_content"] == fetched["page_content"] == chunk["text"]
    for document in (retrieved, fetched):
        assert document["metadata"]["source_type"] == "knowledge_base"
        assert document["metadata"]["knowledge_source_type"] == "local_file"
        assert document["metadata"]["content_is_untrusted"] is True
        assert document["metadata"]["source_uri"] == metadata.get("source_uri", metadata["source_path"])
        for key, value in metadata.items():
            if key != "source_type":
                assert document["metadata"][key] == value
    for key, value in scores.items():
        assert retrieved["metadata"][key] == value
    assert chunk == original  # 只投影，不修改原索引。


def test_kb_identity_ignores_rank_scores_and_runtime_chunk_id():
    chunk = {
        "text": "原文", "chunk_id": 0, "reference_id": "Reference 1",
        "metadata": {"source_id": "source-a", "position": 2, "source_path": "data/a.md"},
    }
    first = chunk_to_evidence_document(chunk)
    assert first.id == chunk_to_evidence_document(chunk).id
    assert first.id == chunk_to_evidence_document({
        **chunk, "chunk_id": 999, "reference_id": "Reference 9", "rerank_score": 0.1,
    }).id
    for changed in ({"source_id": "source-b"}, {"position": 3}):
        assert first.id != chunk_to_evidence_document({
            **chunk, "metadata": {**chunk["metadata"], **changed},
        }).id
    assert "reference_id" not in first.metadata
    assert "snippet" not in first.metadata


def test_kb_chunk_id_fallback_is_deterministic_and_source_scoped():
    chunk = {"chunk_id": 0, "text": "原文"}
    first = chunk_to_evidence_document(chunk)
    assert first.id == chunk_to_evidence_document({**chunk, "chunk_id": "0"}).id
    assert first.id != chunk_to_evidence_document({**chunk, "chunk_id": 1}).id
    assert chunk_to_evidence_document({**chunk, "source_path": "a.md"}).id != (
        chunk_to_evidence_document({**chunk, "source_path": "b.md"}).id
    )
    with pytest.raises(ValueError, match="locator"):
        chunk_to_evidence_document({"text": "无来源"})


def test_kb_source_uri_prefers_original_metadata():
    document = chunk_to_evidence_document({
        "chunk_id": "a", "text": "原文", "source_uri": "https://example.com/projected",
        "metadata": {"source_uri": "https://example.com/original", "source_path": "cache/a.md"},
    })
    assert document.metadata["source_uri"] == "https://example.com/original"
    assert document.metadata["source_path"] == "cache/a.md"


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

    result = invoke_local_tool(
            registry=registry,
            tool_name="search_knowledge_base",
            tool_args={"query": "测试", "top_k": 999},
    )
    assert result["error_code"] == "invalid_arguments"

    assert called["count"] == 0


def test_execution_entry_accepts_explicit_context(tool_registry, monkeypatch):
    """验证唯一执行入口保留显式请求上下文。"""
    monkeypatch.setattr("app.rag.runtime.chunks", [])
    monkeypatch.setattr("app.rag.runtime.RAG_READY", True)
    context = ToolInvocationContext(trace_id="test-trace", permissions=("knowledge_base:read",))
    result = invoke_local_tool(
        registry=tool_registry,
        tool_name="find_documents",
        tool_args={},
        context=context,
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"] == {"query": None, "count": 0, "total": 0, "documents": []}
