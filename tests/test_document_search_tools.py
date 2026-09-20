"""文档发现、精确硬范围检索以及受治理 Agent 引用链路。"""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.agent import framework_agent, framework_tools
from app.agent.final_response import collect_evidence_from_messages
from app.contracts.chat import RAG_DISCOVERY_TOOL_NAMES, RAG_EVIDENCE_TOOL_NAMES, ToolMode
from app.rag import pipeline
from app.rag.store import build_docs_registry_from_chunks
from app.tools.catalog import get_local_tool_specs
from app.tools.context import ToolInvocationContext
from app.tools.schemas import (
    FindDocumentsArgs, FindDocumentsOutput, SearchDocumentArgs, SearchDocumentOutput,
)
from tests.support.agent_runtime import FakeToolCallingModel, build_agent_harness
from tests.support.local_tools import invoke_local_tool


@pytest.fixture
def document_kb(monkeypatch):
    names = [
        "2024研究生奖学金管理办法.pdf", "2026研究生奖学金管理办法.pdf",
        "研究生学籍管理规定.pdf", "本科奖学金实施办法.pdf",
    ]
    chunks = [
        {
            "text": f"{name}：" + ("申请需提交成绩单。" if position == 1 else "申请于九月开始。"),
            "metadata": {
                "source_id": f"source-{index}", "document_uid": f"uid-{index}",
                "file_name": name, "source_uri": f"file:///kb/{name}",
                "source_path": f"/kb/{name}", "file_type": "pdf", "source_type": "local_file",
                "position": position, "chunk_id": index * 2 + position,
            },
        }
        for index, name in enumerate(names) for position in (1, 2)
    ]
    encoder = Mock()
    encoder.predict.side_effect = lambda pairs: [float("九月" in text) for _, text in pairs]
    monkeypatch.setattr(pipeline, "chunks", chunks)
    monkeypatch.setattr(pipeline, "docs_registry", build_docs_registry_from_chunks(chunks))
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "rerank_model", encoder)
    return chunks, encoder


@pytest.fixture
def invoke(settings_factory):
    settings = settings_factory()
    context = ToolInvocationContext(tool_mode=ToolMode.RAG_ONLY, permissions=("knowledge_base:read",))

    def run(name, **args):
        return invoke_local_tool(
            get_local_tool_specs(settings), name, args, context=context, settings=settings,
        )

    return run


def test_discovery_ranks_topic_and_exact_names_without_evidence(document_kb, invoke):
    chunks, encoder = document_kb
    result = invoke("find_documents", query="  研究生奖学金  ")
    assert result["ok"] is True
    output = FindDocumentsOutput.model_validate(result["output"])
    assert output.query == "研究生奖学金"
    assert [item.source_id for item in output.documents[:2]] == ["source-0", "source-1"]
    assert output.count == len(output.documents)
    assert all(item.chunk_count == 2 for item in output.documents)
    assert output.documents[0].document_uid == "uid-0"
    assert output.documents[0].source_uri == chunks[0]["metadata"]["source_uri"]
    assert "evidence" not in result["output"]
    assert "page_content" not in json.dumps(result)
    encoder.predict.assert_not_called()
    message = ToolMessage(content=json.dumps(result), name="find_documents", tool_call_id="find")
    assert collect_evidence_from_messages([message]) == {}
    assert "find_documents" in RAG_DISCOVERY_TOOL_NAMES
    assert "find_documents" not in RAG_EVIDENCE_TOOL_NAMES

    for query in ("2026研究生奖学金管理办法", "2026研究生奖学金管理办法.PDF"):
        exact = invoke("find_documents", query=query, limit=1)["output"]
        assert exact["count"] == len(exact["documents"]) == 1
        assert exact["documents"][0]["source_id"] == "source-1"


@pytest.mark.parametrize("field,query", [
    ("source_id", "source-1"), ("document_uid", "uid-1"),
    ("source_path", "/kb/2026研究生奖学金管理办法.pdf"),
    ("source_uri", "file:///kb/2026研究生奖学金管理办法.pdf"),
])
def test_discovery_matches_stable_metadata(document_kb, invoke, field, query):
    output = invoke("find_documents", query=query)["output"]
    assert output["documents"][0][field] == query
    assert output["documents"][0]["source_id"] == "source-1"


def test_discovery_is_stable_and_does_not_search_body(document_kb, invoke, monkeypatch):
    chunks, _ = document_kb
    before = invoke("find_documents", query="研究生奖学金")["output"]
    monkeypatch.setattr(pipeline, "chunks", list(reversed(chunks)))
    assert invoke("find_documents", query="研究生奖学金")["output"] == before
    assert invoke("find_documents", query="申请需提交成绩单")["output"]["documents"] == []


def test_discovery_never_invents_missing_ids_or_optional_fields(document_kb, invoke):
    chunks, _ = document_kb
    for chunk in chunks[:2]:
        del chunk["metadata"]["document_uid"]
        del chunk["metadata"]["source_uri"]
    chunks.append({"text": "legacy", "metadata": {"file_name": "旧版研究生奖学金.pdf"}})
    output = invoke("find_documents", query="研究生奖学金")["output"]
    item = next(item for item in output["documents"] if item["source_id"] == "source-0")
    assert item["document_uid"] is None and item["source_uri"] is None
    assert all(item["file_name"] != "旧版研究生奖学金.pdf" for item in output["documents"])


@pytest.mark.parametrize("args", [
    {"query": "a" * 501},
    {"query": "x", "limit": 0}, {"query": "x", "limit": 101},
])
def test_discovery_rejects_invalid_arguments(args):
    with pytest.raises(ValidationError):
        FindDocumentsArgs.model_validate(args)


@pytest.mark.parametrize("args", [
    {"document": " ", "query": "q"}, {"document": "s", "query": "\n"},
    {"document": "s", "query": "q", "top_k": 0},
    {"document": "s", "query": "q", "top_k": 11},
])
def test_scoped_search_rejects_invalid_arguments(args):
    with pytest.raises(ValidationError):
        SearchDocumentArgs.model_validate(args)


@pytest.mark.parametrize("document", [
    "source-1", "uid-1", "2026研究生奖学金管理办法.pdf", "2026研究生奖学金管理办法",
    "  ２０２６研究生奖学金管理办法.PDF  ",
])
def test_exact_locators_hard_scope_before_rerank(document_kb, invoke, monkeypatch, document):
    chunks, encoder = document_kb
    original = deepcopy(chunks)
    global_search = Mock(side_effect=AssertionError("scoped search must not use global retrieval"))
    monkeypatch.setattr(pipeline, "search_relevant_chunks", global_search)
    result = invoke("search_document", document=document, query="  何时申请？  ", top_k=1)
    assert result["ok"] is True
    output = SearchDocumentOutput.model_validate(result["output"])
    assert output.source_id == "source-1"
    assert output.count == len(output.evidence) == 1
    assert all(item.metadata["source_id"] == "source-1" for item in output.evidence)
    assert "九月" in output.evidence[0].page_content
    encoder.predict.assert_called_once_with([
        ("何时申请？", chunk["text"]) for chunk in chunks[2:4]
    ])
    global_search.assert_not_called()
    assert chunks == original
    message = ToolMessage(content=json.dumps(result), name="search_document", tool_call_id="search")
    assert collect_evidence_from_messages([message]) == {item.id: item for item in output.evidence}


def test_global_and_scoped_tools_keep_identical_evidence_ids(document_kb, invoke, monkeypatch):
    chunks, _ = document_kb
    scoped = invoke("search_document", document="source-1", query="何时申请？", top_k=2)
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda query, top_k: deepcopy(chunks[2:4]))
    global_result = invoke("search_knowledge_base", query="何时申请？", top_k=2)
    assert scoped["ok"] is global_result["ok"] is True
    assert [item["id"] for item in scoped["output"]["evidence"]] == [
        item["id"] for item in global_result["output"]["evidence"]
    ]


@pytest.mark.parametrize("document", ["missing", "管理办法", "source-", "uid-", "uid-1-extra"])
def test_unresolved_or_fuzzy_locator_fails_before_rerank(document_kb, invoke, document):
    _, encoder = document_kb
    result = invoke("search_document", document=document, query="何时申请？")
    assert result["ok"] is False and result["error_code"] == "invalid_arguments"
    assert result["retryable"] is False
    assert "find_documents" in result["error"]
    encoder.predict.assert_not_called()


@pytest.mark.parametrize("document", ["same.pdf", "same", "shared-uid"])
def test_ambiguous_locator_fails_closed_and_stable_id_disambiguates(document_kb, invoke, document):
    chunks, encoder = document_kb
    for chunk in chunks:
        chunk["metadata"].update(file_name="same.pdf", document_uid="shared-uid", source_path="/same.pdf")
    result = invoke("search_document", document=document, query="何时申请？")
    assert result["ok"] is False and result["error_code"] == "invalid_arguments"
    assert "ambiguous document" in result["error"] and "find_documents" in result["error"]
    encoder.predict.assert_not_called()
    assert invoke("find_documents", query="same.pdf")["output"]["count"] == 4
    resolved = invoke("search_document", document="source-1", query="何时申请？")
    assert resolved["ok"] is True
    assert {item["metadata"]["source_id"] for item in resolved["output"]["evidence"]} == {"source-1"}


def test_locator_priority_prefers_source_id_then_uid(document_kb, invoke):
    chunks, _ = document_kb
    for chunk in chunks[:2]:
        chunk["metadata"].update(document_uid="source-1", file_name="uid-1")
    for locator in ("source-1", "uid-1"):
        result = invoke("search_document", document=locator, query="何时申请？")
        assert result["output"]["source_id"] == "source-1"


@pytest.mark.parametrize("name,args", [
    ("find_documents", {"query": "奖学金"}),
    ("search_document", {"document": "source-1", "query": "何时申请？"}),
])
def test_unready_kb_is_dependency_error(document_kb, invoke, monkeypatch, name, args):
    monkeypatch.setattr(pipeline, "RAG_READY", False)
    result = invoke(name, **args)
    assert result["ok"] is False and result["error_code"] == "dependency_unavailable"


def test_output_counts_and_discovery_evidence_exclusion():
    with pytest.raises(ValidationError, match="count"):
        FindDocumentsOutput(query="q", count=1, total=1, documents=[])
    with pytest.raises(ValidationError, match="evidence"):
        FindDocumentsOutput(query="q", count=0, total=0, documents=[], evidence=[])
    with pytest.raises(ValidationError, match="count"):
        SearchDocumentOutput(query="q", document="s", source_id="s", count=1, evidence=[])


@pytest.mark.asyncio
async def test_agent_discovers_searches_and_cites_in_rag_only(
    document_kb, settings_factory, monkeypatch,
):
    audit = []

    class Sink:
        def write(self, event):
            audit.append(event)

    def search_found_document(model, messages):
        message = messages[-1]
        assert isinstance(message, ToolMessage) and message.name == "find_documents"
        assert collect_evidence_from_messages(messages) == {}
        result = (message.artifact or json.loads(message.content))
        assert result["ok"] is True and result["governance"]["decision"] == "allow"
        source_id = result["output"]["documents"][0]["source_id"]
        return AIMessage(content="", tool_calls=[{
            "name": "search_document", "args": {"document": source_id, "query": "何时申请？", "top_k": 1},
            "id": "scoped-call", "type": "tool_call",
        }])

    def answer(model, messages):
        message = messages[-1]
        assert isinstance(message, ToolMessage) and message.name == "search_document"
        result = (message.artifact or json.loads(message.content))
        assert result["ok"] is True and result["governance"]["decision"] == "allow"
        evidence = collect_evidence_from_messages(messages)
        assert len(evidence) == 1
        assert all(item.metadata["source_id"] == "source-1" for item in evidence.values())
        return model.final_response(outcome="answer", answer="申请于九月开始。", evidence_ids=list(evidence))

    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{
            "name": "find_documents", "args": {"query": "2026研究生奖学金管理办法"},
            "id": "discovery-call", "type": "tool_call",
        }]),
        search_found_document, answer,
    ])
    monkeypatch.setattr(framework_tools, "get_default_tool_audit_sink", lambda settings: Sink())
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    context = ToolInvocationContext(
        permissions=("knowledge_base:read", "web:search"), tool_mode=ToolMode.RAG_ONLY,
    )
    output = [state async for state in build_agent_harness(settings_factory()).astream({
        "original_question": "2026年研究生奖学金何时申请？",
        "effective_question": "2026年研究生奖学金何时申请？",
    }, context=context)][-1]
    result = output["result"]
    assert result["status"] == "OK" and result["answer"] == "申请于九月开始。"
    assert len(model.received_messages) == 3
    assert {"find_documents", "search_document"} <= set(model.bound_tool_names)
    assert "web_search" not in model.bound_tool_names
    evidence = collect_evidence_from_messages(output["messages"])
    assert [ref["reference_id"] for ref in result["references"]] == list(evidence)
    assert result["references"][0]["file_name"] == "2026研究生奖学金管理办法.pdf"
    assert result["tool_result_summary"] == "文档发现完成，找到 4 份候选文档；文档内检索完成，召回 1 个 chunk"
    assert [(event.tool_name, event.event_type.value) for event in audit] == [
        (name, event) for name in ("find_documents", "search_document")
        for event in ("requested", "started", "succeeded")
    ]


@pytest.mark.parametrize("args", [{}, {"query": None}, {"query": ""}, {"query": "  "}])
def test_empty_discovery_lists_metadata_without_evidence(document_kb, invoke, args):
    result = invoke("find_documents", **args)
    assert result["ok"] and result["output"]["count"] == result["output"]["total"] > 0
    assert "evidence" not in result["output"]
    assert all("page_content" not in item for item in result["output"]["documents"])


@pytest.mark.asyncio
async def test_agent_compares_two_scoped_searches(document_kb, settings_factory, monkeypatch):
    def answer(model, messages):
        evidence = collect_evidence_from_messages(messages)
        assert {doc.metadata["source_id"] for doc in evidence.values()} == {"source-0", "source-1"}
        return model.final_response(outcome="answer", answer="两份文档均要求提交成绩单。", evidence_ids=list(evidence))

    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{
            "name": "search_document", "args": {"document": source, "query": "申请要求", "top_k": 1},
            "id": source, "type": "tool_call",
        } for source in ("source-0", "source-1")]),
        answer,
    ])
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    context = ToolInvocationContext(permissions=("knowledge_base:read",))
    output = [state async for state in build_agent_harness(settings_factory()).astream({
        "original_question": "比较两份文档的申请要求", "effective_question": "比较两份文档的申请要求",
    }, context=context)][-1]
    assert output["result"]["status"] == "OK"
    assert len(output["result"]["references"]) == 2
    assert [event["name"] for event in output["governance_events"] if event["kind"] == "tool"] == ["search_document", "search_document"]
