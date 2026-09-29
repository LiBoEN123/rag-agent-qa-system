"""显式同文档上下文窗口、locator 边界以及 Agent 引用链路。"""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.agent import framework_agent
from app.agent.final_response import collect_evidence_from_messages
from app.contracts.chat import RAG_DISCOVERY_TOOL_NAMES, RAG_EVIDENCE_TOOL_NAMES, ToolMode
from app.rag import debug, runtime as pipeline
from app.tools.catalog import get_local_tool_specs
from app.tools.context import ToolInvocationContext
from app.tools.knowledge_base import chunk_to_evidence_document
from app.tools.schemas import GetChunkContextOutput, GetChunkContextArgs
from tests.support.agent_runtime import FakeToolCallingModel, build_agent_harness
from tests.support.local_tools import invoke_local_tool


@pytest.fixture
def context_kb(monkeypatch):
    # chunk_id 故意不连续，runtime 顺序打乱，并混入同名的另一文档。
    chunks = [
        {"text": f"{source} 第 {position} 段正文", "metadata": {
            "source_id": source, "position": position, "chunk_id": chunk_id,
            "file_name": "policy.pdf", "source_path": f"/kb/{source}/policy.pdf",
        }}
        for source, position, chunk_id in [
            ("source-A", 4, 900), ("source-B", 4, 36),
            ("source-A", 2, 800), ("source-A", 3, 35),
            ("source-B", 2, 34), ("source-A", 1, 200),
            ("source-A", 5, 2),
        ]
    ]
    forbidden = Mock(side_effect=AssertionError("context lookup must not retrieve or rerank"))
    monkeypatch.setattr(pipeline, "chunks", chunks)
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "search_relevant_chunks", forbidden)
    monkeypatch.setattr(pipeline, "rerank", forbidden)
    return chunks


@pytest.fixture
def invoke(settings_factory):
    settings = settings_factory()
    context = ToolInvocationContext(tool_mode=ToolMode.RAG_ONLY, permissions=("knowledge_base:read",))

    def run(**args):
        return invoke_local_tool(
            get_local_tool_specs(settings), "get_chunk_context", args,
            context=context, settings=settings,
        )

    return run


@pytest.mark.parametrize("chunk_id,before,after,positions", [
    (35, 1, 1, [2, 3, 4]),
    (200, 2, 2, [1, 2, 3]),
    (2, 2, 3, [3, 4, 5]),
    (35, 3, 3, [1, 2, 3, 4, 5]),
    (35, 0, 2, [3, 4, 5]),
    (35, 2, 0, [1, 2, 3]),
])
def test_window_is_source_scoped_ordered_and_keeps_identity(
    context_kb, invoke, chunk_id, before, after, positions,
):
    original = deepcopy(context_kb)
    result = invoke(chunk_id=chunk_id, before=before, after=after)
    assert result["ok"] is True
    output = GetChunkContextOutput.model_validate(result["output"])
    assert output.anchor_chunk_id == chunk_id
    assert (output.before, output.after) == (before, after)
    assert output.source_id == "source-A"
    assert output.count == len(output.evidence) == len(positions)
    assert [doc.metadata["position"] for doc in output.evidence] == positions
    assert all(doc.metadata["source_id"] == "source-A" for doc in output.evidence)
    originals = {chunk["metadata"]["chunk_id"]: chunk for chunk in context_kb}
    assert output.anchor_position == originals[chunk_id]["metadata"]["position"]
    assert chunk_id in [doc.metadata["chunk_id"] for doc in output.evidence]
    for document in output.evidence:
        assert isinstance(document, Document)
        assert document == chunk_to_evidence_document(originals[document.metadata["chunk_id"]])
    assert context_kb == original


def test_defaults_share_the_exact_existing_locator_contract(context_kb, invoke):
    assert GetChunkContextArgs.model_json_schema()["properties"]["chunk_id"] == (
        GetChunkContextArgs.model_json_schema()["properties"]["chunk_id"]
    )
    assert GetChunkContextArgs(chunk_id="35").model_dump() == {
        "chunk_id": 35, "before": 1, "after": 1,
    }
    output = invoke(chunk_id=35)["output"]
    assert [doc["metadata"]["position"] for doc in output["evidence"]] == [2, 3, 4]


def test_duplicate_chunks_and_missing_positions_are_not_extra_neighbors(context_kb, invoke):
    context_kb.extend([deepcopy(context_kb[3]), deepcopy(context_kb[2])])
    context_kb[:] = [chunk for chunk in context_kb if chunk["metadata"]["position"] != 4]
    output = invoke(chunk_id=35)["output"]
    assert [doc["metadata"]["position"] for doc in output["evidence"]] == [2, 3]
    assert output["count"] == len({doc["id"] for doc in output["evidence"]}) == 2


def test_anchor_and_neighbors_use_the_same_runtime_snapshot(context_kb, invoke, monkeypatch):
    lookup = debug.get_chunk_full_text

    def swap_runtime(chunks, chunk_id):
        anchor = lookup(chunks, chunk_id)
        monkeypatch.setattr(pipeline, "chunks", [])
        return anchor

    monkeypatch.setattr(debug, "get_chunk_full_text", swap_runtime)
    assert invoke(chunk_id=35)["output"]["count"] == 3


def test_missing_anchor_returns_stable_locator_error(context_kb, invoke):
    result = invoke(chunk_id=99999)
    assert result["ok"] is False and result["error_code"] == "invalid_arguments"
    assert result["retryable"] is False
    for text in ("chunk_id=99999", "metadata.chunk_id", "Document.id"):
        assert text in result["error"]


@pytest.mark.parametrize("args", [
    {"chunk_id": 0}, {"chunk_id": -1}, {"chunk_id": 1.5}, {"chunk_id": "missing"},
    {"chunk_id": 35, "before": -1}, {"chunk_id": 35, "before": 4},
    {"chunk_id": 35, "after": -1}, {"chunk_id": 35, "after": 4},
    {"chunk_id": 35, "before": 0, "after": 0},
])
def test_schema_rejects_invalid_locators_and_windows(invoke, args):
    assert invoke(**args)["error_code"] == "invalid_arguments"


def test_anchor_only_returns_exactly_the_anchor(context_kb, invoke):
    result = invoke(chunk_id=35, before=0, after=0)
    assert result["ok"] and result["output"]["count"] == 1
    assert result["output"]["evidence"][0]["metadata"]["chunk_id"] == 35


def test_evidence_id_cannot_be_used_as_chunk_locator(context_kb, invoke):
    document = chunk_to_evidence_document(context_kb[3])
    assert invoke(chunk_id=document.id)["error_code"] == "invalid_arguments"


@pytest.mark.parametrize("field,value", [
    ("source_id", None), ("source_id", ""), ("source_id", "  "), ("source_id", 1),
    ("position", None), ("position", 0), ("position", -1),
    ("position", True), ("position", 3.0), ("position", "3"),
])
def test_invalid_anchor_provenance_fails_closed(context_kb, invoke, field, value):
    metadata = context_kb[3]["metadata"]
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    result = invoke(chunk_id=35)
    assert result["ok"] is False and result["error_code"] == "invalid_arguments"
    assert "source_id" in result["error"] and "position" in result["error"]


def test_unlocatable_neighbors_are_not_inferred_from_list_order(context_kb, invoke):
    context_kb[0]["metadata"].pop("position")
    context_kb[2]["metadata"]["position"] = "2"
    result = invoke(chunk_id=35)
    assert result["ok"] is True
    assert [doc["metadata"]["position"] for doc in result["output"]["evidence"]] == [3]


def test_output_validates_count_and_existing_evidence_contract(context_kb, invoke):
    output = invoke(chunk_id=35)["output"]
    with pytest.raises(ValidationError, match="count"):
        GetChunkContextOutput.model_validate({**output, "count": 1})
    for field, value in (("id", ""), ("page_content", ""), ("content_is_untrusted", False)):
        invalid = deepcopy(output)
        document = invalid["evidence"][0]
        target = document if field in {"id", "page_content"} else document["metadata"]
        target[field] = value
        with pytest.raises(ValidationError, match=field):
            GetChunkContextOutput.model_validate(invalid)


@pytest.mark.parametrize("mode", [ToolMode.RAG_ONLY, ToolMode.FULL])
def test_missing_permission_blocks_before_handler(settings_factory, mode):
    settings = settings_factory()
    handler = Mock(side_effect=AssertionError("unauthorized handler executed"))
    spec = next(spec for spec in get_local_tool_specs(settings) if spec.name == "get_chunk_context")
    result = invoke_local_tool(
        [spec.model_copy(update={"handler": handler})], spec.name, {"chunk_id": 35},
        context=ToolInvocationContext(tool_mode=mode), settings=settings,
    )
    assert result["error_code"] == "permission_denied"
    assert result["governance"]["decision"] == "deny"
    assert result["runtime"]["attempt_count"] == 0
    handler.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("search_name", ["search_knowledge_base", "search_document"])
async def test_agent_explicitly_expands_search_hit_and_cites_neighbors(
    context_kb, settings_factory, monkeypatch, search_name,
):
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda query, top_k: [deepcopy(context_kb[3])])
    monkeypatch.setattr(pipeline, "rerank", lambda chunks, query: sorted(
        chunks, key=lambda chunk: chunk["metadata"]["chunk_id"] != 35,
    ))

    def expand(model, messages):
        message = messages[-1]
        assert isinstance(message, ToolMessage) and message.name == search_name
        output = (message.artifact or json.loads(message.content))["output"]
        assert output["count"] == 1  # search 本身没有自动扩展。
        anchor = output["evidence"][0]
        assert anchor["metadata"]["chunk_id"] == 35
        return AIMessage(content="", tool_calls=[{
            "name": "get_chunk_context", "args": {"chunk_id": anchor["metadata"]["chunk_id"]},
            "id": "context-call", "type": "tool_call",
        }])

    def answer(model, messages):
        message = messages[-1]
        assert isinstance(message, ToolMessage) and message.name == "get_chunk_context"
        assert message.status == "success"
        payload = (message.artifact or json.loads(message.content))
        assert payload["ok"] is True and message.artifact["governance"]["decision"] == "allow"
        evidence = payload["output"]["evidence"]
        assert [doc["metadata"]["position"] for doc in evidence] == [2, 3, 4]
        collected = collect_evidence_from_messages(messages)
        assert len(collected) == 3  # 重复 anchor 按既有 Document.id 去重。
        assert set(collected) == {doc["id"] for doc in evidence}
        return model.final_response(
            scope_match="matched", support_level="full",
            outcome="answer", answer="相邻正文见第 2、4 段。",
            evidence_ids=[evidence[0]["id"], evidence[2]["id"]],
        )

    args = {"query": "上下文是什么？", "top_k": 1}
    if search_name == "search_document":
        args["document"] = "source-A"
    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{
            "name": search_name, "args": args, "id": "search-call", "type": "tool_call",
        }]),
        expand, answer,
    ])
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    context = ToolInvocationContext(tool_mode=ToolMode.RAG_ONLY, permissions=("knowledge_base:read",))
    output = [state async for state in build_agent_harness(settings_factory()).astream({
        "original_question": "上下文是什么？", "effective_question": "上下文是什么？",
    }, context=context)][-1]
    result = output["result"]
    assert result["status"] == "OK"
    assert len(model.received_messages) == 3
    assert "get_chunk_context" in model.bound_tool_names
    assert "get_chunk_context" in RAG_EVIDENCE_TOOL_NAMES
    assert "get_chunk_context" not in RAG_DISCOVERY_TOOL_NAMES
    assert [ref["position"] for ref in result["references"]] == [2, 4]
    assert [ref["reference_id"] for ref in result["references"]] == output["structured_response"].evidence_ids
    trace = [item for item in result["tool_trace"] if item.get("step") == "tool_call"]
    assert [item["tool_name"] for item in trace] == [search_name, "get_chunk_context"]
    assert trace[-1]["summary"] == "上下文扩展完成，返回 3 个同文档 chunk"
