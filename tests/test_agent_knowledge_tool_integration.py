"""真实 KB handler / StructuredTool / create_agent 的证据返回与二次检索。"""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError
from langchain_core.tools import StructuredTool

from app.agent import framework_agent, framework_tools
from app.agent.final_response import FinalAgentResponse
from app.rag import runtime as pipeline
from app.tools.catalog import get_local_tool_specs
from app.tools.context import ToolInvocationContext
from app.tools.knowledge_base import search_knowledge_base
from app.tools.schemas import GetChunkContextArgs, SearchKnowledgeBaseArgs
from tests.support.agent_runtime import FakeToolCallingModel, build_agent_harness
from tests.support.local_tools import local_tools


@pytest.fixture
def retrieval(monkeypatch):
    chunks = [
        {"chunk_id": "a", "text": "申请人需要提交成绩单。", "file_name": "policy.md",
         "hybrid_score": 0.91234, "bm25_score": 5.0, "vector_score": 0.7},
        {"chunk_id": "b", "text": "奖学金申请于九月开始。", "file_name": "policy.md",
         "hybrid_score": 0.81234, "bm25_score": 4.0, "vector_score": 0.6},
    ]
    search = Mock(side_effect=lambda query, top_k: deepcopy(chunks))
    cross_encoder = Mock()
    cross_encoder.predict.return_value = [0.2, 0.9]
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "search_relevant_chunks", search)
    monkeypatch.setattr(pipeline, "rerank_model", cross_encoder)
    yield search, cross_encoder


async def _invoke(monkeypatch, model, settings_factory, **settings_overrides):
    settings = settings_factory(**settings_overrides)
    contexts, args_seen, audit_events = [], [], []

    def handler(args, context):
        args_seen.append(dict(args))
        contexts.append(context)
        return search_knowledge_base(args, context)

    class Sink:
        def write(self, event):
            audit_events.append(event)

    spec = next(spec for spec in get_local_tool_specs(settings)
                if spec.name == "search_knowledge_base")
    monkeypatch.setattr(framework_tools, "get_default_tool_audit_sink", lambda settings: Sink())
    tools = local_tools([spec.model_copy(update={"handler": handler})], settings)
    monkeypatch.setattr(framework_tools, "get_local_tools", lambda settings: tools)
    monkeypatch.setattr(framework_agent, "get_agent_model", lambda: model)
    context = ToolInvocationContext(
        trace_id="trace-kb", session_id="session-kb", principal_id="kb-test",
        permissions=("knowledge_base:read",), source="knowledge_tool_integration_test",
    )
    output = [state async for state in build_agent_harness(settings).astream({
        "original_question": "它何时开始？",
        "effective_question": "奖学金申请条件与开始时间是什么？",
        "trace_id": context.trace_id, "session_id": context.session_id,
    }, context=context)][-1]
    assert "knowledge_base_result" not in output
    assert isinstance(tools[0], StructuredTool)
    assert tools[0].args_schema == SearchKnowledgeBaseArgs.model_json_schema()
    assert "runtime" not in tools[0].tool_call_schema["properties"]
    assert isinstance(model.received_messages[0][0], SystemMessage)
    assert [m.content for m in model.received_messages[0] if isinstance(m, HumanMessage)] == [
        "奖学金申请条件与开始时间是什么？"
    ]
    assert all(c.effective_question == "奖学金申请条件与开始时间是什么？" for c in contexts)
    assert all(not hasattr(c, "top_k") and not hasattr(c, "decompose_top_k") for c in contexts)
    assert all(c.permissions == context.permissions for c in contexts)
    assert {e.trace_id for e in audit_events} == {context.trace_id}
    assert {e.session_id for e in audit_events} == {context.session_id}
    assert {e.principal_id for e in audit_events} == {context.principal_id}
    return output, args_seen, audit_events


def _search(query="奖学金 申请时间", top_k=1, call_id="search-a"):
    return AIMessage(content="", tool_calls=[{
        "name": "search_knowledge_base", "args": {"query": query, "top_k": top_k},
        "id": call_id, "type": "tool_call",
    }])


def _grounded_answer(answer, *, indexes=None):
    def respond(model, messages):
        ids = list(dict.fromkeys(
            document["id"]
            for message in messages if isinstance(message, ToolMessage) and message.name == "search_knowledge_base"
            for document in (message.artifact or json.loads(message.content))["output"]["evidence"]
        ))
        return model.final_response(
            outcome="answer", answer=answer,
            evidence_ids=ids if indexes is None else [ids[i] for i in indexes],
        )
    return respond


def _evidence(model, round_index, call_id):
    message = model.received_messages[round_index][-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == call_id
    assert message.name == "search_knowledge_base"
    assert message.status == "success"
    payload = (message.artifact or json.loads(message.content))
    assert payload["ok"] is True
    assert payload["runtime"]["attempt_count"] == 1
    assert payload["governance"]["decision"] == "allow"
    assert set(payload["output"]) == {"query", "count", "evidence"}
    return payload["output"]


@pytest.mark.asyncio
async def test_search_tool_message_separates_evidence_id_from_chunk_locator(
    monkeypatch, settings_factory, retrieval,
):
    """q054 合同回归：已读取的正文可直接回答，两个 ID 不可互换。"""
    search, cross_encoder = retrieval
    search.side_effect = None
    search.return_value = [{
        "text": "通知书由学院送达学生本人。",
        "metadata": {"chunk_id": 382, "source_id": "academic-warning", "position": 2},
    }]
    cross_encoder.predict.return_value = [0.9]
    model = FakeToolCallingModel([
        _search("学业警示 通知书 模板"), _grounded_answer("通知书由学院送达学生本人。"),
    ])
    output, _, _ = await _invoke(monkeypatch, model, settings_factory)
    document = _evidence(model, 1, "search-a")["evidence"][0]
    assert document["id"].startswith("knowledge_base:")
    assert document["page_content"] == "通知书由学院送达学生本人。"
    with pytest.raises(ValidationError):
        GetChunkContextArgs(chunk_id=document["id"])
    assert GetChunkContextArgs(chunk_id=document["metadata"]["chunk_id"]).chunk_id == 382
    assert output["result"]["status"] == "OK"
    assert output["result"]["references"][0]["reference_id"] == document["id"]


@pytest.mark.asyncio
async def test_search_returns_reranked_evidence_to_next_model(monkeypatch, settings_factory, retrieval):
    search, cross_encoder = retrieval
    model = FakeToolCallingModel([_search(), _grounded_answer("奖学金申请于九月开始。")])
    output, args_seen, audit = await _invoke(monkeypatch, model, settings_factory)

    search.assert_called_once_with("奖学金 申请时间", top_k=1)
    assert args_seen == [{"query": "奖学金 申请时间", "top_k": 1}]
    cross_encoder.predict.assert_called_once_with([
        ("奖学金 申请时间", "申请人需要提交成绩单。"),
        ("奖学金 申请时间", "奖学金申请于九月开始。"),
    ])
    assert len(model.received_messages) == 2
    evidence = _evidence(model, 1, "search-a")
    assert evidence["count"] == 1
    assert evidence["query"] == "奖学金 申请时间"
    document = evidence["evidence"][0]
    assert document["id"] and document["type"] == "Document"
    assert document["page_content"] == "奖学金申请于九月开始。"
    assert document["metadata"]["chunk_id"] == "b"
    assert document["metadata"]["hybrid_score"] == 0.81234
    assert document["metadata"]["rerank_score"] == 0.9
    assert document["metadata"]["source_type"] == "knowledge_base"
    assert document["metadata"]["source_uri"]
    assert document["metadata"]["content_is_untrusted"] is True
    result = output["result"]
    assert result["answer"] == "奖学金申请于九月开始。"
    assert [ref["reference_id"] for ref in result["references"]] == [document["id"]]
    assert result["retrieval_confidence"] == result["answer_confidence"] == 0.0
    assert result["llm_call_count"] == 2
    assert "decompose_llm_used" not in result
    trace = next(s for s in result["tool_trace"] if s["step"] == "tool_call")
    assert trace["final_chunk_count"] == 1
    assert trace["top_chunk_ids"] == ["b"] and trace["top_scores"] == [0.8123]
    assert trace["hits"] == [{"chunk_id": "b", "rerank": 0.9, "score": 0.81234,
                              "bm25": 4.0, "vector": 0.6, "file_name": "policy.md"}]
    assert [e.event_type.value for e in audit] == ["requested", "started", "succeeded"]
    assert audit[-1].sanitized_output_summary == {
        "count": 1, "top_chunk_ids": ["b"], "top_scores": [0.8123],
    }


@pytest.mark.asyncio
async def test_agent_tool_top_k_eight_limits_real_handler_evidence(monkeypatch, settings_factory, retrieval):
    search, cross_encoder = retrieval
    search.side_effect = None
    search.return_value = [
        {"chunk_id": str(i), "text": f"申请材料 {i}", "file_name": "policy.md"}
        for i in range(10)
    ]
    cross_encoder.predict.return_value = list(range(10))
    model = FakeToolCallingModel([
        _search("奖学金 申请材料", top_k=8),
        _grounded_answer("需要提交申请材料。", indexes=[0]),
    ])
    output, args_seen, _ = await _invoke(monkeypatch, model, settings_factory)

    search.assert_called_once_with("奖学金 申请材料", top_k=8)
    assert args_seen == [{"query": "奖学金 申请材料", "top_k": 8}]
    evidence = _evidence(model, 1, "search-a")
    assert evidence["count"] == len(evidence["evidence"]) == 8
    assert [d["metadata"]["chunk_id"] for d in evidence["evidence"]] == [str(i) for i in range(9, 1, -1)]
    assert output["result"]["status"] == "OK"


@pytest.mark.asyncio
async def test_zero_results_are_successful_evidence(monkeypatch, settings_factory, retrieval):
    search, cross_encoder = retrieval
    search.side_effect = None
    search.return_value = []
    final = "无法根据当前证据确定奖学金申请时间。"
    model = FakeToolCallingModel([_search(), FinalAgentResponse(outcome="no_evidence", answer=final)])
    output, _, _ = await _invoke(monkeypatch, model, settings_factory)

    assert len(model.received_messages) == 2
    assert _evidence(model, 1, "search-a") == {
        "query": "奖学金 申请时间", "count": 0, "evidence": [],
    }
    cross_encoder.predict.assert_not_called()
    assert output["result"]["answer"] == final
    assert output["result"]["status"] == "NO_RELEVANT_INFO"
    assert output["result"]["references"] == []
    assert output["result"]["tool_result_summary"] == "知识库检索完成，召回 0 个 chunk"


@pytest.mark.asyncio
async def test_unavailable_kb_returns_error_and_agent_recovers(monkeypatch, settings_factory, retrieval):
    search, cross_encoder = retrieval
    monkeypatch.setattr(pipeline, "RAG_READY", False)
    model = FakeToolCallingModel([_search(), FinalAgentResponse(outcome="answer", answer="知识库暂时不可用，请稍后重试。")])
    output, _, audit = await _invoke(monkeypatch, model, settings_factory)

    assert len(model.received_messages) == 2
    message = model.received_messages[1][-1]
    assert isinstance(message, ToolMessage) and message.tool_call_id == "search-a"
    assert message.status == "error"
    payload = (message.artifact or json.loads(message.content))
    assert payload["ok"] is False and payload["error_code"] == "dependency_unavailable"
    search.assert_not_called()
    cross_encoder.predict.assert_not_called()
    assert output["result"]["status"] == "OK"
    assert output["result"]["answer"] == "知识库暂时不可用，请稍后重试。"
    assert output["result"]["tool_result_summary"] == "工具执行失败"
    assert [e.event_type.value for e in audit] == ["requested", "started", "failed"]


@pytest.mark.asyncio
async def test_agent_can_change_query_and_retrieve_again(monkeypatch, settings_factory, retrieval):
    search, cross_encoder = retrieval
    search.side_effect = [
        [{"chunk_id": "b", "text": "奖学金申请于九月开始。", "file_name": "policy.md"}],
        [{"chunk_id": "a", "text": "申请人需要提交成绩单。", "file_name": "policy.md"}],
    ]
    cross_encoder.predict.side_effect = [[0.9], [0.8]]
    final = "申请于九月开始，需要提交成绩单。"
    model = FakeToolCallingModel([
        _search(top_k=3), _search("奖学金 申请材料 成绩单", top_k=7, call_id="search-b"),
        _grounded_answer(final),
    ])
    output, args_seen, audit = await _invoke(monkeypatch, model, settings_factory)

    assert len(model.received_messages) == 3
    assert search.call_count == cross_encoder.predict.call_count == 2
    assert [(c.args[0], c.kwargs["top_k"]) for c in search.call_args_list] == [
        ("奖学金 申请时间", 3), ("奖学金 申请材料 成绩单", 7),
    ]
    assert args_seen[0]["query"] != args_seen[1]["query"]
    assert _evidence(model, 1, "search-a")["count"] == 1
    assert _evidence(model, 2, "search-b")["count"] == 1
    assert [m.tool_call_id for m in model.received_messages[2] if isinstance(m, ToolMessage)] == [
        "search-a", "search-b",
    ]
    assert [m.type for m in output["messages"]] == ["human", "ai", "tool", "ai", "tool", "ai", "tool"]
    assert output["structured_response"].answer == output["result"]["answer"] == final
    assert len(output["result"]["references"]) == 2
    assert [ref["chunk_id"] for ref in output["result"]["references"]] == ["b", "a"]
    assert output["result"]["llm_call_count"] == 3
    assert output["stop_reason"] == "completed"
    assert sum(e.event_type.value == "succeeded" for e in audit) == 2


@pytest.mark.asyncio
async def test_only_second_retrieved_document_is_referenced(monkeypatch, settings_factory, retrieval):
    model = FakeToolCallingModel([
        _search(top_k=2), _grounded_answer("需要提交成绩单。", indexes=[1]),
    ])
    output, _, audit = await _invoke(monkeypatch, model, settings_factory, agent_max_tool_calls=1)
    documents = _evidence(model, 1, "search-a")["evidence"]
    assert output["result"]["status"] == "OK"
    assert [ref["reference_id"] for ref in output["result"]["references"]] == [documents[1]["id"]]
    assert len(documents) == 2
    assert {event.tool_name for event in audit} == {"search_knowledge_base"}
    assert [step["tool_name"] for step in output["result"]["tool_trace"] if step["step"] == "tool_call"] == [
        "search_knowledge_base",
    ]
    iterations = [step for step in output["result"]["tool_trace"] if step["step"] == "agent_iteration"]
    assert [step["requested_tool_call_count"] for step in iterations] == [1, 0]
    assert all(model.structured_tool_name not in step["available_tool_names"] for step in iterations)


@pytest.mark.asyncio
async def test_native_schema_retry_does_not_execute_or_audit_synthetic_tool(
    monkeypatch, settings_factory, retrieval,
):
    model = FakeToolCallingModel([
        _search(),
        lambda model, messages: model.final_response(outcome="answer", answer=""),
        _grounded_answer("申请于九月开始。"),
    ])
    output, _, audit = await _invoke(monkeypatch, model, settings_factory, agent_max_tool_calls=1)
    assert output["result"]["status"] == "OK"
    assert output["result"]["llm_call_count"] == len(model.received_messages) == 3
    retry = model.received_messages[2][-1]
    assert isinstance(retry, ToolMessage) and retry.name == model.structured_tool_name
    assert "validation" in retry.content.lower()
    assert len(audit) == 3
    assert {event.tool_name for event in audit} == {"search_knowledge_base"}


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[], ["knowledge_base:fake"]])
async def test_real_agent_rejects_ungrounded_structured_answer(
    monkeypatch, settings_factory, retrieval, ids,
):
    model = FakeToolCallingModel([
        _search(), FinalAgentResponse(outcome="answer", answer="UNSUPPORTED", evidence_ids=ids),
    ])
    output, _, _ = await _invoke(monkeypatch, model, settings_factory)
    assert output["result"]["status"] == "MODEL_ERROR"
    assert output["result"]["references"] == []
    assert "UNSUPPORTED" not in str(output["result"])
