"""真实 KB handler / StructuredTool / create_agent 的证据返回与二次检索。"""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent import framework_agent, framework_tools
from app.rag import pipeline
from app.tools.catalog import get_local_tool_specs
from app.tools.context import ToolInvocationContext
from app.tools.knowledge_base import search_knowledge_base
from app.tools.schemas import SearchKnowledgeBaseArgs, SearchKnowledgeBaseOutput
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
    forbidden = Mock(side_effect=AssertionError("KB retrieval must not call legacy RAG/LLM"))
    for path in (
        "app.rag.graph.run_rag_subgraph",
        "app.rag.decomposition.decompose_query_with_llm",
        "app.rag.generation.get_answer_model",
    ):
        monkeypatch.setattr(path, forbidden)
    yield search, cross_encoder
    forbidden.assert_not_called()


async def _invoke(monkeypatch, model, settings_factory):
    settings = settings_factory()
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
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    context = ToolInvocationContext(
        trace_id="trace-kb", session_id="session-kb", principal_id="kb-test",
        permissions=("knowledge_base:read",), source="knowledge_tool_integration_test",
    )
    output = [state async for state in build_agent_harness(settings).astream({
        "original_question": "它何时开始？",
        "effective_question": "奖学金申请条件与开始时间是什么？",
        "top_k": 7, "decompose_top_k": 10,
        "trace_id": context.trace_id, "session_id": context.session_id,
    }, context=context)][-1]
    assert "knowledge_base_result" not in output
    assert isinstance(tools[0], framework_tools.ProjectStructuredTool)
    assert issubclass(tools[0].args_schema, SearchKnowledgeBaseArgs)
    assert tools[0].project_spec.output_model is SearchKnowledgeBaseOutput
    assert "runtime" not in tools[0].tool_call_schema.model_fields
    assert isinstance(model.received_messages[0][0], SystemMessage)
    assert [m.content for m in model.received_messages[0] if isinstance(m, HumanMessage)] == [
        "奖学金申请条件与开始时间是什么？"
    ]
    assert all(c.effective_question == "奖学金申请条件与开始时间是什么？" for c in contexts)
    assert all(c.top_k == 7 and c.decompose_top_k == 10 for c in contexts)
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


def _evidence(model, round_index, call_id):
    message = model.received_messages[round_index][-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == call_id
    assert message.name == "search_knowledge_base"
    assert message.status == "success"
    payload = json.loads(message.content)
    assert payload["ok"] is True
    assert payload["runtime"]["attempt_count"] == 1
    assert payload["governance"]["decision"] == "allow"
    assert set(payload["output"]) == {"count", "chunks", "top_chunk_ids", "top_scores"}
    return payload["output"]


@pytest.mark.asyncio
async def test_search_returns_reranked_evidence_to_next_model(monkeypatch, settings_factory, retrieval):
    search, cross_encoder = retrieval
    model = FakeToolCallingModel([_search(), AIMessage(content="奖学金申请于九月开始。")])
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
    assert evidence["top_chunk_ids"] == ["b"]
    assert evidence["top_scores"] == [0.8123]  # hybrid score，不是 rerank score
    assert evidence["chunks"][0]["rerank_score"] == 0.9
    result = output["result"]
    assert result["answer"] == "奖学金申请于九月开始。"
    assert result["references"] == []
    assert result["retrieval_confidence"] == result["answer_confidence"] == 0.0
    assert result["llm_call_count"] == 2 and result["decompose_llm_used"] is False
    trace = next(s for s in result["tool_trace"] if s["step"] == "tool_call")
    assert trace["final_chunk_count"] == 1
    assert trace["top_chunk_ids"] == ["b"] and trace["top_scores"] == [0.8123]
    assert trace["hits"] == [{"chunk_id": "b", "rerank": 0.9, "score": 0.81234,
                              "bm25": 4.0, "vector": 0.6, "file_name": "policy.md"}]
    assert [e.event_type.value for e in audit] == ["requested", "started", "succeeded"]


@pytest.mark.asyncio
async def test_zero_results_are_successful_evidence(monkeypatch, settings_factory, retrieval):
    search, cross_encoder = retrieval
    search.side_effect = None
    search.return_value = []
    final = "无法根据当前证据确定奖学金申请时间。"
    model = FakeToolCallingModel([_search(), AIMessage(content=final)])
    output, _, _ = await _invoke(monkeypatch, model, settings_factory)

    assert len(model.received_messages) == 2
    assert _evidence(model, 1, "search-a") == {
        "count": 0, "chunks": [], "top_chunk_ids": [], "top_scores": [],
    }
    cross_encoder.predict.assert_not_called()
    assert output["result"]["answer"] == final
    assert output["result"]["status"] == "OK"  # 普通最终 AIMessage，不伪造 RAG 状态
    assert output["result"]["tool_result_summary"] == "知识库检索完成，召回 0 个 chunk"


@pytest.mark.asyncio
async def test_unavailable_kb_returns_error_and_agent_recovers(monkeypatch, settings_factory, retrieval):
    search, cross_encoder = retrieval
    monkeypatch.setattr(pipeline, "RAG_READY", False)
    model = FakeToolCallingModel([_search(), AIMessage(content="知识库暂时不可用，请稍后重试。")])
    output, _, audit = await _invoke(monkeypatch, model, settings_factory)

    assert len(model.received_messages) == 2
    message = model.received_messages[1][-1]
    assert isinstance(message, ToolMessage) and message.tool_call_id == "search-a"
    assert message.status == "error"
    payload = json.loads(message.content)
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
    final = "申请于九月开始，需要提交成绩单。"
    model = FakeToolCallingModel([
        _search(), _search("奖学金 申请材料 成绩单", top_k=2, call_id="search-b"),
        AIMessage(content=final),
    ])
    output, args_seen, audit = await _invoke(monkeypatch, model, settings_factory)

    assert len(model.received_messages) == 3
    assert search.call_count == cross_encoder.predict.call_count == 2
    assert [(c.args[0], c.kwargs["top_k"]) for c in search.call_args_list] == [
        ("奖学金 申请时间", 1), ("奖学金 申请材料 成绩单", 2),
    ]
    assert args_seen[0]["query"] != args_seen[1]["query"]
    assert _evidence(model, 1, "search-a")["count"] == 1
    assert _evidence(model, 2, "search-b")["count"] == 2
    assert [m.tool_call_id for m in model.received_messages[2] if isinstance(m, ToolMessage)] == [
        "search-a", "search-b",
    ]
    assert [m.type for m in output["messages"]] == ["human", "ai", "tool", "ai", "tool", "ai"]
    assert output["messages"][-1].content == output["result"]["answer"] == final
    assert output["result"]["llm_call_count"] == 3
    assert output["stop_reason"] == "completed"
    assert sum(e.event_type.value == "succeeded" for e in audit) == 2
