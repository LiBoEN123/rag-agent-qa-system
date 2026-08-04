"""LangGraph 流式 Runtime 的离线、安全投影测试。"""

import asyncio
import json

from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent.results import create_agent_result
from app.contracts.chat import AgentResultStatus
from app.graph import builder as graph_builder
from app.graph import checkpointing
from app.graph.checkpointing import build_checkpoint_thread_id
from app.graph.streaming import (
    project_graph_stream_part,
    stream_agent_langgraph,
)


def _result(*, answer: str = "完整回答") -> dict:
    return create_agent_result(
        status=AgentResultStatus.OK,
        answer=answer,
        reason="正常回答",
    )


class RecordingGraph:
    def __init__(self, parts):
        self.parts = list(parts)
        self.calls = []
        self.closed = False

    def astream(self, graph_input, config, *, context, **kwargs):
        self.calls.append(
            {
                "graph_input": graph_input,
                "config": config,
                "context": context,
                **kwargs,
            }
        )

        async def generate():
            try:
                for part in self.parts:
                    if isinstance(part, Exception):
                        raise part
                    yield part
            finally:
                self.closed = True

        return generate()


def _message_part(message, *, namespace=("framework_agent:task",)):
    return {
        "type": "messages",
        "ns": namespace,
        "data": (
            message,
            {
                "langgraph_node": "model",
                "lc_agent_name": "framework_agent",
                "run_id": "run-1",
            },
        ),
    }


def _updates(node, update, *, namespace=()):
    return {
        "type": "updates",
        "ns": namespace,
        "data": {node: update},
    }


def test_projector_filters_reasoning_tool_messages_rag_and_raw_tool_data():
    reasoning = AIMessageChunk(
        content=[{"type": "reasoning", "reasoning": "private chain"}],
        id="reasoning-1",
        chunk_position="last",
    )
    rag_json = AIMessageChunk(
        content='{"answer":"不应流出"}',
        id="rag-1",
        chunk_position="last",
    )
    tool_message = ToolMessage(
        content='{"secret":"sk-private","raw_tool":"full body"}',
        tool_call_id="call-1",
        name="web_fetch",
    )
    completed = _updates(
        "tools",
        {
            "governance_events": [
                {
                    "kind": "tool",
                    "name": "web_fetch",
                    "args": {"authorization": "Bearer private"},
                    "result": {
                        "ok": True,
                        "output": {"body": "private page"},
                    },
                    "ok": True,
                    "cost_ms": 12,
                    "tool_call_id": "call-1",
                }
            ]
        },
        namespace=("framework_agent:task",),
    )

    reasoning_projection = project_graph_stream_part(_message_part(reasoning))
    assert all(
        event.event != "answer_delta" for event in reasoning_projection
    )
    assert "private chain" not in json.dumps(
        [event.data for event in reasoning_projection]
    )
    assert project_graph_stream_part(
        _message_part(rag_json, namespace=("rag_subgraph:task",))
    ) == []
    assert project_graph_stream_part(_message_part(tool_message)) == []
    projected = project_graph_stream_part(completed)

    assert len(projected) == 1
    assert projected[0].event == "tool"
    assert projected[0].data == {
        "tool_name": "web_fetch",
        "phase": "completed",
        "status": "success",
        "cost_ms": 12,
    }
    serialized = json.dumps(projected[0].data)
    assert "authorization" not in serialized
    assert "private page" not in serialized


def test_stream_runtime_uses_shared_invocation_and_current_v2_shapes(
    monkeypatch,
    settings_env,
):
    settings_env(
        langgraph_checkpointer="memory",
        enable_agent_trace_store=False,
    )
    parts = [
        _updates("route", {"decision": {"action": "agent"}}),
        _message_part(AIMessageChunk(content="完", id="answer-1")),
        _message_part(
            AIMessageChunk(
                content="整回答",
                id="answer-1",
                chunk_position="last",
            )
        ),
        _updates("finalize", {"result": _result()}),
    ]
    graph = RecordingGraph(parts)

    async def get_graph(_settings=None):
        return graph

    monkeypatch.setattr(
        "app.graph.runtime.get_async_compiled_agent_graph",
        get_graph,
    )

    async def collect():
        return [
            event
            async for event in stream_agent_langgraph(
                write_bad_cases=lambda record: None,
                original_question="原始问题",
                effective_question="改写问题",
                top_k=4,
                decompose_top_k=8,
                trace_id="trace-stream",
                session_id="session-stream",
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())

    assert [event.event for event in events] == [
        "progress",
        "answer_delta",
        "answer_delta",
        "result",
    ]
    assert "".join(
        event.data["text"]
        for event in events
        if event.event == "answer_delta"
    ) == "完整回答"
    assert graph.closed is True
    assert len(graph.calls) == 1
    call = graph.calls[0]
    assert call["graph_input"] == {
        "original_question": "原始问题",
        "effective_question": "改写问题",
        "top_k": 4,
        "decompose_top_k": 8,
        "trace_id": "trace-stream",
        "session_id": "session-stream",
    }
    assert call["config"]["max_concurrency"] == 1
    assert call["config"]["recursion_limit"] >= 25
    assert call["config"]["configurable"]["thread_id"] == (
        build_checkpoint_thread_id(
            trace_id="trace-stream",
            session_id="session-stream",
        )
    )
    assert call["context"].trace_id == "trace-stream"
    assert call["context"].session_id == "session-stream"
    assert call["stream_mode"] == ["updates", "messages"]
    assert call["subgraphs"] is True
    assert call["version"] == "v2"


def test_tool_call_chunks_emit_only_started_event_and_no_arguments():
    message = AIMessageChunk(
        content="",
        id="tool-model-1",
        tool_call_chunks=[
            {
                "name": "search_knowledge_base",
                "args": '{"query":"private question"}',
                "id": "call-1",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
        chunk_position="last",
    )

    projected = project_graph_stream_part(_message_part(message))

    assert projected[0].event == "tool"
    assert projected[0].data == {
        "tool_name": "search_knowledge_base",
        "phase": "started",
        "status": "",
    }
    serialized = json.dumps(
        [event.data for event in projected],
        ensure_ascii=False,
    )
    assert "private question" not in serialized
    assert "tool_call_chunk" not in serialized


def test_stream_runtime_converts_exception_to_valid_internal_result_once(
    monkeypatch,
    settings_env,
):
    settings_env(
        langgraph_checkpointer="disabled",
        enable_agent_trace_store=False,
    )
    graph = RecordingGraph([RuntimeError("api_key=sk-private")])

    async def get_graph(_settings=None):
        return graph

    monkeypatch.setattr(
        "app.graph.runtime.get_async_compiled_agent_graph",
        get_graph,
    )
    bad_cases = []

    async def collect():
        return [
            event
            async for event in stream_agent_langgraph(
                write_bad_cases=bad_cases.append,
                original_question="问题",
                effective_question="问题",
                top_k=5,
                decompose_top_k=10,
                trace_id="trace-error",
                session_id=None,
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())
    result = events[-1].data["result"]

    assert [event.event for event in events] == ["result"]
    assert result["status"] == "INTERNAL_ERROR"
    assert "sk-private" not in json.dumps(result)
    assert len(bad_cases) == 1
    assert graph.closed is True


def test_closing_runtime_stream_closes_graph_iterator(
    monkeypatch,
    settings_env,
):
    settings_env(
        langgraph_checkpointer="disabled",
        enable_agent_trace_store=False,
    )
    graph = RecordingGraph(
        [
            _updates("route", {"decision": {"action": "agent"}}),
            _updates("finalize", {"result": _result()}),
        ]
    )

    async def get_graph(_settings=None):
        return graph

    monkeypatch.setattr(
        "app.graph.runtime.get_async_compiled_agent_graph",
        get_graph,
    )

    async def consume_one_and_close():
        stream = stream_agent_langgraph(
            write_bad_cases=lambda record: None,
            original_question="问题",
            effective_question="问题",
            top_k=5,
            decompose_top_k=10,
            trace_id="trace-close",
            session_id=None,
            bad_case=None,
        )
        first = await anext(stream)
        await stream.aclose()
        return first

    event = asyncio.run(consume_one_and_close())

    assert event.event == "progress"
    assert graph.closed is True


def test_actual_parent_graph_direct_route_streams_offline_without_model(
    settings_env,
):
    """用真实编译父图确认当前 v2 updates 形状，不解析模型或 RAG。"""
    settings_env(
        langgraph_checkpointer="disabled",
        enable_agent_trace_store=False,
    )

    async def collect():
        return [
            event
            async for event in stream_agent_langgraph(
                write_bad_cases=lambda record: None,
                original_question="你能做什么？",
                effective_question="你能做什么？",
                top_k=5,
                decompose_top_k=10,
                trace_id="trace-real-stream",
                session_id=None,
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())
    result = events[-1].data["result"]

    assert events[0].data == {"stage": "routing", "status": "completed"}
    assert events[-1].event == "result"
    assert result["status"] == "OK"
    assert result["action"] == "direct_answer"


def test_actual_parent_graph_astream_uses_async_sqlite_checkpointer(
    settings_env,
    tmp_path,
):
    """回归真实 ``astream + sqlite`` 组合，避免误用同步 SqliteSaver。"""
    settings = settings_env(
        langgraph_checkpointer="sqlite",
        langgraph_checkpoint_db_path=tmp_path / "stream-checkpoint.sqlite3",
        enable_agent_trace_store=False,
    )
    trace_id = "trace-real-sqlite-stream"
    session_id = "session-real-sqlite-stream"
    thread_id = checkpointing.build_checkpoint_thread_id(
        trace_id=trace_id,
        session_id=session_id,
    )

    async def collect():
        try:
            events = [
                event
                async for event in stream_agent_langgraph(
                    write_bad_cases=lambda record: None,
                    original_question="你能做什么？",
                    effective_question="你能做什么？",
                    top_k=5,
                    decompose_top_k=10,
                    trace_id=trace_id,
                    session_id=session_id,
                    bad_case=None,
                )
            ]
            saver = await checkpointing.get_async_checkpointer(settings)
            saved = await saver.aget_tuple(
                {"configurable": {"thread_id": thread_id}}
            )
            return events, saver, saved
        finally:
            await graph_builder.areset_agent_graph_runtime()

    events, saver, saved = asyncio.run(collect())

    assert isinstance(saver, AsyncSqliteSaver)
    assert events[-1].event == "result"
    assert events[-1].data["result"]["status"] == "OK"
    assert saved is not None
    assert saved.checkpoint["channel_values"]["trace_id"] == trace_id
    assert saved.checkpoint["channel_values"]["result"]["status"] == "OK"
