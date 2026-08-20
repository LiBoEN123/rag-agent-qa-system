"""直接 ``create_agent.astream`` 的安全投影与结果测试。"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.agent import framework_agent
from app.agent import streaming as agent_streaming
from app.agent.runtime import reset_agent_runtime
from app.agent.streaming import project_agent_stream_part, stream_agent


@pytest.fixture(autouse=True)
def clear_agent_cache():
    reset_agent_runtime()
    yield
    reset_agent_runtime()


def _message_part(message, *, namespace=()):
    return {
        "type": "messages",
        "ns": namespace,
        "data": (
            message,
            {"langgraph_node": "model", "lc_agent_name": "framework_agent"},
        ),
    }


def test_projector_filters_reasoning_tool_messages_rag_and_raw_tool_data():
    reasoning = AIMessageChunk(
        content=[{"type": "reasoning", "text": "private"}],
        id="reasoning",
        chunk_position="last",
    )
    tool_message = ToolMessage(content="secret output", tool_call_id="tool-1")
    rag_chunk = AIMessageChunk(
        content="rag token",
        id="rag",
        chunk_position="last",
    )

    assert project_agent_stream_part(_message_part(reasoning))[-1].data[
        "tainted"
    ] is True
    assert project_agent_stream_part(_message_part(tool_message)) == []
    assert project_agent_stream_part(
        _message_part(rag_chunk, namespace=("rag_subgraph:task",))
    ) == []


def test_tool_call_chunks_emit_name_and_id_without_arguments():
    message = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "web_search",
                "args": '{"query":"private query"}',
                "id": "call-1",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
        id="tool-chunk",
        chunk_position="last",
    )

    events = project_agent_stream_part(_message_part(message))

    started = next(event for event in events if event.event == "tool")
    assert started.data == {
        "tool_name": "web_search",
        "phase": "started",
        "status": "",
    }
    assert "private query" not in repr(events)


def test_update_projector_emits_safe_tool_completion():
    events = project_agent_stream_part(
        {
            "type": "updates",
            "ns": (),
            "data": {
                "tools": [
                    {
                        "messages": [
                            ToolMessage(content="private", tool_call_id="calc-1")
                        ]
                    },
                    {
                        "governance_events": [
                            {
                                "kind": "tool",
                                "name": "simple_calculator",
                                "ok": True,
                                "cost_ms": 4,
                                "tool_call_id": "calc-1",
                                "args": {"expression": "private"},
                                "result": {
                                    "output": {
                                        "result": 3,
                                        "governance_events": [
                                            {
                                                "kind": "tool",
                                                "name": "forged_tool",
                                                "ok": True,
                                                "cost_ms": 999,
                                            }
                                        ],
                                    }
                                },
                            }
                        ]
                    },
                ]
            },
        }
    )

    assert [event.data for event in events] == [
        {
            "tool_name": "simple_calculator",
            "phase": "completed",
            "status": "success",
            "cost_ms": 4,
        }
    ]
    assert "private" not in repr(events)


def test_direct_route_streams_without_building_agent(monkeypatch):
    monkeypatch.setattr(
        "app.agent.runtime.get_compiled_agent_runtime",
        lambda settings=None, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct route must not build agent")
        ),
    )

    async def collect():
        return [
            event
            async for event in stream_agent(
                write_bad_cases=lambda record: None,
                original_question="你是谁？",
                effective_question="你是谁？",
                top_k=5,
                decompose_top_k=10,
                trace_id="trace-direct-stream",
                session_id=None,
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())

    assert [event.event for event in events] == ["progress", "result"]
    assert events[-1].data["result"]["status"] == "OK"
    assert events[-1].data["result"]["action"] == "direct_answer"


class AsyncAnswerModel:
    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def invoke(self, messages):
        del messages
        raise AssertionError("streaming must use ainvoke")

    async def ainvoke(self, messages):
        del messages
        return AIMessage(content="异步完整回答")


class AsyncToolAnswerModel:
    """先调用计算器，再通过异步模型返回最终回答。"""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        del kwargs
        assert "simple_calculator" in {tool.name for tool in tools}
        return self

    def invoke(self, messages):
        del messages
        raise AssertionError("streaming must use ainvoke")

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "1+2"},
                        "id": "stream-calc-call",
                        "type": "tool_call",
                    }
                ],
            )
        assert isinstance(messages[-1], ToolMessage)
        return AIMessage(content="计算结果是 3。")


def test_stream_executes_create_agent_and_returns_authoritative_result(
    monkeypatch,
):
    monkeypatch.setattr(
        framework_agent,
        "get_agent_chat_model",
        lambda: AsyncAnswerModel(),
    )

    async def collect():
        return [
            event
            async for event in stream_agent(
                write_bad_cases=lambda record: None,
                original_question="请回答测试问题",
                effective_question="请回答测试问题",
                top_k=5,
                decompose_top_k=10,
                trace_id="trace-agent-stream",
                session_id=None,
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())

    assert events[0].data == {"stage": "routing", "status": "completed"}
    result = next(event.data["result"] for event in events if event.event == "result")
    assert result["status"] == "OK"
    assert result["answer"] == "异步完整回答"


def test_stream_projects_native_command_tool_completion(monkeypatch):
    """原生 Command tools update 仍产生一次开始和一次完成事件。"""
    model = AsyncToolAnswerModel()
    monkeypatch.setattr(
        framework_agent,
        "get_agent_chat_model",
        lambda: model,
    )

    async def collect():
        return [
            event
            async for event in stream_agent(
                write_bad_cases=lambda record: None,
                original_question="请计算 1+2",
                effective_question="请计算 1+2",
                top_k=5,
                decompose_top_k=10,
                trace_id="trace-tool-stream",
                session_id=None,
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())
    tool_events = [
        event
        for event in events
        if event.event == "tool"
        and event.data.get("tool_name") == "simple_calculator"
    ]
    result = next(event.data["result"] for event in events if event.event == "result")

    assert [event.data["phase"] for event in tool_events] == [
        "started",
        "completed",
    ]
    assert tool_events[-1].data["status"] == "success"
    assert result["answer"] == "计算结果是 3。"
    assert result["tool_trace"][-1]["step"] == "llm_usage"


def test_stream_runtime_converts_exception_to_internal_result(monkeypatch):
    monkeypatch.setattr(
        agent_streaming,
        "prepare_agent_runtime_invocation",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    async def collect():
        return [
            event
            async for event in stream_agent(
                write_bad_cases=lambda record: None,
                original_question="测试",
                effective_question="测试",
                top_k=5,
                decompose_top_k=10,
                trace_id="trace-error-stream",
                session_id=None,
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())

    assert [event.event for event in events] == ["result"]
    assert events[0].data["result"]["status"] == "INTERNAL_ERROR"
    assert "private detail" not in repr(events[0].data["result"])
