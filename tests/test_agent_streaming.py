"""直接 ``create_agent.astream`` 的安全投影与结果测试。"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.agent import framework_agent
from app.agent.final_response import FinalAgentResponse
from tests.support.agent_runtime import FakeToolCallingModel
from app.agent import runtime as agent_runtime
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

    assert project_agent_stream_part(_message_part(reasoning)) == []
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
                            ToolMessage(content="private", tool_call_id="list-1")
                        ]
                    },
                    {
                        "governance_events": [
                            {
                                "kind": "tool",
                                "name": "find_documents",
                                "ok": True,
                                "cost_ms": 4,
                                "tool_call_id": "list-1",
                                "args": {"private": "private"},
                                "result": {
                                    "output": {
                                        "count": 0,
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
            "tool_name": "find_documents",
            "phase": "completed",
            "status": "success",
            "cost_ms": 4,
        }
    ]
    assert "private" not in repr(events)


@pytest.mark.parametrize(
    ("question", "status", "action"),
    [("你是谁？", "OK", "direct_answer"), ("这个怎么弄", "NEED_CLARIFICATION", "clarify")],
)
def test_direct_route_streams_without_building_agent(monkeypatch, question, status, action):
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
                original_question=question,
                effective_question=question,
                trace_id="trace-direct-stream",
                session_id=None,
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())

    assert [event.event for event in events] == ["progress", "result"]
    result = events[-1].data["result"]
    assert result["status"] == status
    assert result["action"] == action
    assert result["router_llm_used"] is False
    assert result["llm_call_count"] == 0


@pytest.mark.parametrize("failure", [False, True])
def test_stream_executes_create_agent_and_returns_authoritative_result(
    monkeypatch, failure,
):
    model = FakeToolCallingModel([
        RuntimeError("provider secret") if failure else FinalAgentResponse(outcome="answer", answer="异步完整回答"),
    ])
    questions = []
    route = agent_runtime.route_agent_action

    def recording_route(question):
        questions.append(question)
        return route(question)

    monkeypatch.setattr(agent_runtime, "route_agent_action", recording_route)
    monkeypatch.setattr(
        framework_agent,
        "get_answer_model",
        lambda: model,
    )

    async def collect():
        return [
            event
            async for event in stream_agent(
                write_bad_cases=lambda record: None,
                original_question="它是什么？",
                effective_question="请回答测试问题",
                trace_id="trace-agent-stream",
                session_id=None,
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())

    assert events[0].data == {"stage": "routing", "status": "completed"}
    result = next(event.data["result"] for event in events if event.event == "result")
    assert questions == ["请回答测试问题"]
    assert model.received_messages[0][-1].content == "请回答测试问题"
    decide = next(step for step in result["tool_trace"] if step["step"] == "decide")
    assert decide["route_cost_ms"] >= 0
    assert result["status"] == ("MODEL_ERROR" if failure else "OK")
    if failure:
        assert result["reason"] == "agent_model_call_failed"
        assert "provider secret" not in repr(events)
    else:
        assert result["answer"] == "异步完整回答"
        assert [event.data["text"] for event in events if event.event == "answer_delta"] == [result["answer"]]
        assert all(event.event != "tool" for event in events)
        assert events[-2].event == "answer_delta" and events[-1].event == "result"
        assert model.structured_tool_name not in repr(events)
        assert "evidence_ids" not in repr([event for event in events if event.event != "result"])


def test_stream_projects_native_command_tool_completion(monkeypatch):
    """原生 Command tools update 仍产生一次开始和一次完成事件。"""
    monkeypatch.setattr("app.rag.runtime.chunks", [])
    monkeypatch.setattr("app.rag.runtime.RAG_READY", True)
    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{
            "name": "find_documents", "args": {}, "id": "stream-list-call",
        }]),
        FinalAgentResponse(outcome="answer", answer="文档列表读取完成。"),
    ])
    monkeypatch.setattr(
        framework_agent,
        "get_answer_model",
        lambda: model,
    )

    async def collect():
        return [
            event
            async for event in stream_agent(
                write_bad_cases=lambda record: None,
                original_question="请列出知识库文档",
                effective_question="请列出知识库文档",
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
        and event.data.get("tool_name") == "find_documents"
    ]
    result = next(event.data["result"] for event in events if event.event == "result")

    assert [event.data["phase"] for event in tool_events] == [
        "started",
        "completed",
    ]
    assert tool_events[-1].data["status"] == "success"
    assert result["answer"] == "文档列表读取完成。"
    assert result["tool_trace"][-1]["step"] == "llm_usage"


def test_stream_runtime_converts_exception_to_internal_result(monkeypatch):
    bad_cases = []
    monkeypatch.setattr(
        agent_streaming,
        "prepare_agent_runtime_invocation",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    async def collect():
        return [
            event
            async for event in stream_agent(
                write_bad_cases=lambda record: bad_cases.append(dict(record)),
                original_question="测试",
                effective_question="测试",
                trace_id="trace-error-stream",
                session_id=None,
                bad_case=None,
            )
        ]

    events = asyncio.run(collect())

    assert [event.event for event in events] == ["result"]
    assert events[0].data["result"]["status"] == "INTERNAL_ERROR"
    assert "private detail" not in repr(events[0].data["result"])
    assert bad_cases[0]["note"] == "private detail"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_id", [False, True])
async def test_stream_withholds_model_text_until_grounding_passes(monkeypatch, invalid_id):
    def respond(model, messages):
        message = model.final_response(
            outcome="answer", answer="最终回答",
            evidence_ids=["knowledge_base:fake"] if invalid_id else [],
        )
        return message.model_copy(update={"content": "未经校验的模型文本"})

    model = FakeToolCallingModel([respond])
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    events = [event async for event in stream_agent(
        write_bad_cases=None, original_question="测试事实问题", effective_question="测试事实问题",
        trace_id="grounding-stream", session_id=None, bad_case=None,
    )]
    result = events[-1].data["result"]
    assert result["status"] == ("MODEL_ERROR" if invalid_id else "OK")
    assert [event.data["text"] for event in events if event.event == "answer_delta"] == (
        [] if invalid_id else [result["answer"]]
    )
    assert "未经校验的模型文本" not in repr(events)
    assert model.structured_tool_name not in repr(events)
    assert "knowledge_base:fake" not in repr([event for event in events if event.event != "result"])
    assert "evidence_ids" not in repr([event for event in events if event.event != "result"])


def test_plain_text_and_tool_schema_chunks_are_never_answer_deltas():
    assert project_agent_stream_part(_message_part(AIMessageChunk(
        content="尚未校验的回答", id="text", chunk_position="last",
    ))) == []
    assert project_agent_stream_part(_message_part(AIMessageChunk(
        content="", id="schema", chunk_position="last", tool_call_chunks=[{
            "name": FinalAgentResponse.__name__, "args": '{"answer":"private"}',
            "id": "final", "index": 0,
        }],
    ))) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_tool", [False, True])
async def test_total_deadline_interrupts_model_or_sync_tool(monkeypatch, settings_env, slow_tool):
    import time
    from pydantic import BaseModel, RootModel
    from app.agent import framework_tools
    from app.tools.contracts import ToolSpec
    from app.tools.result import tool_success

    settings_env(agent_max_total_cost_ms=30)
    calls = []

    class Args(BaseModel):
        pass

    def handler(args, context):
        calls.append("tool")
        time.sleep(0.15)
        return tool_success("slow_read", args, {})

    spec = ToolSpec(name="slow_read", version="1.0.0", description="slow read",
                    args_model=Args, output_model=RootModel[dict], handler=handler)
    monkeypatch.setattr(framework_tools, "get_local_tools", lambda settings: (
        framework_tools.build_local_tool(spec, settings=settings),
    ))

    class Model(FakeToolCallingModel):
        async def ainvoke(self, messages):
            if not slow_tool:
                await asyncio.sleep(0.15)
            return AIMessage(content="private unfinished text", tool_calls=[{
                "name": "slow_read", "args": {}, "id": "slow-call",
            }])

    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: Model([]))
    started = time.perf_counter()
    events = [event async for event in stream_agent(
        write_bad_cases=None, trace_id="deadline", session_id=None, bad_case=None,
        original_question="执行慢查询", effective_question="执行慢查询",
    )]
    assert time.perf_counter() - started < 0.12
    result = events[-1].data["result"]
    assert result["status"] == "MODEL_ERROR"
    assert result["reason"] == "max_total_cost_reached"
    assert all(event.event != "answer_delta" for event in events)
    assert "private unfinished text" not in repr(events)
    assert calls == (["tool"] if slow_tool else [])
