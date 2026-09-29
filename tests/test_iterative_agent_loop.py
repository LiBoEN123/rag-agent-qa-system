"""迭代式 Agent Loop 的消息、工具治理和预算回归测试。"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import framework_agent, framework_tools
from app.agent.runtime import prepare_agent_runtime_invocation
from app.agent.final_response import collect_evidence_from_messages
from app.config import get_settings
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from tests.support.local_tools import local_tools
from app.tools.result import tool_success
from app.tools.knowledge_base import chunk_to_evidence_document
from app.tools.schemas import SearchKnowledgeBaseArgs, SearchKnowledgeBaseOutput
from tests.support.agent_runtime import FakeToolCallingModel, build_agent_harness


def _grounded_answer(text):
    """显式声明此 fixture 实际收到的证据，通过框架结构化输出结束。"""
    return lambda model, messages: model.final_response(
        scope_match="matched", support_level="full",
        outcome="answer", answer=text,
        evidence_ids=list(collect_evidence_from_messages(messages)),
    )


@pytest.fixture(autouse=True)
def isolate_tool_audit(settings_env, monkeypatch):
    """Agent loop 测试不写生产工具审计文件。"""
    settings_env(tool_audit_enabled=False)
    monkeypatch.setattr("app.rag.runtime.RAG_READY", True)
    monkeypatch.setattr("app.rag.runtime.chunks", [
        {"text": f"正文 {i}", "metadata": {"chunk_id": i, "source_id": f"s-{i}", "position": 1, "file_name": f"{i}.md"}}
        for i in (1, 2, 382, 383, 387)
    ])
    yield


async def _invoke_graph(
    monkeypatch,
    model,
    *,
    max_iterations=6,
    max_tool_calls=10,
    context=None,
):
    """使用确定性模型和独立图执行一次 Agent 请求。"""
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    settings = get_settings().model_copy(
        update={
            "agent_max_iterations": max_iterations,
            "agent_max_tool_calls": max_tool_calls,
        }
    )
    return [state async for state in build_agent_harness(settings).astream(
        {
            "original_question": "请读取 chunk_id=382",
            "effective_question": "请读取 chunk_id=382",
            "trace_id": "trace-loop",
            "session_id": "session-loop",
        },
        context=context
        or ToolInvocationContext(
            trace_id="trace-loop",
            session_id="session-loop",
            principal_id="loop-test",
            permissions=("knowledge_base:read",),
            source="iterative_loop_test",
        ),
    )][-1]


def test_initialize_agent_loop_state_uses_effective_question():
    """Runtime 只把本次有效问题写入 Agent 消息轨迹一次。"""
    invocation = prepare_agent_runtime_invocation(
        original_question="它什么时候开始？",
        effective_question="奖学金申请什么时候开始？",
        trace_id="trace-loop-init",
        session_id="session-loop-init",
    )

    assert invocation.agent_input["messages"] == [
        HumanMessage(content="奖学金申请什么时候开始？")
    ]


@pytest.mark.asyncio
async def test_agent_loop_answers_without_calling_tools(monkeypatch):
    """模型首轮以 structured response 结束，不执行业务工具。"""
    model = FakeToolCallingModel([_grounded_answer("直接回答")])

    output = await _invoke_graph(
        monkeypatch,
        model,
    )

    assert len(model.received_messages) == 1
    assert sum(
        bool(event.get("counted_call"))
        for event in output.get("governance_events", [])
    ) == 0
    assert output["stop_reason"] == "completed"
    assert output["status"] == "ok"
    assert output["result"]["status"] == "OK"
    assert output["result"]["answer"] == "直接回答"


@pytest.mark.asyncio
async def test_agent_loop_returns_single_tool_message_to_model(monkeypatch):
    """一次片段读取调用通过 Runtime 后，以关联 ToolMessage 进入第二轮。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 382},
                        "id": "chunk-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            _grounded_answer("片段读取完成。"),
        ]
    )

    output = await _invoke_graph(
        monkeypatch,
        model,
    )

    assert len(model.received_messages) == 2
    assert "get_chunk_context" in model.bound_tool_names
    tool_message = model.received_messages[1][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "chunk-call-1"
    assert tool_message.name == "get_chunk_context"
    assert not (
        isinstance(tool_message.artifact, dict)
        and "_project_tool_governance" in tool_message.artifact
    )
    payload = (tool_message.artifact or json.loads(tool_message.content))
    assert payload["ok"] is True
    assert payload["status"] == "success"
    assert payload["output"]["count"] == 1
    assert payload["output"]["evidence"][0]["page_content"] == "正文 382"
    assert payload["output"]["evidence"][0]["id"]
    assert payload["runtime"]["attempt_count"] == 1
    assert payload["tool_call_id"]
    assert sum(
        bool(event.get("counted_call"))
        for event in output.get("governance_events", [])
    ) == 1
    assert output["result"]["answer"] == "片段读取完成。"


@pytest.mark.asyncio
async def test_agent_loop_runs_two_sequential_tool_calls(monkeypatch):
    """两轮工具选择各自返回 ToolMessage，第三轮才生成最终回答。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 382},
                        "id": "chunk-seq-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 383},
                        "id": "chunk-seq-2",
                        "type": "tool_call",
                    }
                ],
            ),
            _grounded_answer("两个片段读取完成。"),
        ]
    )

    output = await _invoke_graph(
        monkeypatch,
        model,
    )

    assert len(model.received_messages) == 3
    assert model.received_messages[1][-1].tool_call_id == "chunk-seq-1"
    assert model.received_messages[2][-1].tool_call_id == "chunk-seq-2"
    assert any(
        isinstance(message, ToolMessage)
        and message.tool_call_id == "chunk-seq-1"
        for message in model.received_messages[2]
    )
    assert sum(
        bool(event.get("counted_call"))
        for event in output.get("governance_events", [])
    ) == 2
    assert [
        item["tool_args"]["chunk_id"]
        for item in output["result"]["tool_trace"] if "step_id" in item
    ] == [
        382,
        383,
    ]
    assert output["result"]["answer"] == "两个片段读取完成。"


@pytest.mark.asyncio
async def test_agent_loop_executes_all_tool_calls_from_one_message(monkeypatch):
    """同一 AIMessage 中的多个调用按关联 ID 全部返回，不静默丢失。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 384},
                        "id": "parallel-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 385},
                        "id": "parallel-2",
                        "type": "tool_call",
                    },
                ],
            ),
            _grounded_answer("两个片段读取完成。"),
        ]
    )

    output = await _invoke_graph(
        monkeypatch,
        model,
    )

    returned = [
        message
        for message in model.received_messages[1]
        if isinstance(message, ToolMessage)
    ]
    assert [message.tool_call_id for message in returned] == [
        "parallel-1",
        "parallel-2",
    ]
    assert sum(
        bool(event.get("counted_call"))
        for event in output.get("governance_events", [])
    ) == 2
    assert [
        item["tool_args"]["chunk_id"]
        for item in output["result"]["tool_trace"] if "step_id" in item
    ] == [
        384,
        385,
    ]


@pytest.mark.asyncio
async def test_agent_loop_allows_model_to_correct_failed_tool_arguments(monkeypatch):
    """不存在的 locator 返回参数错误，模型可修正参数后继续。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 999999},
                        "id": "bad-chunk",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 382},
                        "id": "fixed-chunk",
                        "type": "tool_call",
                    }
                ],
            ),
            _grounded_answer("修正后片段读取完成。"),
        ]
    )

    output = await _invoke_graph(
        monkeypatch,
        model,
    )

    first_tool_message = model.received_messages[1][-1]
    first_payload = (first_tool_message.artifact or json.loads(first_tool_message.content))
    assert first_tool_message.status == "error"
    assert first_payload["ok"] is False
    assert first_payload["error_code"] == "invalid_arguments"
    assert first_payload["retryable"] is False
    assert "metadata.chunk_id" in first_payload["error"]
    assert sum(
        bool(event.get("counted_call")) and not event.get("ok")
        for event in output["governance_events"]
    ) == 1
    assert [
        item["success"]
        for item in output["result"]["tool_trace"] if "step_id" in item
    ] == [
        False,
        True,
    ]
    assert output["result"]["status"] == "OK"
    assert output["result"]["answer"] == "修正后片段读取完成。"


@pytest.mark.asyncio
async def test_agent_loop_stops_at_max_iterations(monkeypatch):
    """模型持续请求工具时在模型轮次上限停止，不执行超限调用。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 386},
                        "id": "iteration-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 384},
                        "id": "iteration-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    output = await _invoke_graph(
        monkeypatch,
        model,
        max_iterations=2,
    )

    assert len(model.received_messages) == 2
    assert sum(
        event.get("kind") == "model" for event in output["governance_events"]
    ) == 2
    assert sum(
        bool(event.get("counted_call"))
        for event in output.get("governance_events", [])
    ) == 2
    assert output["stop_reason"] == "max_iterations_reached"
    assert output["status"] == "model_error"
    assert output["result"]["status"] == "MODEL_ERROR"


@pytest.mark.asyncio
async def test_agent_loop_stops_before_exceeding_tool_call_budget(monkeypatch):
    """已用完工具预算后，模型的新调用不会进入本地 handler。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 386},
                        "id": "budget-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 384},
                        "id": "budget-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    output = await _invoke_graph(
        monkeypatch,
        model,
        max_tool_calls=1,
    )

    assert len(model.received_messages) == 2
    assert sum(
        bool(event.get("counted_call"))
        for event in output.get("governance_events", [])
    ) == 1
    assert len([item for item in output["result"]["tool_trace"] if "step_id" in item]) == 1
    assert output["stop_reason"] == "max_tool_calls_reached"
    assert output["result"]["status"] == "MODEL_ERROR"


@pytest.mark.asyncio
async def test_agent_loop_stops_after_forged_hidden_tool_permission_denial(monkeypatch):
    """隐藏工具仍经过执行权限检查，并沿用 permission denied 停止语义。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_documents",
                        "args": {},
                        "id": "denied-list",
                        "type": "tool_call",
                    }
                ],
            ),
            _grounded_answer("hidden tool handled"),
        ]
    )
    denied_context = ToolInvocationContext(
        trace_id="trace-loop",
        session_id="session-loop",
        principal_id="unprivileged-test",
        permissions=(),
        source="iterative_loop_test",
    )

    output = await _invoke_graph(
        monkeypatch,
        model,
        context=denied_context,
    )

    assert len(model.received_messages) == 1
    assert output["stop_reason"] == "fatal_tool_error"
    assert output["result"]["status"] == "TOOL_ERROR"
    tool_trace = next(
        item
        for item in output["result"]["tool_trace"]
        if item.get("step") == "tool_call"
    )
    assert tool_trace["error_code"] == "permission_denied"
    assert tool_trace["handler_executed"] is False


@pytest.mark.asyncio
async def test_agent_loop_preserves_runtime_context_and_audit_chain(monkeypatch):
    """LangChain 注入的请求 Context 会进入项目审计事件。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 387},
                        "id": "governed-chunk",
                        "type": "tool_call",
                    }
                ],
            ),
            _grounded_answer("片段读取完成。"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-governed",
        session_id="session-governed",
        principal_id="governed-test",
        permissions=("knowledge_base:read",),
        source="agent_loop_governance_test",
    )
    audit_events = []

    class Sink:
        def write(self, event):
            audit_events.append(event)

    monkeypatch.setattr(
        framework_tools,
        "get_default_tool_audit_sink",
        lambda settings: Sink(),
    )

    output = await _invoke_graph(
        monkeypatch,
        model,
        context=context,
    )

    assert [event.event_type.value for event in audit_events] == [
        "requested",
        "started",
        "succeeded",
    ]
    assert {event.trace_id for event in audit_events} == {context.trace_id}
    assert {event.session_id for event in audit_events} == {context.session_id}
    assert {event.principal_id for event in audit_events} == {context.principal_id}
    trace = next(
        item
        for item in output["result"]["tool_trace"]
        if item.get("step") == "tool_call"
    )
    assert trace["attempt_count"] == 1
    assert trace["policy_decision"] == "allow"
    assert trace["tool_call_id"]


@pytest.mark.asyncio
async def test_agent_loop_model_failure_after_tool_does_not_replay_tool(monkeypatch):
    """工具后的模型失败受控结束，不通过另一条执行链重放工具。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_chunk_context",
                        "args": {"chunk_id": 388},
                        "id": "before-model-failure",
                        "type": "tool_call",
                    }
                ],
            ),
            RuntimeError("model unavailable"),
        ]
    )

    output = await _invoke_graph(
        monkeypatch,
        model,
    )

    assert len(model.received_messages) == 2
    assert sum(
        bool(event.get("counted_call"))
        for event in output.get("governance_events", [])
    ) == 1
    assert len([item for item in output["result"]["tool_trace"] if "step_id" in item]) == 1
    assert output["stop_reason"] == "agent_model_call_failed"
    assert output["result"]["status"] == "MODEL_ERROR"


@pytest.mark.asyncio
async def test_agent_selected_knowledge_search_returns_evidence_to_model(monkeypatch):
    """知识库与普通工具相同，成功后模型读取 evidence 并回答。"""
    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{
            "name": "search_knowledge_base",
            "args": {"query": "奖学金申请", "top_k": 5},
            "id": "knowledge-call", "type": "tool_call",
        }]),
        _grounded_answer("奖学金申请在九月开始。"),
    ])
    chunk = {"chunk_id": "chunk-1", "text": "奖学金申请在九月开始。"}
    spec = ToolSpec(
        name="search_knowledge_base", version="1.0.0",
        description="从测试知识库检索证据",
        output_model=SearchKnowledgeBaseOutput,
        handler=lambda args, context: tool_success("search_knowledge_base", args, {
            "query": args["query"], "count": 1,
            "evidence": [chunk_to_evidence_document(chunk)],
        }),
        args_model=SearchKnowledgeBaseArgs,
        required_permissions=("knowledge_base:read",),
    )
    monkeypatch.setattr(framework_tools, "get_local_tools", lambda settings: local_tools([spec]))

    output = await _invoke_graph(monkeypatch, model)

    assert len(model.received_messages) == 2
    message = model.received_messages[1][-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "knowledge-call"
    assert (message.artifact or json.loads(message.content))["output"]["evidence"] == [
        chunk_to_evidence_document(chunk).model_dump(mode="json"),
    ]
    assert "knowledge_base_result" not in output
    assert output["result"]["answer"] == "奖学金申请在九月开始。"
    assert [ref["reference_id"] for ref in output["result"]["references"]] == [chunk_to_evidence_document(chunk).id]
    assert output["result"]["llm_call_count"] == 2
    assert output["result"]["tool_trace"][-2]["mode"] == "agent_loop"
