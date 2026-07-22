"""迭代式 Agent Loop 的消息、工具治理和预算回归测试。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import framework_agent, framework_tools
from app.graph import builder as graph_builder
from app.graph import checkpointing, runtime as graph_runtime
from app.graph.builder import build_agent_graph
from app.graph import nodes as graph_nodes
from app.graph.nodes import initialize_agent_state_node
from app.tools.context import ToolInvocationContext
from app.tools import runtime as tool_runtime
from app.tools.contracts import ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry
from app.tools.result import tool_success
from app.tools.schemas import SearchKnowledgeBaseArgs


class FakeToolCallingModel:
    """按顺序返回 AIMessage，并记录工具绑定和每轮完整输入。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tool_names: list[str] = []
        self.received_messages: list[list] = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.received_messages.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _invoke_graph(
    monkeypatch,
    model,
    *,
    decision,
    max_iterations=6,
    max_tool_calls=10,
    context=None,
):
    """使用确定性模型和独立图执行一次 Agent 请求。"""
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_ITERATIONS", max_iterations)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOOL_CALLS", max_tool_calls)
    return build_agent_graph().compile().invoke(
        {
            "original_question": "请计算 1+2",
            "effective_question": "请计算 1+2",
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": decision,
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
    )


def test_initialize_agent_loop_state_uses_effective_question(monkeypatch):
    """初始化只把本次有效问题写入 Agent 消息轨迹一次。"""
    monkeypatch.setattr(
        "app.graph.nodes.Settings.AGENT_MAX_ITERATIONS",
        6,
    )
    monkeypatch.setattr(
        "app.graph.nodes.Settings.AGENT_MAX_TOOL_CALLS",
        10,
    )

    updates = initialize_agent_state_node(
        {
            "original_question": "它什么时候开始？",
            "effective_question": "奖学金申请什么时候开始？",
            "trace_id": "trace-loop-init",
            "session_id": "session-loop-init",
            "decision": {"action": "direct_answer"},
        }
    )

    assert updates["messages"] == [
        HumanMessage(content="奖学金申请什么时候开始？")
    ]
    assert updates["iteration_count"] == 0
    assert updates["tool_call_count"] == 0
    assert updates["tool_error_count"] == 0
    assert updates["stop_reason"] == ""


def test_agent_loop_answers_without_calling_tools(monkeypatch):
    """模型首轮无 tool_calls 时直接经 Composer 与 Finalize 返回。"""
    model = FakeToolCallingModel([AIMessage(content="直接回答")])

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={"action": "agent", "router_llm_used": False},
    )

    assert len(model.received_messages) == 1
    assert output["tool_call_count"] == 0
    assert output["stop_reason"] == "completed"
    assert output["agent_state"].status == "ok"
    assert output["result"]["status"] == "OK"
    assert output["result"]["answer"] == "直接回答"


def test_agent_loop_returns_single_tool_message_to_model(monkeypatch):
    """一次计算调用通过 Runtime 后，以关联 ToolMessage 进入第二轮。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "1+2"},
                        "id": "calc-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="计算结果是 3。"),
        ]
    )

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={
            "action": "simple_calculator",
            "tool_name": "simple_calculator",
            "tool_args": {"expression": "1+2"},
            "router_llm_used": False,
        },
    )

    assert len(model.received_messages) == 2
    assert "simple_calculator" in model.bound_tool_names
    tool_message = model.received_messages[1][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "calc-call-1"
    assert tool_message.name == "simple_calculator"
    payload = json.loads(tool_message.content)
    assert payload["ok"] is True
    assert payload["status"] == "success"
    assert payload["output"]["result"] == 3
    assert payload["runtime"]["attempt_count"] == 1
    assert payload["tool_call_id"]
    assert output["tool_call_count"] == 1
    assert output["result"]["answer"] == "计算结果是 3。"


def test_agent_loop_runs_two_sequential_tool_calls(monkeypatch):
    """两轮工具选择各自返回 ToolMessage，第三轮才生成最终回答。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "1+2"},
                        "id": "calc-seq-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "3*4"},
                        "id": "calc-seq-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="两次计算结果分别是 3 和 12。"),
        ]
    )

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={"action": "simple_calculator", "router_llm_used": False},
    )

    assert len(model.received_messages) == 3
    assert model.received_messages[1][-1].tool_call_id == "calc-seq-1"
    assert model.received_messages[2][-1].tool_call_id == "calc-seq-2"
    assert any(
        isinstance(message, ToolMessage)
        and message.tool_call_id == "calc-seq-1"
        for message in model.received_messages[2]
    )
    assert output["tool_call_count"] == 2
    assert [item.output["result"] for item in output["agent_state"].observations] == [
        3,
        12,
    ]
    assert output["result"]["answer"] == "两次计算结果分别是 3 和 12。"


def test_agent_loop_executes_all_tool_calls_from_one_message(monkeypatch):
    """同一 AIMessage 中的多个调用按关联 ID 全部返回，不静默丢失。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "2+2"},
                        "id": "parallel-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "5*5"},
                        "id": "parallel-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="结果是 4 和 25。"),
        ]
    )

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={"action": "simple_calculator", "router_llm_used": False},
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
    assert output["tool_call_count"] == 2
    assert [item.output["result"] for item in output["agent_state"].observations] == [
        4,
        25,
    ]


def test_agent_loop_allows_model_to_correct_failed_tool_arguments(monkeypatch):
    """普通执行错误作为结构化观察返回，模型可修改参数后继续。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "1/0"},
                        "id": "bad-calc",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "1+2"},
                        "id": "fixed-calc",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="修正后结果是 3。"),
        ]
    )

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={"action": "simple_calculator", "router_llm_used": False},
    )

    first_tool_message = model.received_messages[1][-1]
    first_payload = json.loads(first_tool_message.content)
    assert first_tool_message.status == "error"
    assert first_payload["ok"] is False
    assert first_payload["error_code"] == "execution_error"
    assert output["tool_error_count"] == 1
    assert [item.success for item in output["agent_state"].observations] == [
        False,
        True,
    ]
    assert output["result"]["status"] == "OK"
    assert output["result"]["answer"] == "修正后结果是 3。"


def test_agent_loop_stops_at_max_iterations(monkeypatch):
    """模型持续请求工具时在模型轮次上限停止，不执行超限调用。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "1+1"},
                        "id": "iteration-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "2+2"},
                        "id": "iteration-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={"action": "simple_calculator", "router_llm_used": False},
        max_iterations=2,
    )

    assert len(model.received_messages) == 2
    assert output["iteration_count"] == 2
    assert output["tool_call_count"] == 2
    assert output["stop_reason"] == "max_iterations_reached"
    assert output["agent_state"].status == "model_error"
    assert output["result"]["status"] == "MODEL_ERROR"


def test_agent_loop_stops_before_exceeding_tool_call_budget(monkeypatch):
    """已用完工具预算后，模型的新调用不会进入 ToolRuntime。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "1+1"},
                        "id": "budget-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "2+2"},
                        "id": "budget-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={"action": "simple_calculator", "router_llm_used": False},
        max_tool_calls=1,
    )

    assert len(model.received_messages) == 2
    assert output["tool_call_count"] == 1
    assert len(output["agent_state"].observations) == 1
    assert output["stop_reason"] == "max_tool_calls_reached"
    assert output["result"]["status"] == "MODEL_ERROR"


def test_agent_loop_stops_after_unrecoverable_governance_error(monkeypatch):
    """权限拒绝写回 ToolMessage 与 trace 后停止，不再调用模型。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_documents",
                        "args": {},
                        "id": "denied-list",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    denied_context = ToolInvocationContext(
        trace_id="trace-loop",
        session_id="session-loop",
        principal_id="unprivileged-test",
        permissions=(),
        source="iterative_loop_test",
    )

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={"action": "list_documents", "router_llm_used": False},
        context=denied_context,
    )

    assert len(model.received_messages) == 1
    assert output["stop_reason"] == "fatal_tool_error"
    assert output["agent_state"].status == "tool_error"
    assert output["result"]["status"] == "TOOL_ERROR"
    tool_trace = next(
        item
        for item in output["result"]["tool_trace"]
        if item.get("step") == "tool_call"
    )
    assert tool_trace["error_code"] == "permission_denied"
    assert tool_trace["policy_decision"] == "deny"


def test_agent_loop_preserves_runtime_context_and_audit_chain(monkeypatch):
    """Agent 节点不直调 handler，完整 Context 由 Runtime 与审计事件接收。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "6*7"},
                        "id": "governed-calc",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="结果是 42。"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-governed",
        session_id="session-governed",
        principal_id="governed-test",
        permissions=("knowledge_base:read",),
        source="agent_loop_governance_test",
    )
    execute_contexts = []
    audit_events = []
    original_execute = tool_runtime.ToolRuntime.execute
    original_emit_audit = tool_runtime.ToolRuntime._emit_audit

    def recording_execute(self, tool_name, tool_args=None, *, context=None):
        execute_contexts.append(context)
        return original_execute(
            self,
            tool_name,
            tool_args,
            context=context,
        )

    def recording_emit_audit(self, *, event_type, **kwargs):
        audit_events.append((event_type.value, kwargs["context"]))
        return original_emit_audit(self, event_type=event_type, **kwargs)

    monkeypatch.setattr(tool_runtime.ToolRuntime, "execute", recording_execute)
    monkeypatch.setattr(
        tool_runtime.ToolRuntime,
        "_emit_audit",
        recording_emit_audit,
    )

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={"action": "simple_calculator", "router_llm_used": False},
        context=context,
    )

    assert execute_contexts == [context]
    assert [event for event, _ in audit_events] == [
        "requested",
        "started",
        "succeeded",
    ]
    assert all(event_context is context for _, event_context in audit_events)
    trace = next(
        item
        for item in output["result"]["tool_trace"]
        if item.get("step") == "tool_call"
    )
    assert trace["attempt_count"] == 1
    assert trace["policy_decision"] == "allow"
    assert trace["tool_call_id"]


def test_agent_loop_model_failure_after_tool_does_not_repeat_legacy_plan(monkeypatch):
    """工具后的模型失败受控结束，不降级重放初始 Planner 工具。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "8+9"},
                        "id": "before-model-failure",
                        "type": "tool_call",
                    }
                ],
            ),
            RuntimeError("model unavailable"),
        ]
    )

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={
            "action": "simple_calculator",
            "tool_name": "simple_calculator",
            "tool_args": {"expression": "8+9"},
            "router_llm_used": False,
        },
    )

    assert len(model.received_messages) == 2
    assert output["tool_call_count"] == 1
    assert len(output["agent_state"].observations) == 1
    assert output["stop_reason"] == "agent_model_call_failed"
    assert output["result"]["status"] == "MODEL_ERROR"


def test_agent_selected_knowledge_search_keeps_rag_citation_path(monkeypatch):
    """模型选择知识库检索后转入专用 RAG 子图并保留引用结果。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "奖学金申请", "top_k": 5},
                        "id": "rag-search-call",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    chunk = {
        "chunk_id": "chunk-rag-1",
        "text": "奖学金申请在九月开始。",
        "file_name": "policy.md",
    }
    rag_calls = []

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            version="1.0.0",
            description="从测试知识库检索证据",
            args_schema={"query": "string", "top_k": "integer"},
            output_schema={"chunks": "list"},
            handler=lambda args, context: tool_success(
                "search_knowledge_base",
                args,
                {"count": 1, "chunks": [chunk]},
            ),
            args_model=SearchKnowledgeBaseArgs,
            required_permissions=("knowledge_base:read",),
        )
    )

    def fake_rag_subgraph(**kwargs):
        rag_calls.append(kwargs)
        return {
            "status": "OK",
            "answer": "奖学金申请在九月开始。",
            "references": [{"reference_id": "Reference 1"}],
            "reason": "证据充分",
            "decompose_llm_used": False,
            "answer_llm_used": True,
            "llm_call_count": 1,
            "timings": {},
            "retrieval_confidence": 0.9,
            "answer_confidence": 0.8,
        }

    monkeypatch.setattr(
        framework_tools,
        "get_agent_tool_provider",
        lambda: LocalToolProvider(registry),
    )
    monkeypatch.setattr(graph_nodes, "run_rag_subgraph", fake_rag_subgraph)

    output = _invoke_graph(
        monkeypatch,
        model,
        decision={"action": "agent", "router_llm_used": False},
    )

    assert len(model.received_messages) == 1
    assert len(rag_calls) == 1
    assert rag_calls[0]["retrieved_chunks"] == [chunk]
    assert output["result"]["status"] == "OK"
    assert output["result"]["references"] == [
        {"reference_id": "Reference 1"}
    ]
    assert any(
        item.get("mode") == "rag"
        for item in output["result"]["tool_trace"]
    )


def test_sqlite_checkpointer_serializes_full_agent_message_loop(
    monkeypatch,
    tmp_path,
):
    """SQLite Saver 可往返 Human/AI/Tool/AI 的完整消息轨迹。"""
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "simple_calculator",
                        "args": {"expression": "10+5"},
                        "id": "sqlite-calc",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="结果是 15。"),
        ]
    )
    db_path = tmp_path / "iterative-checkpoint.sqlite3"
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_ITERATIONS", 6)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOOL_CALLS", 10)
    monkeypatch.setattr(
        checkpointing.Settings,
        "LANGGRAPH_CHECKPOINTER",
        "sqlite",
    )
    monkeypatch.setattr(
        checkpointing.Settings,
        "LANGGRAPH_CHECKPOINT_DB_PATH",
        db_path,
    )
    graph_builder.reset_agent_graph_runtime()
    try:
        result = graph_runtime.run_agent_langgraph(
            write_bad_cases=lambda record: None,
            original_question="请计算 10+5",
            effective_question="请计算 10+5",
            top_k=5,
            decompose_top_k=10,
            decision={
                "action": "simple_calculator",
                "tool_name": "simple_calculator",
                "tool_args": {"expression": "10+5"},
                "router_llm_used": False,
            },
            trace_id="trace-sqlite-loop",
            session_id="session-sqlite-loop",
        )
        thread_id = checkpointing.build_checkpoint_thread_id(
            trace_id="trace-sqlite-loop",
            session_id="session-sqlite-loop",
        )
        saved = checkpointing.get_checkpointer().get_tuple(
            {"configurable": {"thread_id": thread_id}}
        )

        assert result["status"] == "OK"
        assert saved is not None
        messages = saved.checkpoint["channel_values"]["messages"]
        assert [message.type for message in messages] == [
            "human",
            "ai",
            "tool",
            "ai",
        ]
        assert messages[2].tool_call_id == "sqlite-calc"
        assert saved.checkpoint["channel_values"]["result"]["answer"] == "结果是 15。"
    finally:
        graph_builder.reset_agent_graph_runtime()
