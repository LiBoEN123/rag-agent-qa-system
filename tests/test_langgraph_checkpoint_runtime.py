"""Checkpointer 编译注入、Runtime 隔离、持久化和生命周期测试。"""

import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent import framework_agent
from app.agent.results import AgentRunResult, create_agent_result
from app.api import lifecycle
from app.config import get_settings
from app.contracts.chat import AgentResultStatus
from app.graph import builder as graph_builder
from app.graph import checkpointing
from app.graph import nodes as graph_nodes
from app.graph import runtime as graph_runtime
from app.tools import runtime as tool_runtime
from app.tools.context import ToolInvocationContext


@pytest.fixture(autouse=True)
def isolate_graph_runtime(monkeypatch, settings_env, tmp_path):
    """隔离 compiled Graph、Saver 和本地数据库配置。"""
    graph_builder.reset_agent_graph_runtime()
    settings_env(
        langgraph_checkpointer="disabled",
        langgraph_checkpoint_db_path=tmp_path / "checkpoint.sqlite3",
        enable_agent_trace_store=False,
        tool_audit_enabled=False,
    )
    tool_runtime.reset_tool_runtime()
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: None)
    monkeypatch.setattr(
        graph_nodes,
        "route_agent_action",
        lambda question: {
            "decision": "direct_answer",
            "action": "direct_answer",
            "reason": question.removeprefix("问题-"),
            "router_source": "rule_based",
            "router_confidence": 1.0,
        },
    )
    yield
    tool_runtime.reset_tool_runtime()
    graph_builder.reset_agent_graph_runtime()


def _run(
    trace_id: str,
    session_id: str = "session-1",
    *,
    original_question: str | None = None,
    effective_question: str | None = None,
) -> dict:
    resolved_effective_question = effective_question or f"问题-{trace_id}"
    resolved_original_question = original_question or resolved_effective_question
    return graph_runtime.run_agent_langgraph(
        write_bad_cases=lambda record: None,
        original_question=resolved_original_question,
        effective_question=resolved_effective_question,
        top_k=5,
        decompose_top_k=10,
        trace_id=trace_id,
        session_id=session_id,
    )


def _valid_recording_result() -> dict:
    """为只关注 Runtime 配置的假 Graph 提供完整结果契约。"""
    return create_agent_result(status=AgentResultStatus.OK)


@pytest.mark.parametrize(
    ("mode", "saver"),
    [
        ("disabled", None),
        ("memory", InMemorySaver()),
        ("sqlite", object()),
    ],
)
def test_builder_compile_injects_only_enabled_checkpointer(
    monkeypatch,
    settings_env,
    mode,
    saver,
):
    """disabled 无参数编译，memory/sqlite 注入工厂返回的 Saver。"""
    calls: list[dict] = []

    class FakeBuilder:
        def compile(self, **kwargs):
            calls.append(kwargs)
            return object()

    settings = settings_env(langgraph_checkpointer=mode)
    monkeypatch.setattr(
        graph_builder,
        "get_checkpointer",
        lambda _settings=None: saver,
    )
    monkeypatch.setattr(
        graph_builder,
        "build_agent_graph",
        lambda _settings=None: FakeBuilder(),
    )
    graph_builder.get_compiled_agent_graph.cache_clear()

    first = graph_builder.get_compiled_agent_graph(settings)
    second = graph_builder.get_compiled_agent_graph(settings)
    assert first is second
    assert calls == ([{}] if saver is None else [{"checkpointer": saver}])


def test_reset_recompiles_graph_and_replaces_memory_saver(settings_env):
    """统一 reset 同时失效 Graph 和 Saver，避免持有已关闭资源。"""
    settings_env(langgraph_checkpointer="memory")
    first_graph = graph_builder.get_compiled_agent_graph()
    first_saver = checkpointing.get_checkpointer()

    graph_builder.reset_agent_graph_runtime()
    second_graph = graph_builder.get_compiled_agent_graph()
    second_saver = checkpointing.get_checkpointer()

    assert first_graph is not second_graph
    assert first_saver is not second_saver


def test_disabled_runtime_invokes_graph_with_context_and_recursion_config(
    monkeypatch,
    settings_env,
):
    """关闭 Saver 时仍传 Context 与框架递归保护，但不传 thread_id。"""
    calls: list[dict] = []

    class RecordingGraph:
        def invoke(self, graph_input, config, *, context):
            calls.append(
                {"input": graph_input, "config": config, "context": context}
            )
            return {"result": _valid_recording_result()}

    settings_env(langgraph_checkpointer="disabled")
    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda _settings=None: RecordingGraph(),
    )

    result = _run("trace-disabled")
    assert result == _valid_recording_result()
    assert len(calls) == 1
    settings = get_settings()
    assert calls[0]["context"] == ToolInvocationContext(
        trace_id="trace-disabled",
        session_id="session-1",
        principal_id=settings.tool_default_principal_id,
        permissions=settings.tool_default_permissions,
        source="chat_api",
    )
    assert calls[0]["config"] == {
        "recursion_limit": 68,
        "max_concurrency": 1,
    }
    assert "permissions" not in calls[0]["input"]


@pytest.mark.parametrize("mode", ["memory", "sqlite"])
def test_enabled_runtime_passes_request_scoped_thread_id(
    monkeypatch,
    settings_env,
    mode,
):
    """启用 Saver 时 Runtime 内部生成 thread_id，API 无需新增字段。"""
    calls: list[dict] = []

    class RecordingGraph:
        def invoke(self, graph_input, config, *, context):
            calls.append(
                {"input": graph_input, "config": config, "context": context}
            )
            return {"result": _valid_recording_result()}

    settings_env(langgraph_checkpointer=mode)
    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda _settings=None: RecordingGraph(),
    )

    result = _run("trace-enabled", "private-session")
    expected = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-enabled",
        session_id="private-session",
    )
    assert result == _valid_recording_result()
    assert calls[0]["config"] == {
        "configurable": {"thread_id": expected},
        "recursion_limit": 68,
        "max_concurrency": 1,
    }
    assert calls[0]["context"].trace_id == "trace-enabled"
    assert calls[0]["context"].session_id == "private-session"
    assert "thread_id" not in calls[0]["input"]
    assert "permissions" not in calls[0]["input"]


def test_memory_checkpoint_serializes_flat_runtime_state(settings_env):
    """默认 Serializer 可真实往返平铺状态、观察结果和 trace。"""
    settings_env(langgraph_checkpointer="memory")
    result = _run("trace-memory")
    thread_id = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-memory",
        session_id="session-1",
    )
    saver = checkpointing.get_checkpointer()
    saved = saver.get_tuple({"configurable": {"thread_id": thread_id}})

    assert result["status"] == "OK"
    assert saved is not None
    values = saved.checkpoint["channel_values"]
    assert values["trace_id"] == "trace-memory"
    assert values["original_question"] == "问题-trace-memory"
    assert values["effective_question"] == "问题-trace-memory"
    assert len(values["messages"]) == 1
    assert values["messages"][0].content == "问题-trace-memory"
    assert values["result"]["status"] == "OK"
    assert isinstance(values["result"], dict)
    assert not isinstance(values["result"], AgentRunResult)
    assert values["status"] == "direct_answer"
    assert values["observations"] == []
    assert values["tool_trace"] == values["result"]["tool_trace"]
    assert "agent_state" not in values
    assert "rewritten_question" not in values
    assert "permissions" not in values
    assert "principal_id" not in values


def test_memory_checkpoint_keeps_rewrite_fields_request_scoped(settings_env):
    """同一 session 的 checkpoint 保留各自问题语义且不会跨请求污染。"""
    settings_env(langgraph_checkpointer="memory")
    rewritten = "研究生奖学金申请什么时候开始"
    _run(
        "trace-rewritten",
        "shared-session",
        original_question="它什么时候开始",
        effective_question=rewritten,
    )
    _run(
        "trace-plain",
        "shared-session",
        original_question="学校什么时候放假",
        effective_question="学校什么时候放假",
    )
    saver = checkpointing.get_checkpointer()
    rewritten_id = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-rewritten",
        session_id="shared-session",
    )
    plain_id = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-plain",
        session_id="shared-session",
    )
    rewritten_values = saver.get_tuple(
        {"configurable": {"thread_id": rewritten_id}}
    ).checkpoint["channel_values"]
    plain_values = saver.get_tuple(
        {"configurable": {"thread_id": plain_id}}
    ).checkpoint["channel_values"]

    assert rewritten_values["original_question"] == "它什么时候开始"
    assert rewritten_values["effective_question"] == rewritten
    assert plain_values["original_question"] == "学校什么时候放假"
    assert plain_values["effective_question"] == "学校什么时候放假"
    assert "rewritten_question" not in rewritten_values
    assert "rewritten_question" not in plain_values
    assert "agent_state" not in rewritten_values
    assert "agent_state" not in plain_values


def test_memory_checkpoint_preserves_agent_and_clarify_flat_channels(
    monkeypatch,
    settings_env,
):
    """memory 模式覆盖普通 Agent 回答和 clarify 的平铺最终状态。"""

    class FinalAnswerModel:
        def bind_tools(self, tools):
            del tools
            return self

        def invoke(self, messages):
            del messages
            return AIMessage(content="普通 Agent 回答")

    settings_env(langgraph_checkpointer="memory")
    monkeypatch.setattr(
        graph_nodes,
        "route_agent_action",
        lambda question: (
            {
                "decision": "clarify",
                "action": "clarify",
                "clarification_question": "请补充具体对象",
                "reason": "问题需要澄清",
            }
            if "澄清" in question
            else {
                "decision": "agent",
                "action": "agent",
                "reason": "进入 Agent",
            }
        ),
    )
    monkeypatch.setattr(
        framework_agent,
        "get_agent_chat_model",
        lambda: FinalAnswerModel(),
    )

    agent_result = _run(
        "trace-memory-agent",
        original_question="普通问题",
        effective_question="普通问题",
    )
    clarify_result = _run(
        "trace-memory-clarify",
        original_question="需要澄清",
        effective_question="需要澄清",
    )
    saver = checkpointing.get_checkpointer()

    def values(trace_id):
        thread_id = checkpointing.build_checkpoint_thread_id(
            trace_id=trace_id,
            session_id="session-1",
        )
        return saver.get_tuple(
            {"configurable": {"thread_id": thread_id}}
        ).checkpoint["channel_values"]

    agent_values = values("trace-memory-agent")
    clarify_values = values("trace-memory-clarify")

    assert agent_result["status"] == "OK"
    assert agent_values["status"] == "ok"
    assert agent_values["final_answer"] == "普通 Agent 回答"
    assert agent_values["observations"] == []
    assert "agent_state" not in agent_values
    assert clarify_result["status"] == "NEED_CLARIFICATION"
    assert clarify_values["status"] == "clarify"
    assert clarify_values["final_answer"] == "请补充具体对象"
    assert "agent_state" not in clarify_values


def test_memory_checkpoint_preserves_plain_tool_observation(
    monkeypatch,
    settings_env,
):
    """memory 模式可序列化完整工具循环中的普通 dict observation。"""

    class ToolModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            del tools
            return self

        def invoke(self, messages):
            del messages
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "simple_calculator",
                            "args": {"expression": "2+3"},
                            "id": "memory-calc",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="结果是 5。")

    model = ToolModel()
    settings_env(langgraph_checkpointer="memory")
    monkeypatch.setattr(
        graph_nodes,
        "route_agent_action",
        lambda question: {
            "decision": "agent",
            "action": "agent",
            "reason": question,
        },
    )
    monkeypatch.setattr(
        framework_agent,
        "get_agent_chat_model",
        lambda: model,
    )

    result = _run(
        "trace-memory-tool",
        original_question="计算 2+3",
        effective_question="计算 2+3",
    )
    thread_id = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-memory-tool",
        session_id="session-1",
    )
    values = checkpointing.get_checkpointer().get_tuple(
        {"configurable": {"thread_id": thread_id}}
    ).checkpoint["channel_values"]

    assert result["status"] == "OK"
    assert values["observations"][0]["tool_name"] == "simple_calculator"
    assert values["observations"][0]["output"]["result"] == 5
    assert type(values["observations"][0]) is dict
    assert any(
        item.get("tool_name") == "simple_calculator"
        for item in values["tool_trace"]
    )
    assert "agent_state" not in values


def test_same_session_requests_use_independent_checkpoint_threads(
    settings_env,
):
    """同一 session 的不同请求不会继承上一轮 Graph 临时状态。"""
    settings_env(langgraph_checkpointer="memory")
    first = _run("trace-a", "shared-session")
    second = _run("trace-b", "shared-session")
    first_id = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-a",
        session_id="shared-session",
    )
    second_id = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-b",
        session_id="shared-session",
    )
    saver = checkpointing.get_checkpointer()
    first_saved = saver.get_tuple(
        {"configurable": {"thread_id": first_id}}
    )
    second_saved = saver.get_tuple(
        {"configurable": {"thread_id": second_id}}
    )

    assert first_id != second_id
    assert first["reason"] == "trace-a"
    assert second["reason"] == "trace-b"
    assert first_saved.checkpoint["channel_values"]["trace_id"] == "trace-a"
    assert second_saved.checkpoint["channel_values"]["trace_id"] == "trace-b"


def test_sqlite_agent_checkpoint_survives_runtime_reset(
    settings_env,
    tmp_path,
):
    """关闭连接并重新初始化后，真实 Agent 最终状态仍存在。"""
    db_path = tmp_path / "agent" / "checkpoint.sqlite3"
    settings_env(
        langgraph_checkpointer="sqlite",
        langgraph_checkpoint_db_path=db_path,
    )
    result = _run("trace-sqlite")
    thread_id = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-sqlite",
        session_id="session-1",
    )
    first = checkpointing.get_checkpointer()
    assert isinstance(first, SqliteSaver)
    assert result["status"] == "OK"
    assert first.get_tuple({"configurable": {"thread_id": thread_id}})

    graph_builder.reset_agent_graph_runtime()
    second = checkpointing.get_checkpointer()
    saved = second.get_tuple({"configurable": {"thread_id": thread_id}})
    assert isinstance(second, SqliteSaver)
    assert second is not first
    values = saved.checkpoint["channel_values"]
    assert values["result"]["status"] == "OK"
    assert len(values["messages"]) == 1
    assert values["messages"][0].content == "问题-trace-sqlite"
    assert values["observations"] == []
    assert values["tool_trace"] == values["result"]["tool_trace"]
    assert "agent_state" not in values
    assert "permissions" not in values
    assert "principal_id" not in values


def test_checkpoint_write_failure_returns_compatible_error_once(
    monkeypatch,
    settings_env,
):
    """Saver/Graph 失败沿用顶层错误结果并只写一次 bad case。"""
    bad_cases: list[dict] = []

    class BrokenGraph:
        def invoke(self, graph_input, config, *, context):
            raise RuntimeError("checkpoint write failed")

    settings_env(langgraph_checkpointer="memory")
    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda _settings=None: BrokenGraph(),
    )
    result = graph_runtime.run_agent_langgraph(
        write_bad_cases=bad_cases.append,
        original_question="失败问题",
        effective_question="失败问题",
        top_k=5,
        decompose_top_k=10,
        trace_id="trace-failed",
        session_id="session-failed",
    )

    assert result["status"] == "INTERNAL_ERROR"
    assert result["reason"] == "Agent 内部执行异常，请稍后重试"
    assert len(bad_cases) == 1


@pytest.mark.parametrize("mode", ["disabled", "memory", "sqlite"])
def test_lifecycle_compiles_every_langgraph_mode(
    monkeypatch,
    settings_factory,
    mode,
):
    """LangGraph 在各 Checkpointer 模式下均预编译一次。"""
    calls: list[str] = []

    def fake_get_compiled_agent_graph(resolved_settings):
        """记录生命周期传入的配置和预编译调用。"""
        assert resolved_settings is settings
        calls.append("compiled")

    fake_get_compiled_agent_graph.cache_clear = lambda: None
    monkeypatch.setattr(
        graph_builder,
        "get_compiled_agent_graph",
        fake_get_compiled_agent_graph,
    )
    settings = settings_factory(langgraph_checkpointer=mode)

    lifecycle.initialize_agent_graph_runtime(settings)
    assert calls == ["compiled"]


def test_lifecycle_rejects_invalid_checkpointer_configuration(
    settings_factory,
):
    """启动校验明确拒绝非法 Checkpointer 模式。"""
    with pytest.raises(ValidationError):
        settings_factory(langgraph_checkpointer="invalid")


def test_lifecycle_propagates_langgraph_compile_failure(
    monkeypatch,
    settings_factory,
):
    """LangGraph 依赖或编译失败会阻止启动，不返回假成功。"""
    def fail_compile(resolved_settings):
        """模拟接收正确配置后发生的图编译失败。"""
        assert resolved_settings is settings
        raise RuntimeError("graph compile failed")

    fail_compile.cache_clear = lambda: None
    monkeypatch.setattr(
        graph_builder,
        "get_compiled_agent_graph",
        fail_compile,
    )
    settings = settings_factory(langgraph_checkpointer="disabled")

    with pytest.raises(RuntimeError, match="graph compile failed"):
        lifecycle.validate_agent_graph_configuration(settings)


def test_lifecycle_close_is_idempotent(monkeypatch):
    """应用关闭可重复调用且每次均走统一资源重置入口。"""
    calls: list[str] = []
    monkeypatch.setattr(
        graph_builder,
        "reset_agent_graph_runtime",
        lambda: calls.append("reset"),
    )

    lifecycle.close_agent_graph_runtime()
    lifecycle.close_agent_graph_runtime()
    assert calls == ["reset", "reset"]
