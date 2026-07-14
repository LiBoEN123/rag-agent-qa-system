"""Checkpointer 编译注入、Runtime 隔离、持久化和生命周期测试。"""

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.api import lifecycle
from app.config import Settings
from app.graph import builder as graph_builder
from app.graph import checkpointing
from app.graph import runtime as graph_runtime


@pytest.fixture(autouse=True)
def isolate_graph_runtime(monkeypatch, tmp_path):
    """隔离 compiled Graph、Saver 和本地数据库配置。"""
    graph_builder.reset_agent_graph_runtime()
    monkeypatch.setattr(Settings, "AGENT_RUNTIME", "langgraph")
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "disabled")
    monkeypatch.setattr(
        Settings,
        "LANGGRAPH_CHECKPOINT_DB_PATH",
        tmp_path / "checkpoint.sqlite3",
    )
    monkeypatch.setattr(Settings, "ENABLE_AGENT_TRACE_STORE", False)
    yield
    graph_builder.reset_agent_graph_runtime()


def _decision(label: str = "能力说明") -> dict:
    return {
        "decision": "direct_answer",
        "action": "direct_answer",
        "tool_name": "",
        "tool_args": {},
        "reason": label,
        "router_source": "rule_based",
        "router_llm_used": False,
    }


def _run(trace_id: str, session_id: str = "session-1") -> dict:
    return graph_runtime.run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question=f"问题-{trace_id}",
        top_k=5,
        decompose_top_k=10,
        decision=_decision(trace_id),
        trace_id=trace_id,
        session_id=session_id,
    )


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
    mode,
    saver,
):
    """disabled 无参数编译，memory/sqlite 注入工厂返回的 Saver。"""
    calls: list[dict] = []

    class FakeBuilder:
        def compile(self, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", mode)
    monkeypatch.setattr(graph_builder, "get_checkpointer", lambda: saver)
    monkeypatch.setattr(graph_builder, "build_agent_graph", FakeBuilder)
    graph_builder.get_compiled_agent_graph.cache_clear()

    first = graph_builder.get_compiled_agent_graph()
    second = graph_builder.get_compiled_agent_graph()
    assert first is second
    assert calls == ([{}] if saver is None else [{"checkpointer": saver}])


def test_reset_recompiles_graph_and_replaces_memory_saver(monkeypatch):
    """统一 reset 同时失效 Graph 和 Saver，避免持有已关闭资源。"""
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "memory")
    first_graph = graph_builder.get_compiled_agent_graph()
    first_saver = checkpointing.get_checkpointer()

    graph_builder.reset_agent_graph_runtime()
    second_graph = graph_builder.get_compiled_agent_graph()
    second_saver = checkpointing.get_checkpointer()

    assert first_graph is not second_graph
    assert first_saver is not second_saver


def test_disabled_runtime_invokes_graph_without_config(monkeypatch):
    """默认关闭时保持旧的单参数 invoke 调用协议。"""
    calls: list[tuple] = []

    class RecordingGraph:
        def invoke(self, graph_input):
            calls.append((graph_input,))
            return {"result": {"status": "OK"}}

    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "disabled")
    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda: RecordingGraph(),
    )

    result = _run("trace-disabled")
    assert result == {"status": "OK"}
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["memory", "sqlite"])
def test_enabled_runtime_passes_request_scoped_thread_id(
    monkeypatch,
    mode,
):
    """启用 Saver 时 Runtime 内部生成 thread_id，API 无需新增字段。"""
    calls: list[dict] = []

    class RecordingGraph:
        def invoke(self, graph_input, config):
            calls.append({"input": graph_input, "config": config})
            return {"result": {"status": "OK"}}

    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", mode)
    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda: RecordingGraph(),
    )

    result = _run("trace-enabled", "private-session")
    expected = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-enabled",
        session_id="private-session",
    )
    assert result == {"status": "OK"}
    assert calls[0]["config"] == {
        "configurable": {"thread_id": expected}
    }
    assert "thread_id" not in calls[0]["input"]


def test_memory_checkpoint_serializes_real_agent_state(monkeypatch):
    """默认 Serializer 可真实往返当前 Pydantic AgentState。"""
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "memory")
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
    assert values["result"]["status"] == "OK"
    assert values["agent_state"].trace_id == "trace-memory"


def test_same_session_requests_use_independent_checkpoint_threads(monkeypatch):
    """同一 session 的不同请求不会继承上一轮 Graph 临时状态。"""
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "memory")
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


def test_sqlite_agent_checkpoint_survives_runtime_reset(monkeypatch, tmp_path):
    """关闭连接并重新初始化后，真实 Agent 最终状态仍存在。"""
    db_path = tmp_path / "agent" / "checkpoint.sqlite3"
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "sqlite")
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINT_DB_PATH", db_path)
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
    assert saved.checkpoint["channel_values"]["result"]["status"] == "OK"


def test_checkpoint_write_failure_returns_compatible_error_once(monkeypatch):
    """Saver/Graph 失败沿用顶层错误结果并只写一次 bad case。"""
    bad_cases: list[dict] = []

    class BrokenGraph:
        def invoke(self, graph_input, config):
            raise RuntimeError("checkpoint write failed")

    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "memory")
    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda: BrokenGraph(),
    )
    result = graph_runtime.run_agent_langgraph(
        write_bad_cases=bad_cases.append,
        question="失败问题",
        top_k=5,
        decompose_top_k=10,
        trace_id="trace-failed",
        session_id="session-failed",
    )

    assert result["status"] == "INTERNAL_ERROR"
    assert result["reason"] == "checkpoint write failed"
    assert len(bad_cases) == 1


@pytest.mark.parametrize(
    ("runtime", "mode", "expected_calls"),
    [
        ("legacy", "memory", 0),
        ("legacy", "sqlite", 0),
        ("langgraph", "disabled", 1),
        ("langgraph", "memory", 1),
        ("langgraph", "sqlite", 1),
    ],
)
def test_lifecycle_compiles_every_langgraph_mode_only(
    monkeypatch,
    runtime,
    mode,
    expected_calls,
):
    """LangGraph 各模式都预编译；legacy 不导入或编译主图。"""
    calls: list[str] = []

    def fake_get_compiled_agent_graph():
        calls.append("compiled")

    fake_get_compiled_agent_graph.cache_clear = lambda: None
    monkeypatch.setattr(
        graph_builder,
        "get_compiled_agent_graph",
        fake_get_compiled_agent_graph,
    )
    settings = SimpleNamespace(
        AGENT_RUNTIME=runtime,
        LANGGRAPH_CHECKPOINTER=mode,
    )

    lifecycle.initialize_agent_graph_runtime(settings)
    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    ("runtime", "mode", "error"),
    [
        ("invalid", "disabled", "AGENT_RUNTIME 配置无效"),
        ("langgraph", "invalid", "LANGGRAPH_CHECKPOINTER 配置无效"),
    ],
)
def test_lifecycle_rejects_invalid_runtime_configuration(
    runtime,
    mode,
    error,
):
    """启动校验明确拒绝非法值，不静默回退。"""
    settings = SimpleNamespace(
        AGENT_RUNTIME=runtime,
        LANGGRAPH_CHECKPOINTER=mode,
    )

    with pytest.raises(ValueError, match=error):
        lifecycle.validate_agent_runtime_configuration(settings)


def test_lifecycle_propagates_langgraph_compile_failure(monkeypatch):
    """LangGraph 依赖或编译失败会阻止启动，不返回假成功。"""
    def fail_compile():
        raise RuntimeError("graph compile failed")

    fail_compile.cache_clear = lambda: None
    monkeypatch.setattr(
        graph_builder,
        "get_compiled_agent_graph",
        fail_compile,
    )
    settings = SimpleNamespace(
        AGENT_RUNTIME="langgraph",
        LANGGRAPH_CHECKPOINTER="disabled",
    )

    with pytest.raises(RuntimeError, match="graph compile failed"):
        lifecycle.validate_agent_runtime_configuration(settings)


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
