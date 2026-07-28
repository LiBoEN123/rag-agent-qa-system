import pytest

from app.agent import framework_agent
from app.graph import builder as graph_builder
from app.graph import conditions as graph_conditions
from app.graph import nodes as graph_nodes
from app.tools.context import ToolInvocationContext
from app.tools.runtime import ToolRuntime


@pytest.fixture(autouse=True)
def clear_compiled_graph_cache():
    graph_builder.get_compiled_agent_graph.cache_clear()
    yield
    graph_builder.get_compiled_agent_graph.cache_clear()


def test_build_and_compile_graph_has_only_framework_execution_nodes(monkeypatch):
    """构图无业务副作用，父图只暴露单一框架执行路径。"""
    calls: list[str] = []

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        calls.append("business_call")
        raise AssertionError("Graph compilation must not call business functions")

    monkeypatch.setattr(graph_nodes, "route_agent_action", fail_if_called)
    monkeypatch.setattr(
        graph_nodes,
        "build_compose_answer_updates",
        fail_if_called,
    )
    monkeypatch.setattr(graph_nodes, "maybe_save_agent_trace", fail_if_called)

    graph = graph_builder.build_agent_graph().compile().get_graph()
    node_names = {name for name in graph.nodes if not name.startswith("__")}
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert calls == []
    assert not hasattr(graph_conditions, "route_condition")
    assert node_names == {
        "route",
        "initialize",
        "framework_agent",
        "adapt_framework",
        "rag_subgraph",
        "compose",
        "finalize",
    }
    assert {"plan", "execute"}.isdisjoint(node_names)
    assert ("__start__", "route") in edges
    assert ("__start__", "initialize") not in edges
    assert ("route", "initialize") in edges
    assert ("framework_agent", "adapt_framework") in edges
    assert ("rag_subgraph", "finalize") in edges
    assert ("compose", "finalize") in edges


@pytest.mark.parametrize(
    ("question", "injected_decision", "expected_status"),
    [
        (
            "你能做什么？",
            {"decision": "clarify", "action": "clarify"},
            "OK",
        ),
        (
            "这个怎么配置",
            {"decision": "direct_answer", "action": "direct_answer"},
            "NEED_CLARIFICATION",
        ),
    ],
)
def test_short_circuits_do_not_initialize_or_call_agent_model(
    monkeypatch,
    question,
    injected_decision,
    expected_status,
):
    """route 覆盖旧注入后，direct_answer 与 clarify 直接进入 compose。"""

    def fail_model():
        raise AssertionError("short circuit must not initialize Agent Model")

    monkeypatch.setattr(framework_agent, "get_agent_chat_model", fail_model)
    output = graph_builder.build_agent_graph().compile().invoke(
        {
            "original_question": question,
            "effective_question": question,
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": injected_decision,
            "trace_id": "trace-short",
        },
        context=ToolInvocationContext(trace_id="trace-short"),
    )

    assert output["result"]["status"] == expected_status
    assert output["iteration_count"] == 0
    assert output["tool_call_count"] == 0
    assert output["governance_events"] == []


@pytest.mark.parametrize(
    "action",
    [
        "rag_fast_path",
        "search_knowledge_base",
        "simple_calculator",
        "use_tool",
        "unknown_action",
    ],
)
def test_legacy_route_actions_fail_closed_without_model_or_tool(
    monkeypatch,
    action,
):
    """旧注入 action 被图内 Router 覆盖，不能触发模型或工具。"""
    calls = {"model": 0, "tool": 0}

    def fail_model():
        calls["model"] += 1
        raise AssertionError("legacy action must not initialize Agent Model")

    def fail_tool(self, tool_name, tool_args=None, *, context=None):
        del self, tool_name, tool_args, context
        calls["tool"] += 1
        raise AssertionError("legacy action must not execute a tool")

    monkeypatch.setattr(framework_agent, "get_agent_chat_model", fail_model)
    monkeypatch.setattr(ToolRuntime, "execute", fail_tool)
    output = graph_builder.build_agent_graph().compile().invoke(
        {
            "original_question": "你能做什么？",
            "effective_question": "你能做什么？",
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": {
                "decision": "use_tool",
                "action": action,
                "tool_name": "simple_calculator",
                "tool_args": {"expression": "1+2"},
            },
            "trace_id": "trace-unsupported",
        },
        context=ToolInvocationContext(trace_id="trace-unsupported"),
    )

    assert calls == {"model": 0, "tool": 0}
    assert output["decision"]["action"] == "direct_answer"
    assert output["result"]["status"] == "OK"


def test_get_compiled_agent_graph_reuses_compiled_instance():
    first = graph_builder.get_compiled_agent_graph()
    second = graph_builder.get_compiled_agent_graph()

    assert first is second


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("running", "framework_agent"),
        ("ok", "compose"),
        ("clarify", "compose"),
        (None, "compose"),
    ],
)
def test_after_initialize_condition_reads_flat_status(status, expected):
    state = {} if status is None else {"status": status}

    assert graph_conditions.after_initialize_condition(state) == expected
