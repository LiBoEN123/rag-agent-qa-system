import pytest

from app.agent import framework_agent
from app.graph import builder as graph_builder
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
    monkeypatch.setattr(graph_nodes, "compose_final_answer", fail_if_called)
    monkeypatch.setattr(graph_nodes, "maybe_save_agent_trace", fail_if_called)

    graph = graph_builder.build_agent_graph().compile().get_graph()
    node_names = {name for name in graph.nodes if not name.startswith("__")}
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert calls == []
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
    assert ("framework_agent", "adapt_framework") in edges
    assert ("rag_subgraph", "finalize") in edges
    assert ("compose", "finalize") in edges


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [
        (
            {
                "decision": "direct_answer",
                "action": "direct_answer",
                "reason": "能力说明",
            },
            "OK",
        ),
        (
            {
                "decision": "clarify",
                "action": "clarify",
                "clarification_question": "请补充具体对象。",
            },
            "NEED_CLARIFICATION",
        ),
    ],
)
def test_short_circuits_do_not_initialize_or_call_agent_model(
    monkeypatch,
    decision,
    expected_status,
):
    """direct_answer 与 clarify 从 initialize 直接进入 compose。"""

    def fail_model():
        raise AssertionError("short circuit must not initialize Agent Model")

    monkeypatch.setattr(framework_agent, "get_agent_chat_model", fail_model)
    output = graph_builder.build_agent_graph().compile().invoke(
        {
            "original_question": "测试问题",
            "effective_question": "测试问题",
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": decision,
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
    """旧内部 action 不再触发工具选择、RAG 回退或框架模型。"""
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
            "original_question": "测试问题",
            "effective_question": "测试问题",
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
    assert output["result"]["status"] == "INTERNAL_ERROR"
    assert output["result"]["answer"] == "Agent 内部执行异常，请稍后重试"
    assert output["result"]["reason"] == f"Unsupported route action: {action}"


def test_get_compiled_agent_graph_reuses_compiled_instance():
    first = graph_builder.get_compiled_agent_graph()
    second = graph_builder.get_compiled_agent_graph()

    assert first is second
