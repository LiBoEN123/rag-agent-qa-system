"""直接 ``create_agent`` 构建与无 Parent Graph 架构测试。"""

from pathlib import Path

import pytest

from app.agent import runtime as agent_runtime
from app.agent.runtime import (
    get_compiled_agent_runtime,
    reset_agent_runtime,
    run_agent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_runtime_cache():
    reset_agent_runtime()
    yield
    reset_agent_runtime()


def test_create_agent_is_the_only_agent_graph():
    node_names = set(get_compiled_agent_runtime().get_graph().nodes)

    assert {"__start__", "model", "tools", "__end__"} <= node_names
    assert {
        "route",
        "initialize",
        "framework_agent",
        "adapt_framework",
        "rag_subgraph",
        "compose",
        "finalize",
    }.isdisjoint(node_names)


@pytest.mark.parametrize(
    ("question", "status", "action"),
    [
        ("你是谁？", "OK", "direct_answer"),
        ("这个怎么弄", "NEED_CLARIFICATION", "clarify"),
    ],
)
def test_short_circuits_do_not_build_or_call_agent(
    monkeypatch,
    question,
    status,
    action,
):
    monkeypatch.setattr(
        agent_runtime,
        "get_compiled_agent_runtime",
        lambda settings=None, **kwargs: (_ for _ in ()).throw(
            AssertionError("short circuit must not build create_agent")
        ),
    )

    result = run_agent(
        write_bad_cases=lambda record: None,
        original_question=question,
        effective_question=question,
        top_k=5,
        decompose_top_k=10,
    )

    assert result["status"] == status
    assert result["action"] == action


def test_compiled_create_agent_is_reused_for_same_settings():
    assert get_compiled_agent_runtime() is get_compiled_agent_runtime()


def test_parent_graph_modules_are_removed():
    graph_root = PROJECT_ROOT / "app" / "graph"

    assert list(graph_root.rglob("*.py")) == []
