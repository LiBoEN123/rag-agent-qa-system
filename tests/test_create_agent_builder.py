"""直接 ``create_agent`` 构建与无 Parent Graph 架构测试。"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from app.agent import framework_agent
from app.prompts import AGENT_SYSTEM_PROMPT
from app.agent.runtime import (
    get_compiled_agent_runtime,
    reset_agent_runtime,
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


def test_compiled_create_agent_is_reused_for_same_settings():
    assert get_compiled_agent_runtime() is get_compiled_agent_runtime()


def test_parent_graph_modules_are_removed():
    graph_root = PROJECT_ROOT / "app" / "graph"

    assert list(graph_root.rglob("*.py")) == []


def test_builder_passes_retrieval_guidance_through_native_system_prompt(monkeypatch):
    create = Mock()
    monkeypatch.setattr(framework_agent, "create_agent", create)

    framework_agent.build_framework_agent(tools=[])

    create.assert_called_once()
    assert create.call_args.kwargs["system_prompt"] is AGENT_SYSTEM_PROMPT
    assert "search_knowledge_base" in AGENT_SYSTEM_PROMPT
    assert "ToolMessage" in AGENT_SYSTEM_PROMPT
    assert "修改 retrieval query" in AGENT_SYSTEM_PROMPT
    assert "不得用模型内部知识补齐" in AGENT_SYSTEM_PROMPT
