"""最小 LangGraph Agent 主图构建测试。"""

import pytest

from app.agent.state import AgentPlanStep, AgentState
from app.graph import builder as graph_builder
from app.graph import nodes as graph_nodes


@pytest.fixture(autouse=True)
def clear_compiled_graph_cache():
    """隔离各测试使用的已编译 Graph。"""
    graph_builder.get_compiled_agent_graph.cache_clear()
    yield
    graph_builder.get_compiled_agent_graph.cache_clear()


def test_build_and_compile_graph_has_expected_public_nodes(monkeypatch):
    """Graph 可无业务副作用地创建、编译，并通过公开描述接口暴露节点。"""
    calls: list[str] = []

    def fail_if_called(*args, **kwargs):
        calls.append("business_call")
        raise AssertionError("Graph compilation must not call business functions")

    monkeypatch.setattr(graph_nodes, "route_agent_action", fail_if_called)
    monkeypatch.setattr(graph_nodes, "build_plan_from_decision", fail_if_called)
    monkeypatch.setattr(graph_nodes, "execute_plan", fail_if_called)
    monkeypatch.setattr(graph_nodes, "compose_final_answer", fail_if_called)
    monkeypatch.setattr(graph_nodes, "maybe_save_agent_trace", fail_if_called)

    uncompiled = graph_builder.build_agent_graph()
    compiled = uncompiled.compile()
    node_names = set(compiled.get_graph().nodes)

    assert calls == []
    assert {
        "route",
        "initialize",
        "plan",
        "execute",
        "rag_subgraph",
        "compose",
        "finalize",
    } <= node_names


@pytest.mark.parametrize("provide_decision", [True, False])
def test_compiled_graph_invokes_each_stage_once(monkeypatch, provide_decision):
    """有无外部 decision 时，Graph 都只执行一次必要的业务阶段。"""
    calls = {
        "route": 0,
        "plan": 0,
        "execute": 0,
        "compose": 0,
        "finalize": 0,
    }
    decision = {
        "decision": "direct_answer",
        "action": "direct_answer",
        "reason": "能力说明",
        "router_source": "rule_based",
        "router_llm_used": False,
    }

    def fake_route(question, top_k, trace_id=""):
        calls["route"] += 1
        assert question == "你能做什么？"
        assert top_k == 5
        assert trace_id == "trace-graph"
        return decision

    def fake_plan(decision, question, top_k):
        calls["plan"] += 1
        assert decision["action"] == "direct_answer"
        return [
            AgentPlanStep(
                step_id=1,
                action="direct_answer",
                reason="能力说明",
            )
        ]

    def fake_execute(state):
        calls["execute"] += 1
        assert isinstance(state, AgentState)
        assert state.session_id == "session-graph"
        state.status = "direct_answer"
        state.reason = "能力说明"
        return state

    def fake_compose(state):
        calls["compose"] += 1
        state.final_answer = "Graph answer"
        return state

    original_build_result = graph_nodes.build_v2_result

    def counted_build_result(state, current_decision):
        calls["finalize"] += 1
        return original_build_result(state, current_decision)

    monkeypatch.setattr(graph_nodes, "route_agent_action", fake_route)
    monkeypatch.setattr(graph_nodes, "build_plan_from_decision", fake_plan)
    monkeypatch.setattr(graph_nodes, "execute_plan", fake_execute)
    monkeypatch.setattr(graph_nodes, "compose_final_answer", fake_compose)
    monkeypatch.setattr(graph_nodes, "build_v2_result", counted_build_result)
    monkeypatch.setattr(graph_nodes, "maybe_save_agent_trace", lambda **kwargs: None)

    graph_input = {
        "question": "你能做什么？",
        "top_k": 5,
        "decompose_top_k": 10,
        "decision": decision if provide_decision else None,
        "trace_id": "trace-graph",
        "session_id": "session-graph",
    }
    output = graph_builder.build_agent_graph().compile().invoke(graph_input)

    assert output["result"]["status"] == "OK"
    assert output["result"]["answer"] == "Graph answer"
    assert calls == {
        "route": 0 if provide_decision else 1,
        "plan": 1,
        "execute": 1,
        "compose": 1,
        "finalize": 1,
    }


def test_get_compiled_agent_graph_reuses_compiled_instance():
    """同一进程内不会为每次请求重复构建和编译主图。"""
    first = graph_builder.get_compiled_agent_graph()
    second = graph_builder.get_compiled_agent_graph()

    assert first is second
