"""LangGraph Runtime 兼容性与异常行为测试。"""

import pytest

from app.agent import answer as agent_answer
from app.agent import executor as agent_executor
from app.agent import runtime as agent_runtime
from app.agent.runtime import run_agent_v2
from app.agent.state import AgentPlanStep, AgentState
from app.graph import builder as graph_builder
from app.graph import nodes as graph_nodes
from app.graph import runtime as graph_runtime
from app.graph.runtime import run_agent_langgraph
from app.observability.metrics import collect_trace_metrics


RESULT_FIELDS = [
    "status",
    "answer",
    "references",
    "need_clarification",
    "clarification_question",
    "reason",
    "action",
    "tool_result_summary",
    "router_llm_used",
    "decompose_llm_used",
    "answer_llm_used",
    "llm_call_count",
]


@pytest.fixture(autouse=True)
def isolate_graph_and_trace_store(monkeypatch):
    """隔离 Graph 缓存和可选 SQLite trace 写入。"""
    graph_builder.get_compiled_agent_graph.cache_clear()
    monkeypatch.setattr(
        agent_runtime.Settings,
        "ENABLE_AGENT_TRACE_STORE",
        False,
    )
    yield
    graph_builder.get_compiled_agent_graph.cache_clear()


def _decision(action: str) -> dict:
    """为等价性测试构造确定性的合法路由决策。"""
    if action == "search_knowledge_base":
        return {
            "decision": "use_knowledge_base",
            "action": action,
            "tool_name": action,
            "tool_args": {"query": "测试知识库", "top_k": 5},
            "reason": "检索知识库",
            "router_llm_used": False,
        }
    if action in {"simple_calculator", "tool_error"}:
        return {
            "decision": "use_tool",
            "action": "simple_calculator",
            "tool_name": "simple_calculator",
            "tool_args": {"expression": "1+2"},
            "reason": "执行计算",
            "router_llm_used": False,
        }
    if action == "clarify":
        return {
            "decision": "clarify",
            "action": "clarify",
            "tool_name": "",
            "tool_args": {},
            "reason": "问题存在指代不明",
            "clarification_question": "请补充具体对象。",
            "router_llm_used": False,
        }
    return {
        "decision": "direct_answer",
        "action": "direct_answer",
        "tool_name": "",
        "tool_args": {},
        "reason": "用户询问系统能力",
        "router_llm_used": False,
    }


def _trace_signature(result: dict) -> list[tuple]:
    """忽略耗时，仅提取新旧 Runtime 应保持一致的 trace 结构。"""
    return [
        (
            item.get("step"),
            item.get("action"),
            item.get("tool_name"),
            item.get("success"),
            item.get("status", item.get("tool_status")),
        )
        for item in result.get("tool_trace", [])
    ]


def test_runtime_without_decision_routes_once_and_forwards_trace_id(monkeypatch):
    """decision=None 时由 Graph 调用现有 Router，且只调用一次。"""
    calls: list[tuple] = []
    decision = _decision("direct_answer")

    def fake_route(question, top_k, trace_id=""):
        calls.append((question, top_k, trace_id))
        return decision

    monkeypatch.setattr(graph_nodes, "route_agent_action", fake_route)

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question="你能做什么？",
        top_k=5,
        decompose_top_k=10,
        decision=None,
        trace_id="trace-route",
    )

    assert result["status"] == "OK"
    assert result["action"] == "direct_answer"
    assert calls == [("你能做什么？", 5, "trace-route")]


def test_runtime_with_decision_initializes_existing_agent_state(monkeypatch):
    """已有 decision 时跳过 Router，并按旧 Runtime 字段初始化 AgentState。"""
    decision = _decision("direct_answer")
    observed: dict = {}

    def fail_route(*args, **kwargs):
        raise AssertionError("Router must not run for an existing decision")

    def fake_plan(decision, question, top_k):
        return [AgentPlanStep(step_id=1, action="direct_answer", reason="能力说明")]

    def fake_execute(state):
        observed.update(
            {
                "type": type(state),
                "trace_id": state.trace_id,
                "session_id": state.session_id,
                "original_question": state.original_question,
                "rewritten_question": state.rewritten_question,
                "top_k": state.top_k,
                "decompose_top_k": state.decompose_top_k,
                "max_steps": state.max_steps,
                "router_llm_used": state.router_llm_used,
            }
        )
        state.status = "direct_answer"
        state.final_answer = "兼容回答"
        return state

    monkeypatch.setattr(graph_nodes, "route_agent_action", fail_route)
    monkeypatch.setattr(graph_nodes, "build_plan_from_decision", fake_plan)
    monkeypatch.setattr(graph_nodes, "execute_plan", fake_execute)
    monkeypatch.setattr(graph_nodes, "compose_final_answer", lambda state: state)

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question="你能做什么？",
        top_k=4,
        decompose_top_k=9,
        decision=decision,
        trace_id="trace-state",
        session_id="session-state",
    )

    assert result["answer"] == "兼容回答"
    assert observed == {
        "type": AgentState,
        "trace_id": "trace-state",
        "session_id": "session-state",
        "original_question": "你能做什么？",
        "rewritten_question": "",
        "top_k": 4,
        "decompose_top_k": 9,
        "max_steps": graph_nodes.Settings.AGENT_MAX_STEPS,
        "router_llm_used": False,
    }


@pytest.mark.parametrize(
    "scenario",
    [
        "search_knowledge_base",
        "simple_calculator",
        "clarify",
        "direct_answer",
        "tool_error",
    ],
)
def test_langgraph_and_legacy_runtime_are_equivalent(monkeypatch, scenario):
    """确定性 RAG、工具、短路和错误场景下，新旧 Runtime 返回保持一致。"""
    dispatch_calls: list[str] = []

    def fake_dispatch(step, trace_id=""):
        dispatch_calls.append(step.tool_name)
        if scenario == "tool_error":
            return {
                "status": "error",
                "ok": False,
                "output": {},
                "error": "mock tool failure",
            }
        if step.tool_name == "search_knowledge_base":
            return {
                "status": "success",
                "ok": True,
                "output": {
                    "count": 1,
                    "chunks": [{"chunk_id": "chunk-1", "text": "证据"}],
                    "top_chunk_ids": ["chunk-1"],
                    "top_scores": [0.9],
                },
                "error": "",
            }
        return {
            "status": "success",
            "ok": True,
            "output": {"expression": "1+2", "result": 3},
            "error": "",
        }

    monkeypatch.setattr(agent_executor, "_call_dispatch_tool", fake_dispatch)
    monkeypatch.setattr(
        agent_answer,
        "_call_ask",
        lambda **kwargs: {
            "status": "OK",
            "answer": "知识库答案",
            "references": [{"reference_id": "Reference 1"}],
            "reason": "证据充分",
            "decompose_llm_used": True,
            "answer_llm_used": True,
            "llm_call_count": 2,
        },
    )
    monkeypatch.setattr(
        graph_nodes,
        "run_rag_subgraph",
        lambda **kwargs: {
            "status": "OK",
            "answer": "知识库答案",
            "references": [{"reference_id": "Reference 1"}],
            "reason": "证据充分",
            "decompose_llm_used": True,
            "answer_llm_used": True,
            "llm_call_count": 2,
        },
    )

    decision = _decision(scenario)
    common_kwargs = {
        "write_bad_cases": lambda record: None,
        "question": "测试问题",
        "top_k": 5,
        "decompose_top_k": 10,
        "decision": decision,
        "trace_id": f"trace-{scenario}",
        "session_id": "session-equivalence",
    }
    legacy_result = run_agent_v2(**common_kwargs)
    graph_result = run_agent_langgraph(**common_kwargs)

    assert {key: graph_result[key] for key in RESULT_FIELDS} == {
        key: legacy_result[key] for key in RESULT_FIELDS
    }
    assert _trace_signature(graph_result) == _trace_signature(legacy_result)
    if scenario in {"clarify", "direct_answer"}:
        assert dispatch_calls == []
    else:
        assert dispatch_calls == [decision["tool_name"], decision["tool_name"]]
    if scenario == "tool_error":
        assert graph_result["status"] == "TOOL_ERROR"
        assert graph_result["answer"] != ""
        tool_steps = [
            item for item in graph_result["tool_trace"]
            if item.get("step") == "tool_call"
        ]
        assert tool_steps[0]["success"] is False
        assert tool_steps[0]["error"] == "mock tool failure"


def test_knowledge_base_path_uses_rag_subgraph_without_legacy_compose(monkeypatch):
    """成功知识库 observation 进入子图，并直接 finalize 而不调用旧 ask。"""
    calls = {"subgraph": 0, "compose": 0}

    monkeypatch.setattr(
        agent_executor,
        "_call_dispatch_tool",
        lambda step, trace_id="": {
            "status": "success",
            "ok": True,
            "output": {
                "count": 1,
                "chunks": [{"chunk_id": "chunk-1", "text": "证据"}],
                "top_chunk_ids": ["chunk-1"],
                "top_scores": [0.9],
            },
            "error": "",
        },
    )

    def fake_subgraph(**kwargs):
        calls["subgraph"] += 1
        assert kwargs["retrieved_chunks"] == [
            {"chunk_id": "chunk-1", "text": "证据"}
        ]
        return {
            "status": "OK",
            "answer": "子图答案",
            "references": [{"reference_id": "Reference 1"}],
            "reason": "证据充分",
            "decompose_llm_used": False,
            "answer_llm_used": True,
            "llm_call_count": 1,
            "timings": {"llm_cost_ms": 3},
            "retrieval_confidence": 0.9,
            "answer_confidence": 0.8,
        }

    def fail_compose(state):
        calls["compose"] += 1
        raise AssertionError("知识库子图路径不应进入普通 compose 节点")

    monkeypatch.setattr(graph_nodes, "run_rag_subgraph", fake_subgraph)
    monkeypatch.setattr(graph_nodes, "compose_final_answer", fail_compose)
    monkeypatch.setattr(
        agent_answer,
        "_call_ask",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("LangGraph 路径不应调用旧 rag.ask")
        ),
    )

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question="测试知识库",
        top_k=5,
        decompose_top_k=10,
        decision=_decision("search_knowledge_base"),
        trace_id="trace-rag-subgraph",
    )

    assert result["status"] == "OK"
    assert result["answer"] == "子图答案"
    assert result["retrieval_confidence"] == 0.9
    assert result["answer_confidence"] == 0.8
    assert result["timings"] == {"llm_cost_ms": 3}
    assert collect_trace_metrics(result)["timings"] == {"llm_cost_ms": 3}
    assert calls == {"subgraph": 1, "compose": 0}
    compose_steps = [
        step
        for step in result["tool_trace"]
        if step.get("step") == "compose_final_answer"
    ]
    assert len(compose_steps) == 1


def test_failed_knowledge_base_observation_does_not_enter_rag_subgraph(monkeypatch):
    """知识库工具失败时保留 TOOL_ERROR，不能伪造 RAG 成功结果。"""
    monkeypatch.setattr(
        agent_executor,
        "_call_dispatch_tool",
        lambda step, trace_id="": {
            "status": "error",
            "ok": False,
            "output": {},
            "error": "mock knowledge base failure",
        },
    )
    monkeypatch.setattr(
        graph_nodes,
        "run_rag_subgraph",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("失败 observation 不应进入 RAG 子图")
        ),
    )

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question="测试知识库",
        top_k=5,
        decompose_top_k=10,
        decision=_decision("search_knowledge_base"),
    )

    assert result["status"] == "TOOL_ERROR"
    assert result["reason"] == "mock knowledge base failure"
    assert "mock knowledge base failure" in result["answer"]


def test_llm_usage_is_normalized_and_appended_once(monkeypatch):
    """Router、Decompose 和 Answer LLM 计数不丢失也不重复。"""
    decision = {
        **_decision("direct_answer"),
        "router_source": "llm",
        "router_llm_used": True,
    }
    monkeypatch.setattr(
        graph_nodes,
        "build_plan_from_decision",
        lambda **kwargs: [AgentPlanStep(step_id=1, action="direct_answer")],
    )

    def fake_execute(state):
        state.status = "direct_answer"
        return state

    def fake_compose(state):
        state.final_answer = "答案"
        state.decompose_llm_used = True
        state.answer_llm_used = True
        state.llm_call_count = 2
        return state

    monkeypatch.setattr(graph_nodes, "execute_plan", fake_execute)
    monkeypatch.setattr(graph_nodes, "compose_final_answer", fake_compose)

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question="测试",
        top_k=5,
        decompose_top_k=10,
        decision=decision,
    )

    usage_steps = [
        item for item in result["tool_trace"] if item.get("step") == "llm_usage"
    ]
    assert result["router_llm_used"] is True
    assert result["decompose_llm_used"] is True
    assert result["answer_llm_used"] is True
    assert result["llm_call_count"] == 3
    assert len(usage_steps) == 1
    assert usage_steps[0]["llm_call_count"] == 3


@pytest.mark.parametrize("malformed_output", [None, {}, {"result": None}])
def test_runtime_rejects_missing_or_invalid_graph_result(monkeypatch, malformed_output):
    """Graph 未生成合法结果时返回明确兼容错误，不暴露原始 Graph State。"""
    class MalformedGraph:
        def invoke(self, graph_input):
            return malformed_output

    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda: MalformedGraph(),
    )

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question="测试",
        top_k=5,
        decompose_top_k=10,
    )

    assert result["status"] == "INTERNAL_ERROR"
    assert result["references"] == []
    assert result["tool_trace"][0]["step"] == "agent_langgraph_error"
    assert "valid result" in result["reason"]


def test_top_level_graph_error_writes_bad_case_and_survives_trace_failure(monkeypatch):
    """节点异常转为兼容错误，bad case 与 trace 失败均不破坏主响应。"""
    bad_cases: list[dict] = []

    class BrokenGraph:
        def invoke(self, graph_input):
            raise RuntimeError("node exploded")

    monkeypatch.setattr(graph_runtime, "get_compiled_agent_graph", lambda: BrokenGraph())
    monkeypatch.setattr(agent_runtime.Settings, "ENABLE_AGENT_TRACE_STORE", True)
    monkeypatch.setattr(
        agent_runtime,
        "save_agent_trace",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("trace store failed")),
    )

    result = run_agent_langgraph(
        write_bad_cases=lambda record: bad_cases.append(record),
        question="测试异常",
        top_k=5,
        decompose_top_k=10,
        trace_id="trace-error",
        session_id="session-error",
    )

    assert result["status"] == "INTERNAL_ERROR"
    assert "LangGraph Agent" in result["answer"]
    assert result["reason"] == "node exploded"
    assert result["tool_trace"] == [
        {
            "step": "agent_langgraph_error",
            "status": "error",
            "error": "node exploded",
        }
    ]
    assert len(bad_cases) == 1
    assert bad_cases[0]["trace_id"] == "trace-error"
    assert bad_cases[0]["question"] == "测试异常"
    assert bad_cases[0]["route"] == "agent_langgraph"
    assert bad_cases[0]["error_type"] == "internal_error"
    assert bad_cases[0]["note"] == "node exploded"
