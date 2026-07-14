"""LangGraph Runtime 兼容性与异常行为测试。"""

import json

import pytest

from app.agent import answer as agent_answer
from app.agent import executor as agent_executor
from app.agent import runtime as agent_runtime
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
    monkeypatch.setattr(
        agent_runtime.Settings,
        "LANGGRAPH_CHECKPOINTER",
        "disabled",
    )
    yield
    graph_builder.get_compiled_agent_graph.cache_clear()


def _decision(action: str) -> dict:
    """为等价性测试构造确定性的合法路由决策。"""
    if action.startswith("search_"):
        return {
            "decision": "use_knowledge_base",
            "action": "search_knowledge_base",
            "tool_name": "search_knowledge_base",
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
    if action == "list_documents":
        return {
            "decision": "use_tool",
            "action": action,
            "tool_name": action,
            "tool_args": {},
            "reason": "列出文档",
            "router_llm_used": False,
        }
    if action == "get_chunk_text":
        return {
            "decision": "use_tool",
            "action": action,
            "tool_name": action,
            "tool_args": {"chunk_id": 1},
            "reason": "读取 chunk",
            "router_llm_used": False,
        }
    if action == "compare_documents":
        return {
            "decision": "use_tool",
            "action": action,
            "tool_name": action,
            "tool_args": {
                "query": "对比规定",
                "doc_a": "a.md",
                "doc_b": "b.md",
                "top_k": 5,
            },
            "reason": "对比文档",
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
    ("original_question", "effective_question"),
    [
        ("学校什么时候放假", "学校什么时候放假"),
        ("  学校什么时候放假  ", "学校什么时候放假"),
    ],
)
def test_explicit_questions_without_rewrite_keep_original_semantics(
    monkeypatch,
    original_question,
    effective_question,
):
    """显式问题字段经 strip 后相同时，不生成虚假的 rewritten question。"""
    observed: dict = {}

    monkeypatch.setattr(
        graph_nodes,
        "route_agent_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("已有 decision 时不应再次执行 Router")
        ),
    )

    def fake_plan(decision, question, top_k):
        observed["planner_question"] = question
        return [
            AgentPlanStep(
                step_id=1,
                action="direct_answer",
                reason="测试",
            )
        ]

    def fake_execute(state):
        observed["original_question"] = state.original_question
        observed["rewritten_question"] = state.rewritten_question
        state.status = "direct_answer"
        state.final_answer = "测试回答"
        return state

    monkeypatch.setattr(graph_nodes, "build_plan_from_decision", fake_plan)
    monkeypatch.setattr(graph_nodes, "execute_plan", fake_execute)
    monkeypatch.setattr(graph_nodes, "compose_final_answer", lambda state: state)

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question=effective_question,
        original_question=original_question,
        effective_question=effective_question,
        top_k=5,
        decompose_top_k=10,
        decision=_decision("direct_answer"),
    )

    assert result["status"] == "OK"
    assert observed == {
        "planner_question": "学校什么时候放假",
        "original_question": "学校什么时候放假",
        "rewritten_question": "",
    }


def test_rewritten_question_drives_plan_executor_rag_and_trace(monkeypatch):
    """改写后问题贯穿执行链，原始问题仅用于 AgentState 与 trace。"""
    original_question = "它什么时候开始"
    effective_question = "研究生奖学金申请什么时候开始"
    observed: dict = {"router_calls": 0}
    trace_records: list[dict] = []
    decision = {
        "decision": "use_knowledge_base",
        "action": "search_knowledge_base",
        "tool_name": "search_knowledge_base",
        "tool_args": {},
        "reason": "检索知识库",
        "router_source": "rule_based",
        "router_llm_used": False,
    }

    def fail_route(*args, **kwargs):
        observed["router_calls"] += 1
        raise AssertionError("ChatService 已提供 Router decision，不应重复路由")

    original_execute = graph_nodes.execute_plan

    def capture_execute(state):
        observed["agent_original_question"] = state.original_question
        observed["agent_rewritten_question"] = state.rewritten_question
        observed["planner_query"] = state.plan[0].tool_args["query"]
        return original_execute(state)

    def fake_dispatch(step, trace_id=""):
        observed["executor_query"] = step.tool_args["query"]
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

    def fake_rag_subgraph(**kwargs):
        observed["rag_question"] = kwargs["question"]
        return {
            "status": "OK",
            "answer": "申请将在九月开始",
            "references": [{"reference_id": "Reference 1"}],
            "reason": "证据充分",
            "decompose_llm_used": False,
            "answer_llm_used": True,
            "llm_call_count": 1,
            "timings": {},
            "retrieval_confidence": 0.9,
            "answer_confidence": 0.8,
        }

    monkeypatch.setattr(graph_nodes, "route_agent_action", fail_route)
    monkeypatch.setattr(graph_nodes, "execute_plan", capture_execute)
    monkeypatch.setattr(agent_executor, "_call_dispatch_tool", fake_dispatch)
    monkeypatch.setattr(graph_nodes, "run_rag_subgraph", fake_rag_subgraph)
    monkeypatch.setattr(agent_runtime.Settings, "ENABLE_AGENT_TRACE_STORE", True)
    monkeypatch.setattr(
        agent_runtime,
        "save_agent_trace",
        lambda **kwargs: trace_records.append(kwargs),
    )

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question=effective_question,
        original_question=original_question,
        effective_question=effective_question,
        top_k=5,
        decompose_top_k=10,
        decision=decision,
        trace_id="trace-rewrite",
        session_id="session-rewrite",
    )

    assert result["status"] == "OK"
    assert observed == {
        "router_calls": 0,
        "agent_original_question": original_question,
        "agent_rewritten_question": effective_question,
        "planner_query": effective_question,
        "executor_query": effective_question,
        "rag_question": effective_question,
    }
    assert len(trace_records) == 1
    assert trace_records[0]["question"] == original_question
    assert trace_records[0]["rewritten_question"] == effective_question


@pytest.mark.parametrize(
    "scenario",
    [
        "search_success",
        "search_no_evidence",
        "search_model_error",
        "simple_calculator",
        "list_documents",
        "get_chunk_text",
        "compare_documents",
        "clarify",
        "direct_answer",
        "tool_error",
    ],
)
def test_langgraph_business_scenarios_preserve_result_protocol(monkeypatch, scenario):
    """确定性 RAG、工具、短路和错误场景保持既有结果协议。"""
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
        if step.tool_name == "list_documents":
            output = {
                "count": 2,
                "documents": [
                    {"file_name": "a.md"},
                    {"file_name": "b.md"},
                ],
            }
        elif step.tool_name == "get_chunk_text":
            output = {"chunk_id": 1, "chunk": {"text": "chunk 证据"}}
        elif step.tool_name == "compare_documents":
            output = {"comparison": "两份文档存在差异"}
        else:
            output = {"expression": "1+2", "result": 3}
        return {
            "status": "success",
            "ok": True,
            "output": output,
            "error": "",
        }

    monkeypatch.setattr(agent_executor, "_call_dispatch_tool", fake_dispatch)

    if scenario == "search_no_evidence":
        rag_result = {
            "status": "NO_RELEVANT_INFO",
            "answer": "没有足够证据",
            "references": [],
            "reason": "证据不足",
            "decompose_llm_used": False,
            "answer_llm_used": False,
            "llm_call_count": 0,
        }
    elif scenario == "search_model_error":
        rag_result = {
            "status": "MODEL_ERROR",
            "answer": "模型暂时不可用",
            "references": [],
            "reason": "模型暂时不可用",
            "decompose_llm_used": False,
            "answer_llm_used": True,
            "llm_call_count": 1,
        }
    else:
        rag_result = {
            "status": "OK",
            "answer": "知识库答案",
            "references": [{"reference_id": "Reference 1"}],
            "reason": "证据充分",
            "decompose_llm_used": True,
            "answer_llm_used": True,
            "llm_call_count": 2,
        }
    monkeypatch.setattr(
        graph_nodes,
        "run_rag_subgraph",
        lambda **kwargs: dict(rag_result),
    )

    decision = _decision(scenario)
    common_kwargs = {
        "write_bad_cases": lambda record: None,
        "question": "测试问题",
        "top_k": 5,
        "decompose_top_k": 10,
        "decision": decision,
        "trace_id": f"trace-{scenario}",
        "session_id": "session-regression",
    }
    graph_result = run_agent_langgraph(**common_kwargs)

    assert set(RESULT_FIELDS).issubset(graph_result)
    assert isinstance(graph_result["answer"], str)
    assert isinstance(graph_result["references"], list)
    assert isinstance(graph_result["need_clarification"], bool)
    assert isinstance(graph_result["tool_trace"], list)
    assert isinstance(graph_result["llm_call_count"], int)
    if scenario in {"clarify", "direct_answer"}:
        assert dispatch_calls == []
    else:
        assert dispatch_calls == [decision["tool_name"]]
    if scenario == "tool_error":
        assert graph_result["status"] == "TOOL_ERROR"
        assert graph_result["answer"] != ""
        tool_steps = [
            item for item in graph_result["tool_trace"]
            if item.get("step") == "tool_call"
        ]
        assert tool_steps[0]["success"] is False
        assert tool_steps[0]["error"] == "mock tool failure"


def test_internal_runtime_errors_keep_compatible_business_fields(monkeypatch):
    """内部异常由 LangGraph 顶层转换为现有兼容错误结构。"""
    decision = _decision("direct_answer")

    def fail_plan(*args, **kwargs):
        raise RuntimeError("shared orchestration failure")

    monkeypatch.setattr(graph_nodes, "build_plan_from_decision", fail_plan)
    kwargs = {
        "write_bad_cases": lambda record: None,
        "question": "异常问题",
        "top_k": 5,
        "decompose_top_k": 10,
        "decision": decision,
        "trace_id": "trace-internal-error",
    }

    graph_result = run_agent_langgraph(**kwargs)

    assert graph_result["status"] == "INTERNAL_ERROR"
    assert graph_result["answer"] == "Agent 内部执行异常，请稍后重试"
    assert graph_result["reason"] == "Agent 内部执行异常，请稍后重试"
    assert graph_result["tool_trace"] == [
        {
            "step": "agent_langgraph_error",
            "status": "error",
            "error": "internal_error",
        }
    ]


def test_knowledge_base_path_uses_rag_subgraph_without_sequential_rag_compose(monkeypatch):
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


@pytest.mark.parametrize(
    ("action", "output"),
    [
        ("simple_calculator", {"expression": "1+2", "result": 3}),
        (
            "list_documents",
            {"count": 1, "documents": [{"file_name": "policy.md"}]},
        ),
        ("get_chunk_text", {"chunk_id": 1, "chunk": {"text": "证据"}}),
        ("compare_documents", {"comparison": "两份文档存在差异"}),
    ],
)
def test_regular_tool_path_dispatches_and_composes_once(
    monkeypatch,
    action,
    output,
):
    """默认 Graph 普通工具路径不进入 RAG，且 dispatch/compose 各一次。"""
    calls = {"dispatch": 0, "compose": 0, "rag": 0}

    def dispatch(step, trace_id=""):
        calls["dispatch"] += 1
        assert step.tool_name == action
        return {
            "status": "success",
            "ok": True,
            "output": output,
            "error": "",
        }

    original_compose = graph_nodes.compose_final_answer

    def compose(state):
        calls["compose"] += 1
        return original_compose(state)

    def fail_rag(**kwargs):
        calls["rag"] += 1
        raise AssertionError("普通工具不应进入 RAG 子图")

    monkeypatch.setattr(agent_executor, "_call_dispatch_tool", dispatch)
    monkeypatch.setattr(graph_nodes, "compose_final_answer", compose)
    monkeypatch.setattr(graph_nodes, "run_rag_subgraph", fail_rag)

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        question="普通工具问题",
        top_k=5,
        decompose_top_k=10,
        decision=_decision(action),
        trace_id=f"trace-{action}",
    )

    assert result["status"] == "OK"
    assert calls == {"dispatch": 1, "compose": 1, "rag": 0}
    assert [
        item["step"] for item in result["tool_trace"]
    ].count("tool_call") == 1
    assert [
        item["step"] for item in result["tool_trace"]
    ].count("compose_final_answer") == 1


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
    assert result["reason"] == "Agent 内部执行异常，请稍后重试"


def test_top_level_graph_error_is_sanitized_but_kept_for_internal_diagnostics(
    monkeypatch,
    caplog,
):
    """节点异常对外脱敏，同时由异常日志和 bad case 保留诊断信息。"""
    bad_cases: list[dict] = []
    sensitive_message = (
        "/home/test-user/project/checkpoints.sqlite: "
        "model request failed with api_key=sk-test-secret"
    )
    sensitive_error = RuntimeError(sensitive_message)

    class BrokenGraph:
        def invoke(self, graph_input):
            raise sensitive_error

    monkeypatch.setattr(graph_runtime, "get_compiled_agent_graph", lambda: BrokenGraph())
    monkeypatch.setattr(agent_runtime.Settings, "ENABLE_AGENT_TRACE_STORE", True)
    monkeypatch.setattr(
        agent_runtime,
        "save_agent_trace",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("trace store failed")),
    )

    exception_calls = []
    original_logger_exception = graph_runtime.logger.exception

    def record_logger_exception(*args, **kwargs):
        exception_calls.append((args, kwargs))
        return original_logger_exception(*args, **kwargs)

    monkeypatch.setattr(
        graph_runtime.logger,
        "exception",
        record_logger_exception,
    )
    monkeypatch.setattr(graph_runtime.logger, "propagate", True)
    with caplog.at_level("ERROR", logger=graph_runtime.logger.name):
        result = run_agent_langgraph(
            write_bad_cases=lambda record: bad_cases.append(record),
            question="研究生奖学金申请什么时候开始",
            original_question="它什么时候开始",
            effective_question="研究生奖学金申请什么时候开始",
            top_k=5,
            decompose_top_k=10,
            trace_id="trace-error",
            session_id="session-error",
        )

    assert result["status"] == "INTERNAL_ERROR"
    assert result["answer"] == "Agent 内部执行异常，请稍后重试"
    assert result["reason"] == "Agent 内部执行异常，请稍后重试"
    assert result["tool_trace"] == [
        {
            "step": "agent_langgraph_error",
            "status": "error",
            "error": "internal_error",
        }
    ]
    response_text = json.dumps(result, ensure_ascii=False)
    assert "Agent 内部执行异常，请稍后重试" in response_text
    assert "/home/test-user/project" not in response_text
    assert "checkpoints.sqlite" not in response_text
    assert "sk-test-secret" not in response_text
    assert "RuntimeError" not in response_text
    assert "model request failed" not in response_text

    assert len(exception_calls) == 1
    exception_records = [
        record
        for record in caplog.records
        if "langgraph_graph_failed" in record.getMessage()
    ]
    assert exception_records
    assert exception_records[0].exc_info is not None
    assert exception_records[0].exc_info[1] is sensitive_error
    assert len(bad_cases) == 1
    assert bad_cases[0]["trace_id"] == "trace-error"
    assert bad_cases[0]["question"] == "它什么时候开始"
    assert bad_cases[0]["route"] == "agent_langgraph"
    assert bad_cases[0]["error_type"] == "internal_error"
    assert bad_cases[0]["note"] == sensitive_message
