import json
import inspect
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.agent import framework_agent
from app.agent.results import create_agent_result
from app.contracts.chat import AgentResultStatus
from app.graph import builder as graph_builder
from app.graph import nodes as graph_nodes
from app.graph import runtime as graph_runtime
from app.graph.runtime import run_agent_langgraph


RESULT_FIELDS = {
    "status",
    "answer",
    "references",
    "need_clarification",
    "clarification_question",
    "reason",
    "action",
    "tool_result_summary",
    "tool_trace",
    "router_llm_used",
    "decompose_llm_used",
    "answer_llm_used",
    "llm_call_count",
}


class FakeModel:
    """返回确定性消息或异常，并记录实际模型输入。"""

    def __init__(self, response):
        self.response = response
        self.received_messages: list[list] = []

    def bind_tools(self, tools):
        del tools
        return self

    def invoke(self, messages):
        self.received_messages.append(list(messages))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture(autouse=True)
def isolate_graph_and_trace_store(settings_env):
    """禁用图检查点与追踪存储，并在用例后清理图缓存。"""
    settings_env(
        enable_agent_trace_store=False,
        langgraph_checkpointer="disabled",
    )
    yield
    graph_builder.get_compiled_agent_graph.cache_clear()


def _run(*, original="测试问题", effective="测试问题"):
    return run_agent_langgraph(
        write_bad_cases=lambda record: None,
        original_question=original,
        effective_question=effective,
        top_k=5,
        decompose_top_k=10,
        trace_id="trace-runtime",
        session_id="session-runtime",
    )


def test_runtime_routes_once_with_effective_question_and_records_cost(monkeypatch):
    calls: list[str] = []
    decision = {
        "decision": "direct_answer",
        "action": "direct_answer",
        "reason": "能力说明",
        "router_source": "rule_based",
        "router_confidence": 1.0,
    }

    def fake_route(question):
        calls.append(question)
        return decision

    clock = iter((10.0, 10.125))
    monkeypatch.setattr(graph_nodes, "route_agent_action", fake_route)
    monkeypatch.setattr(
        graph_nodes,
        "time",
        SimpleNamespace(perf_counter=lambda: next(clock)),
    )
    result = _run(
        original="它能做什么？",
        effective="你能做什么？",
    )

    assert result["status"] == "OK"
    assert result["action"] == "direct_answer"
    assert result["router_llm_used"] is False
    assert result["llm_call_count"] == 0
    assert calls == ["你能做什么？"]
    decide_trace = next(
        step for step in result["tool_trace"] if step.get("step") == "decide"
    )
    assert decide_trace["route_cost_ms"] == 125
    assert decide_trace["tool_name"] == ""
    assert decide_trace["tool_args"] == {}
    assert decide_trace["router_llm_used"] is False


def test_framework_no_tool_answer_uses_effective_question(monkeypatch):
    model = FakeModel(AIMessage(content="框架最终回答"))
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)

    result = _run(
        original="它什么时候开始",
        effective="研究生奖学金申请什么时候开始",
    )

    assert result["status"] == "OK"
    assert result["answer"] == "框架最终回答"
    assert model.received_messages[0][0].content == "研究生奖学金申请什么时候开始"
    assert RESULT_FIELDS <= set(result)
    steps = [item.get("step") for item in result["tool_trace"]]
    assert "agent_iteration" in steps
    assert "compose_final_answer" in steps
    assert "llm_usage" in steps
    assert {"plan", "executor"}.isdisjoint(steps)


def test_framework_model_exception_returns_model_error_without_fallback(
    monkeypatch,
):
    model = FakeModel(RuntimeError("model unavailable"))
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)

    result = _run()

    assert result["status"] == "MODEL_ERROR"
    assert result["answer"] == "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
    assert result["reason"] == "agent_model_call_failed"
    serialized = json.dumps(result, ensure_ascii=False)
    assert "fallback_to_" + "legacy" not in serialized
    assert '"step": "plan"' not in serialized
    assert '"step": "executor"' not in serialized


def test_direct_answer_and_clarify_preserve_public_protocol(monkeypatch):
    monkeypatch.setattr(
        framework_agent,
        "get_agent_chat_model",
        lambda: (_ for _ in ()).throw(
            AssertionError("short circuit must not resolve model")
        ),
    )

    direct = _run(original="你能做什么？", effective="你能做什么？")
    clarify = _run(original="这个怎么配置", effective="这个怎么配置")

    assert RESULT_FIELDS <= set(direct)
    assert direct["status"] == "OK"
    assert direct["answer"]
    assert clarify["status"] == "NEED_CLARIFICATION"
    assert "具体指" in clarify["clarification_question"]


@pytest.mark.parametrize("malformed_output", [None, {}, {"result": None}])
def test_runtime_rejects_missing_or_invalid_graph_result(
    monkeypatch,
    malformed_output,
):
    class MalformedGraph:
        def invoke(self, graph_input, config, *, context):
            del graph_input, config, context
            return malformed_output

    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda: MalformedGraph(),
    )

    result = _run()

    assert result["status"] == "INTERNAL_ERROR"
    assert result["references"] == []
    assert result["tool_trace"][0]["step"] == "agent_langgraph_error"
    assert result["reason"] == "Agent 内部执行异常，请稍后重试"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "UNKNOWN"),
        ("references", ["invalid"]),
        ("retrieval_confidence", 1.5),
        ("llm_call_count", -1),
        ("timings", None),
    ],
)
def test_runtime_rejects_graph_result_that_violates_contract(
    monkeypatch,
    field,
    value,
):
    payload = create_agent_result(status=AgentResultStatus.OK)
    payload[field] = value

    class InvalidResultGraph:
        def invoke(self, graph_input, config, *, context):
            del graph_input, config, context
            return {"result": payload}

    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda: InvalidResultGraph(),
    )

    result = _run()

    assert result["status"] == "INTERNAL_ERROR"
    assert result["tool_trace"][0]["step"] == "agent_langgraph_error"


def test_runtime_rejects_graph_result_with_missing_required_field(monkeypatch):
    payload = create_agent_result(status=AgentResultStatus.OK)
    payload.pop("timings")

    class MissingFieldGraph:
        def invoke(self, graph_input, config, *, context):
            del graph_input, config, context
            return {"result": payload}

    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda: MissingFieldGraph(),
    )

    assert _run()["status"] == "INTERNAL_ERROR"


def test_runtime_signature_rejects_external_decision():
    """Runtime 门面不再公开可绕过图内 Router 的 decision 参数。"""
    assert "decision" not in inspect.signature(run_agent_langgraph).parameters


@pytest.mark.parametrize(
    ("top_k", "decompose_top_k"),
    [(0, 10), (11, 10), (5, 0), (5, 21)],
)
def test_runtime_rejects_out_of_range_counts_before_graph_invocation(
    monkeypatch,
    top_k,
    decompose_top_k,
):
    """删除 Pydantic 嵌套状态后，公开门面仍 fail closed。"""

    def forbidden_graph():
        raise AssertionError("invalid input must not compile or invoke graph")

    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        forbidden_graph,
    )

    result = run_agent_langgraph(
        write_bad_cases=lambda record: None,
        original_question="测试问题",
        effective_question="测试问题",
        top_k=top_k,
        decompose_top_k=decompose_top_k,
        trace_id="trace-invalid-counts",
    )

    assert result["status"] == "INTERNAL_ERROR"
    assert result["tool_trace"][0]["error"] == "internal_error"


def test_top_level_graph_error_is_sanitized_but_kept_for_diagnostics(
    monkeypatch,
    caplog,
):
    """顶层图异常对外脱敏，同时在诊断日志中保留原始信息。"""
    bad_cases: list[dict] = []
    sensitive_message = (
        "/home/test-user/project/checkpoints.sqlite: "
        "model request failed with api_key=sk-test-secret"
    )
    sensitive_error = RuntimeError(sensitive_message)

    class BrokenGraph:
        def invoke(self, graph_input, config, *, context):
            del graph_input, config, context
            raise sensitive_error

    monkeypatch.setattr(
        graph_runtime,
        "get_compiled_agent_graph",
        lambda settings=None: BrokenGraph(),
    )
    monkeypatch.setattr(graph_runtime.logger, "propagate", True)
    with caplog.at_level("ERROR", logger=graph_runtime.logger.name):
        result = run_agent_langgraph(
            write_bad_cases=bad_cases.append,
            original_question="它什么时候开始",
            effective_question="研究生奖学金申请什么时候开始",
            top_k=5,
            decompose_top_k=10,
            trace_id="trace-error",
            session_id="session-error",
        )

    response_text = json.dumps(result, ensure_ascii=False)
    assert result["status"] == "INTERNAL_ERROR"
    assert "/home/test-user/project" not in response_text
    assert "sk-test-secret" not in response_text
    assert len(bad_cases) == 1
    assert bad_cases[0]["note"] == sensitive_message
    exception_records = [
        record
        for record in caplog.records
        if "langgraph_graph_failed" in record.getMessage()
    ]
    assert exception_records
    assert exception_records[0].exc_info[1] is sensitive_error
