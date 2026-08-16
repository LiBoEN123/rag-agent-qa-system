"""无 Parent Graph 的 Agent Runtime 同步入口测试。"""

import inspect

import pytest
from langchain_core.messages import AIMessage

from app.agent import framework_agent
from app.agent import runtime as agent_runtime
from app.agent.runtime import prepare_agent_runtime_invocation, run_agent


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.received_messages = []

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def invoke(self, messages):
        self.received_messages.append(list(messages))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _run(*, original="测试问题", effective="测试问题"):
    return run_agent(
        write_bad_cases=lambda record: None,
        original_question=original,
        effective_question=effective,
        top_k=5,
        decompose_top_k=10,
        trace_id="trace-runtime",
        session_id="session-runtime",
    )


def test_runtime_routes_once_with_effective_question_and_records_cost(monkeypatch):
    model = FakeModel(AIMessage(content="回答"))
    questions = []
    original_route = agent_runtime.route_agent_action

    def recording_route(question):
        questions.append(question)
        return original_route(question)

    monkeypatch.setattr(agent_runtime, "route_agent_action", recording_route)
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)

    result = _run(original="它是什么？", effective="奖学金是什么？")

    assert questions == ["奖学金是什么？"]
    decide = next(step for step in result["tool_trace"] if step["step"] == "decide")
    assert decide["route_cost_ms"] >= 0
    assert result["action"] == "agent"


def test_framework_no_tool_answer_uses_effective_question(monkeypatch):
    model = FakeModel(AIMessage(content="基于有效问题回答"))
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)

    result = _run(original="它呢？", effective="奖学金申请条件是什么？")

    assert model.received_messages[0][0].content == "奖学金申请条件是什么？"
    assert result["status"] == "OK"
    assert result["answer"] == "基于有效问题回答"


def test_framework_model_exception_fails_closed(monkeypatch):
    monkeypatch.setattr(
        framework_agent,
        "get_agent_chat_model",
        lambda: FakeModel(RuntimeError("provider secret")),
    )

    result = _run()

    assert result["status"] == "MODEL_ERROR"
    assert "provider secret" not in result["answer"]
    assert result["reason"] == "agent_model_call_failed"


@pytest.mark.parametrize(
    ("question", "status"),
    [("你是谁？", "OK"), ("这个怎么配置", "NEED_CLARIFICATION")],
)
def test_direct_answer_and_clarify_preserve_public_protocol(question, status):
    result = _run(original=question, effective=question)
    assert result["status"] == status
    assert result["router_llm_used"] is False
    assert result["llm_call_count"] == 0


def test_runtime_signature_rejects_external_decision():
    assert "decision" not in inspect.signature(run_agent).parameters


@pytest.mark.parametrize(
    ("top_k", "decompose_top_k"),
    [(0, 10), (11, 10), (5, 0), (5, 21), (True, 10), (5.5, 10)],
)
def test_prepare_rejects_out_of_range_counts(top_k, decompose_top_k):
    with pytest.raises(ValueError):
        prepare_agent_runtime_invocation(
            original_question="测试",
            effective_question="测试",
            top_k=top_k,
            decompose_top_k=decompose_top_k,
            trace_id="trace-invalid",
            session_id=None,
        )


def test_top_level_runtime_error_is_sanitized_but_kept_for_diagnostics(
    monkeypatch,
):
    bad_cases = []
    monkeypatch.setattr(
        agent_runtime,
        "get_compiled_agent_runtime",
        lambda settings=None, **kwargs: (_ for _ in ()).throw(
            RuntimeError("private runtime detail")
        ),
    )

    result = run_agent(
        write_bad_cases=lambda record: bad_cases.append(dict(record)),
        original_question="测试问题",
        effective_question="测试问题",
        top_k=5,
        decompose_top_k=10,
        trace_id="trace-error",
    )

    assert result["status"] == "INTERNAL_ERROR"
    assert "private runtime detail" not in result["answer"]
    assert bad_cases[0]["note"] == "private runtime detail"
