"""Framework Output 直接构造唯一 AgentRunResult 的公开行为合同。"""

from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.guardrails import AgentStopReason
from app.agent.final_response import FinalAgentResponse
from app.tools.knowledge_base import chunk_to_evidence_document
from app.tools.result import tool_success
from app.api.schemas import ChatReferences
from app.tools.evidence import build_evidence_document

from app.agent.results import (
    AgentRunResult,
    build_framework_agent_result,
    build_short_circuit_result,
    create_agent_result,
    build_internal_error_result,
)
from app.contracts.chat import AgentResultStatus


def _valid_payload(**updates):
    payload = {
        "status": "OK",
        "answer": "测试回答",
        "references": [{"reference_id": "Reference 1"}],
        "retrieval_confidence": 0.8,
        "answer_confidence": 0.7,
        "need_clarification": False,
        "clarification_question": "",
        "reason": "测试通过",
        "action": "agent",
        "tool_result_summary": "工具摘要",
        "tool_trace": [{"step": "llm_usage"}],
        "router_llm_used": False,
        "answer_llm_used": True,
        "llm_call_count": 1,
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("status", "need_clarification"),
    [
        (AgentResultStatus.OK, False),
        (AgentResultStatus.NEED_CLARIFICATION, True),
        (AgentResultStatus.INTERNAL_ERROR, False),
    ],
)
def test_agent_run_result_accepts_complete_results(
    status,
    need_clarification,
):
    payload = _valid_payload(
        status=status.value,
        need_clarification=need_clarification,
        clarification_question="请补充信息" if need_clarification else "",
    )

    result = AgentRunResult.model_validate(payload)
    dumped = result.model_dump(mode="json")

    assert dumped["status"] == status.value
    assert isinstance(dumped["status"], str)
    assert "timings" not in dumped


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "UNKNOWN"},
        {"unexpected": "field"},
        {"decompose_llm_used": False},
        {"timings": {}},
        {"retrieval_confidence": -0.1},
        {"answer_confidence": 1.1},
        {"llm_call_count": -1},
        {"references": ["not-a-dict"]},
        {"tool_trace": ["not-a-dict"]},
        {"router_llm_used": True},
    ],
)
def test_agent_run_result_rejects_invalid_payload(updates):
    with pytest.raises(ValidationError):
        AgentRunResult.model_validate(_valid_payload(**updates))


def test_agent_run_result_requires_every_runtime_field():
    payload = _valid_payload()
    payload.pop("llm_call_count")

    with pytest.raises(ValidationError):
        AgentRunResult.model_validate(payload)


def test_internal_error_result_has_the_complete_contract_and_is_sanitized():
    result = build_internal_error_result(
        RuntimeError("api_key=secret"),
        trace_step="graph_error",
    )

    assert set(result) == set(AgentRunResult.model_fields)
    assert result["status"] == "INTERNAL_ERROR"
    assert "timings" not in result
    assert "secret" not in str(result)
    assert AgentRunResult.model_validate(result).model_dump(mode="json") == result


def _invocation(action="agent", **decision):
    return SimpleNamespace(
        decision={"action": action, **decision},
        route_cost_ms=7,
        original_question="原问题",
        effective_question="改写后的问题",
        trace_id="trace-result",
        session_id="session-result",
    )


def _knowledge_output():
    documents = [chunk_to_evidence_document({
        "chunk_id": key, "text": text, "file_name": "policy.md",
        "source_path": "policy.md", "position": position, "hybrid_score": score,
    }).model_dump(mode="json") for key, text, position, score in (
        ("a", "候选一", 1, 0.9), ("b", "候选二", 2, 0.8),
    )]
    knowledge = tool_success("search_knowledge_base", {}, {
        "query": "测试", "count": 2, "evidence": documents,
    })
    return {
        "governance_events": [
            {"kind": "model", "ok": True, "has_tool_calls": True},
            {
                "kind": "tool", "name": "search_knowledge_base", "ok": True,
                "counted_call": True, "handler_executed": True,
                "cost_ms": 5, "result": knowledge,
            },
            {"kind": "model", "ok": True, "has_tool_calls": False},
        ],
        "stop_reason": "completed",
        "messages": [
            ToolMessage(content=json.dumps(knowledge), name="search_knowledge_base", tool_call_id="kb"),
            AIMessage(content="普通模型文本不是最终回答"),
        ],
        "structured_response": FinalAgentResponse(
            scope_match="matched", support_level="full",
            outcome="answer", answer="最终模型回答", evidence_ids=[documents[1]["id"]],
        ),
    }


def _validate_usage(result, *, answer=False, count=0):
    assert AgentRunResult.model_validate(result).model_dump(mode="json") == result
    assert result["router_llm_used"] is False
    assert result["answer_llm_used"] is answer
    assert result["llm_call_count"] == count
    assert [step for step in result["tool_trace"] if step["step"] == "llm_usage"] == [
        {
            "step": "llm_usage",
            "router_llm_used": False,
            "answer_llm_used": answer,
            "llm_call_count": count,
        }
    ]
    assert result["tool_trace"][-1]["step"] == "llm_usage"


def test_create_agent_result_preserves_all_final_fields_without_mutation():
    payload = _valid_payload()
    original = deepcopy(payload)
    arguments = {
        key: value for key, value in payload.items() if key != "router_llm_used"
    }

    result = create_agent_result(**arguments)

    assert result == original == payload
    assert AgentRunResult.model_validate(result).model_dump(mode="json") == result


@pytest.mark.parametrize(
    ("clarification", "reason", "answer"),
    [
        (" 请说明具体对象。 ", "指代不明", "请说明具体对象。"),
        (" ", " 请补充文档名称 ", "请补充文档名称"),
        (None, "", "问题需要进一步澄清"),
    ],
)
def test_short_circuit_clarify_preserves_contract_and_input(
    clarification, reason, answer
):
    invocation = _invocation(
        "clarify", clarification_question=clarification, reason=reason
    )
    original = deepcopy(invocation)

    result = build_short_circuit_result(invocation)

    assert invocation == original
    assert result["status"] == "NEED_CLARIFICATION"
    assert (
        result["answer"]
        == result["clarification_question"]
        == result["reason"]
        == answer
    )
    assert result["need_clarification"] is True
    assert result["action"] == "clarify"
    assert result["references"] == []
    assert result["tool_result_summary"] == "未调用工具，问题需要澄清"
    assert [step["step"] for step in result["tool_trace"]] == [
        "decide",
        "clarify",
        "compose_final_answer",
        "llm_usage",
    ]
    assert result["tool_trace"][-2] == {
        "step": "compose_final_answer",
        "mode": "clarify",
        "status": "clarify",
        "answer_len": len(answer),
        "reason": answer,
    }
    _validate_usage(result)


def test_short_circuit_direct_answer_keeps_capability_text_and_trace(settings_env):
    settings = settings_env()
    result = build_short_circuit_result(_invocation("direct_answer", reason="能力介绍"))

    assert not settings.sqlite_db_path.exists()
    assert result["status"] == "OK"
    assert result["answer"] == (
        "我是一个 RAG 知识库问答助手。"
        "我可以基于已导入的文档回答问题并提供引用，"
        "也可以列出知识库文档、查看指定 chunk、对比文档内容。"
    )
    assert result["reason"] == "能力介绍"
    assert result["action"] == "direct_answer"
    assert result["references"] == []
    assert result["need_clarification"] is False
    assert result["clarification_question"] == ""
    assert result["retrieval_confidence"] == result["answer_confidence"] == 0.0
    assert "timings" not in result
    assert result["tool_result_summary"] == "未调用工具，直接回答"
    assert [step["step"] for step in result["tool_trace"]] == [
        "decide",
        "direct_answer",
        "compose_final_answer",
        "llm_usage",
    ]
    assert result["tool_trace"][0]["route_cost_ms"] == 7
    assert result["tool_trace"][-2]["mode"] == "direct"
    assert result["tool_trace"][-2]["status"] == "direct_answer"
    _validate_usage(result)


@pytest.mark.parametrize("action", ["unexpected", "agent", "", None])
def test_unsupported_short_circuit_action_fails_closed(action):
    result = build_short_circuit_result(_invocation(action))

    assert result["status"] == "INTERNAL_ERROR"
    assert result["answer"] == "Agent 内部执行异常，请稍后重试"
    assert result["reason"] == f"Unsupported route action: {action or '<missing>'}"
    assert result["tool_result_summary"] == result["reason"]
    assert result["tool_trace"][-2]["mode"] == "error"
    assert result["tool_trace"][-2]["status"] == "error"
    _validate_usage(result)


def test_kb_evidence_projects_only_declared_structured_reference(settings_env):
    settings = settings_env()
    output = _knowledge_output()
    original = deepcopy(output)

    result = build_framework_agent_result(_invocation(), output)

    assert output == original
    assert not settings.sqlite_db_path.exists()
    assert result["status"] == "OK"
    assert result["answer"] == "最终模型回答"
    reference, = result["references"]
    assert reference["reference_id"] == output["structured_response"].evidence_ids[0]
    assert reference["chunk_id"] == "b"
    assert reference["snippet"] == "候选二"
    ChatReferences.model_validate(reference)
    assert result["retrieval_confidence"] == result["answer_confidence"] == 0.0
    assert "timings" not in result
    assert result["tool_result_summary"] == "知识库检索完成，召回 2 个 chunk"
    assert [step["step"] for step in result["tool_trace"]] == [
        "decide", "agent_iteration", "tool_call", "agent_iteration",
        "compose_final_answer", "llm_usage",
    ]
    assert result["tool_trace"][-2]["mode"] == "agent_loop"
    _validate_usage(result, answer=True, count=2)


@pytest.mark.parametrize("model_calls", [0, 1, 3])
def test_kb_tool_does_not_add_llm_calls(model_calls):
    output = _knowledge_output()
    tool_event = output["governance_events"][1]
    output["governance_events"] = [
        *[{"kind": "model", "ok": True} for _ in range(model_calls)],
        tool_event,
        {"kind": "model", "ok": False},
    ]

    result = build_framework_agent_result(_invocation(), output)

    _validate_usage(result, answer=model_calls > 0, count=model_calls + 1)


@pytest.mark.parametrize("stop_reason", ["max_iterations_reached", "fatal_tool_error"])
def test_successful_kb_evidence_cannot_override_terminal_guardrail(stop_reason):
    output = _knowledge_output()
    output["stop_reason"] = stop_reason

    result = build_framework_agent_result(_invocation(), output)

    assert result["status"] != "OK"
    assert result["answer"] != "最终模型回答"
    assert result["reason"] == stop_reason
    assert result["references"] == []


@pytest.mark.parametrize("stop_reason", ["", "completed"])
def test_framework_structured_answer_and_governance_trace_are_preserved(
    stop_reason, settings_env,
):
    settings = settings_env()
    output = {
        "messages": [
            HumanMessage(content="问题"),
            AIMessage(content="旧回答"),
            AIMessage(
                content=[
                    " 第一段 ",
                    {"type": "text", "text": " 第二段 "},
                    {"type": "image_url", "image_url": {"url": "unused"}},
                ]
            ),
            AIMessage(
                content="工具请求",
                tool_calls=[
                    {"name": "find_documents", "args": {}, "id": "call-1"},
                ],
            ),
            ToolMessage(content="工具输出", tool_call_id="call-1"),
        ],
        "stop_reason": stop_reason,
        "structured_response": FinalAgentResponse(scope_match="matched", support_level="full", outcome="answer", answer="第一段\n第二段"),
        "governance_events": [
            {"kind": "model", "ok": True},
            {
                "kind": "tool",
                "name": "custom_tool",
                "ok": True,
                "counted_call": True,
                "handler_executed": True,
                "result": {"ok": True, "summary": "工具摘要"},
            },
        ],
    }
    original = deepcopy(output)

    result = build_framework_agent_result(_invocation(), output)

    assert not settings.sqlite_db_path.exists()
    assert output == original
    assert result["status"] == "OK"
    assert result["answer"] == "第一段\n第二段"
    assert result["reason"] == "Agent model produced structured answer"
    assert result["references"] == []
    assert result["tool_result_summary"] == "工具摘要"
    tool_calls = [step for step in result["tool_trace"] if step["step"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "custom_tool"
    assert tool_calls[0]["summary"] == "工具摘要"
    assert tool_calls[0]["step_id"] == 1
    assert result["tool_trace"][-2] == {
        "step": "compose_final_answer",
        "mode": "agent_loop",
        "status": "ok",
        "answer_len": 7,
        "reason": "Agent model produced structured answer",
    }
    _validate_usage(result, answer=True, count=1)


@pytest.mark.parametrize(
    "messages",
    [
        None,
        (),
        [],
        [AIMessage(content=" ")],
        [AIMessage(content="有旧回答"), AIMessage(content="")],
        [
            AIMessage(
                content="不是最终回答",
                tool_calls=[{"name": "tool", "args": {}, "id": "c"}],
            )
        ],
    ],
)
def test_framework_missing_or_empty_final_answer_fails_closed(messages):
    result = build_framework_agent_result(_invocation(), {"messages": messages})

    assert result["status"] == "MODEL_ERROR"
    assert result["answer"] == "模型未生成可用的最终回答。"
    assert result["reason"] == "Agent structured response is missing or invalid"
    assert result["tool_trace"][-2]["mode"] == "error"
    _validate_usage(result)


@pytest.mark.parametrize(
    "stop_reason",
    [
        "fatal_tool_error",
        "approval_required",

        "runtime_unavailable",
    ],
)
def test_tool_stop_reasons_take_priority_over_final_ai_answer(stop_reason):
    result = build_framework_agent_result(
        _invocation(),
        {
            "stop_reason": stop_reason,
            "messages": [AIMessage(content="忽略此回答")],
            "structured_response": FinalAgentResponse(scope_match="matched", support_level="full", outcome="answer", answer="忽略结构化回答"),
        },
    )

    assert result["status"] == "TOOL_ERROR"
    assert result["answer"] == "工具调用被治理策略或运行时终止，无法继续本次请求。"
    assert result["reason"] == stop_reason
    assert result["tool_trace"][-2]["status"] == "tool_error"
    _validate_usage(result)


@pytest.mark.parametrize(
    "stop_reason",
    [
        AgentStopReason.MAX_ITERATIONS.value,
        AgentStopReason.MAX_TOOL_CALLS.value,
        AgentStopReason.MAX_TOOL_ERRORS.value,
        AgentStopReason.MAX_TOTAL_COST.value,
        AgentStopReason.REPEATED_TOOL_CALL.value,
        AgentStopReason.REPEATED_TOOL_ERROR.value,
        "agent_model_call_failed",
        "unknown_stop",
    ],
)
def test_model_stop_reasons_keep_stable_fail_closed_answers(stop_reason):
    result = build_framework_agent_result(_invocation(), {"stop_reason": stop_reason})

    assert result["status"] == "MODEL_ERROR"
    assert result["reason"] == stop_reason
    assert result["answer"] == (
        "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
        if stop_reason == "agent_model_call_failed"
        else "本次请求达到 Agent 执行上限，已安全停止。请缩小问题范围后重试。"
    )
    assert result["tool_trace"][-2]["mode"] == "error"
    _validate_usage(result)


@pytest.mark.parametrize("summaries", [["普通工具摘要"], ["一", "二", "三", "四", "五"]])
def test_tool_result_summary_uses_last_three_public_tool_summaries(summaries):
    events = []
    for summary in summaries:
        events.extend([
            {"kind": "model", "ok": True},
            {
                "kind": "tool", "name": "custom_tool", "counted_call": True,
                "handler_executed": True,
                "result": {"ok": True, "summary": summary},
            },
            {
                "kind": "tool", "name": "custom_tool", "counted_call": False,
                "result": {"ok": False, "summary": "拦截不进入摘要", "error": "blocked"},
            },
        ])
    result = build_framework_agent_result(
        _invocation(), {"governance_events": events, "messages": [AIMessage(content="完成")]},
    )
    tools = [step for step in result["tool_trace"] if "step_id" in step]
    assert [step["step_id"] for step in tools] == list(range(1, len(summaries) + 1))
    assert [step["summary"] for step in tools] == summaries
    assert result["tool_result_summary"] == "；".join(summaries[-3:])


def test_knowledge_summary_uses_public_final_chunk_count():
    output = _knowledge_output()
    output["governance_events"][1]["result"]["output"].update(
        count=2, evidence=[{"metadata": {"chunk_id": "one-visible-hit"}}],
    )
    result = build_framework_agent_result(_invocation(), output)
    tool_trace = next(step for step in result["tool_trace"] if step["step"] == "tool_call")
    assert len(tool_trace["hits"]) == 1
    assert tool_trace["final_chunk_count"] == 2
    assert result["tool_result_summary"] == "知识库检索完成，召回 2 个 chunk"


@pytest.mark.parametrize("error_code", ["approval_required", "permission_denied", "agent_batch_aborted"])
def test_guardrail_only_summary_falls_back_to_result_reason(error_code):
    result = build_framework_agent_result(
        _invocation(), {
            "stop_reason": "approval_required",
            "governance_events": [
                {"kind": "model", "ok": True},
                {
                    "kind": "tool", "name": "custom_tool", "counted_call": False,
                    "handler_executed": False, "error_code": error_code,
                    "stop_reason": "approval_required",
                    "result": {"ok": False, "error": "blocked", "summary": "不得使用"},
                },
            ],
        },
    )
    assert result["tool_result_summary"] == result["reason"] == "approval_required"
    trace = next(step for step in result["tool_trace"] if step["step"] == "tool_call")
    assert "step_id" not in trace
    assert trace["handler_executed"] is False
    assert trace["will_return_to_agent_model"] is False


@pytest.mark.parametrize("error_code", ["invalid_arguments", "approval_required", "permission_denied"])
def test_counted_pre_handler_failure_keeps_failure_summary(error_code):
    result = build_framework_agent_result(
        _invocation(), {
            "stop_reason": "fatal_tool_error",
            "governance_events": [{
                "kind": "tool", "name": "find_documents", "counted_call": True,
                "handler_executed": False,
                "result": {"ok": False, "error_code": error_code},
            }],
        },
    )
    trace = next(step for step in result["tool_trace"] if step["step"] == "tool_call")
    assert trace["step_id"] == 1
    assert trace["handler_executed"] is False
    assert trace["success"] is False
    assert result["tool_result_summary"] == trace["summary"] == "工具执行失败"


def test_tool_result_summary_reuses_redacted_public_summary():
    result = build_framework_agent_result(
        _invocation(), {
            "messages": [AIMessage(content="完成")],
            "governance_events": [{
                "kind": "tool", "name": "custom_tool", "counted_call": True,
                "handler_executed": True,
                "result": {"ok": True, "summary": "完成 token=raw-token password=raw-password"},
            }],
        },
    )
    trace = next(step for step in result["tool_trace"] if step["step"] == "tool_call")
    assert result["tool_result_summary"] == trace["summary"] == (
        "完成 token=***REDACTED*** password=***REDACTED***"
    )
    assert "raw-token" not in json.dumps(result)
    assert "raw-password" not in json.dumps(result)


@pytest.mark.parametrize("fields", [
    {"outcome": "answer", "answer": " "},
    {"outcome": "answer", "answer": "正文", "clarification_question": "问题"},
    {"outcome": "answer", "answer": "正文", "evidence_ids": ["a", "a"]},
    {"outcome": "no_evidence", "evidence_ids": ["a"]},
    {"outcome": "no_evidence", "clarification_question": "问题"},
    {"outcome": "clarify", "clarification_question": " "},
    {"outcome": "clarify", "clarification_question": "问题", "evidence_ids": ["a"]},
    {"outcome": "tool_error"}, {"outcome": "model_error"}, {"outcome": "internal_error"},
])
def test_final_response_schema_rejects_invalid_static_contract(fields):
    with pytest.raises(ValidationError):
        FinalAgentResponse(
            scope_match="matched",
            support_level="full" if fields["outcome"] == "answer" else "none",
            **fields,
        )


@pytest.mark.parametrize("support", ["none", "partial", "full"])
@pytest.mark.parametrize("scope", ["matched", "mixed", "mismatched"])
@pytest.mark.parametrize("outcome", ["answer", "no_evidence", "clarify"])
def test_final_response_evidence_decision_consistency(support, scope, outcome):
    fields = dict(
        support_level=support, scope_match=scope, outcome=outcome,
        evidence_ids=["document-a"] if outcome == "answer" else [],
        clarification_question="请确认范围" if outcome == "clarify" else "",
    )
    valid = (
        support in {"partial", "full"} and scope in {"matched", "mixed"}
        if outcome == "answer" else support == "none" and scope != "mixed"
    )
    if valid:
        response = FinalAgentResponse(**fields)
        assert response.model_dump(include=set(fields)) == fields
    else:
        with pytest.raises(ValidationError):
            FinalAgentResponse(**fields)


@pytest.mark.parametrize("field", ["scope_match", "support_level"])
def test_final_response_requires_explicit_scope_and_support(field):
    fields = dict(scope_match="matched", support_level="partial", outcome="answer", evidence_ids=["a"])
    del fields[field]
    with pytest.raises(ValidationError, match=field):
        FinalAgentResponse(**fields)


def test_supported_document_answer_requires_evidence_ids():
    with pytest.raises(ValidationError, match="answer requires evidence_ids"):
        FinalAgentResponse(scope_match="matched", support_level="partial", outcome="answer")


@pytest.mark.parametrize("evidence_ids", [[], ["knowledge_base:fake"]])
def test_evidence_answer_requires_real_cited_ids(evidence_ids):
    output = _knowledge_output()
    output["structured_response"] = FinalAgentResponse(
        scope_match="matched", support_level="full",
        outcome="answer", answer="UNSUPPORTED PRIVATE ANSWER", evidence_ids=evidence_ids,
    )
    result = build_framework_agent_result(_invocation(), output)
    assert result["status"] == "MODEL_ERROR"
    assert result["references"] == []
    assert "UNSUPPORTED PRIVATE ANSWER" not in json.dumps(result)
    assert "候选" not in result["reason"]


def test_grounding_mismatch_diagnostics_only_expose_ids():
    output = _knowledge_output()
    payload = json.loads(output["messages"][0].content)
    ids = ["knowledge_base:real-a", "knowledge_base:real-b"]
    for document, evidence_id in zip(payload["output"]["evidence"], ids):
        document["id"] = evidence_id
        document["page_content"] = "PRIVATE SECRET BODY"
        document["metadata"]["private_metadata"] = "PRIVATE METADATA SECRET"
    output["messages"][0].content = json.dumps(payload)
    output["governance_events"][1]["result"] = payload
    output["structured_response"] = FinalAgentResponse(
        scope_match="matched", support_level="full",
        outcome="answer", answer="UNSUPPORTED PRIVATE ANSWER",
        evidence_ids=[ids[0], "knowledge_base:fake"],
    )

    result = build_framework_agent_result(_invocation(), output)

    assert result["status"] == "MODEL_ERROR"
    assert result["references"] == []
    compose = next(step for step in result["tool_trace"] if step["step"] == "compose_final_answer")
    assert compose["mode"] == "error" and compose["status"] == "model_error"
    assert compose["requested_evidence_ids"] == [ids[0], "knowledge_base:fake"]
    assert compose["available_evidence_ids"] == ids
    assert compose["unavailable_evidence_ids"] == ["knowledge_base:fake"]
    assert result["reason"] == compose["reason"] == "Agent structured response referenced unavailable evidence"
    serialized = json.dumps(result)
    for private in ("PRIVATE SECRET BODY", "PRIVATE METADATA SECRET", "private_metadata", "UNSUPPORTED PRIVATE ANSWER"):
        assert private not in serialized
    for key in ("reason", "answer", "tool_result_summary"):
        assert "knowledge_base:fake" not in result[key]


@pytest.mark.parametrize("outcome", ["answer", "no_evidence", "clarify"])
def test_successful_results_do_not_include_grounding_diagnostics(outcome):
    output = _knowledge_output()
    if outcome != "answer":
        output["structured_response"] = FinalAgentResponse(
            scope_match="matched", support_level="full" if outcome == "answer" else "none",
            outcome=outcome, clarification_question="请补充年份" if outcome == "clarify" else "",
        )
    result = build_framework_agent_result(_invocation(), output)
    compose = next(step for step in result["tool_trace"] if step["step"] == "compose_final_answer")
    assert not {"requested_evidence_ids", "available_evidence_ids", "unavailable_evidence_ids"} & compose.keys()


@pytest.mark.parametrize("value", [None, {}, {"outcome": "answer", "answer": "unparsed"},
                                       FinalAgentResponse.model_construct(outcome="answer", answer="")])
def test_unparsed_or_invalid_structured_response_never_falls_back_to_ai(value):
    result = build_framework_agent_result(_invocation(), {
        "structured_response": value, "messages": [AIMessage(content="不得回退的普通答案")],
    })
    assert result["status"] == "MODEL_ERROR"
    assert "不得回退" not in result["answer"]


@pytest.mark.parametrize("answer", ["", "已有资料不足以确定。"])
def test_no_evidence_uses_existing_status_and_fallback(answer):
    output = _knowledge_output()
    output["structured_response"] = FinalAgentResponse(scope_match="matched", support_level="none", outcome="no_evidence", answer=answer)
    result = build_framework_agent_result(_invocation(), output)
    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["answer"] == (answer or "我无法根据当前取得的证据确定答案。")
    assert result["references"] == []
    assert result["need_clarification"] is False
    assert result["clarification_question"] == ""


def test_agent_clarification_preserves_real_tool_summary():
    output = _knowledge_output()
    output["structured_response"] = FinalAgentResponse(
        scope_match="matched", support_level="none",
        outcome="clarify", clarification_question="请确认申请年份。",
    )
    result = build_framework_agent_result(_invocation(), output)
    assert result["status"] == "NEED_CLARIFICATION"
    assert result["need_clarification"] is True
    assert result["clarification_question"] == "请确认申请年份。"
    assert result["references"] == []
    assert result["tool_result_summary"] == "知识库检索完成，召回 2 个 chunk"


@pytest.mark.parametrize("name", ["web_search", "find_documents", "mcp_read"])
def test_non_evidence_tool_outputs_cannot_be_promoted_to_references(name):
    output = _knowledge_output()
    payload = json.loads(output["messages"][0].content)
    payload["tool_name"] = name
    output["messages"] = [ToolMessage(content=json.dumps(payload), name=name, tool_call_id="x")]
    result = build_framework_agent_result(_invocation(), output)
    assert result["status"] == "MODEL_ERROR"
    assert result["references"] == []
    output["structured_response"] = FinalAgentResponse(scope_match="matched", support_level="full", outcome="answer", answer="操作已完成")
    assert build_framework_agent_result(_invocation(), output)["status"] == "OK"


@pytest.mark.parametrize("damage", ["failed_message", "failed_result", "invalid_json", "invalid_document", "wrong_tool"])
def test_failed_or_invalid_evidence_cannot_ground_answer(damage):
    output = _knowledge_output()
    message = output["messages"][0]
    payload = (message.artifact or json.loads(message.content))
    if damage == "failed_message":
        message.status = "error"
    elif damage == "failed_result":
        payload["ok"] = False
    elif damage == "invalid_document":
        payload["output"]["evidence"][1]["metadata"].pop("content_is_untrusted")
    elif damage == "wrong_tool":
        payload["tool_name"] = "web_search"
    message.content = "not json" if damage == "invalid_json" else json.dumps(payload)
    result = build_framework_agent_result(_invocation(), output)
    assert result["status"] == "MODEL_ERROR"
    assert result["references"] == []


def test_web_pdf_references_keep_declared_order_and_existing_api_contract():
    documents = [build_evidence_document(
        page_content="正文 " * 200, source_type=source_type, source_uri=uri,
        identity=(uri,), metadata=metadata,
    ) for source_type, uri, metadata in [
        ("web_page", "https://example.com/page", {"title": "网页"}),
        ("web_pdf", "https://example.com/file.pdf", {"start_page": 3}),
    ]]
    messages = [ToolMessage(
        content=json.dumps(tool_success(name, {}, {"evidence": [doc.model_dump()]})),
        name=name, tool_call_id=name,
    ) for name, doc in zip(["web_fetch", "web_pdf_fetch"], documents)]
    output = {
        "messages": [*messages, messages[0]],
        "structured_response": FinalAgentResponse(
            scope_match="matched", support_level="full",
            outcome="answer", answer="参考网页及 PDF。",
            evidence_ids=[documents[1].id, documents[0].id],
        ),
    }
    result = build_framework_agent_result(_invocation(), output)
    assert result["status"] == "OK"
    pdf, page = [ChatReferences.model_validate(ref) for ref in result["references"]]
    assert pdf.reference_id == pdf.chunk_id == documents[1].id
    assert pdf.source_path == pdf.file_name == "https://example.com/file.pdf"
    assert pdf.position == 3
    assert page.reference_id == page.chunk_id == documents[0].id
    assert page.file_name == "网页" and page.position is None
    assert len(page.snippet) == len(pdf.snippet) == 300
    assert page.score == page.rerank_score == pdf.score == pdf.rerank_score == 0.0


@pytest.mark.parametrize("chunk_id", [0, None])
def test_get_chunk_context_reference_preserves_locator_and_safe_scores(chunk_id):
    document = chunk_to_evidence_document({
        "chunk_id": chunk_id, "text": "原文", "source_id": "source-1", "position": 3,
        "hybrid_score": "invalid", "rerank_score": "nan",
    })
    result = build_framework_agent_result(_invocation(), {
        "messages": [ToolMessage(
            name="get_chunk_context", tool_call_id="chunk",
            content=json.dumps(tool_success("get_chunk_context", {}, {"evidence": [document.model_dump()]})),
        )],
        "structured_response": FinalAgentResponse(
            scope_match="matched", support_level="full",
            outcome="answer", answer="原文回答", evidence_ids=[document.id],
        ),
    })
    assert result["status"] == "OK"
    reference, = [ChatReferences.model_validate(ref) for ref in result["references"]]
    assert reference.reference_id == document.id
    assert reference.chunk_id == (document.id if chunk_id is None else 0)
    assert reference.source_path == document.metadata["source_uri"]
    assert reference.position == 3
    assert reference.score == reference.rerank_score == 0.0


def test_evidence_decision_requires_composition_and_preserves_streamed_whitespace():
    output = _knowledge_output()
    output["structured_response"] = FinalAgentResponse(
        scope_match="matched", support_level="full",
        outcome="answer", evidence_ids=output["structured_response"].evidence_ids,
    )
    assert build_framework_agent_result(_invocation(), output)["status"] == "MODEL_ERROR"
    result = build_framework_agent_result(_invocation(), output, composed_answer=" 答案\n")
    assert result["status"] == "OK" and result["answer"] == " 答案\n"
    output["structured_response"].evidence_ids = ["invented"]
    result = build_framework_agent_result(_invocation(), output, composed_answer="PRIVATE ANSWER")
    assert result["status"] == "MODEL_ERROR" and "PRIVATE ANSWER" not in repr(result)


@pytest.mark.parametrize("answer", ["", " \n"])
def test_empty_composition_never_falls_back_to_structured_draft(answer):
    result = build_framework_agent_result(_invocation(), _knowledge_output(), composed_answer=answer)
    assert result["status"] == "MODEL_ERROR"
