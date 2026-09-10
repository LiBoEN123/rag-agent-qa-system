"""Framework Output 直接构造唯一 AgentRunResult 的公开行为合同。"""

from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.guardrails import AgentStopReason

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
        "decompose_llm_used": False,
        "answer_llm_used": True,
        "llm_call_count": 1,
        "timings": {"llm_cost_ms": 3},
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
    assert dumped["timings"] == {"llm_cost_ms": 3}


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "UNKNOWN"},
        {"unexpected": "field"},
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
    payload.pop("timings")

    with pytest.raises(ValidationError):
        AgentRunResult.model_validate(payload)


def test_internal_error_result_has_the_complete_contract_and_is_sanitized():
    result = build_internal_error_result(
        RuntimeError("api_key=secret"),
        trace_step="graph_error",
    )

    assert set(result) == set(AgentRunResult.model_fields)
    assert result["status"] == "INTERNAL_ERROR"
    assert result["timings"] == {}
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


def _knowledge_output(rag_result):
    knowledge = {
        "ok": True,
        "status": "success",
        "output": {"count": 2, "chunks": [], "rag_result": rag_result},
    }
    return {
        "governance_events": [
            {"kind": "model", "ok": True, "has_tool_calls": True},
            {"kind": "model", "ok": False},
            {
                "kind": "tool",
                "name": "search_knowledge_base",
                "ok": True,
                "counted_call": True,
                "handler_executed": True,
                "cost_ms": 5,
                "result": knowledge,
            },
        ],
        "knowledge_base_result": knowledge,
        # 已完成的 RAG Result 优先于框架停止原因与模型消息。
        "stop_reason": "max_iterations_reached",
        "messages": [AIMessage(content="不应覆盖 RAG 回答")],
    }


def _validate_usage(result, *, decompose=False, answer=False, count=0):
    assert AgentRunResult.model_validate(result).model_dump(mode="json") == result
    assert result["router_llm_used"] is False
    assert result["decompose_llm_used"] is decompose
    assert result["answer_llm_used"] is answer
    assert result["llm_call_count"] == count
    assert [step for step in result["tool_trace"] if step["step"] == "llm_usage"] == [
        {
            "step": "llm_usage",
            "router_llm_used": False,
            "decompose_llm_used": decompose,
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
        "也可以列出知识库文档、查看指定 chunk、对比文档内容，以及进行简单四则运算。"
    )
    assert result["reason"] == "能力介绍"
    assert result["action"] == "direct_answer"
    assert result["references"] == []
    assert result["need_clarification"] is False
    assert result["clarification_question"] == ""
    assert result["retrieval_confidence"] == result["answer_confidence"] == 0.0
    assert result["timings"] == {}
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


@pytest.mark.parametrize(
    ("rag_status", "result_status", "trace_status"),
    [
        ("OK", "OK", "ok"),
        ("NEED_CLARIFICATION", "NEED_CLARIFICATION", "clarify"),
        ("NO_RELEVANT_INFO", "NO_RELEVANT_INFO", "no_evidence"),
        ("MODEL_ERROR", "MODEL_ERROR", "model_error"),
        ("MODEL_BAD_OUTPUT", "MODEL_ERROR", "model_error"),
        ("unexpected", "INTERNAL_ERROR", "error"),
        ("ok", "INTERNAL_ERROR", "error"),
    ],
)
def test_rag_result_maps_directly_without_losing_fields_or_mutating_input(
    rag_status,
    result_status,
    trace_status,
    settings_env,
):
    settings = settings_env()
    rag_result = {
        "status": rag_status,
        "answer": "RAG 回答或澄清问题",
        "references": [{"reference_id": "Reference 1", "chunk_id": "chunk-1"}],
        "reason": "RAG 判断原因",
        "decompose_llm_used": True,
        "answer_llm_used": True,
        "llm_call_count": 2,
        "retrieval_confidence": 0.9,
        "answer_confidence": 0.8,
        "timings": {"llm_cost_ms": 3, "decompose_cost_ms": 4},
    }
    output = _knowledge_output(rag_result)
    invocation = _invocation()
    original_output, original_invocation = deepcopy(output), deepcopy(invocation)

    result = build_framework_agent_result(invocation, output)

    assert not settings.sqlite_db_path.exists()
    assert output == original_output and invocation == original_invocation
    assert result["status"] == result_status
    for field in (
        "answer",
        "references",
        "reason",
        "retrieval_confidence",
        "answer_confidence",
        "timings",
    ):
        assert result[field] == rag_result[field]
    assert result["need_clarification"] is (result_status == "NEED_CLARIFICATION")
    assert result["clarification_question"] == (
        rag_result["answer"] if result_status == "NEED_CLARIFICATION" else ""
    )
    assert result["tool_result_summary"] == (
        "未调用工具，问题需要澄清"
        if result_status == "NEED_CLARIFICATION"
        else "知识库检索完成，召回 2 个 chunk"
    )
    assert [step["step"] for step in result["tool_trace"]] == [
        "decide",
        "agent_iteration",
        "agent_iteration",
        "tool_call",
        "compose_final_answer",
        "llm_usage",
    ]
    assert result["tool_trace"][-2] == {
        "step": "compose_final_answer",
        "mode": "rag",
        "status": trace_status,
        "answer_len": len(rag_result["answer"]),
        "reference_count": 1,
        "reason": rag_result["reason"],
        "retrieval_confidence": 0.9,
        "answer_confidence": 0.8,
        "timings": rag_result["timings"],
        "rag_llm_call_count": 2,
    }
    _validate_usage(result, decompose=True, answer=True, count=3)


@pytest.mark.parametrize(
    ("model_calls", "rag_calls", "decompose", "answer", "expected"),
    [
        (1, 0, False, False, 1),
        (2, 3, True, True, 5),
        (0, 0, True, True, 2),
        (1, -1, False, False, 1),
        (0, 1, False, True, 1),
    ],
)
def test_rag_llm_usage_counts_successful_models_and_rag_once(
    model_calls,
    rag_calls,
    decompose,
    answer,
    expected,
):
    output = _knowledge_output(
        {
            "status": "NO_RELEVANT_INFO",
            "answer": "无可用证据",
            "decompose_llm_used": decompose,
            "answer_llm_used": answer,
            "llm_call_count": rag_calls,
        }
    )
    output["governance_events"] = [
        *[{"kind": "model", "ok": True} for _ in range(model_calls)],
        {"kind": "model", "ok": False},
    ]

    result = build_framework_agent_result(_invocation(), output)

    assert result["answer"] == "无可用证据"
    assert result["references"] == []
    _validate_usage(result, decompose=decompose, answer=answer, count=expected)


@pytest.mark.parametrize(
    "references", [None, "invalid", ["invalid", {"chunk_id": "1"}]]
)
def test_rag_reference_filter_and_timing_defaults_remain_compatible(references):
    output = _knowledge_output(
        {"status": "OK", "references": references, "timings": []}
    )

    result = build_framework_agent_result(_invocation(), output)

    assert result["references"] == (
        [{"chunk_id": "1"}] if isinstance(references, list) else []
    )
    assert result["timings"] == {}
    assert result["retrieval_confidence"] == result["answer_confidence"] == 0.0
    _validate_usage(result, count=1)


@pytest.mark.parametrize("stop_reason", ["", "completed"])
def test_framework_final_ai_content_and_governance_trace_are_preserved(
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
                    {"name": "simple_calculator", "args": {}, "id": "call-1"},
                ],
            ),
            ToolMessage(content="工具输出", tool_call_id="call-1"),
        ],
        "stop_reason": stop_reason,
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
    assert result["reason"] == "Agent model produced final answer"
    assert result["references"] == [] and result["timings"] == {}
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
        "reason": "Agent model produced final answer",
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
    assert result["reason"] == "Agent model returned empty final answer"
    assert result["tool_trace"][-2]["mode"] == "error"
    _validate_usage(result)


@pytest.mark.parametrize(
    "stop_reason",
    [
        "fatal_tool_error",
        "approval_required",
        "missing_idempotency_key",
        "runtime_unavailable",
    ],
)
def test_tool_stop_reasons_take_priority_over_final_ai_answer(stop_reason):
    result = build_framework_agent_result(
        _invocation(),
        {
            "stop_reason": stop_reason,
            "messages": [AIMessage(content="忽略此回答")],
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
    output = _knowledge_output({"status": "OK", "answer": "知识库回答"})
    output["knowledge_base_result"]["output"].update(
        count=2, chunks=[{"chunk_id": "one-visible-hit"}],
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
                "kind": "tool", "name": "simple_calculator", "counted_call": True,
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
