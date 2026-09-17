"""Framework 输出到唯一 Agent Result Contract 的归一化边界。"""

from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agent.final_response import (
    FinalAgentResponse,
    collect_evidence_from_messages,
    document_to_reference,
)
from app.agent.guardrails import AgentStopReason
from app.agent.router import RouteDecision
from app.agent.tool_events import normalize_governance_events
from app.contracts.chat import AgentResultStatus


INTERNAL_ERROR_MESSAGE = "Agent 内部执行异常，请稍后重试"
INTERNAL_ERROR_MARKER = "internal_error"
DIRECT_ANSWER_TEXT = (
    "我是一个 RAG 知识库问答助手。"
    "我可以基于已导入的文档回答问题并提供引用，"
    "也可以列出知识库文档、查看指定 chunk、对比文档内容。"
)

# 仅供公开 trace 序列化，保留既有值；不参与 Result 状态决策。
_TRACE_STATUSES = {
    AgentResultStatus.OK: "ok",
    AgentResultStatus.NEED_CLARIFICATION: "clarify",
    AgentResultStatus.NO_RELEVANT_INFO: "no_evidence",
    AgentResultStatus.TOOL_ERROR: "tool_error",
    AgentResultStatus.MODEL_ERROR: "model_error",
    AgentResultStatus.INTERNAL_ERROR: "error",
}


class AgentRunResult(BaseModel):
    """一次 Agent Runtime 执行的唯一完整结果契约。"""

    model_config = ConfigDict(extra="forbid")

    status: AgentResultStatus
    answer: str
    references: list[dict[str, Any]]
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    answer_confidence: float = Field(ge=0.0, le=1.0)
    need_clarification: bool
    clarification_question: str
    reason: str
    action: str
    tool_result_summary: str
    tool_trace: list[dict[str, Any]]
    router_llm_used: Literal[False]
    answer_llm_used: bool
    llm_call_count: int = Field(ge=0)


class AgentResultInvocation(Protocol):
    """Result builder 所需的一次 Runtime Invocation 最小只读视图。"""

    decision: RouteDecision
    route_cost_ms: int


def summarize_tool_trace_for_result(
    *,
    reason: str,
    tool_trace: list[dict[str, Any]],
) -> str:
    """从唯一公开 trace 中提取计入工具调用的结果摘要。"""
    summaries = []
    for trace in tool_trace:
        # step_id 仅属于 counted call；handler_executed 会排除已计数的失败。
        if trace.get("step") != "tool_call" or "step_id" not in trace:
            continue
        if trace.get("tool_name") == "search_knowledge_base" and trace.get("success"):
            count = trace.get("final_chunk_count", 0)
            summaries.append(f"知识库检索完成，召回 {count} 个 chunk")
            continue
        summary = str(trace.get("summary") or "")
        if summary:
            summaries.append(summary)
    return "；".join(summaries[-3:]) if summaries else reason


def get_route_decision_for_trace(decision: Mapping[str, Any]) -> str:
    """根据路由 action 生成 trace 使用的 route_decision。"""
    action = str(decision.get("action") or "")
    return action if action else "unknown"


def build_decide_trace(
    decision: Mapping[str, Any],
    *,
    route_cost_ms: int = 0,
) -> dict[str, Any]:
    """将最小内部路由协议扩展为稳定的外部兼容 trace。"""
    return {
        "step": "decide",
        "route_decision": get_route_decision_for_trace(decision),
        "decision_type": decision.get("decision", ""),
        "action": decision.get("action", ""),
        "tool_name": "",
        "tool_args": {},
        "sanitized_tool_args": {},
        "reason": decision.get("reason", ""),
        "router_source": decision.get("router_source", ""),
        "router_confidence": decision.get("router_confidence"),
        "router_fallback_reason": "",
        "router_llm_used": False,
        "route_cost_ms": max(0, int(route_cost_ms or 0)),
    }


def upsert_llm_usage_trace(
    tool_trace: list[dict[str, Any]],
    *,
    answer_llm_used: bool,
    llm_call_count: int,
) -> list[dict[str, Any]]:
    """返回包含唯一最新 LLM usage 项的新工具 trace。"""
    usage_trace = {
        "step": "llm_usage",
        "router_llm_used": False,
        "answer_llm_used": bool(answer_llm_used),
        "llm_call_count": max(0, int(llm_call_count or 0)),
    }
    updated: list[dict[str, Any]] = []
    usage_written = False
    for step in tool_trace:
        if not isinstance(step, dict):
            continue
        if step.get("step") == "llm_usage":
            if not usage_written:
                updated.append(usage_trace)
                usage_written = True
            continue
        updated.append(dict(step))
    if not usage_written:
        updated.append(usage_trace)
    return updated


def _stop_result(reason: str) -> tuple[AgentResultStatus, str]:
    """把框架停止原因映射为 API 兼容状态。"""
    tool_errors = {
        AgentStopReason.FATAL_TOOL_ERROR.value,
        AgentStopReason.APPROVAL_REQUIRED.value,
        AgentStopReason.MISSING_IDEMPOTENCY_KEY.value,
        AgentStopReason.RUNTIME_UNAVAILABLE.value,
    }
    if reason in tool_errors:
        status = AgentResultStatus.TOOL_ERROR
        final_answer = "工具调用被治理策略或运行时终止，无法继续本次请求。"
    else:
        status = AgentResultStatus.MODEL_ERROR
        final_answer = "本次请求达到 Agent 执行上限，已安全停止。请缩小问题范围后重试。"
    if reason == "agent_model_call_failed":
        final_answer = "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
    return status, final_answer


def _finalize_agent_result(
    result: AgentRunResult,
) -> dict[str, Any]:
    """序列化并返回已校验的最终 Result。"""
    return result.model_dump(mode="json")


def build_short_circuit_result(
    invocation: AgentResultInvocation,
) -> dict[str, Any]:
    """完成 clarify/direct_answer 或未知 Router action 的确定性结果。"""
    tool_trace = [
        build_decide_trace(invocation.decision, route_cost_ms=invocation.route_cost_ms)
    ]
    action = str(invocation.decision.get("action") or "").strip()
    if action == "clarify":
        answer = reason = (
            str(invocation.decision.get("clarification_question") or "").strip()
            or str(invocation.decision.get("reason") or "").strip()
            or "问题需要进一步澄清"
        )
        status = AgentResultStatus.NEED_CLARIFICATION
        mode = "clarify"
        summary = "未调用工具，问题需要澄清"
    elif action == "direct_answer":
        status = AgentResultStatus.OK
        answer = DIRECT_ANSWER_TEXT
        reason = str(invocation.decision.get("reason") or "该问题可以直接回答")
        mode = "direct"
        summary = "未调用工具，直接回答"
    else:
        status = AgentResultStatus.INTERNAL_ERROR
        answer = INTERNAL_ERROR_MESSAGE
        reason = f"Unsupported route action: {action[:100] or '<missing>'}"
        mode = "error"
        summary = reason
    if action in {"clarify", "direct_answer"}:
        tool_trace.append(
            {
                "step": action,
                "step_id": 1,
                "status": "success",
                "reason": reason,
            }
        )
    tool_trace.append(
        {
            "step": "compose_final_answer",
            "mode": mode,
            "status": "direct_answer" if mode == "direct" else _TRACE_STATUSES[status],
            "answer_len": len(answer),
            "reason": reason,
        }
    )
    result = AgentRunResult(
        status=status,
        answer=answer,
        references=[],
        retrieval_confidence=0.0,
        answer_confidence=0.0,
        need_clarification=status is AgentResultStatus.NEED_CLARIFICATION,
        clarification_question=answer if action == "clarify" else "",
        reason=reason,
        action=str(invocation.decision.get("action") or ""),
        tool_result_summary=summary,
        tool_trace=upsert_llm_usage_trace(
            tool_trace,
            answer_llm_used=False,
            llm_call_count=0,
        ),
        router_llm_used=False,
        answer_llm_used=False,
        llm_call_count=0,
    )
    return _finalize_agent_result(result)


def build_framework_agent_result(
    invocation: AgentResultInvocation,
    framework_output: Mapping[str, Any],
) -> dict[str, Any]:
    """把 ``create_agent`` 最终状态一次性归一化为 AgentRunResult。"""
    raw_events = framework_output.get("governance_events")
    events = (
        [dict(item) for item in raw_events if isinstance(item, Mapping)]
        if isinstance(raw_events, list)
        else []
    )
    event_trace = normalize_governance_events(events)
    successful_models = sum(
        item.get("kind") == "model" and item.get("ok") for item in events
    )
    tool_trace = [
        build_decide_trace(invocation.decision, route_cost_ms=invocation.route_cost_ms),
        *event_trace,
    ]
    answer_llm_used = successful_models > 0
    references = []
    clarification_question = ""
    stop_reason = str(framework_output.get("stop_reason") or "")
    if stop_reason and stop_reason != AgentStopReason.COMPLETED.value:
        status, answer = _stop_result(stop_reason)
        reason = stop_reason
        mode = "error"
    else:
        final = framework_output.get("structured_response")
        status = AgentResultStatus.MODEL_ERROR
        answer = "模型未生成可用的最终回答。"
        reason = "Agent structured response is missing or invalid"
        mode = "error"
        try:
            # 只接受框架已解析的 Pydantic 对象；不解析模型文本或未验证的 dict。
            if not isinstance(final, FinalAgentResponse):
                raise ValueError(reason)
            final = FinalAgentResponse.model_validate(final)
            evidence = collect_evidence_from_messages(framework_output.get("messages"))
            if any(evidence_id not in evidence for evidence_id in final.evidence_ids):
                reason = "Agent structured response referenced unavailable evidence"
                raise ValueError(reason)
            if final.outcome == "answer" and evidence and not final.evidence_ids:
                reason = "Agent structured answer did not cite available evidence"
                raise ValueError(reason)
        except (ValueError, TypeError):
            pass
        else:
            mode = "agent_loop"
            if final.outcome == "answer":
                status = AgentResultStatus.OK
                answer = final.answer
                reason = "Agent model produced structured answer"
                references = [document_to_reference(evidence[key]) for key in final.evidence_ids]
            elif final.outcome == "no_evidence":
                status = AgentResultStatus.NO_RELEVANT_INFO
                answer = final.answer or "我无法根据当前取得的证据确定答案。"
                reason = "Agent reported insufficient evidence"
            else:
                status = AgentResultStatus.NEED_CLARIFICATION
                answer = clarification_question = final.clarification_question
                reason = "Agent requested clarification"
    compose_trace = {
        "step": "compose_final_answer",
        "mode": mode,
        "status": _TRACE_STATUSES[status],
        "answer_len": len(answer),
        "reason": reason,
    }
    tool_trace.append(compose_trace)
    llm_call_count = sum(item.get("kind") == "model" for item in events)
    result = AgentRunResult(
        status=status,
        answer=answer,
        references=references,
        retrieval_confidence=0.0,
        answer_confidence=0.0,
        need_clarification=status is AgentResultStatus.NEED_CLARIFICATION,
        clarification_question=clarification_question,
        reason=reason,
        action=str(invocation.decision.get("action") or ""),
        tool_result_summary=summarize_tool_trace_for_result(
            reason=reason, tool_trace=tool_trace,
        ),
        tool_trace=upsert_llm_usage_trace(
            tool_trace,
            answer_llm_used=answer_llm_used,
            llm_call_count=llm_call_count,
        ),
        router_llm_used=False,
        answer_llm_used=answer_llm_used,
        llm_call_count=llm_call_count,
    )
    return _finalize_agent_result(result)


def create_agent_result(
    *,
    status: AgentResultStatus,
    answer: str = "",
    references: list[dict[str, Any]] | None = None,
    retrieval_confidence: float = 0.0,
    answer_confidence: float = 0.0,
    need_clarification: bool = False,
    clarification_question: str = "",
    reason: str = "",
    action: str = "",
    tool_result_summary: str = "",
    tool_trace: list[dict[str, Any]] | None = None,
    answer_llm_used: bool = False,
    llm_call_count: int = 0,
) -> dict[str, Any]:
    """为服务级特殊分支构造同一完整、经过校验的结果字典。"""
    result = AgentRunResult(
        status=status,
        answer=answer,
        references=list(references or []),
        retrieval_confidence=retrieval_confidence,
        answer_confidence=answer_confidence,
        need_clarification=need_clarification,
        clarification_question=clarification_question,
        reason=reason,
        action=action,
        tool_result_summary=tool_result_summary,
        tool_trace=list(tool_trace or []),
        router_llm_used=False,
        answer_llm_used=answer_llm_used,
        llm_call_count=llm_call_count,
    )
    return result.model_dump(mode="json")


def build_internal_error_result(
    exc: Exception,
    *,
    runtime_label: str = "Agent Runtime",
    trace_step: str = "agent_runtime_error",
    answer: str = INTERNAL_ERROR_MESSAGE,
    reason: str = INTERNAL_ERROR_MESSAGE,
    tool_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造经过统一模型校验的脱敏内部错误结果。"""
    del exc, runtime_label
    resolved_trace = tool_trace
    if resolved_trace is None:
        resolved_trace = [
            {
                "step": trace_step,
                "status": "error",
                "error": INTERNAL_ERROR_MARKER,
            }
        ]
    return create_agent_result(
        status=AgentResultStatus.INTERNAL_ERROR,
        answer=answer,
        references=[],
        reason=reason,
        tool_trace=resolved_trace,
    )


def _default_bad_case(trace_id: str, question: str) -> dict[str, Any]:
    """构造 Agent Runtime 使用的 bad case 默认结构。"""
    return {
        "trace_id": trace_id,
        "question": question,
        "answer": "",
        "references": [],
        "route": "",
        "error_type": "",
        "note": "",
    }


def _write_internal_bad_case(
    write_bad_cases: Callable[[dict[str, Any]], None] | None,
    bad_case: dict[str, Any] | None,
    trace_id: str,
    question: str,
    exc: Exception,
    *,
    runtime_route: str = "agent_runtime",
    runtime_label: str = "Agent Runtime",
) -> None:
    """Runtime 顶层异常时写入 bad case，写入失败不继续抛错。"""
    if not callable(write_bad_cases):
        return

    record = dict(bad_case or _default_bad_case(trace_id, question))
    record["answer"] = f"系统内部执行 {runtime_label} 时发生错误，无法生成可靠回答。"
    record["references"] = []
    record["route"] = runtime_route
    record["error_type"] = "internal_error"
    record["note"] = str(exc)

    try:
        write_bad_cases(record)
    except Exception:
        return


def build_agent_runtime_error_result(
    *,
    write_bad_cases: Callable[[dict[str, Any]], None] | None,
    bad_case: dict[str, Any] | None,
    trace_id: str,
    session_id: str | None,
    original_question: str,
    effective_question: str,
    exc: Exception,
) -> dict[str, Any]:
    """把流式 Runtime 异常转换成脱敏内部错误结果。"""
    _write_internal_bad_case(
        write_bad_cases=write_bad_cases,
        bad_case=bad_case,
        trace_id=trace_id,
        question=original_question,
        exc=exc,
        runtime_route="agent_runtime",
        runtime_label="Agent Runtime",
    )
    return build_internal_error_result(
        exc,
        runtime_label="Agent Runtime",
        trace_step="agent_runtime_error",
    )


def build_empty_question_result() -> dict[str, Any]:
    """构造 EMPTY_QUESTION 服务级分支的完整澄清结果。"""
    clarification = (
        "请直接输入你想查询的具体问题，例如：年假申请流程是什么？"
    )
    return create_agent_result(
        status=AgentResultStatus.NEED_CLARIFICATION,
        need_clarification=True,
        clarification_question=clarification,
        reason="用户未输入问题",
    )


__all__ = [
    "AgentResultInvocation",
    "AgentRunResult",
    "build_agent_runtime_error_result",
    "build_decide_trace",
    "build_empty_question_result",
    "build_framework_agent_result",
    "build_internal_error_result",
    "build_short_circuit_result",
    "create_agent_result",
    "get_route_decision_for_trace",
    "upsert_llm_usage_trace",
]
