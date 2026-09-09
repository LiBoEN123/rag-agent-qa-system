"""Framework 输出到唯一 Agent Result Contract 的归一化边界。"""

from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, Field

from app.agent.guardrails import AgentStopReason
from app.agent.router import RouteDecision
from app.agent.tool_events import normalize_governance_events
from app.common.logging import get_logger
from app.config import get_settings
from app.contracts.chat import AgentResultStatus
from app.storage.traces import save_agent_trace


INTERNAL_ERROR_MESSAGE = "Agent 内部执行异常，请稍后重试"
INTERNAL_ERROR_MARKER = "internal_error"
DIRECT_ANSWER_TEXT = (
    "我是一个 RAG 知识库问答助手。"
    "我可以基于已导入的文档回答问题并提供引用，"
    "也可以列出知识库文档、查看指定 chunk、对比文档内容，以及进行简单四则运算。"
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

logger = get_logger("chat_service")


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
    decompose_llm_used: bool
    answer_llm_used: bool
    llm_call_count: int = Field(ge=0)
    timings: dict[str, Any]


class AgentResultInvocation(Protocol):
    """Result builder 所需的一次 Runtime Invocation 最小只读视图。"""

    decision: RouteDecision
    route_cost_ms: int
    original_question: str
    effective_question: str
    trace_id: str
    session_id: str | None


def summarize_tool_trace_for_result(
    *,
    status: AgentResultStatus,
    reason: str,
    tool_trace: list[dict[str, Any]],
) -> str:
    """从唯一公开 trace 中提取计入工具调用的结果摘要。"""
    if status is AgentResultStatus.NEED_CLARIFICATION:
        return "未调用工具，问题需要澄清"
    summaries = []
    for trace in tool_trace:
        # step_id 仅属于 counted call；handler_executed 会排除已计数的失败。
        if trace.get("step") != "tool_call" or "step_id" not in trace:
            continue
        if trace.get("tool_name") == "search_knowledge_base":
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


def resolve_llm_call_count(
    *,
    decompose_llm_used: bool,
    answer_llm_used: bool,
    current_count: int,
) -> int:
    """解析 Agent Model 与 RAG 回答侧已经累计的精确调用数。"""
    return max(
        int(bool(decompose_llm_used)) + int(bool(answer_llm_used)),
        max(0, int(current_count or 0)),
    )


def upsert_llm_usage_trace(
    tool_trace: list[dict[str, Any]],
    *,
    decompose_llm_used: bool,
    answer_llm_used: bool,
    llm_call_count: int,
) -> list[dict[str, Any]]:
    """返回包含唯一最新 LLM usage 项的新工具 trace。"""
    usage_trace = {
        "step": "llm_usage",
        "router_llm_used": False,
        "decompose_llm_used": bool(decompose_llm_used),
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


def _normalize_ai_content(content: object) -> str:
    """把模型支持的文本内容形式归一化为字符串。"""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content or "").strip()
    parts = [
        item if isinstance(item, str) else item.get("text", "")
        for item in content
        if isinstance(item, (str, dict))
    ]
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def _last_final_ai(messages: object) -> AIMessage | None:
    """查找最近一条不再请求工具的最终模型消息。"""
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message
    return None


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


def maybe_save_agent_trace(
    *,
    trace_id: str,
    session_id: str | None,
    question: str,
    rewritten_question: str = "",
    final_answer: str = "",
    status: str = "",
    reason: str = "",
    tool_trace: list[dict[str, Any]] | None = None,
    total_cost_ms: int = 0,
) -> None:
    """按配置保存已构造的 Agent Result trace，失败不影响主流程。"""
    if not get_settings().enable_agent_trace_store:
        return

    try:
        save_agent_trace(
            trace_id=trace_id,
            session_id=session_id,
            question=question,
            rewritten_question=rewritten_question,
            final_answer=final_answer,
            status=status,
            reason=reason,
            tool_trace=tool_trace or [],
            total_cost_ms=total_cost_ms,
        )
    except Exception as exc:
        logger.warning("保存 Agent trace 失败，忽略持久化错误: %s", exc)


def _finalize_agent_result(
    invocation: AgentResultInvocation,
    result: AgentRunResult,
) -> dict[str, Any]:
    """序列化已校验的最终 Result，并持久化同一份公开 trace。"""
    serialized = result.model_dump(mode="json")
    maybe_save_agent_trace(
        trace_id=invocation.trace_id,
        session_id=invocation.session_id,
        question=invocation.original_question,
        rewritten_question=(
            invocation.effective_question
            if invocation.effective_question != invocation.original_question
            else ""
        ),
        final_answer=serialized["answer"],
        status=serialized["status"],
        reason=serialized["reason"],
        tool_trace=serialized["tool_trace"],
    )
    return serialized


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
            decompose_llm_used=False,
            answer_llm_used=False,
            llm_call_count=0,
        ),
        router_llm_used=False,
        decompose_llm_used=False,
        answer_llm_used=False,
        llm_call_count=0,
        timings={},
    )
    return _finalize_agent_result(invocation, result)


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
    decompose_llm_used = False
    answer_llm_used = successful_models > 0
    rag_llm_call_count = 0
    references = []
    retrieval_confidence = answer_confidence = 0.0
    timings = {}

    raw_knowledge_result = framework_output.get("knowledge_base_result")
    knowledge_result = (
        raw_knowledge_result if isinstance(raw_knowledge_result, Mapping) else {}
    )
    knowledge_output = knowledge_result.get("output")
    rag_result = (
        knowledge_output.get("rag_result")
        if isinstance(knowledge_output, Mapping)
        else None
    )
    if isinstance(rag_result, Mapping):
        status = {
            "OK": AgentResultStatus.OK,
            "NEED_CLARIFICATION": AgentResultStatus.NEED_CLARIFICATION,
            "NO_RELEVANT_INFO": AgentResultStatus.NO_RELEVANT_INFO,
            "MODEL_ERROR": AgentResultStatus.MODEL_ERROR,
            "MODEL_BAD_OUTPUT": AgentResultStatus.MODEL_ERROR,
        }.get(str(rag_result.get("status") or "").strip(), AgentResultStatus.INTERNAL_ERROR)
        answer = str(rag_result.get("answer") or "")
        reason = str(rag_result.get("reason") or "")
        raw_references = rag_result.get("references")
        references = (
            [dict(item) for item in raw_references if isinstance(item, dict)]
            if isinstance(raw_references, list)
            else []
        )
        retrieval_confidence = rag_result.get("retrieval_confidence", 0.0)
        answer_confidence = rag_result.get("answer_confidence", 0.0)
        raw_timings = rag_result.get("timings")
        timings = dict(raw_timings) if isinstance(raw_timings, dict) else {}
        decompose_llm_used = bool(rag_result.get("decompose_llm_used"))
        answer_llm_used = bool(rag_result.get("answer_llm_used"))
        rag_llm_call_count = max(0, int(rag_result.get("llm_call_count", 0) or 0))
        mode = "rag"
    else:
        stop_reason = str(framework_output.get("stop_reason") or "")
        if stop_reason and stop_reason != AgentStopReason.COMPLETED.value:
            status, answer = _stop_result(stop_reason)
            reason = stop_reason
            mode = "error"
        else:
            message = _last_final_ai(framework_output.get("messages"))
            answer = _normalize_ai_content(message.content if message else "")
            status = AgentResultStatus.OK if answer else AgentResultStatus.MODEL_ERROR
            reason = (
                "Agent model produced final answer"
                if answer
                else "Agent model returned empty final answer"
            )
            mode = "agent_loop" if answer else "error"
            answer = answer or "模型未生成可用的最终回答。"
    compose_trace = {
        "step": "compose_final_answer",
        "mode": mode,
        "status": _TRACE_STATUSES[status],
        "answer_len": len(answer),
        "reason": reason,
    }
    if mode == "rag":
        compose_trace.update(
            reference_count=len(references),
            retrieval_confidence=retrieval_confidence,
            answer_confidence=answer_confidence,
            timings=timings,
            rag_llm_call_count=rag_llm_call_count,
        )
    tool_trace.append(compose_trace)
    llm_call_count = resolve_llm_call_count(
        decompose_llm_used=decompose_llm_used,
        answer_llm_used=answer_llm_used,
        current_count=successful_models + rag_llm_call_count,
    )
    result = AgentRunResult(
        status=status,
        answer=answer,
        references=references,
        retrieval_confidence=retrieval_confidence,
        answer_confidence=answer_confidence,
        need_clarification=status is AgentResultStatus.NEED_CLARIFICATION,
        clarification_question=(
            answer if status is AgentResultStatus.NEED_CLARIFICATION else ""
        ),
        reason=reason,
        action=str(invocation.decision.get("action") or ""),
        tool_result_summary=summarize_tool_trace_for_result(
            status=status, reason=reason, tool_trace=tool_trace,
        ),
        tool_trace=upsert_llm_usage_trace(
            tool_trace,
            decompose_llm_used=decompose_llm_used,
            answer_llm_used=answer_llm_used,
            llm_call_count=llm_call_count,
        ),
        router_llm_used=False,
        decompose_llm_used=decompose_llm_used,
        answer_llm_used=answer_llm_used,
        llm_call_count=llm_call_count,
        timings=timings,
    )
    return _finalize_agent_result(invocation, result)


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
    decompose_llm_used: bool = False,
    answer_llm_used: bool = False,
    llm_call_count: int = 0,
    timings: dict[str, Any] | None = None,
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
        decompose_llm_used=decompose_llm_used,
        answer_llm_used=answer_llm_used,
        llm_call_count=llm_call_count,
        timings=dict(timings or {}),
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
    """把同步与流式 Runtime 异常转换成同一脱敏内部错误结果。"""
    _write_internal_bad_case(
        write_bad_cases=write_bad_cases,
        bad_case=bad_case,
        trace_id=trace_id,
        question=original_question,
        exc=exc,
        runtime_route="agent_runtime",
        runtime_label="Agent Runtime",
    )
    result = build_internal_error_result(
        exc,
        runtime_label="Agent Runtime",
        trace_step="agent_runtime_error",
    )
    maybe_save_agent_trace(
        trace_id=trace_id,
        session_id=session_id,
        question=original_question,
        rewritten_question=(
            effective_question if effective_question != original_question else ""
        ),
        final_answer=result.get("answer", ""),
        status=result.get("status", ""),
        reason=result.get("reason", ""),
        tool_trace=result.get("tool_trace", []),
    )
    return result


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
    "maybe_save_agent_trace",
    "resolve_llm_call_count",
    "upsert_llm_usage_trace",
]
