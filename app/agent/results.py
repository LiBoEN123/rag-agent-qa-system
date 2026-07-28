"""Agent 结果模型、状态转换和 JSON-safe 结果构造。"""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.state import AgentStatus
from app.contracts.chat import AgentResultStatus


INTERNAL_ERROR_MESSAGE = "Agent 内部执行异常，请稍后重试"
INTERNAL_ERROR_MARKER = "internal_error"


class AgentResultContractError(ValueError):
    """Graph 输出不满足公开 Agent 结果契约。"""


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


def map_agent_status_to_result_status(
    status: AgentStatus | str,
) -> AgentResultStatus:
    """将平铺 Graph 内部状态映射为唯一 Result status。"""
    mapping = {
        "ok": AgentResultStatus.OK,
        "direct_answer": AgentResultStatus.OK,
        "clarify": AgentResultStatus.NEED_CLARIFICATION,
        "tool_error": AgentResultStatus.TOOL_ERROR,
        "model_error": AgentResultStatus.MODEL_ERROR,
        "no_evidence": AgentResultStatus.NO_RELEVANT_INFO,
        "error": AgentResultStatus.INTERNAL_ERROR,
        "running": AgentResultStatus.INTERNAL_ERROR,
    }
    return mapping.get(
        str(status or ""),
        AgentResultStatus.INTERNAL_ERROR,
    )


def map_rag_status_to_agent_status(status: str) -> AgentStatus:
    """将现有 RAG 状态映射为平铺 AgentGraphState 状态。"""
    mapping: dict[str, AgentStatus] = {
        "OK": "ok",
        "NEED_CLARIFICATION": "clarify",
        "NO_RELEVANT_INFO": "no_evidence",
        "MODEL_ERROR": "model_error",
        "MODEL_BAD_OUTPUT": "model_error",
        "ok": "ok",
        "clarify": "clarify",
        "no_evidence": "no_evidence",
        "model_error": "model_error",
    }
    return mapping.get(str(status or "").strip(), "error")


def summarize_observations_for_result(
    *,
    status: str,
    reason: str,
    observations: list[dict[str, Any]],
) -> str:
    """从 observations 中提取 API 响应使用的工具结果摘要。"""
    if status == "clarify":
        return "未调用工具，问题需要澄清"
    if status == "direct_answer":
        return "未调用工具，直接回答"
    if not observations:
        return reason

    summaries = []
    for observation in observations:
        if observation.get("tool_name") == "search_knowledge_base":
            output = observation.get("output")
            output = output if isinstance(output, dict) else {}
            chunks = output.get("chunks", [])
            count = output.get(
                "count",
                len(chunks) if isinstance(chunks, list) else 0,
            )
            summaries.append(f"知识库检索完成，召回 {count} 个 chunk")
            continue
        summary = str(observation.get("summary") or "")
        if summary:
            summaries.append(summary)
    return "；".join(summaries[-3:]) if summaries else reason


def build_agent_result(state: Mapping[str, Any]) -> AgentRunResult:
    """从平铺 AgentGraphState 构造并校验唯一 Runtime 结果。"""
    status = str(state.get("status") or "")
    reason = str(state.get("reason") or "")
    final_answer = str(state.get("final_answer") or "")
    references = state.get("references")
    references = references if isinstance(references, list) else []
    observations = [
        item
        for item in state.get("observations", [])
        if isinstance(item, dict)
    ]
    tool_trace = state.get("tool_trace")
    tool_trace = tool_trace if isinstance(tool_trace, list) else []
    decision = state.get("decision")
    decision = decision if isinstance(decision, dict) else {}
    need_clarification = status == "clarify"
    rag_trace = next(
        (
            step
            for step in reversed(tool_trace)
            if isinstance(step, dict)
            and step.get("step") == "compose_final_answer"
            and step.get("mode") == "rag"
        ),
        {},
    )
    timings = rag_trace.get("timings")

    return AgentRunResult(
        status=map_agent_status_to_result_status(status),
        answer=final_answer,
        references=references,
        retrieval_confidence=rag_trace.get("retrieval_confidence", 0.0),
        answer_confidence=rag_trace.get("answer_confidence", 0.0),
        need_clarification=need_clarification,
        clarification_question=final_answer if need_clarification else "",
        reason=reason,
        action=str(decision.get("action") or ""),
        tool_result_summary=summarize_observations_for_result(
            status=status,
            reason=reason,
            observations=observations,
        ),
        tool_trace=tool_trace,
        router_llm_used=False,
        decompose_llm_used=bool(state.get("decompose_llm_used")),
        answer_llm_used=bool(state.get("answer_llm_used")),
        llm_call_count=max(0, int(state.get("llm_call_count", 0) or 0)),
        timings=dict(timings) if isinstance(timings, dict) else {},
    )


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


def validate_agent_result_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """校验并返回 JSON-safe 普通字典，不把校验输入写入异常。"""
    try:
        result = AgentRunResult.model_validate(dict(payload))
    except (TypeError, ValidationError) as exc:
        error_count = (
            len(exc.errors(include_input=False))
            if isinstance(exc, ValidationError)
            else 1
        )
        raise AgentResultContractError(
            f"Agent result contract validation failed ({error_count} errors)"
        ) from None
    return result.model_dump(mode="json")


__all__ = [
    "AgentResultContractError",
    "AgentRunResult",
    "build_agent_result",
    "build_empty_question_result",
    "build_internal_error_result",
    "create_agent_result",
    "map_agent_status_to_result_status",
    "map_rag_status_to_agent_status",
    "validate_agent_result_payload",
]
