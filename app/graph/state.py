"""LangGraph 主图状态定义。"""

from typing import Any, TypedDict

from app.agent.state import AgentState


class AgentGraphState(TypedDict, total=False):
    """在 LangGraph 节点间传递请求上下文和现有 Agent 业务状态。"""

    # ``question`` 是旧调用协议中的执行问题别名，保留用于兼容直接调用。
    question: str
    original_question: str
    effective_question: str
    top_k: int
    decompose_top_k: int
    decision: dict[str, Any] | None
    trace_id: str
    session_id: str | None
    agent_state: AgentState
    result: dict[str, Any]


def resolve_question_inputs(
    question: str | None = None,
    *,
    original_question: str | None = None,
    effective_question: str | None = None,
) -> tuple[str, str]:
    """统一规整新旧调用协议，返回原始问题和实际执行问题。"""
    legacy_question = (question or "").strip()
    normalized_original = (original_question or "").strip()
    normalized_effective = (effective_question or "").strip()

    resolved_effective = (
        normalized_effective
        or legacy_question
        or normalized_original
    )
    resolved_original = (
        normalized_original
        or legacy_question
        or resolved_effective
    )
    return resolved_original, resolved_effective


__all__ = ["AgentGraphState", "resolve_question_inputs"]
