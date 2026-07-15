"""LangGraph 主图状态定义。"""

from typing import Any, TypedDict

from app.agent.state import AgentState


class AgentGraphState(TypedDict, total=False):
    """在 LangGraph 节点间传递请求上下文和现有 Agent 业务状态。"""

    original_question: str
    effective_question: str
    top_k: int
    decompose_top_k: int
    decision: dict[str, Any] | None
    trace_id: str
    session_id: str | None
    agent_state: AgentState
    result: dict[str, Any]

__all__ = ["AgentGraphState"]
