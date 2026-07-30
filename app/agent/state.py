"""Agent 业务状态中的轻量类型。"""

from typing import Any, Literal, TypedDict


AgentStatus = Literal[
    "running",
    "ok",
    "clarify",
    "direct_answer",
    "tool_error",
    "model_error",
    "no_evidence",
    "error",
]


class AgentObservation(TypedDict):
    """单个工具调用对应的普通数据观察结果。"""

    step_id: int
    tool_name: str
    success: bool
    summary: str
    output: dict[str, Any]
    error: str
    cost_ms: int


__all__ = [
    "AgentObservation",
    "AgentStatus",
]
