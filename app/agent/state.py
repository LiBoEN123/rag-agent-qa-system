"""工具治理事件投影使用的轻量 observation 类型。"""

from typing import Any, TypedDict


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
]
