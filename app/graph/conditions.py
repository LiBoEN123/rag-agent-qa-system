"""LangGraph 父图的无副作用条件边。"""

from typing import Literal

from app.graph.state import AgentGraphState


def after_initialize_condition(
    state: AgentGraphState,
) -> Literal["framework_agent", "compose"]:
    """只有运行中的 Agent 状态进入框架模型，其余确定性状态直接合成。"""
    return "framework_agent" if state.get("status") == "running" else "compose"


def after_framework_condition(
    state: AgentGraphState,
) -> Literal["rag_subgraph", "compose"]:
    """根据框架子图结果选择 RAG 或答案合成路径。"""
    if state.get("pending_rag_result"):
        return "rag_subgraph"
    return "compose"


__all__ = [
    "after_framework_condition",
    "after_initialize_condition",
]
