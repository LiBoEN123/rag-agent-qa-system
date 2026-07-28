"""LangGraph 父图的无副作用条件边。"""

from typing import Literal

from app.graph.state import AgentGraphState


def route_condition(state: AgentGraphState) -> Literal["route", "initialize"]:
    """仅在调用方没有提供路由决策时执行 Router。"""
    return "route" if state.get("decision") is None else "initialize"


def after_initialize_condition(
    state: AgentGraphState,
) -> Literal["framework_agent", "compose"]:
    """只有运行中的 Agent 状态进入框架模型，其余确定性状态直接合成。"""
    agent_state = state.get("agent_state")
    return (
        "framework_agent"
        if agent_state is not None and agent_state.status == "running"
        else "compose"
    )


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
    "route_condition",
]
