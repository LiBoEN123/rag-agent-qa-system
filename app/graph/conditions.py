"""LangGraph 主图的条件边函数。"""

from typing import Literal

from app.agent.state import AgentState
from app.graph.state import AgentGraphState


def route_condition(state: AgentGraphState) -> Literal["route", "initialize"]:
    """仅在调用方未提供路由决策时进入 Router 节点。"""
    return "route" if state.get("decision") is None else "initialize"


def after_execute_condition(
    state: AgentGraphState,
) -> Literal["rag_subgraph", "compose"]:
    """成功知识库 observation 进入 RAG 子图，其余状态继续旧 Compose。"""
    agent_state = state.get("agent_state")
    if not isinstance(agent_state, AgentState):
        return "compose"
    if agent_state.status in {
        "clarify",
        "direct_answer",
        "tool_error",
        "model_error",
        "error",
    }:
        return "compose"

    for observation in agent_state.observations:
        if observation.tool_name != "search_knowledge_base":
            continue
        if not observation.success or not isinstance(observation.output, dict):
            continue
        if isinstance(observation.output.get("chunks"), list):
            return "rag_subgraph"
    return "compose"


__all__ = ["after_execute_condition", "route_condition"]
