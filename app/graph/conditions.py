"""LangGraph 父图的无副作用条件边。"""

from typing import Literal

from app.agent.state import AgentState
from app.graph.state import AgentGraphState


def route_condition(state: AgentGraphState) -> Literal["route", "initialize"]:
    """仅在调用方没有提供路由决策时执行 Router。"""
    return "route" if state.get("decision") is None else "initialize"


def after_initialize_condition(
    state: AgentGraphState,
) -> Literal["framework_agent", "legacy_plan"]:
    """显式兼容动作保留旧快路径，其余全部进入框架 Agent。"""
    decision = state.get("decision")
    if isinstance(decision, dict) and decision.get("action") in {
        "rag_fast_path",
        "search_knowledge_base",
        "clarify",
        "direct_answer",
    }:
        return "legacy_plan"
    return "framework_agent"


def after_framework_condition(
    state: AgentGraphState,
) -> Literal["legacy_plan", "rag_subgraph", "compose"]:
    """根据框架子图结果选择兼容、RAG 或答案合成路径。"""
    if state.get("fallback_to_legacy"):
        return "legacy_plan"
    if state.get("pending_rag_result"):
        return "rag_subgraph"
    return "compose"


def after_execute_condition(
    state: AgentGraphState,
) -> Literal["rag_subgraph", "compose"]:
    """旧 Executor 的成功知识库 observation 继续进入现有 RAG 子图。"""
    agent_state = state.get("agent_state")
    if not isinstance(agent_state, AgentState) or agent_state.status in {
        "clarify",
        "direct_answer",
        "tool_error",
        "model_error",
        "error",
    }:
        return "compose"
    for observation in agent_state.observations:
        if (
            observation.tool_name == "search_knowledge_base"
            and observation.success
            and isinstance(observation.output.get("chunks"), list)
        ):
            return "rag_subgraph"
    return "compose"


__all__ = [
    "after_execute_condition",
    "after_framework_condition",
    "after_initialize_condition",
    "route_condition",
]
