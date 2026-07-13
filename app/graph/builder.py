"""最小 LangGraph Agent 主图构建器。"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.conditions import route_condition
from app.graph.nodes import (
    compose_node,
    execute_node,
    finalize_node,
    initialize_agent_state_node,
    plan_node,
    route_node,
)
from app.graph.state import AgentGraphState


def build_agent_graph(
) -> StateGraph[AgentGraphState, None, AgentGraphState, AgentGraphState]:
    """创建尚未编译的最小 Agent StateGraph，不触发任何业务调用。"""
    builder = StateGraph(AgentGraphState)
    builder.add_node("route", route_node)
    builder.add_node("initialize", initialize_agent_state_node)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("compose", compose_node)
    builder.add_node("finalize", finalize_node)

    builder.add_conditional_edges(
        START,
        route_condition,
        {
            "route": "route",
            "initialize": "initialize",
        },
    )
    builder.add_edge("route", "initialize")
    builder.add_edge("initialize", "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "compose")
    builder.add_edge("compose", "finalize")
    builder.add_edge("finalize", END)
    return builder


@lru_cache(maxsize=1)
def get_compiled_agent_graph(
) -> CompiledStateGraph[AgentGraphState, None, AgentGraphState, AgentGraphState]:
    """返回进程内复用的已编译 Agent Graph。"""
    return build_agent_graph().compile()


__all__ = ["build_agent_graph", "get_compiled_agent_graph"]
