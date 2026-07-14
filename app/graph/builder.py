"""最小 LangGraph Agent 主图构建器。"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.checkpointing import get_checkpointer
from app.graph.conditions import after_execute_condition, route_condition
from app.graph.nodes import (
    compose_node,
    execute_node,
    finalize_node,
    initialize_agent_state_node,
    plan_node,
    rag_subgraph_node,
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
    builder.add_node("rag_subgraph", rag_subgraph_node)
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
    builder.add_conditional_edges(
        "execute",
        after_execute_condition,
        {
            "rag_subgraph": "rag_subgraph",
            "compose": "compose",
        },
    )
    builder.add_edge("rag_subgraph", "finalize")
    builder.add_edge("compose", "finalize")
    builder.add_edge("finalize", END)
    return builder


@lru_cache(maxsize=1)
def get_compiled_agent_graph(
) -> CompiledStateGraph[AgentGraphState, None, AgentGraphState, AgentGraphState]:
    """按配置注入 Checkpointer，并返回进程内复用的已编译 Agent Graph。"""
    checkpointer = get_checkpointer()
    builder = build_agent_graph()
    if checkpointer is None:
        return builder.compile()
    return builder.compile(checkpointer=checkpointer)


def reset_agent_graph_runtime() -> None:
    """先失效 compiled Graph，再关闭并清理其 Checkpointer。"""
    get_compiled_agent_graph.cache_clear()
    from app.graph.checkpointing import clear_checkpointer_cache

    clear_checkpointer_cache()


__all__ = [
    "build_agent_graph",
    "get_compiled_agent_graph",
    "reset_agent_graph_runtime",
]
