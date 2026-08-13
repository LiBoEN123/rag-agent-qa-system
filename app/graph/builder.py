"""最小 LangGraph Agent 主图构建器。"""

import asyncio
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.framework_agent import build_framework_agent
from app.config import AppSettings, get_settings
from app.graph.checkpointing import (
    clear_async_checkpointer_cache,
    clear_checkpointer_cache,
    get_async_checkpointer,
    get_checkpointer,
    has_async_checkpointer,
)
from app.graph.conditions import (
    after_framework_condition,
    after_initialize_condition,
)
from app.graph.nodes import (
    adapt_framework_agent_node,
    compose_node,
    finalize_node,
    initialize_node,
    rag_subgraph_node,
    route_node,
)
from app.graph.state import AgentGraphState
from app.tools.context import ToolInvocationContext


_async_compiled_agent_graph: CompiledStateGraph | None = None
_async_compiled_settings: AppSettings | None = None
_async_graph_owner_loop: asyncio.AbstractEventLoop | None = None
_async_graph_lock: asyncio.Lock | None = None
_async_graph_lock_loop: asyncio.AbstractEventLoop | None = None


def build_agent_graph(
    settings: AppSettings | None = None,
) -> StateGraph[
    AgentGraphState,
    ToolInvocationContext,
    AgentGraphState,
    AgentGraphState,
]:
    """创建尚未编译的最小 Agent StateGraph，不触发任何业务调用。"""
    resolved = settings or get_settings()
    builder = StateGraph(
        AgentGraphState,
        context_schema=ToolInvocationContext,
    )
    builder.add_node("route", route_node)
    builder.add_node("initialize", initialize_node)
    builder.add_node("framework_agent", build_framework_agent(resolved))
    builder.add_node("adapt_framework", adapt_framework_agent_node)
    builder.add_node("rag_subgraph", rag_subgraph_node)
    builder.add_node("compose", compose_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "route")
    builder.add_edge("route", "initialize")
    builder.add_conditional_edges(
        "initialize",
        after_initialize_condition,
        {
            "framework_agent": "framework_agent",
            "compose": "compose",
        },
    )
    builder.add_edge("framework_agent", "adapt_framework")
    builder.add_conditional_edges(
        "adapt_framework",
        after_framework_condition,
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
    settings: AppSettings | None = None,
) -> CompiledStateGraph[
    AgentGraphState,
    ToolInvocationContext,
    AgentGraphState,
    AgentGraphState,
]:
    """按配置注入 Checkpointer，并返回进程内复用的已编译 Agent Graph。"""
    resolved = settings or get_settings()
    checkpointer = get_checkpointer(resolved)
    builder = build_agent_graph(resolved)
    if checkpointer is None:
        return builder.compile()
    return builder.compile(checkpointer=checkpointer)


def _get_async_graph_lock() -> asyncio.Lock:
    """返回绑定当前事件循环的流式 Graph 初始化锁。"""
    global _async_graph_lock, _async_graph_lock_loop

    loop = asyncio.get_running_loop()
    if _async_graph_lock is None:
        _async_graph_lock = asyncio.Lock()
        _async_graph_lock_loop = loop
    elif _async_graph_lock_loop is not loop:
        if _async_compiled_agent_graph is not None:
            raise RuntimeError(
                "LangGraph 流式 Runtime 的事件循环已变化，"
                "请先重置 Agent Graph Runtime"
            )
        _async_graph_lock = asyncio.Lock()
        _async_graph_lock_loop = loop
    return _async_graph_lock


async def get_async_compiled_agent_graph(
    settings: AppSettings | None = None,
) -> CompiledStateGraph[
    AgentGraphState,
    ToolInvocationContext,
    AgentGraphState,
    AgentGraphState,
]:
    """为 ``astream`` 注入异步 Saver，并在事件循环内复用编译结果。"""
    global _async_compiled_agent_graph, _async_compiled_settings
    global _async_graph_owner_loop

    resolved = settings or get_settings()
    loop = asyncio.get_running_loop()
    if _async_compiled_agent_graph is not None:
        if _async_compiled_settings != resolved:
            raise RuntimeError(
                "LangGraph 流式 Runtime 配置已变化，请先重置 Runtime"
            )
        if _async_graph_owner_loop is not loop:
            raise RuntimeError(
                "LangGraph 流式 Runtime 的事件循环已变化，"
                "请先重置 Agent Graph Runtime"
            )
        return _async_compiled_agent_graph

    lock = _get_async_graph_lock()
    async with lock:
        if _async_compiled_agent_graph is not None:
            return _async_compiled_agent_graph
        checkpointer = await get_async_checkpointer(resolved)
        builder = build_agent_graph(resolved)
        graph = (
            builder.compile()
            if checkpointer is None
            else builder.compile(checkpointer=checkpointer)
        )
        _async_compiled_agent_graph = graph
        _async_compiled_settings = resolved
        _async_graph_owner_loop = loop
        return graph


def clear_async_compiled_agent_graph_cache() -> None:
    """失效流式 Graph 缓存；Saver 由异步重置入口另行关闭。"""
    global _async_compiled_agent_graph, _async_compiled_settings
    global _async_graph_lock, _async_graph_lock_loop
    global _async_graph_owner_loop

    _async_compiled_agent_graph = None
    _async_compiled_settings = None
    _async_graph_owner_loop = None
    _async_graph_lock = None
    _async_graph_lock_loop = None


async def areset_agent_graph_runtime() -> None:
    """在异步生命周期中失效两类 Graph，并关闭同步、异步 Saver。"""
    get_compiled_agent_graph.cache_clear()
    clear_async_compiled_agent_graph_cache()
    clear_checkpointer_cache()
    await clear_async_checkpointer_cache()


def reset_agent_graph_runtime() -> None:
    """在同步上下文中失效两类 Graph，并关闭同步、异步 Saver。"""
    if has_async_checkpointer():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "异步 Checkpointer 必须使用 await "
                "areset_agent_graph_runtime() 关闭"
            )

    get_compiled_agent_graph.cache_clear()
    clear_async_compiled_agent_graph_cache()
    clear_checkpointer_cache()
    if has_async_checkpointer():
        asyncio.run(clear_async_checkpointer_cache())


__all__ = [
    "areset_agent_graph_runtime",
    "build_agent_graph",
    "clear_async_compiled_agent_graph_cache",
    "get_async_compiled_agent_graph",
    "get_compiled_agent_graph",
    "reset_agent_graph_runtime",
]
