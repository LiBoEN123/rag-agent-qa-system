"""应用启动生命周期模块。

负责校验配置，并在服务启动时判断加载缓存还是重建知识库索引。
"""

from dataclasses import dataclass
from typing import Callable

from app.common.logging import get_logger
from app.config import AppSettings


logger = get_logger("chat_service")


@dataclass(frozen=True)
class StartupDependencies:
    settings: AppSettings
    needs_rebuild: Callable
    index_cache_exists: Callable
    load_index_cache: Callable
    rebuild_and_persist_index: Callable


def initialize_rag(deps: StartupDependencies) -> None:
    """在启动时加载或重建 RAG 索引。"""
    data_dir = str(deps.settings.rag_file_path)
    needs_rebuild, reason = deps.needs_rebuild(data_dir)
    logger.info("[startup] needs_rebuild=%s reason=%s", needs_rebuild, reason)

    should_rebuild = (
        needs_rebuild
        or not deps.index_cache_exists()
        or deps.settings.auto_rebuild_on_startup
    )
    if should_rebuild:
        ok, message = deps.rebuild_and_persist_index(
            data_dir=data_dir,
            chunk_size=deps.settings.chunk_size,
            chunk_overlap=deps.settings.chunk_overlap,
        )
        logger.info("[startup] %s", message)
        if not ok:
            logger.warning("[startup] 重建失败，RAG 可能不可用")
        return

    loaded, message = deps.load_index_cache()
    logger.info("[startup] %s", message)
    if loaded:
        return

    logger.warning("[startup] 缓存加载失败，降级为重建")
    _, message = deps.rebuild_and_persist_index(
        data_dir=data_dir,
        chunk_size=deps.settings.chunk_size,
        chunk_overlap=deps.settings.chunk_overlap,
    )
    logger.info("[startup] %s", message)


def validate_agent_graph_configuration(settings: AppSettings) -> str:
    """校验 Checkpointer，并预编译无副作用的 LangGraph 主图。"""
    mode = settings.langgraph_checkpointer
    from app.graph.builder import get_compiled_agent_graph

    get_compiled_agent_graph(settings)
    logger.info(
        "[startup] agent_runtime=langgraph checkpointer_mode=%s validated",
        mode,
    )
    return mode


def initialize_agent_graph_runtime(settings: AppSettings) -> None:
    """执行 Checkpointer 配置校验和 LangGraph 预编译。"""
    validate_agent_graph_configuration(settings)


async def initialize_mcp_runtime(settings: AppSettings) -> None:
    """在启动事件循环中完成 MCP 配置读取和 Tool discovery。"""
    from app.tools.mcp.client import initialize_mcp_runtime as initialize

    await initialize(settings)


def close_agent_graph_runtime() -> None:
    """关闭 Agent Graph Runtime；关闭异常只记录日志，不阻塞其他清理。"""
    try:
        from app.graph.builder import reset_agent_graph_runtime

        reset_agent_graph_runtime()
    except Exception as exc:
        logger.warning(
            "[shutdown] langgraph_runtime_close_failed error=%s",
            str(exc)[:200],
        )


async def aclose_agent_graph_runtime() -> None:
    """在应用事件循环中关闭同步与流式 Agent Graph Runtime。"""
    try:
        from app.graph.builder import areset_agent_graph_runtime

        await areset_agent_graph_runtime()
    except Exception as exc:
        logger.warning(
            "[shutdown] async_langgraph_runtime_close_failed error=%s",
            str(exc)[:200],
        )


async def close_mcp_runtime() -> None:
    """清理 MCP client 引用和 Tool cache；失败不阻塞其他资源关闭。"""
    try:
        from app.tools.mcp.client import close_mcp_runtime as close_runtime

        await close_runtime()
    except Exception as exc:
        logger.warning(
            "[shutdown] mcp_runtime_close_failed error_type=%s",
            type(exc).__name__,
        )


def close_tool_runtime() -> None:
    """关闭工具线程池；关闭异常只记录日志，不阻塞其他资源清理。"""
    try:
        from app.tools.runtime import close_tool_runtime as close_runtime

        close_runtime()
    except Exception as exc:
        logger.warning(
            "[shutdown] tool_runtime_close_failed error=%s",
            str(exc)[:200],
        )


def close_sql_engine_runtime() -> None:
    """dispose 惰性 SQL Engine 缓存；未启用或未调用时保持空操作。"""
    try:
        from app.tools.sql_readonly.engine import clear_sql_engine_cache

        clear_sql_engine_cache()
    except Exception as exc:
        logger.warning(
            "[shutdown] sql_engine_close_failed error_type=%s",
            type(exc).__name__,
        )
