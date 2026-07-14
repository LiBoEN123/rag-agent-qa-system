"""应用启动生命周期模块。

负责校验配置，并在服务启动时判断加载缓存还是重建知识库索引。
"""

from dataclasses import dataclass
from typing import Any, Callable

from app.common.logging import get_logger


logger = get_logger("chat_service")


@dataclass(frozen=True)
class StartupDependencies:
    settings: Any
    needs_rebuild: Callable
    index_cache_exists: Callable
    load_index_cache: Callable
    rebuild_and_persist_index: Callable


def initialize_rag(deps: StartupDependencies) -> None:
    """校验配置，并在启动时加载或重建 RAG 索引。"""
    deps.settings.validate()
    data_dir = str(deps.settings.RAG_FILE_PATH)
    needs_rebuild, reason = deps.needs_rebuild(data_dir)
    logger.info("[startup] needs_rebuild=%s reason=%s", needs_rebuild, reason)

    should_rebuild = (
        needs_rebuild
        or not deps.index_cache_exists()
        or deps.settings.AUTO_REBUILD_ON_STARTUP
    )
    if should_rebuild:
        ok, message = deps.rebuild_and_persist_index(
            data_dir=data_dir,
            chunk_size=deps.settings.CHUNK_SIZE,
            chunk_overlap=deps.settings.CHUNK_OVERLAP,
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
        chunk_size=deps.settings.CHUNK_SIZE,
        chunk_overlap=deps.settings.CHUNK_OVERLAP,
    )
    logger.info("[startup] %s", message)


def validate_agent_runtime_configuration(settings: Any) -> str:
    """校验 Runtime/Checkpointer，并为 LangGraph 预编译无副作用主图。"""
    from app.config import (
        validate_agent_runtime,
        validate_langgraph_checkpointer,
    )

    runtime = validate_agent_runtime(settings.AGENT_RUNTIME)
    mode = validate_langgraph_checkpointer(settings.LANGGRAPH_CHECKPOINTER)
    if runtime == "legacy":
        logger.info(
            "[startup] agent_runtime=legacy checkpointer_mode=disabled validated"
        )
        return runtime

    from app.graph.builder import get_compiled_agent_graph

    get_compiled_agent_graph()
    logger.info(
        "[startup] agent_runtime=langgraph checkpointer_mode=%s validated",
        mode,
    )
    return runtime


def initialize_agent_graph_runtime(settings: Any) -> None:
    """兼容现有启动入口，执行 Runtime 配置校验和 LangGraph 预编译。"""
    validate_agent_runtime_configuration(settings)


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
