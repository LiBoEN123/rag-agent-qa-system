"""应用启动生命周期模块。

启动时增量同步知识库索引，并管理 Agent 等运行时资源。
"""

from dataclasses import dataclass
from typing import Callable

from app.common.logging import get_logger
from app.config import AppSettings


logger = get_logger("chat_service")


# 保存启动阶段使用的配置、索引重建判断和重建执行依赖。
@dataclass(frozen=True)
class StartupDependencies:
    settings: AppSettings
    needs_rebuild: Callable
    rebuild_and_persist_index: Callable


def initialize_rag(deps: StartupDependencies) -> None:
    """启动时始终执行增量 sync；仅配置/存储变化触发 force rebuild。"""
    data_dir = str(deps.settings.rag_file_path)
    needs_force_rebuild, reason = deps.needs_rebuild(data_dir)
    force_rebuild = needs_force_rebuild or deps.settings.auto_rebuild_on_startup
    logger.info(
        "[startup] force_rebuild=%s reason=%s",
        force_rebuild,
        reason,
    )
    ok, message = deps.rebuild_and_persist_index(
        data_dir=data_dir,
        chunk_size=deps.settings.chunk_size,
        chunk_overlap=deps.settings.chunk_overlap,
        force_rebuild=force_rebuild,
    )
    logger.info("[startup] %s", message)
    if not ok:
        logger.warning("[startup] 增量同步失败，保留原 RAG runtime")


def initialize_agent_runtime(settings: AppSettings) -> None:
    """预编译无外部调用的唯一 ``create_agent`` Runtime。"""
    from app.agent.runtime import get_compiled_agent_runtime

    get_compiled_agent_runtime(settings)
    logger.info("[startup] agent_runtime=create_agent initialized")


async def initialize_mcp_runtime(settings: AppSettings) -> None:
    """在启动事件循环中完成 MCP 配置读取和 Tool discovery。"""
    from app.tools.mcp.client import initialize_mcp_runtime as initialize

    await initialize(settings)


def close_agent_runtime() -> None:
    """失效 Agent Runtime 缓存；关闭异常不阻塞其他清理。"""
    try:
        from app.agent.runtime import reset_agent_runtime

        reset_agent_runtime()
    except Exception as exc:
        logger.warning(
            "[shutdown] agent_runtime_close_failed error=%s",
            str(exc)[:200],
        )


async def aclose_agent_runtime() -> None:
    """为 FastAPI lifespan 提供一致的异步关闭入口。"""
    close_agent_runtime()


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


