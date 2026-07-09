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
