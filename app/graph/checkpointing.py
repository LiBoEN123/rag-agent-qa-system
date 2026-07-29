"""LangGraph Checkpointer 的可选资源管理与请求线程标识。"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.common.logging import get_logger
from app.config import AppSettings, get_settings


logger = get_logger("chat_service")

_lock = threading.RLock()
_checkpointer: BaseCheckpointSaver | None = None
_sqlite_connection: sqlite3.Connection | None = None
_instance_key: tuple[str, str] | None = None


def get_checkpointer_mode(settings: AppSettings | None = None) -> str:
    """返回当前经过校验的 Checkpointer 配置模式。"""
    return (settings or get_settings()).langgraph_checkpointer


def _effective_instance_key(
    settings: AppSettings,
) -> tuple[str, str] | None:
    """生成当前 Saver 配置签名；未启用时返回 ``None``。"""
    mode = get_checkpointer_mode(settings)
    if mode == "disabled":
        return None
    if mode == "memory":
        return (mode, "")
    path = settings.langgraph_checkpoint_db_path
    return (mode, str(path))


def _create_sqlite_checkpointer(
    path: Path,
) -> tuple[BaseCheckpointSaver, sqlite3.Connection]:
    """创建同步 SQLite Saver，并立即验证数据库可初始化。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(path),
            check_same_thread=False,
            timeout=30.0,
        )
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(
            f"无法打开 LangGraph Checkpoint SQLite 数据库: {path}"
        ) from exc

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        saver = SqliteSaver(connection)
        saver.setup()
    except Exception as exc:
        connection.close()
        raise RuntimeError(
            f"无法初始化 LangGraph SQLite Checkpointer: {path}"
        ) from exc
    return saver, connection


def initialize_checkpointer(
    settings: AppSettings | None = None,
) -> BaseCheckpointSaver | None:
    """按配置惰性创建并复用 Checkpointer；disabled 返回 ``None``。"""
    global _checkpointer, _instance_key, _sqlite_connection

    resolved = settings or get_settings()
    requested_key = _effective_instance_key(resolved)
    if requested_key is None:
        return None

    with _lock:
        if _checkpointer is not None:
            if _instance_key != requested_key:
                raise RuntimeError(
                    "LangGraph Checkpointer 配置已变化，请先重置 Agent Graph Runtime"
                )
            return _checkpointer

        mode, raw_path = requested_key
        if mode == "memory":
            saver: BaseCheckpointSaver = InMemorySaver()
            connection = None
        else:
            saver, connection = _create_sqlite_checkpointer(Path(raw_path))

        _checkpointer = saver
        _sqlite_connection = connection
        _instance_key = requested_key
        logger.info(
            "[langgraph] checkpointer_initialized mode=%s db_path=%s",
            mode,
            raw_path or "-",
        )
        return _checkpointer


def get_checkpointer(
    settings: AppSettings | None = None,
) -> BaseCheckpointSaver | None:
    """返回当前生命周期内复用的 Checkpointer。"""
    return initialize_checkpointer(settings)


def close_checkpointer() -> None:
    """安全失效 compiled Graph，并幂等关闭 Checkpointer 资源。"""
    global _checkpointer, _instance_key, _sqlite_connection

    # Saver 关闭后 compiled Graph 不能继续持有旧引用。延迟导入避免模块循环。
    try:
        from app.graph.builder import get_compiled_agent_graph

        get_compiled_agent_graph.cache_clear()
    except (ImportError, AttributeError):
        pass

    with _lock:
        connection = _sqlite_connection
        mode = _instance_key[0] if _instance_key else "disabled"
        _checkpointer = None
        _sqlite_connection = None
        _instance_key = None
        if connection is None:
            return
        try:
            connection.close()
            logger.info("[langgraph] checkpointer_closed mode=%s", mode)
        except sqlite3.Error as exc:
            logger.warning(
                "[langgraph] checkpointer_close_failed mode=%s error=%s",
                mode,
                str(exc)[:200],
            )


def clear_checkpointer_cache() -> None:
    """关闭并清理全局 Saver，供测试和配置重载使用。"""
    close_checkpointer()


def build_checkpoint_thread_id(
    *,
    trace_id: str,
    session_id: str | None = None,
) -> str:
    """为一次 Graph 执行生成稳定、安全且不暴露 session 的 thread ID。"""
    normalized_trace_id = (trace_id or "").strip() or uuid.uuid4().hex
    normalized_session_id = (session_id or "").strip()
    payload = (
        f"{len(normalized_session_id)}:{normalized_session_id}|"
        f"{len(normalized_trace_id)}:{normalized_trace_id}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
    return f"agent-run-{digest}"


__all__ = [
    "build_checkpoint_thread_id",
    "clear_checkpointer_cache",
    "close_checkpointer",
    "get_checkpointer",
    "get_checkpointer_mode",
    "initialize_checkpointer",
]
