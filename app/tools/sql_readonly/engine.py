"""只读 SQLAlchemy Engine 的惰性创建、缓存和释放。"""

from __future__ import annotations

import hashlib
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from sqlalchemy import Engine, URL, create_engine, event
from sqlalchemy.engine import Connection, make_url

from app.config import AppSettings, secret_value, sqlite_database_path
from app.tools.errors import ToolErrorCode
from app.tools.sql_readonly.errors import SqlReadonlyError


_engine_lock = threading.Lock()
_engine_cache: dict[tuple[str, int, float], Engine] = {}


@dataclass(slots=True)
class StatementTimeoutState:
    """记录 SQLite progress handler 是否实际触发了中断。"""

    timed_out: bool = False


def _readonly_sqlite_url(settings: AppSettings) -> URL:
    """把受控 SQLite 文件路径重建为 ``mode=ro`` URI。"""
    path = sqlite_database_path(settings.sql_database_url)
    if path is None:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_NOT_CONFIGURED,
            "只读 SQL 数据源未正确配置",
        )
    raw_url = secret_value(settings.sql_database_url)
    try:
        configured = make_url(raw_url or "")
    except Exception as exc:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_NOT_CONFIGURED,
            "只读 SQL 数据源配置无效",
        ) from exc
    drivername = configured.drivername
    if drivername == "sqlite":
        drivername = "sqlite+pysqlite"
    return URL.create(
        drivername=drivername,
        database=f"file:{path.as_posix()}",
        query={"mode": "ro", "uri": "true"},
    )


def _configure_sqlite_readonly(engine: Engine) -> None:
    """为连接池中的每个 SQLite DBAPI 连接启用 query_only。"""

    @event.listens_for(engine, "connect")
    def set_sqlite_query_only(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA query_only = ON")
        finally:
            cursor.close()


def _create_sql_engine(settings: AppSettings) -> Engine:
    """只创建 Engine 和小型连接池，不建立数据库连接。"""
    raw_url = secret_value(settings.sql_database_url)
    if not settings.sql_tool_enabled or not raw_url:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_NOT_CONFIGURED,
            "只读 SQL 工具未启用或数据源未配置",
        )
    try:
        configured = make_url(raw_url)
        is_sqlite = configured.get_backend_name() == "sqlite"
        engine_url: URL | str = (
            _readonly_sqlite_url(settings) if is_sqlite else raw_url
        )
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        engine = create_engine(
            engine_url,
            connect_args=connect_args,
            echo=False,
            hide_parameters=True,
            max_overflow=0,
            pool_pre_ping=True,
            pool_size=2,
            pool_timeout=max(1.0, min(settings.sql_tool_timeout_seconds, 30.0)),
        )
        if is_sqlite:
            _configure_sqlite_readonly(engine)
        return engine
    except SqlReadonlyError:
        raise
    except Exception as exc:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_CONNECTION_ERROR,
            "无法初始化只读数据库连接",
        ) from exc


def get_sql_engine(settings: AppSettings) -> Engine:
    """按不暴露 URL 原文的配置指纹惰性复用 SQLAlchemy Engine。"""
    raw_url = secret_value(settings.sql_database_url)
    if not settings.sql_tool_enabled or not raw_url:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_NOT_CONFIGURED,
            "只读 SQL 工具未启用或数据源未配置",
        )
    fingerprint = hashlib.sha256(raw_url.encode("utf-8")).hexdigest()
    cache_key = (
        fingerprint,
        settings.sql_statement_timeout_ms,
        settings.sql_tool_timeout_seconds,
    )
    with _engine_lock:
        engine = _engine_cache.get(cache_key)
        if engine is None:
            engine = _create_sql_engine(settings)
            _engine_cache[cache_key] = engine
        return engine


@contextmanager
def statement_timeout(
    connection: Connection,
    timeout_ms: int,
) -> Iterator[StatementTimeoutState]:
    """在一次 SQLite 查询的 execute/fetch 全阶段施加可中断超时。"""
    state = StatementTimeoutState()
    if connection.dialect.name != "sqlite":
        yield state
        return

    driver_connection = connection.connection.driver_connection
    deadline = time.monotonic() + timeout_ms / 1000.0

    def interrupt_when_expired() -> int:
        if time.monotonic() < deadline:
            return 0
        state.timed_out = True
        return 1

    driver_connection.set_progress_handler(interrupt_when_expired, 1000)
    try:
        yield state
    finally:
        driver_connection.set_progress_handler(None, 0)


def clear_sql_engine_cache() -> None:
    """移除并 dispose 全部缓存 Engine，供 shutdown 与隔离测试调用。"""
    with _engine_lock:
        engines = list(_engine_cache.values())
        _engine_cache.clear()
    for engine in engines:
        engine.dispose()


__all__ = [
    "StatementTimeoutState",
    "clear_sql_engine_cache",
    "get_sql_engine",
    "statement_timeout",
]
