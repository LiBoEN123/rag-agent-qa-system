"""Application SQLite 的共享 SQLAlchemy Engine 和 Session。"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


_engines: dict[Path, Engine] = {}
_engine_lock = Lock()


def now_iso() -> str:
    """返回适合 SQLite 文本字段存储的本地时间。"""
    return datetime.now().isoformat(timespec="milliseconds")


def get_db_path() -> Path:
    """返回当前配置的 SQLite 数据库路径。"""
    return get_settings().sqlite_db_path


def get_engine(db_path: Path | None = None) -> Engine:
    """按绝对路径复用 Engine，支持测试隔离和线程调用。"""
    path = Path(db_path if db_path is not None else get_db_path()).resolve()
    with _engine_lock:
        if path not in _engines:
            path.parent.mkdir(parents=True, exist_ok=True)
            _engines[path] = create_engine(
                URL.create("sqlite", database=str(path)),
                connect_args={"check_same_thread": False},
            )
        return _engines[path]


# 按数据库引擎缓存会话工厂，并在提交后保留已加载的对象属性。
@lru_cache(maxsize=None)
def _session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def get_session_factory(db_path: Path | None = None) -> sessionmaker[Session]:
    """返回与指定数据库 Engine 共享的同步 Session 工厂。"""
    return _session_factory(get_engine(db_path))


def clear_storage_engine_cache() -> None:
    """测试结束或应用停止使用 storage 后释放连接池和工厂缓存。"""
    with _engine_lock:
        for engine in _engines.values():
            engine.dispose()
        _session_factory.cache_clear()
        _engines.clear()


__all__ = ["clear_storage_engine_cache", "get_db_path", "get_engine",
           "get_session_factory", "now_iso"]
