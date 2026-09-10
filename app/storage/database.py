"""Application SQLite 的共享 SQLAlchemy schema、Engine 和 Session。"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock

from sqlalchemy import Column, Index, Integer, Table, Text, create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


# 作为应用存储模型的 SQLAlchemy 声明基类，统一持有表元数据。
class Base(DeclarativeBase): ...


# nullable=True 保留旧 SQLite PRIMARY KEY 的 PRAGMA 标记；不新增 FK。
sessions = Table(
    "sessions", Base.metadata,
    Column("session_id", Text, primary_key=True, nullable=True),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
messages = Table(
    "messages", Base.metadata,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("session_id", Text, nullable=False),
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Index("idx_messages_session_id_id", "session_id", "id"),
    sqlite_autoincrement=True,
)


# 映射入库候选表，保存查询字段和完整的候选契约 JSON。
class _IngestionCandidateRecord(Base):
    __tablename__ = "ingestion_candidates"

    candidate_id: Mapped[str] = mapped_column(Text, primary_key=True, nullable=True)
    status: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(Text)
    source_uri: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    requested_at: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[str] = mapped_column(Text)
    candidate_json: Mapped[str] = mapped_column(Text)
    __table_args__ = (
        Index("idx_ingestion_candidates_status_requested", status, requested_at.desc()),
        Index("idx_ingestion_candidates_source_requested", source_type, requested_at.desc()),
    )


# 映射索引运行表，保存执行状态、时间、变更计数和错误信息。
class _IngestionIndexRunRecord(Base):
    __tablename__ = "ingestion_index_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=True)
    candidate_id: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    started_at: Mapped[str] = mapped_column(Text)
    completed_at: Mapped[str | None] = mapped_column(Text)
    added: Mapped[int | None] = mapped_column(Integer)
    updated: Mapped[int | None] = mapped_column(Integer)
    deleted: Mapped[int | None] = mapped_column(Integer)
    skipped: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        Index("idx_ingestion_index_runs_candidate_id", candidate_id, id.desc()),
        Index("idx_ingestion_index_runs_status_id", status, id.desc()),
        {"sqlite_autoincrement": True},
    )


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


def init_db(db_path: Path | None = None) -> None:
    """创建当前应用声明的全部表及索引，保留已有数据。"""
    Base.metadata.create_all(get_engine(db_path))


def clear_storage_engine_cache() -> None:
    """测试结束或应用停止使用 storage 后释放连接池和工厂缓存。"""
    with _engine_lock:
        for engine in _engines.values():
            engine.dispose()
        _session_factory.cache_clear()
        _engines.clear()


__all__ = ["clear_storage_engine_cache", "get_db_path", "get_engine",
           "get_session_factory", "init_db", "now_iso"]
