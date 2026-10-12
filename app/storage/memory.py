"""使用共享 SQLAlchemy Session 保存、查询和清理会话消息。"""

from pathlib import Path

from sqlalchemy import Column, Index, Integer, MetaData, Table, Text, delete, insert, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.storage.database import get_engine, get_session_factory, now_iso


_metadata = MetaData()

# nullable=True 保留旧 SQLite PRIMARY KEY 的 PRAGMA 标记；不新增 FK。
sessions = Table(
    "sessions", _metadata,
    Column("session_id", Text, primary_key=True, nullable=True),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
messages = Table(
    "messages", _metadata,
    Column("id", Integer, primary_key=True, nullable=True),
    Column("session_id", Text, nullable=False),
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Index("idx_messages_session_id_id", "session_id", "id"),
    sqlite_autoincrement=True,
)


def init_session_storage(db_path: Path | None = None) -> None:
    """只初始化 Chat session schema，不触及 ingestion 表或兼容迁移。"""
    engine = get_engine(db_path)
    with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        _metadata.create_all(connection)


_ALLOWED_ROLES = {"user", "assistant"}


def append_message_sqlite(
    session_id: str | None,
    role: str,
    content: str,
    db_path: Path | None = None,
) -> None:
    """把一条会话消息写入 SQLite。"""
    normalized_session_id = (session_id or "").strip()
    normalized_content = (content or "").strip()
    if not normalized_session_id or not normalized_content:
        return
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"invalid role: {role}")

    init_session_storage(db_path)
    timestamp = now_iso()
    with get_session_factory(db_path).begin() as session:
        session.execute(
            sqlite_insert(sessions).values(
                session_id=normalized_session_id, created_at=timestamp, updated_at=timestamp,
            ).on_conflict_do_update(
                index_elements=["session_id"], set_={"updated_at": timestamp},
            )
        )
        session.execute(insert(messages).values(
            session_id=normalized_session_id, role=role,
            content=normalized_content, created_at=timestamp,
        ))


def get_history_sqlite(
    session_id: str | None,
    limit: int = 6,
    db_path: Path | None = None,
) -> list[dict]:
    """按旧到新顺序返回会话最近的消息。"""
    normalized_session_id = (session_id or "").strip()
    if not normalized_session_id or limit <= 0:
        return []

    init_session_storage(db_path)
    with get_session_factory(db_path)() as session:
        rows = session.execute(
            select(messages.c.role, messages.c.content)
            .where(messages.c.session_id == normalized_session_id)
            .order_by(messages.c.id.desc()).limit(int(limit))
        ).mappings().all()
    return [dict(row) for row in reversed(rows)]


def clear_session_sqlite(
    session_id: str | None,
    db_path: Path | None = None,
) -> None:
    """只删除指定 session 的消息和会话记录。"""
    normalized_session_id = (session_id or "").strip()
    if not normalized_session_id:
        return

    init_session_storage(db_path)
    with get_session_factory(db_path).begin() as session:
        session.execute(delete(messages).where(messages.c.session_id == normalized_session_id))
        session.execute(delete(sessions).where(sessions.c.session_id == normalized_session_id))


__all__ = ["append_message_sqlite", "clear_session_sqlite", "get_history_sqlite"]
