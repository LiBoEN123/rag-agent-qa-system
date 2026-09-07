"""使用共享 SQLAlchemy Session 保存、查询和清理会话消息。"""

from pathlib import Path

from sqlalchemy import delete, insert, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.storage.database import get_session_factory, init_db, messages, now_iso, sessions


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

    init_db(db_path)
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

    init_db(db_path)
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

    init_db(db_path)
    with get_session_factory(db_path).begin() as session:
        session.execute(delete(messages).where(messages.c.session_id == normalized_session_id))
        session.execute(delete(sessions).where(sessions.c.session_id == normalized_session_id))


__all__ = ["append_message_sqlite", "clear_session_sqlite", "get_history_sqlite"]
