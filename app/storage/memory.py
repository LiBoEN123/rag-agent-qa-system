"""会话记忆持久化模块。

使用 SQLite 保存、查询和清理会话消息，为多轮问题改写提供历史上下文。
"""

from pathlib import Path

from app.storage.database import get_connection, init_db, now_iso


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
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id)
                DO UPDATE SET updated_at=excluded.updated_at
                """,
                (normalized_session_id, timestamp, timestamp),
            )
            conn.execute(
                """
                INSERT INTO messages(session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized_session_id, role, normalized_content, timestamp),
            )
    finally:
        conn.close()


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
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (normalized_session_id, int(limit)),
        ).fetchall()
    finally:
        conn.close()

    return [
        {"role": row["role"], "content": row["content"]}
        for row in reversed(rows)
    ]


def clear_session_sqlite(
    session_id: str | None,
    db_path: Path | None = None,
) -> None:
    """只删除指定 session 的消息和会话记录。"""
    normalized_session_id = (session_id or "").strip()
    if not normalized_session_id:
        return

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (normalized_session_id,),
            )
            conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (normalized_session_id,),
            )
    finally:
        conn.close()


__all__ = [
    "append_message_sqlite",
    "clear_session_sqlite",
    "get_history_sqlite",
]
