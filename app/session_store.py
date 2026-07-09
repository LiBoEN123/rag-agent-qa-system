from app.memory_store import (
    append_message_sqlite,
    clear_session_sqlite,
    get_history_sqlite,
)
from app.untils import get_logger


_MAX_HISTORY_MESSAGES = 6
_ALLOWED_ROLES = {"user", "assistant"}
logger = get_logger("chat_service")


def get_history(session_id: str | None) -> list[dict]:
    """从 SQLite 读取指定会话最近几条历史消息。"""
    try:
        return get_history_sqlite(
            session_id,
            limit=_MAX_HISTORY_MESSAGES,
        )
    except Exception as exc:
        logger.warning("读取 SQLite 会话历史失败: %s", exc)
        return []


def append_message(session_id: str | None, role: str, content: str) -> None:
    """向 SQLite 会话历史追加一条用户或助手消息。"""
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"invalid role: {role}")

    try:
        append_message_sqlite(session_id, role, content)
    except Exception as exc:
        logger.warning("写入 SQLite 会话历史失败: %s", exc)


def clear_session(session_id: str | None) -> None:
    """从 SQLite 清理指定会话历史。"""
    try:
        clear_session_sqlite(session_id)
    except Exception as exc:
        logger.warning("清理 SQLite 会话历史失败: %s", exc)
