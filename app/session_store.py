from collections import defaultdict, deque


_MAX_HISTORY_MESSAGES = 6
_ALLOWED_ROLES = {"user", "assistant"}
_SESSIONS = defaultdict(lambda: deque(maxlen=_MAX_HISTORY_MESSAGES))


def _normalize_session_id(session_id: str | None) -> str:
    """规整会话 ID，空值表示不启用历史。"""
    return (session_id or "").strip()


def get_history(session_id: str | None) -> list[dict]:
    """读取指定会话最近几条历史消息。"""
    normalized_session_id = _normalize_session_id(session_id)
    if not normalized_session_id:
        return []

    return [dict(message) for message in _SESSIONS.get(normalized_session_id, [])]


def append_message(session_id: str | None, role: str, content: str) -> None:
    """向指定会话追加一条用户或助手消息。"""
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"invalid role: {role}")

    normalized_session_id = _normalize_session_id(session_id)
    normalized_content = (content or "").strip()

    if not normalized_session_id or not normalized_content:
        return

    _SESSIONS[normalized_session_id].append({
        "role": role,
        "content": normalized_content,
    })


def clear_session(session_id: str | None) -> None:
    """清理指定会话历史，主要用于测试或手动重置。"""
    normalized_session_id = _normalize_session_id(session_id)
    if not normalized_session_id:
        return

    _SESSIONS.pop(normalized_session_id, None)
