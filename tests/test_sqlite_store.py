import sqlite3

import pytest


def test_init_db_creates_tables(tmp_path):
    """验证初始化数据库会创建全部持久化表。"""
    from app.storage.database import init_db

    db_path = tmp_path / "test.sqlite"
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

    names = {row[0] for row in rows}

    assert "sessions" in names
    assert "messages" in names
    assert "agent_runs" not in names
    assert "agent_steps" not in names
    assert "ingestion_candidates" in names
    assert "ingestion_index_runs" in names


def test_sqlite_memory_append_and_get_history(tmp_path):
    """验证 SQLite 会话消息可以按写入顺序读取。"""
    from app.storage.memory import append_message_sqlite, get_history_sqlite

    db_path = tmp_path / "test.sqlite"

    append_message_sqlite("s1", "user", "你好", db_path=db_path)
    append_message_sqlite("s1", "assistant", "你好，有什么可以帮你？", db_path=db_path)

    history = get_history_sqlite("s1", db_path=db_path)

    assert history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]


def test_sqlite_memory_get_history_limit_order(tmp_path):
    """验证历史查询仅返回最近消息且保持旧到新顺序。"""
    from app.storage.memory import append_message_sqlite, get_history_sqlite

    db_path = tmp_path / "test.sqlite"

    for i in range(10):
        append_message_sqlite("s1", "user", f"q{i}", db_path=db_path)

    history = get_history_sqlite("s1", limit=3, db_path=db_path)

    assert [item["content"] for item in history] == ["q7", "q8", "q9"]


def test_clear_session_sqlite(tmp_path):
    """验证清理会话时不会影响其他 session。"""
    from app.storage.memory import (
        append_message_sqlite,
        clear_session_sqlite,
        get_history_sqlite,
    )

    db_path = tmp_path / "test.sqlite"

    append_message_sqlite("s1", "user", "q1", db_path=db_path)
    append_message_sqlite("s2", "user", "q2", db_path=db_path)

    clear_session_sqlite("s1", db_path=db_path)

    assert get_history_sqlite("s1", db_path=db_path) == []
    assert get_history_sqlite("s2", db_path=db_path) == [
        {"role": "user", "content": "q2"}
    ]


def test_append_message_sqlite_rejects_invalid_role(tmp_path):
    """验证 SQLite 会话存储拒绝非法消息角色。"""
    from app.storage.memory import append_message_sqlite

    db_path = tmp_path / "test.sqlite"

    with pytest.raises(ValueError):
        append_message_sqlite("s1", "system", "bad", db_path=db_path)
