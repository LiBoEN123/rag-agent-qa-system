import pytest

from app.storage.memory import (
    append_message_sqlite,
    clear_session_sqlite,
    get_history_sqlite,
)


@pytest.fixture(autouse=True)
def use_temporary_sqlite_database(settings_env, tmp_path):
    """让每个 session_store 测试使用独立的临时 SQLite 数据库。"""
    settings_env(
        sqlite_db_path=tmp_path / "session_store.sqlite",
    )


def test_session_store_ignores_empty_session_id():
    """验证空 session_id 不保存历史。"""
    append_message_sqlite(None, "user", "hello")
    append_message_sqlite("", "assistant", "hi")
    append_message_sqlite("   ", "user", "hello")

    assert get_history_sqlite(None) == []
    assert get_history_sqlite("") == []
    assert get_history_sqlite("   ") == []


def test_session_store_appends_and_reads_history():
    """验证 append 后可以读取历史。"""
    clear_session_sqlite("s_append")

    append_message_sqlite(" s_append ", "user", " q1 ")
    append_message_sqlite("s_append", "assistant", "a1")
    append_message_sqlite("s_append", "user", "")
    append_message_sqlite("s_append", "assistant", "   ")

    assert get_history_sqlite(" s_append ") == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]

    clear_session_sqlite("s_append")


def test_session_store_keeps_recent_messages():
    """验证每个 session 最多保留最近 6 条消息。"""
    clear_session_sqlite("s_recent")

    for i in range(10):
        append_message_sqlite("s_recent", "user", f"q{i}")

    history = get_history_sqlite("s_recent")
    assert history == [
        {"role": "user", "content": f"q{i}"} for i in range(4, 10)
    ]

    clear_session_sqlite("s_recent")


def test_session_store_clear_session():
    """验证只清理指定会话，空 session_id 不影响已有历史。"""
    append_message_sqlite("s_clear", "user", "q1")
    append_message_sqlite("s_keep", "user", "keep")
    clear_session_sqlite(" s_clear ")
    clear_session_sqlite(None)
    clear_session_sqlite("")

    assert get_history_sqlite("s_clear") == []
    assert get_history_sqlite("s_keep") == [{"role": "user", "content": "keep"}]


def test_session_store_rejects_invalid_role():
    """验证非法 role 会被拒绝。"""
    with pytest.raises(ValueError):
        append_message_sqlite("s_bad_role", "system", "content")

    clear_session_sqlite("s_bad_role")
