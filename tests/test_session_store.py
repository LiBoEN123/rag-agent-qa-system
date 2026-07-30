import pytest

from app.storage.session import append_message, clear_session, get_history


@pytest.fixture(autouse=True)
def use_temporary_sqlite_database(settings_env, tmp_path):
    """让每个 session_store 测试使用独立的临时 SQLite 数据库。"""
    settings_env(
        sqlite_db_path=tmp_path / "session_store.sqlite",
    )


def test_session_store_ignores_empty_session_id():
    """验证空 session_id 不保存历史。"""
    append_message(None, "user", "hello")
    append_message("", "assistant", "hi")

    assert get_history(None) == []
    assert get_history("") == []


def test_session_store_appends_and_reads_history():
    """验证 append 后可以读取历史。"""
    clear_session("s_append")

    append_message("s_append", "user", "q1")
    append_message("s_append", "assistant", "a1")

    assert get_history("s_append") == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]

    clear_session("s_append")


def test_session_store_keeps_recent_messages():
    """验证每个 session 最多保留最近 6 条消息。"""
    clear_session("s_recent")

    for i in range(10):
        append_message("s_recent", "user", f"q{i}")

    history = get_history("s_recent")
    assert len(history) == 6
    assert history[0]["content"] == "q4"
    assert history[-1]["content"] == "q9"

    clear_session("s_recent")


def test_session_store_clear_session():
    """验证 clear_session 可以清理历史。"""
    append_message("s_clear", "user", "q1")
    clear_session("s_clear")

    assert get_history("s_clear") == []


def test_session_store_rejects_invalid_role():
    """验证非法 role 会被拒绝。"""
    with pytest.raises(ValueError):
        append_message("s_bad_role", "system", "content")

    clear_session("s_bad_role")
