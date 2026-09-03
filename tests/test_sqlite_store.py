import pytest


def test_init_db_creates_tables(tmp_path):
    """验证初始化数据库会创建全部持久化表。"""
    from app.storage.database import get_connection, init_db

    db_path = tmp_path / "test.sqlite"
    init_db(db_path)

    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

    names = {row["name"] for row in rows}

    assert "sessions" in names
    assert "messages" in names
    assert "agent_runs" in names
    assert "agent_steps" in names
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


def test_save_agent_trace_writes_run_and_steps(tmp_path):
    """验证 Agent 运行摘要和步骤能够完整写入 SQLite。"""
    from app.storage.traces import get_agent_run, get_agent_steps, save_agent_trace

    db_path = tmp_path / "test.sqlite"

    tool_trace = [
        {
            "step": "decide",
            "action": "search_knowledge_base",
            "tool_name": "search_knowledge_base",
            "tool_args": {"query": "测试", "top_k": 5},
            "status": "success",
        },
        {
            "step": "tool_call",
            "step_id": 1,
            "action": "tool_call",
            "tool_name": "search_knowledge_base",
            "tool_args": {"query": "测试", "top_k": 5},
            "tool_status": "success",
            "success": True,
            "cost_ms": 123,
        },
    ]

    save_agent_trace(
        trace_id="trace_001",
        session_id="session_001",
        question="测试问题",
        rewritten_question="测试问题",
        final_answer="测试答案",
        status="OK",
        reason="ok",
        tool_trace=tool_trace,
        total_cost_ms=456,
        db_path=db_path,
    )

    run = get_agent_run("trace_001", db_path=db_path)
    steps = get_agent_steps("trace_001", db_path=db_path)

    assert run is not None
    assert run["trace_id"] == "trace_001"
    assert run["session_id"] == "session_001"
    assert run["question"] == "测试问题"
    assert run["final_answer"] == "测试答案"
    assert run["status"] == "OK"
    assert run["total_cost_ms"] == 456

    assert len(steps) == 2
    assert steps[0]["trace_id"] == "trace_001"
    assert steps[1]["tool_name"] == "search_knowledge_base"
    assert steps[1]["cost_ms"] == 123


def test_save_agent_trace_replaces_existing_steps(tmp_path):
    """验证重复保存同一 trace 时旧步骤会被替换。"""
    from app.storage.traces import get_agent_steps, save_agent_trace

    db_path = tmp_path / "test.sqlite"

    save_agent_trace(
        trace_id="trace_001",
        session_id=None,
        question="q",
        tool_trace=[{"step": "one"}],
        db_path=db_path,
    )

    save_agent_trace(
        trace_id="trace_001",
        session_id=None,
        question="q",
        tool_trace=[{"step": "two"}, {"step": "three"}],
        db_path=db_path,
    )

    steps = get_agent_steps("trace_001", db_path=db_path)

    assert len(steps) == 2
    assert "two" in steps[0]["observation_json"]
    assert "three" in steps[1]["observation_json"]
