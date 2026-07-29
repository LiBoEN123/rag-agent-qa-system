"""LangGraph Checkpointer 配置、工厂、持久化与线程标识测试。"""

from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

import pytest
from pydantic import ValidationError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.config import (
    validate_startup_environment,
)
from app.graph import builder as graph_builder
from app.graph import checkpointing


class _CheckpointState(TypedDict):
    value: str


@pytest.fixture(autouse=True)
def isolate_checkpoint_runtime():
    """让每个测试使用独立配置和 Saver 生命周期。"""
    graph_builder.reset_agent_graph_runtime()
    yield
    graph_builder.reset_agent_graph_runtime()


def _build_checkpoint_graph(saver):
    builder = StateGraph(_CheckpointState)
    builder.add_node(
        "finish",
        lambda state: {"value": f"{state['value']}-done"},
    )
    builder.add_edge(START, "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=saver)


def test_checkpointer_mode_defaults_and_normalization(settings_factory):
    """空值默认关闭，合法值会去除空格并转为小写。"""
    assert settings_factory(langgraph_checkpointer="").langgraph_checkpointer == (
        "disabled"
    )
    assert settings_factory(
        langgraph_checkpointer="  MEMORY "
    ).langgraph_checkpointer == "memory"
    assert settings_factory(
        langgraph_checkpointer=" SQLite "
    ).langgraph_checkpointer == "sqlite"


def test_invalid_checkpointer_mode_has_clear_error(settings_factory):
    """非法模式不会静默回退。"""
    with pytest.raises(ValidationError):
        settings_factory(langgraph_checkpointer="redis")


def test_settings_rejects_sqlite_directory_path(settings_factory, tmp_path):
    """启用 SQLite 时，目录不能被误当作数据库文件。"""
    settings = settings_factory(
        langgraph_checkpointer="sqlite",
        langgraph_checkpoint_db_path=tmp_path,
    )
    with pytest.raises(ValueError, match="必须指向文件"):
        validate_startup_environment(settings)


def test_disabled_returns_none_without_creating_database(
    settings_factory,
    tmp_path,
):
    """disabled 模式不创建 Saver、连接、目录或数据库文件。"""
    db_path = tmp_path / "missing" / "checkpoint.sqlite3"
    settings = settings_factory(
        langgraph_checkpointer="disabled",
        langgraph_checkpoint_db_path=db_path,
    )

    assert checkpointing.get_checkpointer(settings) is None
    assert checkpointing.get_checkpointer(settings) is None
    checkpointing.close_checkpointer()
    checkpointing.close_checkpointer()
    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_memory_saver_is_reused_and_reset_creates_new_instance(
    settings_factory,
    tmp_path,
):
    """Memory Saver 在同一生命周期复用，重置后换成新实例且不建文件。"""
    db_path = tmp_path / "unused.sqlite3"
    settings = settings_factory(
        langgraph_checkpointer="memory",
        langgraph_checkpoint_db_path=db_path,
    )

    first = checkpointing.get_checkpointer(settings)
    second = checkpointing.get_checkpointer(settings)
    assert isinstance(first, InMemorySaver)
    assert first is second
    assert not db_path.exists()

    graph_builder.reset_agent_graph_runtime()
    third = checkpointing.get_checkpointer(settings)
    assert isinstance(third, InMemorySaver)
    assert third is not first


def test_memory_saver_round_trip(settings_factory):
    """Memory Saver 可以读取一次真实 Graph 执行保存的状态。"""
    settings = settings_factory(langgraph_checkpointer="memory")
    saver = checkpointing.get_checkpointer(settings)
    graph = _build_checkpoint_graph(saver)
    config = {"configurable": {"thread_id": "memory-thread"}}

    assert graph.invoke({"value": "memory"}, config=config)["value"] == (
        "memory-done"
    )
    saved = saver.get_tuple(config)
    assert saved is not None
    assert saved.checkpoint["channel_values"]["value"] == "memory-done"


def test_sqlite_saver_creates_parent_and_survives_reopen(
    settings_factory,
    tmp_path,
):
    """SQLite Saver 自动建目录，关闭重开后仍可读取状态。"""
    db_path = tmp_path / "nested" / "checkpoint.sqlite3"
    settings = settings_factory(
        langgraph_checkpointer="sqlite",
        langgraph_checkpoint_db_path=db_path,
    )

    first = checkpointing.get_checkpointer(settings)
    assert isinstance(first, SqliteSaver)
    assert db_path.is_file()
    graph = _build_checkpoint_graph(first)
    config = {"configurable": {"thread_id": "sqlite-thread"}}
    assert graph.invoke({"value": "sqlite"}, config=config)["value"] == (
        "sqlite-done"
    )
    assert first.get_tuple(config) is not None

    graph_builder.reset_agent_graph_runtime()
    second = checkpointing.get_checkpointer(settings)
    assert isinstance(second, SqliteSaver)
    assert second is not first
    saved = second.get_tuple(config)
    assert saved is not None
    assert saved.checkpoint["channel_values"]["value"] == "sqlite-done"


def test_sqlite_saver_handles_parallel_distinct_threads(
    settings_factory,
    tmp_path,
):
    """共享同步 Saver 可由多个线程写入彼此独立的 Graph thread。"""
    db_path = tmp_path / "parallel.sqlite3"
    settings = settings_factory(
        langgraph_checkpointer="sqlite",
        langgraph_checkpoint_db_path=db_path,
    )
    saver = checkpointing.get_checkpointer(settings)
    graph = _build_checkpoint_graph(saver)

    def invoke(index: int) -> str:
        config = {"configurable": {"thread_id": f"parallel-{index}"}}
        return graph.invoke({"value": str(index)}, config=config)["value"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(invoke, range(8)))

    assert results == [f"{index}-done" for index in range(8)]
    for index in range(8):
        config = {"configurable": {"thread_id": f"parallel-{index}"}}
        assert saver.get_tuple(config) is not None


def test_sqlite_initialization_error_is_not_silently_downgraded(
    settings_factory,
    tmp_path,
):
    """父路径是文件时给出清晰错误，不回退到 Memory Saver。"""
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("fixture", encoding="utf-8")
    settings = settings_factory(
        langgraph_checkpointer="sqlite",
        langgraph_checkpoint_db_path=parent_file / "checkpoint.sqlite3",
    )

    with pytest.raises(RuntimeError, match="无法打开 LangGraph Checkpoint"):
        checkpointing.get_checkpointer(settings)


def test_checkpoint_thread_id_is_stable_safe_and_request_scoped():
    """thread ID 不暴露 session，并由每次请求的 trace 保证隔离。"""
    first = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-123",
        session_id="sensitive/session value",
    )
    same = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-123",
        session_id="sensitive/session value",
    )
    other = checkpointing.build_checkpoint_thread_id(
        trace_id="trace-456",
        session_id="sensitive/session value",
    )
    generated = checkpointing.build_checkpoint_thread_id(
        trace_id="",
        session_id="sensitive/session value",
    )
    long_value = checkpointing.build_checkpoint_thread_id(
        trace_id="trace/../" * 1000,
        session_id="session " * 1000,
    )

    assert first == same
    assert first != other
    assert first.startswith("agent-run-")
    assert len(first) == 50
    assert len(generated) == 50
    assert len(long_value) == 50
    assert first.replace("agent-run-", "").isalnum()
    assert "sensitive" not in first
    assert all(char not in long_value for char in " /\\..")
